"""SparAlignFL: locate-and-concentrate backdoor attack (single-file v2)."""
from __future__ import annotations

import csv
import os
import random
import re
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Set

import torch
import torch.nn as nn

from core.attack.base import BaseAttacker
from core.attack.lga import project_model_layerwise_
from core.types import ClientContext, ClientUpdate, split_model_state
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model
from data.loader import derive_stable_seed

LayerScore = Tuple[float, str]
_CONV_TO_BN_RE = re.compile(r"(^|\.)conv(\d+)$")
_CSV_LOCK = threading.Lock()


def _bn_module_names(model: nn.Module) -> Set[str]:
    return {
        name
        for name, mod in model.named_modules()
        if name and isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d))
    }


def _module_prefix(param_name: str) -> str:
    return param_name.rsplit(".", 1)[0] if "." in param_name else param_name


def _is_weight_layer(name: str, params: Mapping[str, torch.Tensor], bn_mods: Set[str]) -> bool:
    if not name.endswith(".weight") or name not in params:
        return False
    val = params[name]
    return val.ndim >= 2 and _module_prefix(name) not in bn_mods


def _eligible_weight_keys(params: Mapping[str, torch.Tensor], bn_mods: Set[str]) -> List[str]:
    return [k for k in params if _is_weight_layer(k, params, bn_mods)]


def _paired_bn_prefix(weight_key: str, bn_mods: Set[str]) -> Optional[str]:
    prefix = _module_prefix(weight_key)
    if prefix.endswith("shortcut.0"):
        cand = prefix[:-1] + "1"
        return cand if cand in bn_mods else None
    if _CONV_TO_BN_RE.search(prefix) is None:
        return None
    cand = _CONV_TO_BN_RE.sub(r"\1bn\2", prefix)
    return cand if cand in bn_mods else None


def _expand_scored_weight(
    weight_key: str,
    params: Mapping[str, torch.Tensor],
    bn_mods: Set[str],
) -> List[str]:
    ordered = _eligible_weight_keys(params, bn_mods)
    idx = ordered.index(weight_key) if weight_key in ordered else -1
    prev_key = ordered[idx - 1] if idx > 0 else None

    expanded: List[str] = []
    seen: Set[str] = set()
    for cand in (prev_key, weight_key):
        if cand is None:
            continue
        w_pref = _module_prefix(cand)
        bn_pref = _paired_bn_prefix(cand, bn_mods)
        for p_name in params:
            mod_name = _module_prefix(p_name)
            if mod_name == w_pref or (bn_pref is not None and mod_name == bn_pref):
                if p_name not in seen:
                    seen.add(p_name)
                    expanded.append(p_name)
    return expanded


@torch.no_grad()
def score_contrastive_layers(
    global_params: Mapping[str, torch.Tensor],
    benign_params: Mapping[str, torch.Tensor],
    malicious_params: Mapping[str, torch.Tensor],
    bn_mods: Set[str],
    eps: float = 1e-8,
) -> List[LayerScore]:
    scores: List[LayerScore] = []
    for name in _eligible_weight_keys(global_params, bn_mods):
        if name not in benign_params or name not in malicious_params:
            continue
        base = global_params[name].detach().float()
        db = benign_params[name].detach().to(device=base.device, dtype=torch.float32) - base
        dm = malicious_params[name].detach().to(device=base.device, dtype=torch.float32) - base
        num = (dm - db).square().sum()
        den = dm.square().sum() + db.square().sum() + float(eps)
        scores.append((float((num / den).item()), name))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


def select_critical_layers(
    ranked: Sequence[LayerScore],
    num_critical_layers: int,
    params: Mapping[str, torch.Tensor],
    bn_mods: Set[str],
    strategy: str = "expanded_topk",
    random_seed: Optional[int] = None,
) -> List[str]:
    if num_critical_layers <= 0 or not ranked:
        return []

    count = min(int(num_critical_layers), len(ranked))
    if strategy == "random_k":
        rng = random.Random(0 if random_seed is None else int(random_seed))
        return [w for _, w in rng.sample(list(ranked), count)]
    if strategy == "bottom_k":
        return [w for _, w in ranked[-count:]]
    if strategy == "weight_topk":
        return [w for _, w in ranked[:count]]

    selected: List[str] = []
    selected_set: Set[str] = set()
    for _, w_key in ranked:
        for k in _expand_scored_weight(w_key, params, bn_mods):
            if k not in selected_set:
                selected_set.add(k)
                selected.append(k)
        n_weights = sum(1 for k in selected if _is_weight_layer(k, params, bn_mods))
        if n_weights >= int(num_critical_layers):
            break
    return selected


@torch.no_grad()
def build_benign_active_masks(
    global_params: Mapping[str, torch.Tensor],
    benign_params: Mapping[str, torch.Tensor],
    critical_layers: Iterable[str],
    mask_ratio: float,
) -> Dict[str, torch.Tensor]:
    entries = []
    flat_scores = []
    for name in critical_layers:
        if name not in global_params or name not in benign_params:
            continue
        ref = benign_params[name].detach()
        base = global_params[name].detach().to(device=ref.device, dtype=torch.float32)
        delta_abs = (ref.float() - base).abs().reshape(-1)
        entries.append((name, ref, delta_abs.numel()))
        flat_scores.append(delta_abs.cpu())

    if not entries:
        return {}

    scores = torch.cat(flat_scores)
    blocked = torch.zeros(scores.numel(), dtype=torch.bool)
    num_masked = min(scores.numel(), int(scores.numel() * mask_ratio))
    if num_masked > 0:
        blocked[torch.topk(scores, k=num_masked, largest=True, sorted=False).indices] = True

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, ref, numel in entries:
        keep = ~blocked[offset : offset + numel]
        masks[name] = keep.reshape_as(ref).to(device=ref.device, dtype=ref.dtype)
        offset += numel
    return masks


def _log_sparalign_csvs(
    run_dir: str,
    round_idx: int,
    client_id: int,
    ranked: Sequence[LayerScore],
    critical_layers: Iterable[str],
    masks: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    os.makedirs(run_dir, exist_ok=True)
    crit_set = set(critical_layers)

    score_csv = os.path.join(run_dir, "sparalignfl_layer_scores.csv")
    mask_csv = os.path.join(run_dir, "sparalignfl_layer_masks.csv")
    n_total = 0
    n_masked = 0

    with _CSV_LOCK:
        write_s_hdr = not os.path.exists(score_csv) or os.path.getsize(score_csv) == 0
        with open(score_csv, "a", newline="", encoding="utf-8") as fs:
            writer = csv.DictWriter(
                fs, fieldnames=["round", "client_id", "rank", "layer", "score", "selected"]
            )
            if write_s_hdr:
                writer.writeheader()
            for rank, (score, name) in enumerate(ranked):
                writer.writerow(
                    {
                        "round": round_idx,
                        "client_id": client_id,
                        "rank": rank,
                        "layer": name,
                        "score": f"{score:.9g}",
                        "selected": int(name in crit_set),
                    }
                )

        if masks:
            write_m_hdr = not os.path.exists(mask_csv) or os.path.getsize(mask_csv) == 0
            with open(mask_csv, "a", newline="", encoding="utf-8") as fm:
                m_writer = csv.DictWriter(
                    fm, fieldnames=["round", "client_id", "layer", "n_total", "n_masked", "ratio"]
                )
                if write_m_hdr:
                    m_writer.writeheader()
                for name, mask in masks.items():
                    tot = int(mask.numel())
                    blk = int((mask == 0).sum().item())
                    n_total += tot
                    n_masked += blk
                    m_writer.writerow(
                        {
                            "round": round_idx,
                            "client_id": client_id,
                            "layer": name,
                            "n_total": tot,
                            "n_masked": blk,
                            "ratio": f"{(blk / tot if tot > 0 else 0.0):.6f}",
                        }
                    )

    return {
        "safl/mask_ratio": (n_masked / n_total) if n_total > 0 else 0.0,
        "safl/mask_n_masked": float(n_masked),
    }


@ATTACK_REGISTRY.register("sparalignfl")
class SparAlignFLAttack(BaseAttacker):
    """Two-stage contrastive critical-layer localization & concentrated backdoor training."""

    def __init__(
        self,
        num_critical_layers: int = 5,
        mask_ratio: float = 0.1,
        probe_epochs: int = 3,
        probe_poison_ratio: float = 1.0,
        eps: float = 1e-8,
        process_critical_layers: bool = True,
        process_non_critical_layers: bool = True,
        layer_selection_strategy: str = "expanded_topk",
        log_stats: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        strat = str(layer_selection_strategy).lower().replace("-", "_")
        if strat in {"topk", "top_k", "score_topk"}:
            strat = "weight_topk"
        elif strat in {"expanded_top_k", "default"}:
            strat = "expanded_topk"
        if strat not in {"expanded_topk", "random_k", "bottom_k", "weight_topk"}:
            raise ValueError(f"Unknown SparAlignFL layer_selection_strategy: {layer_selection_strategy!r}")

        self.num_critical_layers = int(num_critical_layers)
        self.mask_ratio = float(mask_ratio)
        self.probe_epochs = int(probe_epochs)
        self.probe_poison_ratio = float(probe_poison_ratio)
        self.eps = float(eps)
        self.process_critical_layers = bool(process_critical_layers)
        self.process_non_critical_layers = bool(process_non_critical_layers)
        self.layer_selection_strategy = strat
        self.log_stats = bool(log_stats)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        bn_mods = _bn_module_names(ctx.model)
        probe_poisoner = self.clone_poisoner(poison_ratio=self.probe_poison_ratio)

        # Stage I-a: Clean probe from Wg
        train_model(ctx.model, ctx.loader(seed=self.seed), ctx.train_cfg, epochs=self.probe_epochs)
        benign_params, _ = split_model_state(ctx.model)

        # Stage I-b: 100% poison probe from Wg
        ctx.reset_model()
        train_model(
            ctx.model,
            ctx.loader(poisoner=probe_poisoner, seed=self.seed),
            ctx.train_cfg,
            epochs=self.probe_epochs,
        )
        malicious_params, _ = split_model_state(ctx.model)

        # Contrastive layer ranking & critical mask construction
        ranked = score_contrastive_layers(
            ctx.initial_params, benign_params, malicious_params, bn_mods, eps=self.eps
        )
        r_seed = (
            derive_stable_seed(self.seed or 0, ctx.client_id, ctx.round_idx)
            if self.layer_selection_strategy == "random_k"
            else None
        )
        critical_layers = select_critical_layers(
            ranked,
            self.num_critical_layers,
            ctx.initial_params,
            bn_mods,
            strategy=self.layer_selection_strategy,
            random_seed=r_seed,
        )
        grad_masks = (
            build_benign_active_masks(
                ctx.initial_params, benign_params, critical_layers, self.mask_ratio
            )
            if self.process_critical_layers
            else {}
        )

        # Stage II: Retrain from Wg with coordinate masks on L_c and LGA projection outside L_c
        ctx.reset_model()
        prev_delta = ctx.prev_global_delta
        post_fn = (
            (
                lambda _ep, m: project_model_layerwise_(
                    m,
                    ctx.initial_params,
                    prev_delta,
                    eps=self.eps,
                    skip_keys=critical_layers,
                )
            )
            if self.process_non_critical_layers
            else None
        )
        stats = train_model(
            ctx.model,
            ctx.loader(poisoner=self.poisoner, seed=self.seed),
            ctx.train_cfg,
            grad_mask=grad_masks,
            post_epoch_fn=post_fn,
        )

        metrics: Dict[str, float] = {"loss": stats.loss}
        if self.log_stats and ctx.run_dir:
            metrics.update(
                _log_sparalign_csvs(
                    ctx.run_dir, ctx.round_idx, ctx.client_id, ranked, critical_layers, grad_masks
                )
            )
        return ctx.pack_update(metrics=metrics)
