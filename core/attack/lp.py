"""LP-Attack: Layer-wise Poisoning via Backdoor-Critical (BC) layers (single-file v2)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate, StateDict, load_model_state, split_model_state
from core.utils import functional as F
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model

DEFAULT_BLS_EPS = 0.005
LayerScore = Tuple[float, str]


@dataclass(frozen=True)
class AtomicGroup:
    """Selectable module group: Conv + paired BN (weight/bias) or Linear (weight/bias)."""

    name: str
    keys: Tuple[str, ...]


@torch.no_grad()
def _eval_acc(model: nn.Module, loader: Iterable[Any]) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        bx, by = batch[0], batch[1]
        preds = model(bx).argmax(dim=1)
        correct += int(preds.eq(by).sum().item())
        total += int(by.size(0))
    if was_training:
        model.train()
    return (correct / total) if total > 0 else 0.0


def build_atomic_groups(
    model: nn.Module,
    available_params: Mapping[str, torch.Tensor],
) -> List[AtomicGroup]:
    """Group learnable parameter keys into Conv(+BN) and Linear atomic units."""
    avail = set(available_params.keys())
    conv_to_bn = dict(F.find_conv_bn_pairs(model))
    conv_names: List[str] = []
    linear_names: List[str] = []

    for name, mod in model.named_modules():
        if not name:
            continue
        if isinstance(mod, nn.Conv2d):
            conv_names.append(name)
        elif isinstance(mod, nn.Linear):
            linear_names.append(name)

    groups: List[AtomicGroup] = []
    for conv_name in conv_names:
        keys = sorted(k for k in avail if k.startswith(f"{conv_name}."))
        bn_name = conv_to_bn.get(conv_name)

        if bn_name is not None:
            for suffix in ("weight", "bias"):
                k = f"{bn_name}.{suffix}"
                if k in avail and k not in keys:
                    keys.append(k)
            bn_tail = bn_name.split(".")[-1]
            group_name = (
                f"{conv_name}+bn"
                if (bn_tail == "1" and conv_name.endswith("shortcut.0"))
                else f"{conv_name}+{bn_tail}"
            )
        else:
            group_name = conv_name

        if keys:
            groups.append(AtomicGroup(name=group_name, keys=tuple(keys)))

    for lin_name in linear_names:
        keys = [k for k in (f"{lin_name}.weight", f"{lin_name}.bias") if k in avail]
        if keys:
            groups.append(AtomicGroup(name=lin_name, keys=tuple(keys)))

    return groups


def expand_group_keys(
    group_names: Sequence[str],
    group_map: Mapping[str, Sequence[str]],
) -> List[str]:
    out: List[str] = []
    seen = set()
    for g in group_names:
        for k in group_map.get(g, ()):
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def _swap_params(
    base_params: Mapping[str, torch.Tensor],
    src_params: Mapping[str, torch.Tensor],
    keys: Iterable[str],
) -> StateDict:
    mixed = {k: v for k, v in base_params.items()}
    for k in keys:
        if k in src_params:
            mixed[k] = src_params[k]
    return mixed


@torch.no_grad()
def _run_fls(
    model: nn.Module,
    benign_params: Mapping[str, torch.Tensor],
    mal_params: Mapping[str, torch.Tensor],
    mal_buffers: Mapping[str, torch.Tensor],
    bsr_mal: float,
    val_loader: Iterable[Any],
    groups: Sequence[AtomicGroup],
) -> Tuple[List[str], List[float], Dict[str, Tuple[str, ...]]]:
    group_map = {g.name: g.keys for g in groups}
    names: List[str] = []
    deltas: List[float] = []
    for g in groups:
        mixed = _swap_params(mal_params, benign_params, g.keys)
        load_model_state(model, mixed, mal_buffers)
        bsr = _eval_acc(model, val_loader)
        names.append(g.name)
        deltas.append(float(bsr) - float(bsr_mal))
    return names, deltas, group_map


@torch.no_grad()
def _run_bls(
    model: nn.Module,
    key_arr: Sequence[str],
    val_arr: Sequence[float],
    benign_params: Mapping[str, torch.Tensor],
    benign_buffers: Mapping[str, torch.Tensor],
    mal_params: Mapping[str, torch.Tensor],
    bsr_mal: float,
    val_loader: Iterable[Any],
    group_map: Mapping[str, Sequence[str]],
    *,
    tau: float = 0.95,
    eps: float = DEFAULT_BLS_EPS,
) -> Tuple[List[str], float]:
    if not key_arr:
        return [], 0.0

    threshold = float(bsr_mal) * float(tau)
    delta = {key_arr[i]: float(val_arr[i]) for i in range(len(key_arr))}
    neg = [g for g in key_arr if delta[g] < -eps]
    zero = [g for g in key_arr if abs(delta[g]) <= eps]
    pos = [g for g in key_arr if delta[g] > eps]

    def _eval_sel(sel: Sequence[str]) -> float:
        mixed = _swap_params(benign_params, mal_params, expand_group_keys(sel, group_map))
        load_model_state(model, mixed, benign_buffers)
        return _eval_acc(model, val_loader)

    def _l2(name: str) -> float:
        return sum(
            float((mal_params[k].float() - benign_params[k].float()).pow(2).sum().item())
            for k in group_map.get(name, ())
            if k in benign_params and k in mal_params
        )

    selected: List[str] = []
    selected_set = set()
    cur_bsr = _eval_sel(selected)

    # Phase 1: Warm-start greedy on FLS-negative groups
    for g in sorted(neg, key=lambda n: (delta[n], n)):
        if cur_bsr >= threshold:
            break
        selected.append(g)
        selected_set.add(g)
        cur_bsr = _eval_sel(selected)

    def _best_single(cands: Sequence[str]) -> Tuple[Optional[str], float, float]:
        best_g: Optional[str] = None
        best_gain = float("-inf")
        best_bsr = cur_bsr
        best_l2 = float("-inf")
        for g in cands:
            bsr = _eval_sel(selected + [g])
            gain = bsr - cur_bsr
            l2 = _l2(g)
            if best_g is None or gain > best_gain + 1e-15 or (abs(gain - best_gain) <= 1e-15 and l2 > best_l2):
                best_g, best_gain, best_bsr, best_l2 = g, gain, bsr, l2
        return best_g, best_gain, best_bsr

    def _best_pair(cands: Sequence[str]) -> Tuple[Optional[Tuple[str, str]], float, float]:
        best_p: Optional[Tuple[str, str]] = None
        best_gain = float("-inf")
        best_bsr = cur_bsr
        best_l2 = float("-inf")
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a, b = cands[i], cands[j]
                bsr = _eval_sel(selected + [a, b])
                gain = bsr - cur_bsr
                l2 = _l2(a) + _l2(b)
                if best_p is None or gain > best_gain + 1e-15 or (abs(gain - best_gain) <= 1e-15 and l2 > best_l2):
                    best_p, best_gain, best_bsr, best_l2 = (a, b), gain, bsr, l2
        return best_p, best_gain, best_bsr

    # Phase 2: Plateau / complementary expansion
    candidates = [g for g in zero if g not in selected_set]
    pos_pending = [g for g in pos if g not in selected_set]
    pos_merged = False

    while cur_bsr < threshold:
        pool = [g for g in candidates if g not in selected_set]
        if not pool:
            if not pos_merged and pos_pending:
                candidates = list(dict.fromkeys(candidates + pos_pending))
                pos_merged = True
                continue
            break

        g_star, gain, new_bsr = _best_single(pool)
        if g_star is not None and gain > eps:
            selected.append(g_star)
            selected_set.add(g_star)
            cur_bsr = new_bsr
            continue

        if not pos_merged and pos_pending:
            candidates = list(dict.fromkeys(candidates + pos_pending))
            pos_merged = True
            continue

        pair, p_gain, p_bsr = _best_pair(pool)
        if pair is not None and p_gain > eps:
            for g in pair:
                if g not in selected_set:
                    selected.append(g)
                    selected_set.add(g)
            cur_bsr = p_bsr
            continue

        if g_star is None:
            break
        selected.append(g_star)
        selected_set.add(g_star)
        cur_bsr = new_bsr

    # Phase 3: Backtracking prune
    changed = True
    while changed and selected:
        changed = False
        for idx in range(len(selected) - 1, -1, -1):
            trial = selected[:idx] + selected[idx + 1 :]
            bsr = _eval_sel(trial)
            if bsr >= threshold:
                selected = trial
                cur_bsr = bsr
                changed = True
                break

    return selected, float(cur_bsr)


def craft_lp_params(
    benign_params: Mapping[str, torch.Tensor],
    mal_params: Mapping[str, torch.Tensor],
    global_params: Mapping[str, torch.Tensor],
    selected_keys: Iterable[str],
    lambda_scale: float = 1.0,
) -> StateDict:
    selected = set(selected_keys)
    lam = float(lambda_scale)
    relu_term = max(0.0, 1.0 - lam)
    crafted: StateDict = {}
    for k, b_val in benign_params.items():
        if k in selected and k in mal_params and k in global_params:
            g = global_params[k]
            crafted[k] = (g + lam * (mal_params[k] - g) + relu_term * (b_val - g)).detach().clone()
        else:
            crafted[k] = b_val.detach().clone()
    return crafted


@ATTACK_REGISTRY.register("lp")
class LPAttack(BaseAttacker):
    """Layer-wise Poisoning via reference FLS + warm-start greedy BLS on atomic modules."""

    def __init__(
        self,
        tau: float = 0.95,
        val_ratio: float = 0.2,
        lambda_scale: float = 1.0,
        ref_benign_acc_threshold: float = 0.80,
        ref_malicious_asr_threshold: float = 0.90,
        ref_malicious_max_epochs: int = 32,
        eps: float = DEFAULT_BLS_EPS,
        log_selected_layers: bool = False,
        layer_score_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.tau = float(tau)
        self.val_ratio = float(val_ratio)
        self.lambda_scale = float(lambda_scale)
        self.ref_benign_acc_threshold = float(ref_benign_acc_threshold)
        self.ref_malicious_asr_threshold = float(ref_malicious_asr_threshold)
        self.ref_malicious_max_epochs = max(1, int(ref_malicious_max_epochs))
        self.eps = float(eps)
        self.log_selected_layers = bool(log_selected_layers)
        self.layer_score_dir = layer_score_dir

    def _log_selection(self, ctx: ClientContext, summary: Mapping[str, Any]) -> None:
        if not self.log_selected_layers:
            return
        score_dir = self.layer_score_dir or (
            os.path.join(ctx.run_dir, "lp_layer_scores") if ctx.run_dir else None
        )
        if not score_dir:
            return
        os.makedirs(score_dir, exist_ok=True)
        path = os.path.join(score_dir, f"selections_client_{ctx.client_id}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def step(self, ctx: ClientContext) -> ClientUpdate:
        # 5-line zero-copy split & loader assembly!
        train_view, val_view = ctx.data.split(self.val_ratio, seed=self.seed)
        clean_train_ld = ctx.loader(train_view, seed=self.seed)
        poison_train_ld = ctx.loader(train_view, poisoner=self.poisoner, seed=self.seed)
        poison_val_ld = ctx.loader(val_view, poisoner=self.poisoner, mode="eval", shuffle=False)
        clean_full_ld = ctx.loader(seed=self.seed)
        poison_full_ld = ctx.loader(poisoner=self.poisoner, seed=self.seed)

        # 1. Reference long-train on train split to locate BC atomic groups
        ref_ben_epochs = 0
        acc = _eval_acc(ctx.model, clean_train_ld)
        while acc < self.ref_benign_acc_threshold and ref_ben_epochs < 64:
            train_model(ctx.model, clean_train_ld, ctx.train_cfg, epochs=1)
            ref_ben_epochs += 1
            acc = _eval_acc(ctx.model, clean_train_ld)
        ref_benign_params, ref_benign_buffers = split_model_state(ctx.model)

        ref_mal_epochs = 0
        asr = _eval_acc(ctx.model, poison_val_ld)
        while asr < self.ref_malicious_asr_threshold and ref_mal_epochs < self.ref_malicious_max_epochs:
            train_model(ctx.model, poison_train_ld, ctx.train_cfg, epochs=1)
            ref_mal_epochs += 1
            asr = _eval_acc(ctx.model, poison_val_ld)
        ref_mal_params, ref_mal_buffers = split_model_state(ctx.model)

        groups = build_atomic_groups(ctx.model, ref_benign_params)
        key_arr, val_arr, group_map = _run_fls(
            ctx.model, ref_benign_params, ref_mal_params, ref_mal_buffers, asr, poison_val_ld, groups
        )
        attack_groups, final_bsr = _run_bls(
            ctx.model,
            key_arr,
            val_arr,
            ref_benign_params,
            ref_benign_buffers,
            ref_mal_params,
            asr,
            poison_val_ld,
            group_map,
            tau=self.tau,
            eps=self.eps,
        )
        selected_keys = expand_group_keys(attack_groups, group_map)

        self._log_selection(
            ctx,
            {
                "round": ctx.round_idx,
                "client_id": ctx.client_id,
                "bsr_malicious": asr,
                "final_bsr": final_bsr,
                "ref_benign_epochs": ref_ben_epochs,
                "ref_malicious_epochs": ref_mal_epochs,
                "num_selected": len(attack_groups),
                "selected_layers": attack_groups,
            },
        )

        # 2. Local benign branch from Wg on full clean data (craft base + benign BN stats)
        ctx.reset_model()
        train_model(ctx.model, clean_full_ld, ctx.train_cfg)
        benign_params, benign_buffers = split_model_state(ctx.model)

        # 3. Local malicious branch from Wg on full poison data
        ctx.reset_model()
        mal_stats = train_model(ctx.model, poison_full_ld, ctx.train_cfg)
        mal_params, _ = split_model_state(ctx.model)

        # 4. Graft BC parameter groups onto local benign model
        crafted_params = craft_lp_params(
            benign_params, mal_params, ctx.initial_params, selected_keys, self.lambda_scale
        )
        param_delta = {k: crafted_params[k] - ctx.initial_params[k] for k in ctx.initial_params}

        return ctx.pack_update(
            param_delta=param_delta,
            buffers=benign_buffers,
            metrics={
                "loss": mal_stats.loss,
                "lp/num_selected": float(len(attack_groups)),
            },
        )
