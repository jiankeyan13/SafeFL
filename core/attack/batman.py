"""Batman: BadSP / AlphaEdit SVD null-space alignment attack (single-file v2)."""
from __future__ import annotations

import json
import logging
import math
import os
import random
import threading
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate, split_model_state
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model
from data.loader import derive_stable_seed

_LAYER_LOG_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)

LayerScore = Tuple[float, str]


def count_eligible_layers(state: Mapping[str, torch.Tensor]) -> int:
    return sum(1 for p in state.values() if torch.is_tensor(p) and p.ndim >= 2)


def resolve_num_selected_layers(
    eligible_count: int,
    num_selected_layers: int,
    selected_layer_ratio: Optional[float],
) -> int:
    if selected_layer_ratio is not None:
        if not 0.0 < selected_layer_ratio <= 1.0:
            raise ValueError(f"selected_layer_ratio must be in (0, 1], got {selected_layer_ratio}")
        if eligible_count <= 0:
            return 0
        k = int(math.floor(eligible_count * selected_layer_ratio + 0.5))
        return max(1, min(eligible_count, k))
    return max(0, min(eligible_count, int(num_selected_layers)))


@torch.no_grad()
def _score_by_deviation(
    poisoned: Mapping[str, torch.Tensor],
    global_params: Mapping[str, torch.Tensor],
) -> List[LayerScore]:
    scores: List[LayerScore] = []
    for name, p in poisoned.items():
        if p.ndim < 2 or name not in global_params:
            continue
        score = float(torch.norm(p.detach().float() - global_params[name].detach().float()).item())
        scores.append((score, name))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


@torch.no_grad()
def _score_by_diff_top(
    poisoned: Mapping[str, torch.Tensor],
    global_params: Mapping[str, torch.Tensor],
) -> List[LayerScore]:
    scores: List[LayerScore] = []
    for name, p in poisoned.items():
        if p.ndim <= 2 or name not in global_params:
            continue
        diff = torch.abs(p.detach().float() - global_params[name].detach().float())
        total = float(diff.sum().item())
        score = 0.0 if total == 0.0 else float(torch.mean(diff / total).item())
        scores.append((score, name))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores


def rank_critical_layers(
    poisoned: Mapping[str, torch.Tensor],
    global_params: Mapping[str, torch.Tensor],
    strategy: str = "global_deviation",
    rng: Optional[random.Random] = None,
) -> List[LayerScore]:
    if strategy == "diff_top":
        return _score_by_diff_top(poisoned, global_params)
    if strategy == "global_deviation":
        return _score_by_deviation(poisoned, global_params)
    if strategy == "random":
        if rng is None:
            raise ValueError("layer_selection='random' requires an RNG")
        names = [k for k, v in poisoned.items() if v.ndim >= 2]
        rng.shuffle(names)
        n = len(names)
        return [(float(n - i), name) for i, name in enumerate(names)]
    raise ValueError(f"Unknown Batman layer_selection strategy: {strategy!r}")


def _shape_restruct(param: torch.Tensor) -> torch.Tensor:
    if param.ndim > 2:
        return param.reshape(param.shape[0] * param.shape[2], param.shape[1] * param.shape[3])
    return param


@torch.no_grad()
def _safe_svd(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    try:
        res = torch.linalg.svd(matrix, full_matrices=False)
        if all(torch.isfinite(v).all() for v in res):
            return res
        raise RuntimeError("SVD non-finite")
    except RuntimeError as first_err:
        if matrix.device.type == "cuda":
            try:
                res = torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")
                if all(torch.isfinite(v).all() for v in res):
                    return res
            except RuntimeError:
                pass
        work = torch.nan_to_num(
            matrix.detach().to(device="cpu", dtype=torch.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        scale = work.abs().amax()
        if torch.isfinite(scale) and scale > 0:
            work = work / scale
        try:
            u, s, vh = torch.linalg.svd(work, full_matrices=False)
            if not all(torch.isfinite(v).all() for v in (u, s, vh)):
                raise RuntimeError("CPU SVD non-finite")
            if torch.isfinite(scale) and scale > 0:
                s = s * scale
        except RuntimeError:
            _logger.warning("Batman SVD fallback failed; using identity: %s", first_err)
            rows, cols = work.shape
            r = min(rows, cols)
            u = torch.eye(rows, r, dtype=work.dtype)
            s = torch.zeros(r, dtype=work.dtype)
            vh = torch.eye(r, cols, dtype=work.dtype)
        return (
            u.to(device=matrix.device, dtype=matrix.dtype),
            s.to(device=matrix.device, dtype=matrix.dtype),
            vh.to(device=matrix.device, dtype=matrix.dtype),
        )


@torch.no_grad()
def _compute_perturbation(
    w_malicious: torch.Tensor,
    w_standard: torch.Tensor,
    lambda_reg: float,
) -> torch.Tensor:
    u, s, _ = _safe_svd(w_malicious)
    zero_basis = u[:, (s < 1e-6).nonzero(as_tuple=True)[0]]
    p_m = zero_basis @ zero_basis.transpose(0, 1)
    residual = w_standard - w_malicious
    eye = torch.eye(w_malicious.shape[0], device=w_malicious.device, dtype=w_malicious.dtype)
    h = p_m @ (w_malicious @ w_malicious.transpose(0, 1)) + float(lambda_reg) * eye
    return torch.linalg.solve(h, p_m @ residual)


@torch.no_grad()
def _align_layer(
    poisoned_w: torch.Tensor,
    clean_w: torch.Tensor,
    global_w: torch.Tensor,
    rank: int,
    lamda_reg: float,
    beta_reg: float,
) -> torch.Tensor:
    if poisoned_w.ndim < 2:
        return poisoned_w.clone()

    orig_shape, orig_dtype = poisoned_w.shape, poisoned_w.dtype
    wp = _shape_restruct(poisoned_w.detach().float())
    wc = _shape_restruct(clean_w.detach().float())
    wg = _shape_restruct(global_w.detach().float())

    eff_rank = min(int(rank), wp.shape[0], wp.shape[1])
    u, s, vh = _safe_svd(wp)
    w_first = u[:, :eff_rank] @ torch.diag(s[:eff_rank]) @ vh[:eff_rank, :]
    w_last = wp - w_first

    delta_tg = _compute_perturbation(w_first, wg, lamda_reg)
    delta_tc = _compute_perturbation(w_first, wc, beta_reg)
    aligned = w_first + 0.5 * (delta_tg + delta_tc) + w_last
    if not torch.isfinite(aligned).all():
        return clean_w.detach().clone()
    return aligned.reshape(orig_shape).to(orig_dtype)


@torch.no_grad()
def align_selected_layers(
    poisoned_params: Mapping[str, torch.Tensor],
    clean_params: Mapping[str, torch.Tensor],
    global_params: Mapping[str, torch.Tensor],
    selected_layers: Iterable[str],
    *,
    rank: int,
    lamda_reg: float,
    beta_reg: float,
) -> OrderedDict[str, torch.Tensor]:
    selected = set(selected_layers)
    out: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, p_val in poisoned_params.items():
        if name not in selected or p_val.ndim < 2:
            out[name] = p_val.clone()
        else:
            out[name] = _align_layer(
                p_val, clean_params[name], global_params[name], rank, lamda_reg, beta_reg
            )
    return out


@ATTACK_REGISTRY.register("batman")
class BatmanAttack(BaseAttacker):
    """Sequential clean-then-poison local training followed by SVD null-space alignment."""

    def __init__(
        self,
        rank: int = 4,
        regularization: Optional[float] = None,
        lamda_reg: Optional[float] = None,
        beta_reg: Optional[float] = None,
        num_selected_layers: int = 5,
        selected_layer_ratio: Optional[float] = None,
        layer_selection: str = "global_deviation",
        layer_selection_seed: Optional[int] = None,
        log_selected_layers: bool = True,
        layer_selection_log_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # Pop unused legacy stats args gracefully if present in old YAML configs.
        kwargs.pop("stats_top_layers", None)
        kwargs.pop("stats_layer_ratio", None)
        super().__init__(**kwargs)
        self.rank = int(rank)
        if lamda_reg is None and beta_reg is None and regularization is not None:
            self.lamda_reg = float(regularization)
            self.beta_reg = float(regularization)
        else:
            self.lamda_reg = 20.0 if lamda_reg is None else float(lamda_reg)
            self.beta_reg = 15.0 if beta_reg is None else float(beta_reg)
        self.num_selected_layers = int(num_selected_layers)
        self.selected_layer_ratio = selected_layer_ratio
        self.layer_selection = str(layer_selection)
        self.layer_selection_seed = (
            int(layer_selection_seed)
            if layer_selection_seed is not None
            else (int(self.seed) if self.seed is not None else 0)
        )
        self.log_selected_layers = bool(log_selected_layers)
        self.layer_selection_log_path = layer_selection_log_path

    def _log_selection(
        self,
        ctx: ClientContext,
        selected: Sequence[str],
        ranked: Sequence[LayerScore],
    ) -> None:
        if not self.log_selected_layers:
            return
        if self.layer_selection_log_path:
            path = self.layer_selection_log_path
        elif ctx.run_dir:
            path = os.path.join(ctx.run_dir, f"batman_selected_layers_{self.layer_selection}.jsonl")
        else:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        record = {
            "round": ctx.round_idx,
            "client_id": ctx.client_id,
            "strategy": self.layer_selection,
            "num_selected": len(selected),
            "selected_layers": list(selected),
            "ranked_layers": [{"name": n, "score": s} for s, n in ranked],
        }
        with _LAYER_LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def step(self, ctx: ClientContext) -> ClientUpdate:
        # 1. Clean trajectory: Wg -> Wc
        train_model(ctx.model, ctx.loader(seed=self.seed), ctx.train_cfg)
        clean_params, _ = split_model_state(ctx.model)

        # 2. Continued poison trajectory on same model: Wc -> Wp
        stats = train_model(
            ctx.model,
            ctx.loader(poisoner=self.poisoner, seed=self.seed),
            ctx.train_cfg,
        )
        poisoned_params, poisoned_buffers = split_model_state(ctx.model)

        # 3. Rank & align top critical layers in left null-space
        eligible = count_eligible_layers(poisoned_params)
        num_align = resolve_num_selected_layers(
            eligible, self.num_selected_layers, self.selected_layer_ratio
        )
        rng = (
            random.Random(
                derive_stable_seed(self.layer_selection_seed, ctx.client_id, ctx.round_idx)
            )
            if self.layer_selection == "random"
            else None
        )
        ranked = rank_critical_layers(
            poisoned_params, ctx.initial_params, strategy=self.layer_selection, rng=rng
        )
        selected = list(dict.fromkeys(name for _, name in ranked[:num_align]))
        self._log_selection(ctx, selected, ranked)

        aligned_params = align_selected_layers(
            poisoned_params,
            clean_params,
            ctx.initial_params,
            selected,
            rank=self.rank,
            lamda_reg=self.lamda_reg,
            beta_reg=self.beta_reg,
        )
        param_delta = {k: aligned_params[k] - ctx.initial_params[k] for k in ctx.initial_params}
        return ctx.pack_update(
            param_delta=param_delta,
            buffers=poisoned_buffers,
            metrics={"loss": stats.loss},
        )
