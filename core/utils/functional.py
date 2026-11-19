"""Stateless tensor, StateDict, flat-matrix, and BN operators shared across SafeFL v2."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from core.types import ClientUpdate, StateDict


def ordered_keys(param_dict: StateDict) -> List[str]:
    """Return deterministic parameter key order."""
    return list(param_dict.keys())


def flatten_delta(
    delta: StateDict,
    keys: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Flatten a parameter dict into a 1D ``float32`` vector."""
    key_list = ordered_keys(delta) if keys is None else list(keys)
    if not key_list:
        return torch.empty(0, dtype=torch.float32, device=device)
    parts = [
        (
            delta[k].detach().to(device=device, dtype=torch.float32).reshape(-1)
            if device is not None
            else delta[k].detach().float().reshape(-1)
        )
        for k in key_list
        if k in delta
    ]
    if not parts:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(parts, dim=0)


def stack_flat_deltas(
    updates: Sequence[ClientUpdate],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Stack all ``updates[i].param_delta`` into a 2D ``[N, D]`` ``float32`` matrix."""
    if not updates:
        raise ValueError("updates must be non-empty")
    keys = ordered_keys(updates[0].param_delta)
    mat = torch.stack([flatten_delta(u.param_delta, keys, device=device) for u in updates], dim=0)
    return mat, keys


def unflatten_delta(
    flat: torch.Tensor,
    template: StateDict,
    keys: Optional[Sequence[str]] = None,
) -> StateDict:
    """Reshape a 1D ``flat`` vector back into a ``StateDict`` matching ``template`` shapes."""
    key_list = ordered_keys(template) if keys is None else list(keys)
    out: StateDict = {}
    offset = 0
    for k in key_list:
        ref = template[k]
        numel = ref.numel()
        out[k] = flat[offset : offset + numel].view(ref.shape).to(dtype=ref.dtype)
        offset += numel
    return out


def delta_l2_norms(
    updates: Sequence[ClientUpdate],
    device: Optional[torch.device] = None,
) -> List[float]:
    """Compute per-client L2 norm over learnable ``param_delta``."""
    norms: List[float] = []
    for u in updates:
        vec = flatten_delta(u.param_delta, device=device)
        norms.append(float(torch.norm(vec, p=2).item()) if vec.numel() > 0 else 0.0)
    return norms


def pairwise_cosine_similarity(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute ``[N, N]`` cosine similarity matrix for row vectors in ``mat``."""
    norms = torch.norm(mat, p=2, dim=1, keepdim=True).clamp_min(eps)
    normed = mat / norms
    return torch.mm(normed, normed.t()).clamp(-1.0, 1.0)


def pairwise_cosine_distance_np(mat: torch.Tensor, eps: float = 1e-12) -> np.ndarray:
    """Compute ``[N, N]`` ``float64`` cosine distance matrix (for ``hdbscan(metric='precomputed')``)."""
    cos_sim = pairwise_cosine_similarity(mat, eps=eps)
    return (1.0 - cos_sim).clamp(min=0.0, max=2.0).detach().cpu().numpy().astype(np.float64)


def zero_delta_like(template: StateDict, device: Optional[torch.device] = None) -> StateDict:
    return {
        k: torch.zeros_like(v, device=device if device is not None else v.device)
        for k, v in template.items()
    }


def scale_delta(
    delta: StateDict,
    scale: float,
    device: Optional[torch.device] = None,
) -> StateDict:
    s = float(scale)
    return {
        k: (v.to(device=device, dtype=torch.float32) if device is not None else v.float()) * s
        for k, v in delta.items()
    }


def clip_deltas_by_norm(
    deltas: Sequence[StateDict],
    max_norm: float,
    norms: Optional[Sequence[float]] = None,
    *,
    device: Optional[torch.device] = None,
    eps: float = 1e-12,
) -> List[StateDict]:
    """Clip each parameter delta dict so its L2 norm does not exceed ``max_norm``."""
    cap = float(max_norm)
    if cap <= 0.0:
        return [
            {k: v.to(device) if device is not None else v for k, v in d.items()}
            for d in deltas
        ]

    out: List[StateDict] = []
    for idx, d in enumerate(deltas):
        norm_i = (
            float(norms[idx])
            if norms is not None
            else float(torch.norm(flatten_delta(d, device=device), p=2).item())
        )
        gamma = min(1.0, cap / max(norm_i, eps))
        if gamma < 1.0:
            out.append(scale_delta(d, gamma, device=device))
        else:
            out.append({k: v.to(device) if device is not None else v for k, v in d.items()})
    return out


def weighted_average_dicts(
    dicts: Sequence[StateDict],
    weights: Sequence[float],
    device: Optional[torch.device] = None,
) -> StateDict:
    """Compute normalized weighted average across ``StateDict``s (supports float & int buffers)."""
    if not dicts:
        raise ValueError("dicts must be non-empty")
    total_w = float(sum(weights))
    if total_w <= 0.0:
        return zero_delta_like(dicts[0], device=device)

    norm_w = [float(w) / total_w for w in weights]
    dev = device if device is not None else next(iter(dicts[0].values())).device
    w_t = torch.tensor(norm_w, dtype=torch.float32, device=dev)

    out: StateDict = {}
    for k, ref in dicts[0].items():
        stacked = torch.stack([d[k].to(device=dev, dtype=torch.float32) for d in dicts], dim=0)
        view_shape = [len(dicts)] + [1] * (stacked.dim() - 1)
        avg = torch.sum(stacked * w_t.view(*view_shape), dim=0)
        if torch.is_floating_point(ref):
            out[k] = avg.to(dtype=ref.dtype)
        else:
            out[k] = torch.round(avg).to(dtype=ref.dtype)
    return out


def aggregate_updates(
    updates: Sequence[ClientUpdate],
    weights: Optional[Sequence[float]] = None,
    *,
    weight_by_samples: bool = True,
    device: Optional[torch.device] = None,
) -> Tuple[StateDict, StateDict]:
    """Aggregate selected/scored ``ClientUpdate``s into ``(param_delta, buffers)``."""
    if not updates:
        raise ValueError("updates must be non-empty")

    n = len(updates)
    base_w = [1.0] * n if weights is None else [float(w) for w in weights]
    if weight_by_samples:
        eff_w = [w * max(int(u.num_samples), 1) for w, u in zip(base_w, updates)]
    else:
        eff_w = base_w

    if sum(eff_w) <= 0.0:
        return zero_delta_like(updates[0].param_delta, device=device), {}

    param_delta = weighted_average_dicts([u.param_delta for u in updates], eff_w, device=device)
    buffers = (
        weighted_average_dicts([u.buffers for u in updates], eff_w, device=device)
        if updates[0].buffers
        else {}
    )
    return param_delta, buffers


def average_buffers(
    updates: Sequence[ClientUpdate],
    indices: Optional[Sequence[int]] = None,
    *,
    weights: Optional[Sequence[float]] = None,
    weight_by_samples: bool = False,
    device: Optional[torch.device] = None,
) -> Optional[StateDict]:
    """Average BatchNorm ``buffers`` across selected ``updates`` (returns ``None`` if empty)."""
    selected = list(updates) if indices is None else [updates[i] for i in indices]
    buf_list = [u.buffers for u in selected if u.buffers]
    if not buf_list:
        return None
    if weights is None:
        eff_w = (
            [float(max(int(u.num_samples), 1)) for u in selected if u.buffers]
            if weight_by_samples
            else [1.0] * len(buf_list)
        )
    else:
        raw_w = [float(w) for w, u in zip(weights, selected) if u.buffers]
        eff_w = (
            [w * max(int(u.num_samples), 1) for w, u in zip(weights, selected) if u.buffers]
            if weight_by_samples
            else raw_w
        )
    return weighted_average_dicts(buf_list, eff_w, device=device)


def median_buffers(
    updates: Sequence[ClientUpdate],
    device: torch.device,
) -> Optional[StateDict]:
    """Compute coordinate-wise median over BatchNorm ``buffers`` across ``updates``."""
    buf_list = [u.buffers for u in updates if u.buffers]
    if not buf_list:
        return None
    out: StateDict = {}
    for k, ref in buf_list[0].items():
        stacked = torch.stack([b[k].to(device=device, dtype=torch.float32) for b in buf_list], dim=0)
        med = torch.median(stacked, dim=0).values
        out[k] = (
            med.to(dtype=ref.dtype)
            if torch.is_floating_point(ref)
            else torch.round(med).to(dtype=ref.dtype)
        )
    return out


def clip_and_average_flat(
    delta_mat: torch.Tensor,
    indices: Sequence[int],
    max_norm: float,
    template: StateDict,
    keys: Sequence[str],
    *,
    norms: Optional[torch.Tensor] = None,
    weights: Optional[Sequence[float]] = None,
    eps: float = 1e-12,
) -> StateDict:
    """Vectorized L2 norm clipping + weighted average directly on a ``[N, D]`` matrix."""
    if not indices:
        return zero_delta_like(template, device=delta_mat.device)

    idx_t = torch.as_tensor(list(indices), dtype=torch.long, device=delta_mat.device)
    sub_mat = delta_mat.index_select(0, idx_t)
    sub_norms = (
        norms.index_select(0, idx_t)
        if norms is not None
        else torch.norm(sub_mat, p=2, dim=1)
    )
    cap = float(max_norm)
    if cap > 0.0:
        gammas = torch.clamp(cap / sub_norms.clamp_min(eps), max=1.0)
        sub_mat = sub_mat * gammas.unsqueeze(1)

    if weights is None:
        avg_vec = sub_mat.mean(dim=0)
    else:
        w_t = torch.as_tensor(list(weights), dtype=torch.float32, device=delta_mat.device)
        w_sum = w_t.sum()
        if float(w_sum.item()) <= 0.0:
            return zero_delta_like(template, device=delta_mat.device)
        avg_vec = (sub_mat * (w_t / w_sum).unsqueeze(1)).sum(dim=0)

    return unflatten_delta(avg_vec, template, keys)


def add_gaussian_noise(
    param_delta: StateDict,
    std: float,
) -> StateDict:
    """Add i.i.d. ``N(0, std^2)`` noise strictly to learnable parameters."""
    sigma = float(std)
    if sigma <= 0.0:
        return param_delta
    return {
        k: (v + torch.randn_like(v) * sigma if torch.is_floating_point(v) else v)
        for k, v in param_delta.items()
    }


@torch.no_grad()
def calibrate_bn(
    model: nn.Module,
    proxy_loader: Iterable[Any],
    device: torch.device,
) -> None:
    """Recalibrate BatchNorm running statistics using unlabeled/proxy batches."""
    model.train()
    model.to(device)
    for batch in proxy_loader:
        bx = batch[0]
        if bx.device != device:
            bx = bx.to(device, non_blocking=True)
        model(bx)
