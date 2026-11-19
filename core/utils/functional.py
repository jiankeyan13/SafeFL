"""Stateless tensor, StateDict, flat-matrix, BN, and model-topology operators for SafeFL v2."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as tF

from core.types import ClientUpdate, StateDict

_CLASSIFIER_ATTRS = ("linear", "fc", "classifier")


# =============================================================================
# 1. Flat Vector / Matrix <-> StateDict Operators
# =============================================================================


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
    present = [k for k in key_list if k in delta]
    if not present:
        return torch.empty(0, dtype=torch.float32, device=device)

    dev = device if device is not None else delta[present[0]].device
    total_dim = sum(int(delta[k].numel()) for k in present)
    out = torch.empty(total_dim, dtype=torch.float32, device=dev)
    offset = 0
    for k in present:
        t = delta[k].detach()
        n = int(t.numel())
        out[offset : offset + n].copy_(t.reshape(-1))
        offset += n
    return out


def stack_flat_deltas(
    updates: Sequence[ClientUpdate],
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, List[str]]:
    """Pack ``updates[i].param_delta`` directly into a single pre-allocated ``[N, D]`` ``float32`` matrix."""
    if not updates:
        raise ValueError("updates must be non-empty")

    template = updates[0].param_delta
    keys = ordered_keys(template)
    if not keys:
        return torch.empty((len(updates), 0), dtype=torch.float32, device=device), keys

    dev = device if device is not None else template[keys[0]].device
    total_dim = sum(int(template[k].numel()) for k in keys)
    mat = torch.empty((len(updates), total_dim), dtype=torch.float32, device=dev)

    for row, u in enumerate(updates):
        offset = 0
        for k in keys:
            n = int(template[k].numel())
            if k in u.param_delta:
                mat[row, offset : offset + n].copy_(u.param_delta[k].detach().reshape(-1))
            else:
                mat[row, offset : offset + n].zero_()
            offset += n
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
    """Compute per-client L2 norm over learnable ``param_delta`` without materializing flat vectors."""
    norms: List[float] = []
    for u in updates:
        sq = 0.0
        for v in u.param_delta.values():
            t = v.detach().to(device=device, dtype=torch.float32) if device is not None else v.detach().float()
            sq += float(t.square().sum().item())
        norms.append(sq ** 0.5)
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


# =============================================================================
# 2. Scaling, Clipping, Aggregation & Noise Operators
# =============================================================================


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
    *,
    seed: Optional[int] = None,
) -> StateDict:
    """Add i.i.d. ``N(0, std^2)`` noise to learnable parameters using an optional isolated seed."""
    sigma = float(std)
    if sigma <= 0.0:
        return param_delta

    gens: dict[torch.device, torch.Generator] = {}
    out: StateDict = {}
    for k, v in param_delta.items():
        if not torch.is_floating_point(v):
            out[k] = v
            continue
        if seed is None:
            noise = torch.randn_like(v) * sigma
        else:
            gen = gens.get(v.device)
            if gen is None:
                gen = torch.Generator(device=v.device)
                gen.manual_seed(int(seed))
                gens[v.device] = gen
            noise = torch.randn(v.shape, dtype=v.dtype, device=v.device, generator=gen) * sigma
        out[k] = v + noise
    return out


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


# =============================================================================
# 3. Model Topology Introspection Helpers (Regex-Free)
# =============================================================================


def find_conv_bn_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """Return ``(conv_name, bn_name)`` pairs by walking ``model.named_modules()`` topologically."""
    pairs: List[Tuple[str, str]] = []
    pending: Optional[Tuple[str, nn.Conv2d]] = None
    for name, mod in model.named_modules():
        if not name:
            continue
        if isinstance(mod, nn.Conv2d):
            pending = (name, mod)
        elif isinstance(mod, nn.BatchNorm2d):
            if pending is not None and pending[1].out_channels == mod.num_features:
                pairs.append((pending[0], name))
            pending = None
    return pairs


def find_classifier_head(model: nn.Module) -> Tuple[str, nn.Linear]:
    """Locate the final ``nn.Linear`` classification layer ``(module_name, module)``."""
    for attr in _CLASSIFIER_ATTRS:
        cand = getattr(model, attr, None)
        if isinstance(cand, nn.Linear):
            return attr, cand

    linear_layers = [
        (name, mod)
        for name, mod in model.named_modules()
        if name and isinstance(mod, nn.Linear)
    ]
    if not linear_layers:
        raise AttributeError(f"Model {type(model).__name__} has no nn.Linear classifier head.")
    return linear_layers[-1]


def resolve_classifier_keys(
    param_keys: Sequence[str],
    model: Optional[nn.Module] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ``(weight_key, bias_key)`` of the output classifier layer."""
    key_set = set(param_keys)
    if model is not None:
        try:
            head_name, _ = find_classifier_head(model)
            w_k, b_k = f"{head_name}.weight", f"{head_name}.bias"
            if w_k in key_set and b_k in key_set:
                return w_k, b_k
        except AttributeError:
            pass

    for prefix in _CLASSIFIER_ATTRS:
        if f"{prefix}.weight" in key_set and f"{prefix}.bias" in key_set:
            return f"{prefix}.weight", f"{prefix}.bias"
    for k in reversed(param_keys):
        if k.endswith(".weight"):
            b_k = k[: -len("weight")] + "bias"
            if b_k in key_set:
                return k, b_k
    return None, None


def extract_penultimate_features(
    model: nn.Module,
    inputs: torch.Tensor,
    head_mod: nn.Module,
) -> torch.Tensor:
    """Capture L2-normalized penultimate features fed into ``head_mod`` via forward pre-hook."""
    captured: List[torch.Tensor] = []

    def _pre_hook(_mod: nn.Module, args: Tuple[Any, ...]) -> None:
        captured.append(args[0])

    handle = head_mod.register_forward_pre_hook(_pre_hook)
    try:
        model(inputs)
    finally:
        handle.remove()

    if not captured:
        raise RuntimeError("Forward pass did not invoke the classifier head module.")
    feats = captured[-1]
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)
    return tF.normalize(feats, dim=1)
