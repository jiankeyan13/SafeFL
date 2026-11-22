"""FDCR defense: Fisher-weighted gradient divergence + FINCH clustering (NeurIPS 2024, single-file v2)."""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

from core.defense.base import BaseDefense, build_client_weights
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


def _finch_partition0(scores: np.ndarray) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Run first-level FINCH clustering on 1D/2D feature array."""
    n = scores.shape[0]
    if n < 2:
        return [], []
    nbrs = NearestNeighbors(n_neighbors=2, metric="euclidean", n_jobs=1).fit(scores)
    connectivity = nbrs.kneighbors_graph(scores)
    connectivity = connectivity @ connectivity.T
    connectivity.setdiag(0)
    connectivity.eliminate_zeros()

    n_comp, labels = connected_components(csgraph=connectivity)
    if n_comp <= 1:
        return [], []

    centers: List[np.ndarray] = []
    core_indices: List[List[int]] = []
    for cid in range(n_comp):
        idx = np.where(labels == cid)[0]
        core_indices.append(idx.tolist())
        centers.append(scores[idx].mean(axis=0))
    return centers, core_indices


@DEFENSE_REGISTRY.register("fdcr")
class FDCRDefense(BaseDefense):
    """Fisher-Information-weighted divergence filtering + sigmoid parameter rescaling."""

    client_extras = ("fisher",)

    def __init__(self, eps: float = 1e-12, seed: int = 42, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eps = float(eps)
        self.seed = int(seed)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        n = len(updates)
        if n <= 2:
            param_delta, buffers = F.aggregate_updates(updates, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(
                param_delta=param_delta,
                buffers=buffers,
                metrics={"defense/kept_clients": float(n)},
                client_weights=build_client_weights(updates, weight_by_samples=True),
            )

        delta_mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        if delta_mat.shape[1] == 0:
            param_delta, buffers = F.aggregate_updates(updates, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(
                param_delta=param_delta,
                buffers=buffers,
                metrics={"defense/kept_clients": float(n)},
                client_weights=build_client_weights(updates, weight_by_samples=True),
            )

        lr = max(float(ctx.client_lr), self.eps)
        grad_mat = -delta_mat / lr

        # Collect and min-max normalize each client's Fisher vector.
        norm_fish_list: List[torch.Tensor] = []
        gen = torch.Generator(device=ctx.device)
        gen.manual_seed(self.seed + ctx.round_idx)
        for u in updates:
            fisher_dict = u.extra.get("fisher") if isinstance(u.extra, dict) else None
            if isinstance(fisher_dict, dict) and all(k in fisher_dict for k in keys):
                f_vec = F.flatten_delta(fisher_dict, keys, device=ctx.device)
            else:
                f_vec = torch.randn(delta_mat.shape[1], device=ctx.device, dtype=delta_mat.dtype, generator=gen)
            f_min = f_vec.min()
            f_max = f_vec.max()
            denom = (f_max - f_min).clamp_min(self.eps)
            norm_fish_list.append((f_vec - f_min) / denom)

        norm_fish_mat = torch.stack(norm_fish_list, dim=0)
        weighted_grad_mat = grad_mat * norm_fish_mat

        sample_counts = torch.tensor(
            [float(max(int(u.num_samples), 1)) for u in updates],
            device=ctx.device,
            dtype=delta_mat.dtype,
        )
        freq = sample_counts / sample_counts.sum()
        weighted_global_grad = (weighted_grad_mat * freq.unsqueeze(1)).sum(dim=0, keepdim=True)

        div_scores = torch.norm(weighted_grad_mat - weighted_global_grad, p=2, dim=1, keepdim=True)
        centers, core_indices = _finch_partition0(div_scores.detach().cpu().numpy())

        if not centers:
            param_delta, buffers = F.aggregate_updates(updates, weight_by_samples=True, device=ctx.device)
            return DefenseOutput(
                param_delta=param_delta,
                buffers=buffers,
                metrics={"defense/kept_clients": float(n)},
                client_weights=build_client_weights(updates, weight_by_samples=True),
            )

        evils_cluster_idx = int(np.argmax([float(c[0]) for c in centers]))
        evils_set = set(core_indices[evils_cluster_idx])
        benign_idx = [i for i in range(n) if i not in evils_set] or list(range(n))

        # Rescale surviving benign deltas by sigmoid(norm_fish) * 2.
        rescaled_updates: List[ClientUpdate] = []
        for i in benign_idx:
            scale_vec = torch.sigmoid(norm_fish_mat[i]) * 2.0
            scaled_delta_vec = delta_mat[i] * scale_vec
            scaled_delta = F.unflatten_delta(scaled_delta_vec, updates[i].param_delta, keys)
            rescaled_updates.append(
                ClientUpdate(
                    client_id=updates[i].client_id,
                    param_delta=scaled_delta,
                    buffers=updates[i].buffers,
                    num_samples=updates[i].num_samples,
                    metrics=updates[i].metrics,
                )
            )

        param_delta, buffers = F.aggregate_updates(
            rescaled_updates,
            weight_by_samples=True,
            device=ctx.device,
        )
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(rescaled_updates))},
            client_weights=build_client_weights(updates, kept_indices=benign_idx, weight_by_samples=True),
        )
