"""FLAME defense: HDBSCAN cosine clustering + median norm clipping + Gaussian noise (USENIX Security 2022)."""
from __future__ import annotations

from typing import Any, List

import hdbscan
import numpy as np
import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, split_model_state
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("flame")
class FlameDefense(BaseDefense):
    """Single-file FLAME defense replacing HdbscanScreener + FlameAggregator + NoiseRefiner."""

    def __init__(self, noise: float = 0.001, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.noise = float(noise)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        n = len(updates)
        delta_mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        norms = torch.norm(delta_mat, p=2, dim=1).tolist()

        # 1. HDBSCAN cosine clustering over reconstructed local weights (W_g + delta_i)
        if n < 2:
            benign_idx = list(range(n))
        else:
            global_params, _ = split_model_state(global_model)
            global_vec = F.flatten_delta(global_params, keys, device=ctx.device)
            w_locals = delta_mat + global_vec.unsqueeze(0)
            cos_dist = F.pairwise_cosine_distance_np(w_locals, eps=1e-9)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=n // 2 + 1,
                min_samples=1,
                allow_single_cluster=True,
                metric="precomputed",
            ).fit(cos_dist)
            labels = clusterer.labels_

            valid = labels >= 0
            if np.any(valid):
                uniq, counts = np.unique(labels[valid], return_counts=True)
                best_cluster = uniq[np.argmax(counts)]
                benign_idx = np.where(labels == best_cluster)[0].tolist()
            else:
                benign_idx = list(range(n))

        # 2. Norm clipping by median update norm of the benign cluster
        benign_norms = [norms[i] for i in benign_idx]
        clip_value = float(np.median(benign_norms)) if benign_norms else 1.0

        benign_deltas = [updates[i].param_delta for i in benign_idx]
        clipped_deltas = F.clip_deltas_by_norm(
            benign_deltas,
            max_norm=clip_value,
            norms=benign_norms,
            device=ctx.device,
        )

        # 3. Uniform average over clipped benign updates + calibrated Gaussian noise
        uniform_w = [1.0] * len(clipped_deltas)
        avg_delta = F.weighted_average_dicts(clipped_deltas, uniform_w, device=ctx.device)
        noisy_delta = F.add_gaussian_noise(avg_delta, std=self.noise * clip_value)

        benign_buffers = [updates[i].buffers for i in benign_idx if updates[i].buffers]
        buffers = (
            F.weighted_average_dicts(benign_buffers, [1.0] * len(benign_buffers), device=ctx.device)
            if benign_buffers
            else None
        )
        return DefenseOutput(
            param_delta=noisy_delta,
            buffers=buffers,
            metrics={
                "defense/kept_clients": float(len(benign_idx)),
                "defense/clip_value": clip_value,
            },
        )
