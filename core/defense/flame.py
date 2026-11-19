"""FLAME defense: HDBSCAN cosine clustering + median norm clipping + Gaussian noise (USENIX Security 2022)."""
from __future__ import annotations

from typing import Any, List

import hdbscan
import numpy as np
import torch
import torch.nn as nn

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, split_model_state
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY
from data.loader import derive_stable_seed


@DEFENSE_REGISTRY.register("flame")
class FlameDefense(BaseDefense):
    """Single-file FLAME defense with vectorized matrix norm clipping."""

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
        norms = torch.norm(delta_mat, p=2, dim=1)

        # 1. HDBSCAN cosine clustering over reconstructed local weights (W_g + delta_i)
        if n < 2:
            benign_idx = list(range(n))
        else:
            global_params, _ = split_model_state(global_model)
            global_vec = F.flatten_delta(global_params, keys, device=ctx.device)
            w_locals = delta_mat + global_vec.unsqueeze(0)
            cos_dist = F.pairwise_cosine_distance_np(w_locals, eps=1e-9)

            labels = hdbscan.HDBSCAN(
                min_cluster_size=n // 2 + 1,
                min_samples=1,
                allow_single_cluster=True,
                metric="precomputed",
            ).fit_predict(cos_dist)

            valid = labels >= 0
            if np.any(valid):
                uniq, counts = np.unique(labels[valid], return_counts=True)
                best_cluster = uniq[np.argmax(counts)]
                benign_idx = np.where(labels == best_cluster)[0].tolist()
            else:
                benign_idx = list(range(n))

        # 2. Vectorized norm clipping + uniform average + calibrated Gaussian noise
        benign_norms = [float(norms[i].item()) for i in benign_idx]
        clip_value = float(np.median(benign_norms)) if benign_norms else 1.0

        avg_delta = F.clip_and_average_flat(
            delta_mat,
            benign_idx,
            clip_value,
            updates[0].param_delta,
            keys,
            norms=norms,
        )
        noise_seed = derive_stable_seed(ctx.seed, "flame_noise", ctx.round_idx)
        noisy_delta = F.add_gaussian_noise(
            avg_delta,
            std=self.noise * clip_value,
            seed=noise_seed,
        )
        buffers = F.average_buffers(updates, benign_idx, device=ctx.device)

        return DefenseOutput(
            param_delta=noisy_delta,
            buffers=buffers,
            metrics={
                "defense/kept_clients": float(len(benign_idx)),
                "defense/clip_value": clip_value,
            },
        )
