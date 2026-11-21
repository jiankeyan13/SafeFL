"""SPMC (SPMD) leave-one-out cosine-similarity elastic-weighting defense (ICML 2025, single-file v2)."""
from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, split_model_state
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("spmc")
class SPMCDefense(BaseDefense):
    """Leave-one-out parameter cosine similarity + softmax elastic aggregation."""

    def __init__(self, eps: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eps = float(eps)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        n = len(updates)
        if n == 0:
            raise ValueError("updates must be non-empty")

        delta_mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        if n < 2 or delta_mat.shape[1] == 0:
            param_delta, buffers = F.aggregate_updates(
                updates,
                weight_by_samples=False,
                device=ctx.device,
            )
            return DefenseOutput(
                param_delta=param_delta,
                buffers=buffers,
                metrics={"defense/kept_clients": float(n)},
            )

        global_params, _ = split_model_state(global_model)
        global_vec = F.flatten_delta(global_params, keys, device=ctx.device)

        # Reconstruct each client's local parameter vector and compute leave-one-out mean.
        local_mat = delta_mat + global_vec.unsqueeze(0)
        total_vec = local_mat.sum(dim=0, keepdim=True)
        loo_mat = (total_vec - local_mat) / float(n - 1)

        dots = (local_mat * loo_mat).sum(dim=1)
        client_norms = torch.norm(local_mat, p=2, dim=1)
        loo_norms = torch.norm(loo_mat, p=2, dim=1)
        cos_scores = dots / (client_norms * loo_norms + self.eps)

        exp_scores = torch.exp(cos_scores - cos_scores.max() + self.eps)
        elastic_weights = (exp_scores / exp_scores.sum()).tolist()

        param_delta, buffers = F.aggregate_updates(
            updates,
            weights=elastic_weights,
            weight_by_samples=False,
            device=ctx.device,
        )
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(n)},
        )
