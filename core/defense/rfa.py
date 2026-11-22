"""Robust Federated Aggregation (RFA) via smoothed Weiszfeld geometric median (single-file v2)."""
from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("rfa")
class RFADefense(BaseDefense):
    """Weighted geometric median aggregation (Pillutla et al., IEEE TSP 2022)."""

    def __init__(self, num_iters: int = 4, nu: float = 1e-6, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1")
        if nu <= 0:
            raise ValueError("nu must be > 0")
        self.num_iters = int(num_iters)
        self.nu = float(nu)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        alpha = torch.tensor(
            [float(max(u.num_samples, 1)) for u in updates],
            dtype=torch.float32,
            device=ctx.device,
        )
        y = (mat * alpha.unsqueeze(1)).sum(dim=0) / alpha.sum().clamp_min(1e-12)
        beta = alpha

        for _ in range(self.num_iters):
            dists = torch.norm(mat - y.unsqueeze(0), dim=1).clamp_min(self.nu)
            beta = alpha / dists
            y = (mat * beta.unsqueeze(1)).sum(dim=0) / beta.sum().clamp_min(1e-12)

        param_delta = F.unflatten_delta(y, updates[0].param_delta, keys)
        beta_list = beta.tolist()
        return DefenseOutput(
            param_delta=param_delta,
            buffers=F.median_buffers(updates, ctx.device),
            client_weights={u.client_id: float(w) for u, w in zip(updates, beta_list)},
        )
