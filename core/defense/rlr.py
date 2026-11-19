"""Robust Learning Rate (RLR) sign-voting defense (Ozdayi et al., AAAI 2021, single-file v2)."""
from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("rlr")
class RLRDefense(BaseDefense):
    """Coordinate-wise sign consensus mask over sample-weighted parameter deltas."""

    def __init__(self, robustLR_threshold: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if robustLR_threshold < 1:
            raise ValueError("robustLR_threshold must be >= 1")
        self.robustLR_threshold = int(robustLR_threshold)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        delta_mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        w = torch.tensor(
            [float(max(int(u.num_samples), 1)) for u in updates],
            dtype=torch.float32,
            device=ctx.device,
        )
        avg_vec = (delta_mat * (w / w.sum()).unsqueeze(1)).sum(dim=0)

        sum_signs = torch.sign(delta_mat).sum(dim=0)
        sign_mask = torch.where(
            sum_signs.abs() >= self.robustLR_threshold,
            torch.ones_like(sum_signs),
            -torch.ones_like(sum_signs),
        )
        param_delta = F.unflatten_delta(avg_vec * sign_mask, updates[0].param_delta, keys)
        buffers = F.average_buffers(updates, weight_by_samples=True, device=ctx.device)
        return DefenseOutput(param_delta=param_delta, buffers=buffers)
