"""Robust Learning Rate (RLR) sign-voting defense (Ozdayi et al., AAAI 2021, single-file v2)."""
from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, StateDict
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
        avg_delta, buffers = F.aggregate_updates(updates, device=ctx.device)

        masked_delta: StateDict = {}
        for k, avg_t in avg_delta.items():
            stacked = torch.stack(
                [u.param_delta[k].to(device=ctx.device, dtype=torch.float32) for u in updates],
                dim=0,
            )
            sum_signs = torch.sign(stacked).sum(dim=0)
            sign_mask = torch.where(
                sum_signs.abs() >= self.robustLR_threshold,
                torch.ones_like(sum_signs),
                -torch.ones_like(sum_signs),
            )
            masked_delta[k] = (avg_t.float() * sign_mask).to(dtype=avg_t.dtype)

        return DefenseOutput(param_delta=masked_delta, buffers=buffers)
