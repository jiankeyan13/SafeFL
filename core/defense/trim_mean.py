"""Coordinate-wise Trimmed Mean and Median defenses (single-file v2)."""
from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("trim_mean")
class TrimMeanDefense(BaseDefense):
    """Coordinate-wise Trimmed Mean aggregation (Yin et al., ICML 2018)."""

    def __init__(self, trim_ratio: float = 0.2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not 0.0 <= float(trim_ratio) < 0.5:
            raise ValueError(f"trim_ratio must be in [0.0, 0.5), got {trim_ratio}")
        self.trim_ratio = float(trim_ratio)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        n = len(updates)
        num_cut = int(n * self.trim_ratio)
        if n - 2 * num_cut <= 0:
            raise ValueError(f"Too many clients trimmed: n={n}, trim_ratio={self.trim_ratio}")

        mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        sorted_mat, _ = torch.sort(mat, dim=0)
        trimmed_vec = sorted_mat[num_cut : n - num_cut].mean(dim=0)
        param_delta = F.unflatten_delta(trimmed_vec, updates[0].param_delta, keys)
        return DefenseOutput(param_delta=param_delta, buffers=F.median_buffers(updates, ctx.device))


@DEFENSE_REGISTRY.register("median")
class MedianDefense(BaseDefense):
    """Coordinate-wise Median aggregation (Yin et al., ICML 2018)."""

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        med_vec = torch.median(mat, dim=0).values
        param_delta = F.unflatten_delta(med_vec, updates[0].param_delta, keys)
        return DefenseOutput(param_delta=param_delta, buffers=F.median_buffers(updates, ctx.device))
