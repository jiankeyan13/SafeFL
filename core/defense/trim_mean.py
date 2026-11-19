"""Coordinate-wise Trimmed Mean and Median defenses (single-file v2)."""
from __future__ import annotations

from typing import Any, List

import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, StateDict
from core.utils.registry import DEFENSE_REGISTRY


def _coordinate_median_buffers(updates: List[ClientUpdate], device: torch.device) -> StateDict:
    if not updates or not updates[0].buffers:
        return {}
    out: StateDict = {}
    for k, ref in updates[0].buffers.items():
        stacked = torch.stack([u.buffers[k].to(device=device, dtype=torch.float32) for u in updates], dim=0)
        med = torch.median(stacked, dim=0).values
        out[k] = med.to(dtype=ref.dtype) if torch.is_floating_point(ref) else torch.round(med).to(dtype=ref.dtype)
    return out


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

        param_delta: StateDict = {}
        for k, ref in updates[0].param_delta.items():
            stacked = torch.stack(
                [u.param_delta[k].to(device=ctx.device, dtype=torch.float32) for u in updates],
                dim=0,
            )
            sorted_t, _ = torch.sort(stacked, dim=0)
            param_delta[k] = sorted_t[num_cut : n - num_cut].mean(dim=0).to(dtype=ref.dtype)

        buffers = _coordinate_median_buffers(updates, ctx.device)
        return DefenseOutput(param_delta=param_delta, buffers=buffers)


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
        if not updates:
            raise ValueError("updates must be non-empty")

        param_delta: StateDict = {}
        for k, ref in updates[0].param_delta.items():
            stacked = torch.stack(
                [u.param_delta[k].to(device=ctx.device, dtype=torch.float32) for u in updates],
                dim=0,
            )
            param_delta[k] = torch.median(stacked, dim=0).values.to(dtype=ref.dtype)

        buffers = _coordinate_median_buffers(updates, ctx.device)
        return DefenseOutput(param_delta=param_delta, buffers=buffers)
