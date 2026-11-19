"""Iterative Multi-Krum defense (Blanchard et al., NeurIPS 2017, single-file v2)."""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("multi_krum")
@DEFENSE_REGISTRY.register("krum")
class MultiKrumDefense(BaseDefense):
    """Iterative Multi-Krum screening followed by sample-weighted averaging."""

    def __init__(self, f: int = 0, m: Optional[int] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.f = int(f)
        self.m = (2 * self.f + 2) if m is None else int(m)

    def _krum_scores(self, mat: torch.Tensor) -> List[float]:
        n = mat.shape[0]
        dists_sq = torch.cdist(mat, mat, p=2).pow(2)
        k = max(n - self.f - 2, 1)
        scores: List[float] = []
        for i in range(n):
            sorted_d, _ = torch.sort(dists_sq[i])
            scores.append(float(sorted_d[1 : k + 1].sum().item()))
        return scores

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        n = len(updates)
        if n <= 2 * self.f + 2:
            kept_updates = updates
        else:
            mat, _ = F.stack_flat_deltas(updates, device=ctx.device)
            remaining = list(range(n))
            target_m = max(1, min(self.m, n))
            while len(remaining) > target_m:
                scores = self._krum_scores(mat[remaining])
                worst_pos = max(range(len(remaining)), key=lambda i: scores[i])
                remaining.pop(worst_pos)
            kept_updates = [updates[i] for i in remaining]

        param_delta, buffers = F.aggregate_updates(kept_updates, device=ctx.device)
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(kept_updates))},
        )
