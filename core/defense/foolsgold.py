"""FoolsGold Sybil-resistant historical cosine defense (Fung et al., RAID 2020, single-file v2)."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("foolsgold")
class FoolsGoldDefense(BaseDefense):
    """Dynamic aggregation weighting based on historical update cosine similarity & pardoning."""

    def __init__(
        self,
        use_history: bool = True,
        logit_center: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.use_history = bool(use_history)
        self.logit_center = float(logit_center)
        self._history: Dict[int, torch.Tensor] = {}

    @staticmethod
    def _pardon(cs: torch.Tensor) -> torch.Tensor:
        cs_no_diag = cs.clone().fill_diagonal_(0.0)
        v = cs_no_diag.max(dim=1).values
        v_col, v_row = v.unsqueeze(1), v.unsqueeze(0)
        ratio = torch.clamp(
            torch.where(v_row > v_col, v_col / v_row.clamp_min(1e-8), torch.ones_like(v_row)),
            0.0,
            1.0,
        )
        return (cs * ratio).fill_diagonal_(0.0)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        if len(updates) < 2:
            param_delta, buffers = F.aggregate_updates(updates, device=ctx.device)
            return DefenseOutput(param_delta=param_delta, buffers=buffers)

        keys = F.ordered_keys(updates[0].param_delta)
        feats: List[torch.Tensor] = []
        for u in updates:
            vec = F.flatten_delta(u.param_delta, keys, device=ctx.device)
            if self.use_history:
                prev = self._history.get(u.client_id)
                hist = vec if prev is None else (prev.to(ctx.device) + vec)
                # Store history on CPU so server VRAM stays flat for large client pools.
                self._history[u.client_id] = hist.detach().cpu()
                feats.append(hist)
            else:
                feats.append(vec)

        mat = torch.stack(feats, dim=0)
        cs = F.pairwise_cosine_similarity(mat, eps=1e-8)
        cs_pardoned = self._pardon(cs)

        alpha_raw = 1.0 - cs_pardoned.max(dim=1).values
        alpha_norm = (alpha_raw / alpha_raw.max().clamp_min(1e-8)).clamp(0.0, 1.0)
        alpha_norm[alpha_norm >= 1.0] = 0.99

        eps = 1e-7
        clamped = alpha_norm.clamp(eps, 1.0 - eps)
        weights = (torch.log(clamped / (1.0 - clamped)) + self.logit_center).clamp(0.0, 1.0).tolist()

        param_delta, buffers = F.aggregate_updates(updates, weights=weights, device=ctx.device)
        return DefenseOutput(param_delta=param_delta, buffers=buffers)
