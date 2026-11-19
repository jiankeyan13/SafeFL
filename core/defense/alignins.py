"""AlignIns defense: TDA + MPSA Z-score filtering + benign-median clipping (single-file v2)."""
from __future__ import annotations

import math
from typing import Any, List, Sequence

import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, split_model_state
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("alignins")
class AlignInsDefense(BaseDefense):
    """Direction & sign alignment inspection followed by norm-clipped benign averaging."""

    def __init__(
        self,
        top_ratio: float = 0.3,
        tda_threshold: float = 1.0,
        mpsa_threshold: float = 1.0,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 < float(top_ratio) <= 1.0:
            raise ValueError("top_ratio must be in (0, 1]")
        self.top_ratio = float(top_ratio)
        self.tda_threshold = float(tda_threshold)
        self.mpsa_threshold = float(mpsa_threshold)
        self.eps = float(eps)

    def _zscores(self, values: Sequence[float]) -> List[float]:
        t = torch.tensor(values, dtype=torch.float32)
        if t.numel() == 0:
            return []
        med = torch.median(t)
        std = torch.std(t, unbiased=False)
        if float(std.item()) <= self.eps:
            return [0.0] * t.numel()
        return (torch.abs(t - med) / std.clamp_min(self.eps)).tolist()

    def _compute_mpsa(self, vec: torch.Tensor, principal_sign: torch.Tensor) -> float:
        d = int(vec.numel())
        if d == 0:
            return 1.0
        top_k = max(1, int(math.ceil(d * self.top_ratio)))
        top_idx = torch.topk(vec.abs(), k=top_k, largest=True, sorted=False).indices
        mismatch = (torch.sign(vec[top_idx]) != principal_sign[top_idx]).sum().item()
        return 1.0 - float(mismatch) / float(top_k)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        n = len(updates)
        delta_mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        norms = torch.norm(delta_mat, p=2, dim=1).tolist()

        if n < 2 or delta_mat.shape[1] == 0:
            benign_idx = list(range(n))
        else:
            global_params, _ = split_model_state(global_model)
            global_vec = F.flatten_delta(global_params, keys, device=ctx.device)
            g_norm = global_vec.norm(p=2)

            tda_scores: List[float] = []
            for i in range(n):
                denom = norms[i] * float(g_norm.item())
                if denom <= self.eps:
                    tda_scores.append(1.0)
                else:
                    tda_scores.append(float((torch.dot(delta_mat[i], global_vec) / denom).item()))

            principal_sign = torch.sign(torch.sign(delta_mat).sum(dim=0))
            if bool((principal_sign == 0).any()):
                raw_sign = torch.sign(delta_mat.sum(dim=0))
                principal_sign = torch.where(principal_sign == 0, raw_sign, principal_sign)
                principal_sign = torch.where(
                    principal_sign == 0, torch.ones_like(principal_sign), principal_sign
                )

            mpsa_scores = [self._compute_mpsa(delta_mat[i], principal_sign) for i in range(n)]
            tda_z = self._zscores(tda_scores)
            mpsa_z = self._zscores(mpsa_scores)

            benign_idx = [
                i
                for i, (tz, mz) in enumerate(zip(tda_z, mpsa_z))
                if tz <= self.tda_threshold and mz <= self.mpsa_threshold
            ] or list(range(n))

        benign_norms = [norms[i] for i in benign_idx]
        clip_value = float(torch.median(torch.tensor(benign_norms)).item()) if benign_norms else 0.0

        benign_deltas = [updates[i].param_delta for i in benign_idx]
        clipped = F.clip_deltas_by_norm(
            benign_deltas,
            max_norm=clip_value,
            norms=benign_norms,
            device=ctx.device,
            eps=self.eps,
        )
        param_delta = F.weighted_average_dicts(clipped, [1.0] * len(clipped), device=ctx.device)

        benign_buffers = [updates[i].buffers for i in benign_idx if updates[i].buffers]
        buffers = (
            F.weighted_average_dicts(benign_buffers, [1.0] * len(benign_buffers), device=ctx.device)
            if benign_buffers
            else None
        )
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={
                "defense/kept_clients": float(len(benign_idx)),
                "defense/clip_value": clip_value,
            },
        )
