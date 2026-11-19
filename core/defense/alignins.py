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
        norms = torch.norm(delta_mat, p=2, dim=1)

        if n < 2 or delta_mat.shape[1] == 0:
            benign_idx = list(range(n))
        else:
            global_params, _ = split_model_state(global_model)
            global_vec = F.flatten_delta(global_params, keys, device=ctx.device)
            g_norm = global_vec.norm(p=2)
            denom = norms * g_norm
            dots = delta_mat @ global_vec
            tda_t = torch.where(denom <= self.eps, torch.ones_like(dots), dots / denom.clamp_min(self.eps))
            tda_scores = tda_t.tolist()

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

        benign_norms = norms[benign_idx]
        clip_value = float(torch.median(benign_norms).item()) if benign_norms.numel() > 0 else 0.0

        param_delta = F.clip_and_average_flat(
            delta_mat,
            benign_idx,
            clip_value,
            updates[0].param_delta,
            keys,
            norms=norms,
            eps=self.eps,
        )
        buffers = F.average_buffers(updates, benign_idx, device=ctx.device)

        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={
                "defense/kept_clients": float(len(benign_idx)),
                "defense/clip_value": clip_value,
            },
        )
