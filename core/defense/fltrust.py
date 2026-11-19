"""FLTrust server-root-of-trust defense (Cao et al., NDSS 2021, single-file v2)."""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from core.defense.base import BaseDefense
from core.types import (
    ClientUpdate,
    DefenseOutput,
    ServerContext,
    compute_param_delta,
    load_model_state,
    split_model_state,
)
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY
from core.utils.trainer import train_model


@DEFENSE_REGISTRY.register("fltrust")
class FLTrustDefense(BaseDefense):
    """ReLU cosine trust scoring + norm normalization anchored by server proxy update."""

    def __init__(
        self,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        momentum: Optional[float] = None,
        weight_decay: Optional[float] = None,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.eps = float(eps)

    def _compute_reference_delta(
        self,
        global_model: nn.Module,
        ctx: ServerContext,
    ):
        if ctx.proxy_data is None or len(ctx.proxy_data) == 0:
            raise ValueError(
                "FLTrust requires server proxy data. Set `data.enable_proxy=true` in config."
            )

        init_params, init_buffers = split_model_state(global_model)
        server_cfg = ctx.train_cfg.with_overrides(
            lr=self.lr if self.lr is not None else ctx.client_lr,
            epochs=self.epochs if self.epochs is not None else ctx.train_cfg.epochs,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        proxy_loader = ctx.proxy_data.loader(
            batch_size=server_cfg.batch_size,
            shuffle=True,
            device=ctx.device,
            round_idx=ctx.round_idx,
        )
        try:
            train_model(global_model, proxy_loader, server_cfg, device=ctx.device)
            delta_0, _ = compute_param_delta(global_model, init_params)
        finally:
            load_model_state(global_model, init_params, init_buffers)
        return delta_0

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        delta_0 = self._compute_reference_delta(global_model, ctx)
        delta_mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        flat_0 = F.flatten_delta(delta_0, keys, device=ctx.device)
        ref_norm = float(torch.norm(flat_0, p=2).item())

        if ref_norm < 1e-6:
            return DefenseOutput(
                param_delta=F.zero_delta_like(updates[0].param_delta, device=ctx.device),
                buffers=None,
                metrics={"defense/accepted_clients": 0.0, "defense/ref_norm": ref_norm},
            )

        norms = torch.norm(delta_mat, p=2, dim=1)
        cos_sim = torch.clamp(
            (delta_mat @ flat_0) / ((ref_norm + self.eps) * (norms + self.eps)),
            -1.0,
            1.0,
        )
        ts = torch.relu(cos_sim)
        keep_mask = ts > 0.0
        ts_sum = float(ts[keep_mask].sum().item())

        if not bool(keep_mask.any()) or ts_sum <= 0.0:
            return DefenseOutput(
                param_delta=F.zero_delta_like(updates[0].param_delta, device=ctx.device),
                buffers=None,
                metrics={"defense/accepted_clients": 0.0, "defense/ref_norm": ref_norm},
            )

        # Each accepted client's normalized contribution is: ts_i * ((ref_norm / norm_i) * delta_i)
        scales = (ref_norm + self.eps) / (norms[keep_mask] + self.eps)
        eff_coeff = (ts[keep_mask] * scales) / ts_sum
        agg_vec = (delta_mat[keep_mask] * eff_coeff.unsqueeze(1)).sum(dim=0)
        param_delta = F.unflatten_delta(agg_vec, updates[0].param_delta, keys)

        kept_idx = keep_mask.nonzero(as_tuple=True)[0].tolist()
        kept_ts = ts[keep_mask].tolist()
        buffers = F.average_buffers(updates, kept_idx, weights=kept_ts, device=ctx.device)

        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={
                "defense/accepted_clients": float(len(kept_idx)),
                "defense/ref_norm": ref_norm,
            },
        )
