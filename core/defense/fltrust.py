"""FLTrust server-root-of-trust defense (Cao et al., NDSS 2021, single-file v2)."""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import (
    ClientUpdate,
    DefenseOutput,
    ServerContext,
    compute_param_delta,
    load_model_state,
    split_model_state,
)
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
        keys = F.ordered_keys(delta_0)
        flat_0 = F.flatten_delta(delta_0, keys, device=ctx.device)
        ref_norm = float(torch.norm(flat_0, p=2).item())

        if ref_norm < 1e-6:
            return DefenseOutput(
                param_delta=F.zero_delta_like(updates[0].param_delta, device=ctx.device),
                buffers=None,
                metrics={"defense/accepted_clients": 0.0, "defense/ref_norm": ref_norm},
            )

        scaled_deltas = []
        accepted_buffers = []
        trust_weights: List[float] = []

        for u in updates:
            flat_i = F.flatten_delta(u.param_delta, keys, device=ctx.device)
            norm_i = float(torch.norm(flat_i, p=2).item())
            cos_sim = float(
                torch.clamp(
                    (flat_0 @ flat_i) / ((ref_norm + self.eps) * (norm_i + self.eps)),
                    -1.0,
                    1.0,
                ).item()
            )
            ts = max(0.0, cos_sim)
            if ts <= 0.0:
                continue
            scale = (ref_norm + self.eps) / (norm_i + self.eps)
            scaled_deltas.append(F.scale_delta(u.param_delta, scale, device=ctx.device))
            if u.buffers:
                accepted_buffers.append(u.buffers)
            trust_weights.append(ts)

        if not scaled_deltas or sum(trust_weights) <= 0.0:
            return DefenseOutput(
                param_delta=F.zero_delta_like(updates[0].param_delta, device=ctx.device),
                buffers=None,
                metrics={"defense/accepted_clients": 0.0, "defense/ref_norm": ref_norm},
            )

        param_delta = F.weighted_average_dicts(scaled_deltas, trust_weights, device=ctx.device)
        buffers = (
            F.weighted_average_dicts(accepted_buffers, trust_weights, device=ctx.device)
            if accepted_buffers
            else None
        )
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={
                "defense/accepted_clients": float(len(trust_weights)),
                "defense/ref_norm": ref_norm,
            },
        )
