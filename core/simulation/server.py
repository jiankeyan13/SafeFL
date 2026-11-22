"""Standard Federated Learning Server for SafeFL v2."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from core.defense import BaseDefense, FedAvgDefense
from core.types import (
    ClientUpdate,
    ServerBroadcast,
    ServerContext,
    StateDict,
    apply_update_to_model,
    split_model_state,
)
from core.utils import functional as F


class Server:
    """Clean FL Server managing global model, broadcast, and defense aggregation."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        defense: Optional[BaseDefense] = None,
        *,
        seed: int = 42,
        calibrate_bn_on_proxy: bool = False,
    ) -> None:
        self.global_model = model.to(device)
        self.device = device
        self.defense: BaseDefense = defense or FedAvgDefense()
        self.rng = random.Random(int(seed))
        self.calibrate_bn_on_proxy = bool(calibrate_bn_on_proxy)
        self.last_aggregated_delta: Optional[StateDict] = None
        self.last_client_weights: Optional[Dict[int, float]] = None

    def select_clients(self, client_ids: Sequence[int], num_select: int) -> List[int]:
        """Uniformly sample ``num_select`` participants from ``client_ids``."""
        k = max(1, min(int(num_select), len(client_ids)))
        return self.rng.sample(list(client_ids), k)

    def broadcast(
        self,
        round_idx: int,
        lr: float,
        *,
        include_prev_delta: bool = False,
        ctx: Optional[ServerContext] = None,
    ) -> ServerBroadcast:
        """Package current global model state & round metadata into ``ServerBroadcast``."""
        override = self.defense.pre_broadcast(self.global_model, ctx) if ctx is not None else None
        if override is not None:
            params, buffers = override
        else:
            params, buffers = split_model_state(self.global_model)
        extra = (
            {"client_extras": tuple(self.defense.client_extras)}
            if self.defense.client_extras
            else {}
        )
        return ServerBroadcast(
            round_idx=int(round_idx),
            params=params,
            buffers=buffers,
            lr=float(lr),
            prev_delta=self.last_aggregated_delta if include_prev_delta else None,
            extra=extra,
        )

    def step(
        self,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> Dict[str, float]:
        """Run ``self.defense`` on client updates and update ``self.global_model``."""
        if not updates:
            self.last_client_weights = None
            return {}

        out = self.defense.defend(self.global_model, updates, ctx)
        self.last_client_weights = (
            {int(k): float(v) for k, v in out.client_weights.items()}
            if out.client_weights is not None
            else None
        )
        apply_update_to_model(self.global_model, out.param_delta, out.buffers)

        if self.calibrate_bn_on_proxy and ctx.proxy_data is not None and len(ctx.proxy_data) > 0:
            proxy_loader = ctx.proxy_data.loader(
                batch_size=ctx.train_cfg.batch_size,
                shuffle=True,
                device=self.device,
                round_idx=ctx.round_idx,
                seed=ctx.seed,
            )
            F.calibrate_bn(self.global_model, proxy_loader, self.device)

        self.last_aggregated_delta = {k: v.detach() for k, v in out.param_delta.items()}
        return dict(out.metrics)
