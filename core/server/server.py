"""Standard Federated Learning Server for SafeFL v2."""
from __future__ import annotations

import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from core.defense import BaseDefense, FedAvgDefense
from core.defense import functional as F
from core.types import (
    ClientUpdate,
    ServerBroadcast,
    ServerContext,
    StateDict,
    apply_update_to_model,
    split_model_state,
)
from core.utils.evaluator import Evaluator


class Server:
    """Clean FL Server managing global model, broadcast, defense aggregation, and evaluation."""

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

    def select_clients(self, client_ids: Sequence[int], num_select: int) -> List[int]:
        """Uniformly sample ``num_select`` participants from ``client_ids``."""
        k = max(1, min(int(num_select), len(client_ids)))
        return self.rng.sample(list(client_ids), k)

    def broadcast(self, round_idx: int, lr: float) -> ServerBroadcast:
        """Package current global model state & round metadata into ``ServerBroadcast``."""
        params, buffers = split_model_state(self.global_model)
        return ServerBroadcast(
            round_idx=int(round_idx),
            params=params,
            buffers=buffers,
            lr=float(lr),
            prev_delta=self.last_aggregated_delta,
        )

    def step(
        self,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> Dict[str, float]:
        """Run ``self.defense`` on client updates and update ``self.global_model``."""
        if not updates:
            return {}

        out = self.defense.defend(self.global_model, updates, ctx)
        apply_update_to_model(self.global_model, out.param_delta, out.buffers)

        if self.calibrate_bn_on_proxy and ctx.proxy_data is not None and len(ctx.proxy_data) > 0:
            proxy_loader = ctx.proxy_data.loader(
                batch_size=ctx.train_cfg.batch_size,
                shuffle=True,
                device=self.device,
            )
            F.calibrate_bn(self.global_model, proxy_loader, self.device)

        self.last_aggregated_delta = {
            k: v.detach().clone() for k, v in out.param_delta.items()
        }
        return dict(out.metrics)

    def evaluate(
        self,
        evaluator: Evaluator,
        dataloader: Iterable[Any],
        criterion: Optional[nn.Module],
    ) -> Dict[str, float]:
        """Evaluate ``self.global_model`` on ``dataloader``."""
        return evaluator.evaluate(self.global_model, dataloader, criterion, self.device)
