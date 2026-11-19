"""BadNets dual-branch diagnostic attack (single-file v2 implementation)."""
from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Optional

import torch

from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate, split_model_state, to_cpu_state
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model


def _state_diff_cpu(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    return {
        k: (left[k].detach() - right[k].detach()).cpu()
        for k in left
        if k in right
    }


@ATTACK_REGISTRY.register("badnets_dual")
class BadNetsDualAttack(BaseAttacker):
    """Trains independent clean (Wc) and poison (Wb) branches from Wg; uploads Wb - Wg."""

    def __init__(
        self,
        delta_log_interval: int = 10,
        delta_log_dir: Optional[str] = None,
        delta_log_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.delta_log_interval = int(delta_log_interval)
        self.delta_log_dir = delta_log_dir
        self.delta_log_enabled = bool(delta_log_enabled)

    def _should_log_deltas(self, round_idx: int) -> bool:
        return (
            self.delta_log_enabled
            and self.delta_log_interval > 0
            and round_idx > 0
            and round_idx % self.delta_log_interval == 0
        )

    def _save_round_deltas(
        self,
        ctx: ClientContext,
        wb_params: Mapping[str, torch.Tensor],
        wc_params: Mapping[str, torch.Tensor],
    ) -> None:
        base_dir = self.delta_log_dir or (
            os.path.join(ctx.run_dir, "delta_logs") if ctx.run_dir else os.path.abspath("delta_logs")
        )
        save_dir = os.path.join(base_dir, f"round_{ctx.round_idx:03d}", f"client_{ctx.client_id}")
        os.makedirs(save_dir, exist_ok=True)

        torch.save(_state_diff_cpu(wb_params, wc_params), os.path.join(save_dir, "wb_minus_wc.pt"))
        torch.save(
            _state_diff_cpu(wc_params, ctx.initial_params), os.path.join(save_dir, "wc_minus_wg.pt")
        )
        torch.save(
            to_cpu_state(ctx.prev_global_delta) or {}, os.path.join(save_dir, "wg_minus_wg1.pt")
        )

    def step(self, ctx: ClientContext) -> ClientUpdate:
        # 1. Independent clean branch: Wg -> Wc
        train_model(ctx.model, ctx.loader(seed=self.seed), ctx.train_cfg)
        wc_params, _ = split_model_state(ctx.model)

        # 2. Reset to Wg, independent poison branch: Wg -> Wb
        ctx.reset_model()
        stats = train_model(
            ctx.model,
            ctx.loader(poisoner=self.poisoner, seed=self.seed),
            ctx.train_cfg,
        )
        wb_params, wb_buffers = split_model_state(ctx.model)

        if self._should_log_deltas(ctx.round_idx):
            self._save_round_deltas(ctx, wb_params, wc_params)

        param_delta = {k: wb_params[k] - ctx.initial_params[k] for k in ctx.initial_params}
        return ctx.pack_update(
            param_delta=param_delta,
            buffers=wb_buffers,
            metrics={"loss": stats.loss},
        )
