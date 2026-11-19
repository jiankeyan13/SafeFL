"""LGA: Adaptive Layer-wise Gradient Alignment (ICCV 2025, single-file v2)."""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import torch
import torch.nn as nn

from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import build_optimizer, train_model


@torch.no_grad()
def project_model_layerwise_(
    model: nn.Module,
    global_params: Mapping[str, torch.Tensor],
    prev_global_delta: Optional[Mapping[str, torch.Tensor]],
    *,
    eps: float = 1e-8,
    skip_keys: Optional[Iterable[str]] = None,
) -> None:
    """Scale each layer's parameter delta in-place so ``||Delta_m^l|| <= ||Delta_g^{t-1,l}||``."""
    if prev_global_delta is None:
        return
    skipped = set(skip_keys or ())

    for name, param in model.named_parameters():
        if name in skipped or name not in global_params or name not in prev_global_delta:
            continue
        base = global_params[name].to(device=param.device, dtype=param.dtype)
        ref = prev_global_delta[name].to(device=param.device, dtype=torch.float32)
        delta = param.detach() - base
        delta_norm = float(torch.norm(delta.float()).item())
        ref_norm = float(torch.norm(ref).item())
        scale = min(1.0, ref_norm / (delta_norm + float(eps)))
        if scale < 1.0:
            param.copy_(base + delta * scale)


@ATTACK_REGISTRY.register("lga")
class LGAAttack(BaseAttacker):
    """Two-phase clean pretrain + poisoned training with per-epoch layerwise alignment."""

    def __init__(self, epoch_clean: int = 0, eps: float = 1e-8, **kwargs: Any) -> None:
        if epoch_clean < 0:
            raise ValueError(f"epoch_clean must be >= 0, got {epoch_clean}")
        super().__init__(**kwargs)
        self.epoch_clean = int(epoch_clean)
        self.eps = float(eps)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        # Share one optimizer across clean + poison phases to preserve momentum buffer.
        opt = build_optimizer(ctx.model, ctx.train_cfg)
        total_loss = 0.0
        total_samples = 0

        if self.epoch_clean > 0:
            clean_stats = train_model(
                ctx.model,
                ctx.loader(seed=self.seed),
                ctx.train_cfg,
                epochs=self.epoch_clean,
                optimizer=opt,
            )
            total_loss += clean_stats.loss * clean_stats.num_samples
            total_samples += clean_stats.num_samples

        prev_delta = ctx.prev_global_delta
        poison_stats = train_model(
            ctx.model,
            ctx.loader(poisoner=self.poisoner, seed=self.seed),
            ctx.train_cfg,
            optimizer=opt,
            post_epoch_fn=lambda _ep, m: project_model_layerwise_(
                m, ctx.initial_params, prev_delta, eps=self.eps
            ),
        )
        total_loss += poison_stats.loss * poison_stats.num_samples
        total_samples += poison_stats.num_samples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return ctx.pack_update(metrics={"loss": avg_loss})
