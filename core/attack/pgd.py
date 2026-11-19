"""PGD-constrained backdoor attack (single-file v2 implementation)."""
from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model


def compute_parameter_l2_norm(model: nn.Module, state: Mapping[str, torch.Tensor]) -> float:
    """Return L2 norm of trainable parameters in ``state``."""
    sq_norm: Optional[torch.Tensor] = None
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in state:
            continue
        val = state[name].detach().to(device=param.device, dtype=torch.float64)
        term = val.square().sum()
        sq_norm = term if sq_norm is None else sq_norm + term
    return 0.0 if sq_norm is None else float(sq_norm.sqrt().item())


@torch.no_grad()
def project_model_l2_(
    model: nn.Module,
    global_params: Mapping[str, torch.Tensor],
    epsilon: float,
) -> Tuple[float, float]:
    """Project trainable parameters in-place into an L2 ball of radius ``epsilon`` around ``global_params``."""
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon}")

    entries = []
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in global_params:
            continue
        ref = global_params[name].to(device=param.device, dtype=param.dtype)
        entries.append((param, ref, param.detach() - ref))

    if not entries:
        return 0.0, 1.0

    sq_norm = torch.zeros((), device=entries[0][0].device, dtype=torch.float64)
    for _, _, delta in entries:
        sq_norm.add_(delta.double().square().sum())
    distance = float(sq_norm.sqrt().item())
    scale = min(1.0, epsilon / distance) if distance > 0.0 else 1.0

    if scale < 1.0:
        for param, ref, delta in entries:
            param.copy_(ref + delta * scale)
    return distance, scale


@ATTACK_REGISTRY.register("pgd")
class PGDAttack(BaseAttacker):
    """Poisoned local training with per-epoch L2 projection onto ``||w - W_g||_2 <= eps``."""

    def __init__(
        self,
        epsilon_scale: float = 0.1,
        epsilon: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon_scale = float(epsilon_scale)
        self.fixed_epsilon = None if epsilon is None else float(epsilon)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        epsilon = (
            self.fixed_epsilon
            if self.fixed_epsilon is not None
            else self.epsilon_scale * compute_parameter_l2_norm(ctx.model, ctx.initial_params)
        )
        poison_loader = ctx.loader(poisoner=self.poisoner)
        stats = train_model(
            ctx.model,
            poison_loader,
            ctx.train_cfg,
            post_epoch_fn=lambda _ep, m: project_model_l2_(m, ctx.initial_params, epsilon),
        )
        return ctx.pack_update(metrics={"loss": stats.loss})
