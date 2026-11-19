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


def resolve_pgd_epsilon(
    model: nn.Module,
    global_params: Mapping[str, torch.Tensor],
    *,
    dataset_name: Optional[str] = None,
    cifar10_epsilon_scale: float = 0.1,
    default_epsilon_scale: float = 1.0,
) -> float:
    w0_norm = compute_parameter_l2_norm(model, global_params)
    is_cifar10 = bool(dataset_name and dataset_name.strip().lower() == "cifar10")
    scale = cifar10_epsilon_scale if is_cifar10 else default_epsilon_scale
    return float(scale) * w0_norm


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
        dataset: Optional[str] = None,
        cifar10_epsilon_scale: float = 0.1,
        default_epsilon_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset
        self.cifar10_epsilon_scale = float(cifar10_epsilon_scale)
        self.default_epsilon_scale = float(default_epsilon_scale)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        dataset_name = self.dataset or ctx.data.source.spec.name
        epsilon = resolve_pgd_epsilon(
            ctx.model,
            ctx.initial_params,
            dataset_name=dataset_name,
            cifar10_epsilon_scale=self.cifar10_epsilon_scale,
            default_epsilon_scale=self.default_epsilon_scale,
        )
        poison_loader = ctx.loader(poisoner=self.poisoner, seed=self.seed)
        stats = train_model(
            ctx.model,
            poison_loader,
            ctx.train_cfg,
            post_epoch_fn=lambda _ep, m: project_model_l2_(m, ctx.initial_params, epsilon),
        )
        return ctx.pack_update(metrics={"loss": stats.loss})
