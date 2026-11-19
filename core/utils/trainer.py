"""Standalone local training utilities for BenignClient, Attacker, and Server proxy steps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Optional, Union

import torch
import torch.nn as nn

from core.types import TrainConfig


@dataclass
class TrainStats:
    """Summary returned by ``train_model``."""

    loss: float
    num_samples: int
    num_steps: int


def build_optimizer(
    model_or_params: Union[nn.Module, Iterable[nn.Parameter]],
    train_cfg: Optional[TrainConfig] = None,
    *,
    lr: Optional[float] = None,
) -> torch.optim.Optimizer:
    """Build a PyTorch optimizer from ``TrainConfig`` for a model or parameter subset."""
    cfg = train_cfg or TrainConfig()
    if isinstance(model_or_params, nn.Module):
        params: List[nn.Parameter] = [p for p in model_or_params.parameters() if p.requires_grad]
    else:
        params = [p for p in model_or_params if p.requires_grad]

    if not params:
        raise ValueError("build_optimizer received no trainable parameters.")

    opt_name = cfg.optimizer
    effective_lr = float(cfg.lr if lr is None else lr)
    kwargs: dict[str, Any] = {"lr": effective_lr}

    if opt_name == "SGD":
        kwargs["momentum"] = float(cfg.momentum)
        kwargs["weight_decay"] = float(cfg.weight_decay)
    elif opt_name in {"Adam", "AdamW"}:
        kwargs["weight_decay"] = float(cfg.weight_decay)

    kwargs.update(cfg.extra.get("optimizer_kwargs", {}))
    opt_cls = getattr(torch.optim, opt_name)
    return opt_cls(params, **kwargs)


def build_criterion(train_cfg: Optional[TrainConfig] = None) -> nn.Module:
    """Build a PyTorch loss criterion from ``TrainConfig``."""
    cfg = train_cfg or TrainConfig()
    kwargs = cfg.extra.get("criterion_kwargs", {})
    crit_cls = getattr(nn, cfg.criterion)
    return crit_cls(**kwargs)


def _infer_model_device(model: nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    for buf in model.buffers():
        return buf.device
    return torch.device("cpu")


def train_model(
    model: nn.Module,
    loader: Iterable[Any],
    train_cfg: Optional[TrainConfig] = None,
    *,
    epochs: Optional[int] = None,
    lr: Optional[float] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    device: Optional[Union[str, torch.device]] = None,
    grad_mask: Optional[Mapping[str, torch.Tensor]] = None,
    post_epoch_fn: Optional[Callable[[int, nn.Module], None]] = None,
) -> TrainStats:
    """Run standard supervised local training for ``epochs`` passes over ``loader``.

    Optional lightweight hooks cover common backdoor interventions without subclassing:
      - ``grad_mask``     : per-parameter mask multiplied onto ``param.grad`` before step
      - ``post_epoch_fn`` : ``fn(epoch_idx, model)`` run at the end of each local epoch
    """
    cfg = train_cfg or TrainConfig()
    num_epochs = int(cfg.epochs if epochs is None else epochs)
    if num_epochs <= 0:
        return TrainStats(loss=0.0, num_samples=0, num_steps=0)

    dev = torch.device(device) if device is not None else _infer_model_device(model)
    model.train()

    opt = optimizer or build_optimizer(model, cfg, lr=lr)
    loss_fn = criterion or build_criterion(cfg)
    max_grad_norm = cfg.grad_clip_norm

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    named_params = dict(model.named_parameters()) if grad_mask else {}

    total_loss = 0.0
    total_samples = 0
    total_steps = 0

    for epoch_idx in range(num_epochs):
        for bx, by in loader:
            if bx.device != dev:
                bx = bx.to(dev, non_blocking=True)
            if by.device != dev:
                by = by.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss_val = loss_fn(logits, by)
            loss_val.backward()

            if grad_mask:
                with torch.no_grad():
                    for name, mask in grad_mask.items():
                        param = named_params.get(name)
                        if param is not None and param.grad is not None:
                            param.grad.mul_(mask.to(device=param.grad.device, dtype=param.grad.dtype))

            if max_grad_norm is not None and max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(max_grad_norm))

            opt.step()

            bsz = int(by.size(0))
            total_loss += float(loss_val.detach().item()) * bsz
            total_samples += bsz
            total_steps += 1

        if post_epoch_fn is not None:
            post_epoch_fn(epoch_idx, model)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return TrainStats(loss=avg_loss, num_samples=total_samples, num_steps=total_steps)
