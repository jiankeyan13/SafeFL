"""Standard honest Federated Learning client for SafeFL v2."""
from __future__ import annotations

import torch
import torch.nn as nn

from core.types import ClientContext, ClientUpdate, StateDict
from core.utils.trainer import build_criterion, train_model


def _compute_diagonal_fisher(
    model: nn.Module,
    ctx: ClientContext,
) -> StateDict:
    """Estimate per-parameter diagonal Fisher information on local data."""
    fisher: StateDict = {
        name: torch.zeros_like(param)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    if not fisher:
        return fisher

    was_training = model.training
    model.eval()
    criterion = build_criterion(ctx.train_cfg)
    total = 0
    for bx, by in ctx.loader(shuffle=False):
        bx = bx.to(ctx.device, non_blocking=True)
        by = by.to(ctx.device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        loss = criterion(model(bx), by)
        loss.backward()
        bsz = int(by.size(0))
        total += bsz
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None and name in fisher:
                fisher[name].add_(param.grad.detach().pow(2), alpha=float(bsz))
    if total > 0:
        for t in fisher.values():
            t.div_(float(total))
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return fisher


_CLIENT_EXTRA_HANDLERS = {
    "fisher": _compute_diagonal_fisher,
}


class BenignClient:
    """Honest FL participant: trains locally on clean data and packages update."""

    def step(self, ctx: ClientContext) -> ClientUpdate:
        stats = train_model(ctx.model, ctx.loader(), ctx.train_cfg)
        extra = {}
        for key in ctx.broadcast.extra.get("client_extras", ()):
            handler = _CLIENT_EXTRA_HANDLERS.get(key)
            if handler is not None:
                extra[key] = handler(ctx.model, ctx)
        return ctx.pack_update(metrics={"loss": stats.loss}, extra=extra)
