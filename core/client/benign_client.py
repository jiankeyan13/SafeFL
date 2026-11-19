"""Standard honest Federated Learning client for SafeFL v2."""
from __future__ import annotations

from core.types import ClientContext, ClientUpdate
from core.utils.trainer import train_model


class BenignClient:
    """Honest FL participant: trains locally on clean data and packages update."""

    def step(self, ctx: ClientContext) -> ClientUpdate:
        stats = train_model(ctx.model, ctx.loader(), ctx.train_cfg)
        return ctx.pack_update(metrics={"loss": stats.loss})
