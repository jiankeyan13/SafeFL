"""Single-class BaseDefense, FedAvgDefense, and defense builder for SafeFL v2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch.nn as nn

from core.types import ClientUpdate, DefenseOutput, ServerContext, StateDict
from core.utils import functional as F
from core.utils.config import ComponentConfig
from core.utils.registry import DEFENSE_REGISTRY


class BaseDefense(ABC):
    """Base class for all server-side defenses (1 defense = 1 file)."""

    client_extras: Tuple[str, ...] = ()

    def __init__(self, **_: Any) -> None:
        pass

    def pre_broadcast(
        self,
        global_model: nn.Module,
        ctx: ServerContext,
    ) -> Optional[Tuple[StateDict, StateDict]]:
        """Optional hook run before ``Server.broadcast`` (returns ``(params, buffers)`` override)."""
        del global_model, ctx
        return None

    @abstractmethod
    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        """Sanitize/aggregate client ``updates`` and return ``DefenseOutput``."""


@DEFENSE_REGISTRY.register("fedavg")
@DEFENSE_REGISTRY.register("vanilla")
class FedAvgDefense(BaseDefense):
    """Standard Federated Averaging (weighted by client sample count by default)."""

    def __init__(self, weight_by_samples: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.weight_by_samples = bool(weight_by_samples)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        param_delta, buffers = F.aggregate_updates(
            updates,
            weight_by_samples=self.weight_by_samples,
            device=ctx.device,
        )
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            client_weights=build_client_weights(updates, weight_by_samples=self.weight_by_samples),
        )


def build_client_weights(
    updates: Sequence[ClientUpdate],
    *,
    kept_updates: Optional[Sequence[ClientUpdate]] = None,
    kept_indices: Optional[Iterable[int]] = None,
    weight_by_samples: bool = False,
) -> Dict[int, float]:
    """Build per-client aggregation weights (0.0 for excluded clients)."""
    if kept_indices is not None:
        kept_ids = {updates[i].client_id for i in kept_indices}
    elif kept_updates is not None:
        kept_ids = {u.client_id for u in kept_updates}
    else:
        kept_ids = {u.client_id for u in updates}

    return {
        u.client_id: (
            (float(max(u.num_samples, 1)) if weight_by_samples else 1.0)
            if u.client_id in kept_ids
            else 0.0
        )
        for u in updates
    }


def build_defense(cfg: ComponentConfig, *, seed: int = 42) -> BaseDefense:
    """Instantiate a registered ``BaseDefense`` from ``ComponentConfig``."""
    params = dict(cfg.params)
    params.setdefault("seed", int(seed))
    return DEFENSE_REGISTRY.build(cfg.name, **params)
