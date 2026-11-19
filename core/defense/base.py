"""Single-class BaseDefense, FedAvgDefense, and defense builder for SafeFL v2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping

import torch.nn as nn

from core.config import ComponentConfig
from core.defense import functional as F
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils.registry import DEFENSE_REGISTRY


class BaseDefense(ABC):
    """Base class for all server-side defenses (1 defense = 1 file).

    Replaces the fragmented v1 ``Screener + Aggregator + Refiner + Algorithm``
    4-file split and untyped ``context`` dict with a single clean ``defend`` method.
    """

    def __init__(self, **_: Any) -> None:
        pass

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
        return DefenseOutput(param_delta=param_delta, buffers=buffers)


def flatten_legacy_defense_params(raw_params: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten legacy ``screener / aggregator / refiner`` YAML sections into flat kwargs."""
    flat: Dict[str, Any] = {}
    nested_sections = {"screener", "aggregator", "refiner"}
    for key, value in raw_params.items():
        if key in nested_sections and isinstance(value, Mapping):
            inner = value.get("params") if isinstance(value.get("params"), Mapping) else value
            flat.update(inner)
        elif key == "client" and isinstance(value, Mapping):
            continue
        else:
            flat[key] = value
    return flat


def build_defense(cfg: ComponentConfig, *, seed: int = 42) -> BaseDefense:
    """Instantiate a registered ``BaseDefense`` from ``ComponentConfig``."""
    params = flatten_legacy_defense_params(cfg.params)
    params.setdefault("seed", int(seed))
    return DEFENSE_REGISTRY.build(cfg.name.strip().lower(), **params)
