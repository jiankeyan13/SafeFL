"""Single-layer BaseAttacker and built-in data-poisoning attacks for SafeFL v2.

``BaseAttacker`` is a peer of ``BenignClient`` (both implement ``step(ctx) -> ClientUpdate``)
and never inherits from ``BenignClient``. Every attacker composes a ``BasePoisoner``
component out of the box, so subclasses only write their own training/crafting logic.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Union

from core.attack.data.poisoner import BasePoisoner, build_poisoner
from core.attack.data.triggers import PatchValue
from core.types import ClientContext, ClientUpdate
from core.utils.config import AttackStrategyConfig
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model


class BaseAttacker:
    """Base class for all backdoor attackers (1 attack = 1 file, zero Client subclassing).

    Subclasses only need to accept their own algorithm hyperparameters and forward
    ``**kwargs`` to ``super().__init__(**kwargs)`` to inherit full ``Poisoner``
    configuration (``target_label``, ``poison_ratio``, ``patch_size``, ``poisoner``, etc.).
    """

    default_poisoner_name: str = "patch"
    needs_prev_global_delta: bool = False

    def __init__(
        self,
        *,
        poisoner: Optional[Union[BasePoisoner, Mapping[str, Any], str]] = None,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: PatchValue = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **extra_poisoner_kwargs: Any,
    ) -> None:
        self.seed = seed
        base_kwargs: Dict[str, Any] = {
            "target_label": int(target_label),
            "poison_ratio": float(poison_ratio),
            "patch_size": int(patch_size),
            "patch_value": patch_value,
            "patch_location": str(patch_location),
            "seed": seed,
            "exclude_target_class": bool(exclude_target_class),
            **extra_poisoner_kwargs,
        }

        if isinstance(poisoner, BasePoisoner):
            self.poisoner = poisoner
        elif isinstance(poisoner, str):
            self.poisoner = build_poisoner(poisoner, **base_kwargs)
        elif isinstance(poisoner, Mapping):
            p_cfg = dict(poisoner)
            p_name = str(p_cfg.pop("name", self.default_poisoner_name))
            merged = {**base_kwargs, **p_cfg}
            self.poisoner = build_poisoner(p_name, **merged)
        else:
            self.poisoner = build_poisoner(self.default_poisoner_name, **base_kwargs)

    @property
    def target_label(self) -> int:
        return self.poisoner.target_label

    @property
    def poison_ratio(self) -> float:
        return self.poisoner.poison_ratio

    def clone_poisoner(self, *, poison_ratio: Optional[float] = None) -> BasePoisoner:
        """Clone ``self.poisoner`` (optionally overriding ``poison_ratio``, e.g. for 100% probes)."""
        cloned = copy.deepcopy(self.poisoner)
        if poison_ratio is not None:
            cloned.poison_ratio = float(poison_ratio)
        return cloned

    def step(self, ctx: ClientContext) -> ClientUpdate:
        """Default backdoor step: train on ``self.poisoner``-poisoned local data."""
        loader = ctx.loader(poisoner=self.poisoner)
        stats = train_model(ctx.model, loader, ctx.train_cfg)
        return ctx.pack_update(metrics={"loss": stats.loss})


@ATTACK_REGISTRY.register("badnets")
class BadNetsAttack(BaseAttacker):
    """Standard BadNets corner-patch data poisoning attack."""

    default_poisoner_name = "patch"


@ATTACK_REGISTRY.register("dba")
class DBAAttack(BaseAttacker):
    """Distributed Backdoor Attack (sub-block triggers during train, full patch at eval)."""

    default_poisoner_name = "dba"

    def __init__(self, num_blocks: int = 4, **kwargs: Any) -> None:
        super().__init__(num_blocks=int(num_blocks), **kwargs)


@ATTACK_REGISTRY.register("blend")
class BlendAttack(BaseAttacker):
    """Blended pattern data poisoning attack."""

    default_poisoner_name = "blend"

    def __init__(self, alpha: float = 0.2, pattern_seed: int = 2024, **kwargs: Any) -> None:
        super().__init__(alpha=float(alpha), pattern_seed=int(pattern_seed), **kwargs)


def build_attacker(
    strategy: AttackStrategyConfig,
    *,
    seed: Optional[int] = None,
) -> BaseAttacker:
    """Instantiate a registered ``BaseAttacker`` directly from ``AttackStrategyConfig``."""
    params = dict(strategy.params)
    if strategy.poisoner is not None and "poisoner" not in params:
        params["poisoner"] = dict(strategy.poisoner)
    if seed is not None:
        params.setdefault("seed", int(seed))
    return ATTACK_REGISTRY.build(strategy.name.strip().lower(), **params)
