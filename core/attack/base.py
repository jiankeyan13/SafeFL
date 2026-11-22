"""Single-layer BaseAttacker and built-in data-poisoning attacks for SafeFL v2.

``BaseAttacker`` is a peer of ``BenignClient`` (both implement ``step(ctx) -> ClientUpdate``)
and never inherits from ``BenignClient``. Every attacker composes a ``BasePoisoner``
component out of the box, so subclasses only write their own training/crafting logic.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch

from core.attack.poisoner import BasePoisoner, PatchValue, build_poisoner
from core.types import ClientContext, ClientUpdate
from core.utils.config import AttackStrategyConfig
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model


def collect_raw_batches(
    ctx: ClientContext,
    max_batches: Optional[int] = None,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Fetch local batches in physical ``[0, 1]`` pixel space plus normalization stats."""
    spec = ctx.data.source.spec
    c = len(spec.mean)
    mean_t = torch.tensor(spec.mean, device=ctx.device, dtype=torch.float32).view(1, c, 1, 1)
    std_t = torch.tensor(spec.std, device=ctx.device, dtype=torch.float32).view(1, c, 1, 1)

    indices = ctx.data.indices
    bs = max(1, int(ctx.batch_size))
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(indices), bs):
        if max_batches is not None and len(batches) >= max_batches:
            break
        raw_x, raw_y = ctx.data.source.fetch_batch(
            indices[start : start + bs], device=ctx.device
        )
        batches.append((raw_x, raw_y))
    return batches, mean_t, std_t


class BaseAttacker:
    """Base class for all backdoor attackers (1 attack = 1 file, zero Client subclassing)."""

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
            self.poisoner = build_poisoner(p_name, **{**base_kwargs, **p_cfg})
        else:
            self.poisoner = self._build_default_poisoner(base_kwargs)

    def _build_default_poisoner(self, base_kwargs: Dict[str, Any]) -> BasePoisoner:
        return build_poisoner(self.default_poisoner_name, **base_kwargs)

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


@ATTACK_REGISTRY.register("semantic")
class SemanticAttack(BaseAttacker):
    """Semantic backdoor data poisoning attack."""

    default_poisoner_name = "semantic"


@ATTACK_REGISTRY.register("edge_case")
class EdgeCaseAttack(BaseAttacker):
    """Edge-case OOD backdoor data poisoning attack."""

    default_poisoner_name = "edge_case"


def build_attacker(
    strategy: AttackStrategyConfig,
    *,
    seed: Optional[int] = None,
    attacker_rank: Optional[int] = None,
) -> BaseAttacker:
    """Instantiate a registered ``BaseAttacker`` directly from ``AttackStrategyConfig``."""
    params = dict(strategy.params)
    if strategy.poisoner is not None and "poisoner" not in params:
        params["poisoner"] = dict(strategy.poisoner)
    if seed is not None:
        params.setdefault("seed", int(seed))
    attacker: BaseAttacker = ATTACK_REGISTRY.build(strategy.name, **params)
    if attacker_rank is not None:
        attacker.poisoner.attacker_rank = int(attacker_rank)
    return attacker
