"""FCBA: Full-Combination Backdoor Attack (AAAI 2024, single-file v2)."""
from __future__ import annotations

import itertools
from typing import Any, List, Optional, Sequence, Tuple

import torch

from core.attack.base import BaseAttacker
from core.attack.poisoner import (
    BasePoisoner,
    PatchValue,
    _VALID_LOCATIONS,
    stamp_dba_block_,
    stamp_patch_,
)
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model


def _fcba_combinations(num_blocks: int) -> List[Tuple[int, ...]]:
    """Enumerate all non-empty proper subsets of ``[0, num_blocks)``.

    For ``num_blocks == 4``, preserves the exact ordering from the official FCBA
    implementation (4 singletons, 4 adjacent pairs, 2 opposite pairs, 4 triples = 14).
    """
    if num_blocks < 2:
        return [(0,)]
    if num_blocks == 4:
        return [
            (0,), (1,), (2,), (3,),
            (0, 1), (1, 2), (2, 3), (3, 0),
            (0, 2), (1, 3),
            (0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1),
        ]
    combos: List[Tuple[int, ...]] = []
    for r in range(1, num_blocks):
        combos.extend(itertools.combinations(range(num_blocks), r))
    return combos


class FCBAPoisoner(BasePoisoner):
    """Combinatorial distributed trigger poisoner for FCBA.

    - Train mode: stamps a combination of sub-blocks determined by ``attacker_rank``
      (covering C(m, 1) .. C(m, m-1) combinations).
    - Eval mode: stamps the full assembled trigger patch to measure global ASR.
    """

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: PatchValue = 1.0,
        patch_location: str = "bottom_right",
        num_blocks: int = 4,
        combo_id: Optional[int] = None,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        if patch_location not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid location {patch_location!r}. Expected {_VALID_LOCATIONS}.")
        if patch_size <= 0 or num_blocks < 2:
            raise ValueError("patch_size must be positive and num_blocks >= 2.")
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.patch_size = int(patch_size)
        self.patch_value = patch_value
        self.patch_location = patch_location
        self.num_blocks = int(num_blocks)
        self.combo_id = None if combo_id is None else int(combo_id)
        self._combos = _fcba_combinations(self.num_blocks)

    def _resolve_blocks(self, client_id: Optional[int]) -> Tuple[int, ...]:
        if self.combo_id is not None:
            idx = self.combo_id
        elif self.attacker_rank is not None:
            idx = self.attacker_rank
        elif client_id is not None:
            idx = int(client_id)
        else:
            idx = 0
        return self._combos[idx % len(self._combos)]

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        for bid in self._resolve_blocks(client_id):
            stamp_dba_block_(
                bx,
                block_id=bid,
                num_blocks=self.num_blocks,
                patch_size=self.patch_size,
                patch_value=self.patch_value,
                location=self.patch_location,
                mask=mask,
            )
        return bx

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return stamp_patch_(
            bx,
            self.patch_size,
            self.patch_value,
            self.patch_location,
            mask=None,
        )


@ATTACK_REGISTRY.register("fcba")
class FCBAAttack(BaseAttacker):
    """FCBA: Full-Combination Backdoor Attack across sub-trigger combinations."""

    def __init__(
        self,
        num_blocks: int = 4,
        combo_id: Optional[int] = None,
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if "poisoner" not in kwargs or kwargs["poisoner"] is None:
            poisoner_keys = {
                "target_label",
                "poison_ratio",
                "patch_size",
                "patch_value",
                "patch_location",
                "seed",
                "exclude_target_class",
            }
            p_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in poisoner_keys}
            kwargs["poisoner"] = FCBAPoisoner(
                num_blocks=int(num_blocks),
                combo_id=combo_id,
                **p_kwargs,
            )
        super().__init__(**kwargs)
        self.scale = float(scale)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        if ctx.attacker_rank is not None:
            self.poisoner.attacker_rank = int(ctx.attacker_rank)
        stats = train_model(ctx.model, ctx.loader(poisoner=self.poisoner), ctx.train_cfg)
        if self.scale == 1.0:
            return ctx.pack_update(metrics={"loss": stats.loss})
        delta = {
            k: (p.detach() - ctx.initial_params[k]) * self.scale
            for k, p in ctx.model.named_parameters()
            if k in ctx.initial_params
        }
        return ctx.pack_update(param_delta=delta, metrics={"loss": stats.loss})
