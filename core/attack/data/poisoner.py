"""Composable batch-vectorized Poisoners implementing ``data.BatchPoisoner``.

Decouples *how data is poisoned* (trigger + target label + poison ratio) from
*how the attacker trains or crafts model updates*, while executing directly on
``[B, C, H, W]`` GPU/CPU batches in physical ``[0, 1]`` pixel space *before*
dataset ``Normalize``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch

from core.attack.data.triggers import (
    BlendTrigger,
    PartialPatchTrigger,
    PatchTrigger,
    PatchValue,
)
from core.utils.registry import Registry
from data.loader import BatchPoisoner, derive_stable_seed

POISONER_REGISTRY = Registry("Poisoners")


class BasePoisoner(BatchPoisoner, ABC):
    """Base batch-level poisoner shared by training and global ASR evaluation."""

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
    ) -> None:
        if poison_ratio < 0.0:
            raise ValueError(f"poison_ratio must be >= 0, got {poison_ratio}")
        self.target_label = int(target_label)
        self.poison_ratio = float(poison_ratio)
        self.seed = seed
        self.exclude_target_class = bool(exclude_target_class)

    def build_mask(
        self,
        num_samples: int,
        *,
        targets: Optional[np.ndarray] = None,
        client_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Select a deterministic fixed subset of local samples to poison."""
        mask = np.zeros(int(num_samples), dtype=bool)
        if num_samples <= 0 or self.poison_ratio <= 0.0:
            return mask
        if self.poison_ratio >= 1.0 and not self.exclude_target_class:
            mask[:] = True
            return mask

        if self.exclude_target_class and targets is not None:
            candidates = np.where(np.asarray(targets) != self.target_label)[0]
        else:
            candidates = np.arange(num_samples, dtype=np.int64)

        if candidates.size == 0:
            return mask

        num_poison = min(int(num_samples * self.poison_ratio), int(candidates.size))
        if num_poison <= 0:
            return mask

        base_seed = self.seed if seed is None else seed
        eff_seed = (
            None
            if base_seed is None
            else derive_stable_seed(int(base_seed), "poison_mask", client_id)
        )
        rng = np.random.default_rng(eff_seed)
        chosen = rng.choice(candidates, size=num_poison, replace=False)
        mask[chosen] = True
        return mask

    @abstractmethod
    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply trigger in-place onto ``mask``-selected images in ``[0, 1]`` ``bx``."""

    @abstractmethod
    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply full evaluation trigger in-place onto all images in ``[0, 1]`` ``bx``."""

    def apply_train_batch(
        self,
        bx: torch.Tensor,
        by: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bool(mask.any()):
            bx = self._stamp_train(bx, mask, client_id=client_id, round_idx=round_idx)
            by[mask] = self.target_label
        return bx, by

    def apply_eval_batch(
        self,
        bx: torch.Tensor,
        by: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        orig_y = by
        bx = self._stamp_eval(bx, round_idx=round_idx)
        target_y = torch.full_like(orig_y, self.target_label)
        return bx, target_y, orig_y


@POISONER_REGISTRY.register("patch")
@POISONER_REGISTRY.register("badnets")
class PatchPoisoner(BasePoisoner):
    """Standard BadNets corner-patch poisoner."""

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: PatchValue = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.trigger = PatchTrigger(
            patch_size=patch_size,
            patch_value=patch_value,
            location=patch_location,
        )

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del client_id, round_idx
        return self.trigger(bx, mask=mask, inplace=True)

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return self.trigger(bx, mask=None, inplace=True)


@POISONER_REGISTRY.register("dba")
class DBAPoisoner(BasePoisoner):
    """Distributed Backdoor Attack (DBA) poisoner.

    - Train mode: stamps one sub-block (``client_block_map[client_id]`` when
      assigned by an ``AttackCoordinator``, otherwise ``client_id % num_blocks``).
    - Eval mode: stamps the full assembled patch to measure global ASR.
    """

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: PatchValue = 1.0,
        patch_location: str = "bottom_right",
        num_blocks: int = 4,
        block_id: Optional[int] = None,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.num_blocks = int(num_blocks)
        self.fixed_block_id = block_id
        self.client_block_map: Dict[int, int] = {}
        self.eval_trigger = PatchTrigger(
            patch_size=patch_size,
            patch_value=patch_value,
            location=patch_location,
        )
        self._block_triggers = [
            PartialPatchTrigger(
                patch_size=patch_size,
                patch_value=patch_value,
                location=patch_location,
                block_id=bid,
                num_blocks=self.num_blocks,
            )
            for bid in range(self.num_blocks)
        ]

    def set_client_block_map(self, mapping: Mapping[int, int]) -> None:
        """Allow an ``AttackCoordinator`` to dynamically assign sub-blocks per round."""
        self.client_block_map = {int(k): int(v) % self.num_blocks for k, v in mapping.items()}

    def _resolve_block_id(self, client_id: Optional[int]) -> int:
        if self.fixed_block_id is not None:
            return int(self.fixed_block_id) % self.num_blocks
        if client_id is not None and int(client_id) in self.client_block_map:
            return self.client_block_map[int(client_id)]
        return (0 if client_id is None else int(client_id)) % self.num_blocks

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        bid = self._resolve_block_id(client_id)
        return self._block_triggers[bid](bx, mask=mask, inplace=True)

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return self.eval_trigger(bx, mask=None, inplace=True)


@POISONER_REGISTRY.register("blend")
class BlendPoisoner(BasePoisoner):
    """Blended backdoor poisoner in ``[0, 1]`` pixel space."""

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        alpha: float = 0.2,
        pattern_seed: int = 2024,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.trigger = BlendTrigger(alpha=alpha, pattern_seed=pattern_seed)

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del client_id, round_idx
        return self.trigger(bx, mask=mask, inplace=True)

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return self.trigger(bx, mask=None, inplace=True)


def build_poisoner(name: str = "patch", **params: Any) -> BasePoisoner:
    """Build a ``BasePoisoner`` by registry name (default ``'patch'``)."""
    return POISONER_REGISTRY.build(name.strip().lower(), **params)
