"""Composable batch-vectorized Poisoners implementing ``data.BatchPoisoner``.

Operates directly on ``[B, C, H, W]`` tensors in physical ``[0, 1]`` pixel space
*before* dataset ``Normalize``, decoupling trigger stamping from attacker training.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from core.utils.registry import Registry
from data.loader import BatchPoisoner, derive_stable_seed

POISONER_REGISTRY = Registry("Poisoners")
PatchValue = Union[float, Sequence[float]]
_VALID_LOCATIONS = ("bottom_right", "bottom_left", "top_right", "top_left")


# =============================================================================
# 1. Pure Vectorized Trigger Stamping Functions (in [0, 1] pixel space)
# =============================================================================


def _patch_origin(location: str, h: int, w: int, patch_size: int) -> Tuple[int, int]:
    if location == "bottom_right":
        return h - patch_size, w - patch_size
    if location == "bottom_left":
        return h - patch_size, 0
    if location == "top_right":
        return 0, w - patch_size
    if location == "top_left":
        return 0, 0
    raise ValueError(f"Invalid location {location!r}. Expected one of {_VALID_LOCATIONS}.")


def _block_slices(patch_size: int, block_id: int, num_blocks: int = 4) -> Tuple[int, int, int, int]:
    """Return relative ``(row_start, row_end, col_start, col_end)`` for a DBA sub-block."""
    if num_blocks == 4:
        if block_id not in (0, 1, 2, 3):
            raise ValueError(f"block_id must be in [0, 3], got {block_id}")
        mid = patch_size // 2
        if block_id == 0:
            return 0, mid, 0, mid
        if block_id == 1:
            return 0, mid, mid, patch_size
        if block_id == 2:
            return mid, patch_size, 0, mid
        return mid, patch_size, mid, patch_size

    if num_blocks <= 0 or not (0 <= block_id < num_blocks):
        raise ValueError(f"Invalid block_id={block_id} for num_blocks={num_blocks}")
    edges = [int(round(i * patch_size / num_blocks)) for i in range(num_blocks + 1)]
    r0, r1 = edges[block_id], max(edges[block_id] + 1, edges[block_id + 1])
    return min(r0, patch_size - 1), min(r1, patch_size), 0, patch_size


def _format_patch_value(
    patch_value: PatchValue,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Union[float, torch.Tensor]:
    if isinstance(patch_value, (int, float)):
        return float(patch_value)
    vals = list(patch_value)
    if len(vals) == 1:
        return float(vals[0])
    if len(vals) != channels:
        raise ValueError(
            f"patch_value length ({len(vals)}) does not match image channels ({channels})."
        )
    return torch.tensor(vals, device=device, dtype=dtype).view(1, channels, 1, 1)


def stamp_patch_(
    bx: torch.Tensor,
    patch_size: int = 5,
    patch_value: PatchValue = 1.0,
    location: str = "bottom_right",
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stamp a square corner patch onto ``bx`` (``[B, C, H, W]`` in ``[0, 1]``)."""
    if bx.ndim != 4:
        raise ValueError(f"Expected 4D [B, C, H, W] batch tensor, got shape {tuple(bx.shape)}")
    if bx.shape[0] == 0 or (mask is not None and not bool(mask.any())):
        return bx

    h, w = int(bx.shape[-2]), int(bx.shape[-1])
    ps = min(int(patch_size), h, w)
    r0, c0 = _patch_origin(location, h, w, ps)
    val = _format_patch_value(patch_value, int(bx.shape[1]), bx.device, bx.dtype)
    if mask is None:
        bx[:, :, r0 : r0 + ps, c0 : c0 + ps] = val
    else:
        bx[mask, :, r0 : r0 + ps, c0 : c0 + ps] = val
    return bx


def stamp_dba_block_(
    bx: torch.Tensor,
    block_id: int,
    num_blocks: int = 4,
    patch_size: int = 5,
    patch_value: PatchValue = 1.0,
    location: str = "bottom_right",
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Stamp one DBA sub-block onto ``bx`` (``[B, C, H, W]`` in ``[0, 1]``)."""
    if bx.ndim != 4:
        raise ValueError(f"Expected 4D [B, C, H, W] batch tensor, got shape {tuple(bx.shape)}")
    if bx.shape[0] == 0 or (mask is not None and not bool(mask.any())):
        return bx

    h, w = int(bx.shape[-2]), int(bx.shape[-1])
    ps = min(int(patch_size), h, w)
    r0, c0 = _patch_origin(location, h, w, ps)
    rs, re, cs, ce = _block_slices(ps, int(block_id), int(num_blocks))
    val = _format_patch_value(patch_value, int(bx.shape[1]), bx.device, bx.dtype)
    if mask is None:
        bx[:, :, r0 + rs : r0 + re, c0 + cs : c0 + ce] = val
    else:
        bx[mask, :, r0 + rs : r0 + re, c0 + cs : c0 + ce] = val
    return bx


# =============================================================================
# 2. Composable Batch Poisoners
# =============================================================================


class BasePoisoner(BatchPoisoner, ABC):
    """Base batch-level poisoner shared by local training and global ASR evaluation."""

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        attacker_rank: Optional[int] = None,
    ) -> None:
        if poison_ratio < 0.0:
            raise ValueError(f"poison_ratio must be >= 0, got {poison_ratio}")
        self.target_label = int(target_label)
        self.poison_ratio = float(poison_ratio)
        self.seed = seed
        self.exclude_target_class = bool(exclude_target_class)
        self.attacker_rank = attacker_rank

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

    def load_state(self, state: Mapping[str, Any]) -> None:
        """Optional hook to sync dynamic trigger state from persisted attacker state."""
        del state

    def bind_data_source(self, source: Any) -> None:
        """Optional hook called by ``FastBatchLoader`` to expose the active ``SharedDataSource``."""
        del source

    def excluded_train_indices(self) -> Sequence[int]:
        """Optional train-split indices to exclude from benign client partitions."""
        return ()

    def make_eval_loader(
        self,
        hub: Any,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Any:
        """Build the global ASR eval loader (default: full ``hub.test()`` with ``self``)."""
        return hub.test().loader(
            batch_size=batch_size,
            shuffle=False,
            device=device,
            poisoner=self,
            mode="eval",
            return_orig_label=True,
        )


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
        if patch_location not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid location {patch_location!r}. Expected {_VALID_LOCATIONS}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.patch_size = int(patch_size)
        self.patch_value = patch_value
        self.patch_location = patch_location

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del client_id, round_idx
        return stamp_patch_(
            bx, self.patch_size, self.patch_value, self.patch_location, mask=mask
        )

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return stamp_patch_(
            bx, self.patch_size, self.patch_value, self.patch_location, mask=None
        )


@POISONER_REGISTRY.register("dba")
class DBAPoisoner(BasePoisoner):
    """Distributed Backdoor Attack (DBA) poisoner.

    - Train mode: stamps sub-block ``attacker_rank % num_blocks`` (collision-free across
      sampled attackers in the round, falling back to ``client_id % num_blocks`` standalone).
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
        if patch_location not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid location {patch_location!r}. Expected {_VALID_LOCATIONS}.")
        if patch_size <= 0 or num_blocks <= 0:
            raise ValueError("patch_size and num_blocks must be positive.")
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
        self.fixed_block_id = block_id

    def _resolve_block_id(self, client_id: Optional[int]) -> int:
        if self.fixed_block_id is not None:
            return int(self.fixed_block_id) % self.num_blocks
        if self.attacker_rank is not None:
            return int(self.attacker_rank) % self.num_blocks
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
        return stamp_dba_block_(
            bx,
            block_id=self._resolve_block_id(client_id),
            num_blocks=self.num_blocks,
            patch_size=self.patch_size,
            patch_value=self.patch_value,
            location=self.patch_location,
            mask=mask,
        )

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return stamp_patch_(
            bx, self.patch_size, self.patch_value, self.patch_location, mask=None
        )


@POISONER_REGISTRY.register("blend")
class BlendPoisoner(BasePoisoner):
    """Blended random-pattern poisoner ``(1 - alpha) * x + alpha * pattern`` in ``[0, 1]``."""

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
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.alpha = float(alpha)
        self.pattern_seed = int(pattern_seed)
        self._pattern_cache: Dict[Tuple[Tuple[int, int, int], torch.device, torch.dtype], torch.Tensor] = {}

    def _get_pattern(self, chw: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (chw, device, dtype)
        pat = self._pattern_cache.get(key)
        if pat is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.pattern_seed)
            pat = torch.rand(chw, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype).unsqueeze(0)
            self._pattern_cache[key] = pat
        return pat

    def _blend(self, bx: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if bx.ndim != 4:
            raise ValueError(f"Expected 4D [B, C, H, W] batch tensor, got shape {tuple(bx.shape)}")
        if bx.shape[0] == 0:
            return bx
        chw = (int(bx.shape[1]), int(bx.shape[2]), int(bx.shape[3]))
        pat = self._get_pattern(chw, bx.device, bx.dtype)
        a = self.alpha
        if mask is None:
            return bx.mul_(1.0 - a).add_(pat, alpha=a)
        if bool(mask.any()):
            bx[mask] = bx[mask] * (1.0 - a) + pat * a
        return bx

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del client_id, round_idx
        return self._blend(bx, mask=mask)

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return self._blend(bx, mask=None)


def stamp_patch_differentiable(
    raw_x: torch.Tensor,
    trigger: torch.Tensor,
    location: str,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
) -> torch.Tensor:
    """Stamp ``trigger`` onto ``raw_x`` in ``[0, 1]`` space (differentiably) and normalize."""
    h, w = int(raw_x.shape[-2]), int(raw_x.shape[-1])
    ps = min(int(trigger.shape[-2]), int(trigger.shape[-1]), h, w)
    r0, c0 = _patch_origin(location, h, w, ps)
    patched = raw_x.clone()
    patched[:, :, r0 : r0 + ps, c0 : c0 + ps] = (
        trigger[:, :ps, :ps].to(dtype=raw_x.dtype).unsqueeze(0)
    )
    return (patched - mean_t) / std_t


@POISONER_REGISTRY.register("learnable_patch")
class LearnablePatchPoisoner(BasePoisoner):
    """Localized corner-patch poisoner holding an optimizable ``[C, ps, ps]`` trigger tensor."""

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_location: str = "bottom_right",
        init_value: float = 0.5,
        seed: Optional[int] = 42,
        exclude_target_class: bool = False,
        **_: Any,
    ) -> None:
        if patch_location not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid location {patch_location!r}. Expected {_VALID_LOCATIONS}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        super().__init__(
            target_label=target_label,
            poison_ratio=poison_ratio,
            seed=seed,
            exclude_target_class=exclude_target_class,
        )
        self.patch_size = int(patch_size)
        self.patch_location = patch_location
        self.init_value = float(init_value)
        self.trigger: Optional[torch.Tensor] = None

    def ensure_trigger(self, channels: int, device: torch.device) -> torch.Tensor:
        expected = (int(channels), self.patch_size, self.patch_size)
        if self.trigger is None or tuple(self.trigger.shape) != expected:
            self.trigger = torch.full(
                expected, self.init_value, dtype=torch.float32, device="cpu"
            )
        return self.trigger.to(device=device, dtype=torch.float32)

    def load_state(self, state: Mapping[str, Any]) -> None:
        trig = state.get("trigger")
        if torch.is_tensor(trig):
            self.trigger = trig.detach().cpu().clone()

    def _stamp(self, bx: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if bx.ndim != 4:
            raise ValueError(f"Expected 4D [B, C, H, W] batch tensor, got shape {tuple(bx.shape)}")
        if bx.shape[0] == 0 or (mask is not None and not bool(mask.any())):
            return bx

        h, w = int(bx.shape[-2]), int(bx.shape[-1])
        ps = min(self.patch_size, h, w)
        r0, c0 = _patch_origin(self.patch_location, h, w, ps)
        trig = (
            self.ensure_trigger(int(bx.shape[1]), bx.device)[:, :ps, :ps]
            .to(dtype=bx.dtype)
            .unsqueeze(0)
        )
        if mask is None:
            bx[:, :, r0 : r0 + ps, c0 : c0 + ps] = trig
        else:
            bx[mask, :, r0 : r0 + ps, c0 : c0 + ps] = trig
        return bx

    def _stamp_train(
        self,
        bx: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del client_id, round_idx
        return self._stamp(bx, mask)

    def _stamp_eval(
        self,
        bx: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> torch.Tensor:
        del round_idx
        return self._stamp(bx, None)


def build_poisoner(name: str = "patch", **params: Any) -> BasePoisoner:
    """Build a ``BasePoisoner`` by registry name (default ``'patch'``)."""
    return POISONER_REGISTRY.build(name, **params)
