"""Vectorized batch triggers operating in-place on ``[B, C, H, W]`` tensors in ``[0, 1]`` space."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

_VALID_LOCATIONS = ("bottom_right", "bottom_left", "top_right", "top_left")
PatchValue = Union[float, Sequence[float]]


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


def block_slices(patch_size: int, block_id: int, num_blocks: int = 4) -> Tuple[int, int, int, int]:
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


def _stamp_rect_(
    bx: torch.Tensor,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    patch_value: PatchValue,
    *,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if bx.ndim != 4:
        raise ValueError(f"Expected 4D [B, C, H, W] batch tensor, got shape {tuple(bx.shape)}")
    if bx.shape[0] == 0:
        return bx
    val = _format_patch_value(patch_value, int(bx.shape[1]), bx.device, bx.dtype)
    if mask is None:
        bx[:, :, r0:r1, c0:c1] = val
    elif bool(mask.any()):
        bx[mask, :, r0:r1, c0:c1] = val
    return bx


class PatchTrigger(nn.Module):
    """Square corner patch trigger in ``[0, 1]`` pixel space."""

    def __init__(
        self,
        patch_size: int = 5,
        patch_value: PatchValue = 1.0,
        location: str = "bottom_right",
    ) -> None:
        super().__init__()
        if location not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid location {location!r}. Expected one of {_VALID_LOCATIONS}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        self.patch_size = int(patch_size)
        self.patch_value = patch_value
        self.location = location

    def forward(self, bx: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = int(bx.shape[-2]), int(bx.shape[-1])
        ps = min(self.patch_size, h, w)
        r0, c0 = _patch_origin(self.location, h, w, ps)
        return _stamp_rect_(bx, r0, r0 + ps, c0, c0 + ps, self.patch_value, mask=mask)


class PartialPatchTrigger(nn.Module):
    """DBA sub-block patch trigger in ``[0, 1]`` pixel space."""

    def __init__(
        self,
        patch_size: int = 5,
        patch_value: PatchValue = 1.0,
        location: str = "bottom_right",
        block_id: int = 0,
        num_blocks: int = 4,
    ) -> None:
        super().__init__()
        if location not in _VALID_LOCATIONS:
            raise ValueError(f"Invalid location {location!r}. Expected one of {_VALID_LOCATIONS}.")
        self.patch_size = int(patch_size)
        self.patch_value = patch_value
        self.location = location
        self.block_id = int(block_id)
        self.num_blocks = int(num_blocks)

    def forward(self, bx: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h, w = int(bx.shape[-2]), int(bx.shape[-1])
        ps = min(self.patch_size, h, w)
        r0, c0 = _patch_origin(self.location, h, w, ps)
        rs, re, cs, ce = block_slices(ps, self.block_id, self.num_blocks)
        return _stamp_rect_(bx, r0 + rs, r0 + re, c0 + cs, c0 + ce, self.patch_value, mask=mask)


class BlendTrigger(nn.Module):
    """Blended random-pattern trigger ``(1 - alpha) * x + alpha * pattern`` in ``[0, 1]``."""

    def __init__(self, alpha: float = 0.2, pattern_seed: int = 2024) -> None:
        super().__init__()
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.pattern_seed = int(pattern_seed)
        self._pattern_cache: dict[Tuple[Tuple[int, int, int], torch.device, torch.dtype], torch.Tensor] = {}

    def _get_pattern(self, chw: Tuple[int, int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (chw, device, dtype)
        pat = self._pattern_cache.get(key)
        if pat is None:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.pattern_seed)
            pat = torch.rand(chw, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype).unsqueeze(0)
            self._pattern_cache[key] = pat
        return pat

    def forward(self, bx: torch.Tensor, *, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
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
