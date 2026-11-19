"""Batch-vectorized data view, augmentation, poison hook protocol, and loader."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Iterator, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F

from data.source import SharedDataSource
from data.spec import AugmentSpec


def derive_stable_seed(base_seed: int, *parts: Any) -> int:
    """Derive a deterministic 31-bit seed from ``(base_seed, *parts)``."""
    raw = "|".join([str(int(base_seed)), *(str(p) for p in parts)])
    digest = hashlib.md5(raw.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little") & 0x7FFFFFFF


@runtime_checkable
class BatchPoisoner(Protocol):
    """Protocol for batch-level data poisoning in physical ``[0, 1]`` pixel space.

    The data layer stays completely agnostic to specific backdoor algorithms and
    only invokes this protocol right *before* batch augmentation & normalization.
    """

    def build_mask(
        self,
        num_samples: int,
        *,
        targets: Optional[np.ndarray] = None,
        client_id: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Return boolean mask of shape ``(num_samples,)`` selecting poisoned samples."""
        ...

    def apply_train_batch(
        self,
        bx: torch.Tensor,
        by: torch.Tensor,
        mask: torch.Tensor,
        *,
        client_id: Optional[int] = None,
        round_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply trigger & target label to ``mask``-selected entries in ``[0, 1]`` ``bx``."""
        ...

    def apply_eval_batch(
        self,
        bx: torch.Tensor,
        by: torch.Tensor,
        *,
        round_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Poison all samples in ``[0, 1]`` ``bx``; return ``(bx_poisoned, target_y, orig_y)``."""
        ...


def apply_batch_augment(
    bx: torch.Tensor,
    spec: AugmentSpec,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply independent per-sample random flip and random crop on ``[B, C, H, W]``."""
    if not spec.enabled or bx.shape[0] == 0:
        return bx

    b, c, h, w = bx.shape
    dev = bx.device

    if spec.random_horizontal_flip:
        flip_mask = torch.rand(b, device=dev, generator=generator) < 0.5
        if bool(flip_mask.any()):
            bx[flip_mask] = bx[flip_mask].flip(-1)

    pad = int(spec.random_crop_padding)
    if pad > 0:
        padded = F.pad(bx, (pad, pad, pad, pad), mode="constant", value=0.0)
        dy = torch.randint(0, 2 * pad + 1, (b, 1, 1, 1), device=dev, generator=generator)
        dx = torch.randint(0, 2 * pad + 1, (b, 1, 1, 1), device=dev, generator=generator)
        b_idx = torch.arange(b, device=dev).view(b, 1, 1, 1)
        c_idx = torch.arange(c, device=dev).view(1, c, 1, 1)
        r_idx = dy + torch.arange(h, device=dev).view(1, 1, h, 1)
        w_idx = dx + torch.arange(w, device=dev).view(1, 1, 1, w)
        bx = padded[b_idx, c_idx, r_idx, w_idx].contiguous()

    return bx


class FastBatchLoader:
    """Zero-worker, batch-vectorized iterator over a ``DataView``.

    Execution order on every batch (for ``SharedDataSource``):
      1. Gather ``uint8 [B, C, H, W]`` & ``int64 [B]`` -> transfer to ``device`` -> ``float32 [0, 1]``
      2. (Optional) ``poisoner`` in true ``[0, 1]`` pixel space
      3. (Optional) vectorized per-sample batch augmentation (``RandomCrop`` / ``HorizontalFlip``)
      4. Channel-wise ``Normalize(mean, std)``
    """

    def __init__(
        self,
        view: "DataView",
        batch_size: int = 64,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        poisoner: Optional[BatchPoisoner] = None,
        mode: str = "train",
        augment: Optional[bool] = None,
        return_orig_label: bool = False,
        round_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if mode not in ("train", "eval"):
            raise ValueError(f"mode must be 'train' or 'eval', got {mode!r}")

        self.view = view
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.device = torch.device(device) if device is not None else None
        self.poisoner = poisoner
        self.mode = mode
        self.return_orig_label = bool(return_orig_label)
        self.round_idx = round_idx
        self.seed = seed

        aug_spec = view.source.default_augment
        if augment is None:
            self.use_augment = bool(
                mode == "train"
                and view.default_use_augmentation
                and aug_spec is not None
                and aug_spec.enabled
            )
        else:
            self.use_augment = bool(augment and aug_spec is not None and aug_spec.enabled)
        self._augment_spec = aug_spec

        self._rng = np.random.default_rng(seed)
        self._aug_seed = None if seed is None else derive_stable_seed(seed, "augment", view.client_id)
        self._aug_step = 0

        self._poison_mask: Optional[np.ndarray] = None
        if self.poisoner is not None and self.mode == "train" and len(self.view) > 0:
            self._poison_mask = np.asarray(
                self.poisoner.build_mask(
                    len(self.view),
                    targets=self.view.targets,
                    client_id=self.view.client_id,
                    seed=seed,
                ),
                dtype=bool,
            )
            if self._poison_mask.shape != (len(self.view),):
                raise ValueError(
                    f"Poisoner mask shape {self._poison_mask.shape} != ({len(self.view)},)"
                )

    @property
    def num_samples(self) -> int:
        return len(self.view)

    def __len__(self) -> int:
        n = len(self.view)
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def _next_aug_generator(self, device: torch.device) -> Optional[torch.Generator]:
        if self._aug_seed is None:
            return None
        gen = torch.Generator(device=device)
        gen.manual_seed(derive_stable_seed(self._aug_seed, self._aug_step))
        self._aug_step += 1
        return gen

    def __iter__(self) -> Iterator[Any]:
        src = self.view.source
        n = len(self.view)
        if n == 0:
            return

        if self.shuffle:
            pos_order = self._rng.permutation(n)
        else:
            pos_order = np.arange(n, dtype=np.int64)

        limit = (n // self.batch_size) * self.batch_size if self.drop_last else n
        dev = self.device or torch.device("cpu")

        for start in range(0, limit, self.batch_size):
            pos = pos_order[start : start + self.batch_size]
            global_idx = self.view.indices[pos]

            # 1. Gather physical [0, 1] float32 batch + int64 targets on target device.
            bx, by = src.fetch_batch(global_idx, device=dev)

            # 2. Optional poisoning in physical [0, 1] pixel space (BEFORE Normalize!).
            if self.poisoner is not None:
                if self.mode == "train":
                    assert self._poison_mask is not None
                    mask_t = torch.as_tensor(
                        self._poison_mask[pos], dtype=torch.bool, device=bx.device
                    )
                    bx, by = self.poisoner.apply_train_batch(
                        bx,
                        by,
                        mask_t,
                        client_id=self.view.client_id,
                        round_idx=self.round_idx,
                    )
                    if self.use_augment and self._augment_spec is not None:
                        bx = apply_batch_augment(
                            bx, self._augment_spec, self._next_aug_generator(bx.device)
                        )
                    src.normalize_(bx)
                    yield bx, by
                else:
                    bx, target_y, orig_y = self.poisoner.apply_eval_batch(
                        bx, by, round_idx=self.round_idx
                    )
                    src.normalize_(bx)
                    if self.return_orig_label:
                        yield bx, target_y, orig_y
                    else:
                        yield bx, target_y
            else:
                # 3. Clean path: optional augmentation -> Normalize.
                if self.use_augment and self._augment_spec is not None:
                    bx = apply_batch_augment(
                        bx, self._augment_spec, self._next_aug_generator(bx.device)
                    )
                src.normalize_(bx)
                yield bx, by


class DataView:
    """Zero-copy index slice over a ``SharedDataSource``.

    Provides 1-line splitting and microsecond ``FastBatchLoader`` assembly for
    benign clients, complex multi-branch attackers, and server evaluators.
    """

    def __init__(
        self,
        source: SharedDataSource,
        indices: Optional[Sequence[int] | np.ndarray] = None,
        *,
        client_id: Optional[int] = None,
        default_use_augmentation: bool = False,
    ) -> None:
        self.source = source
        if indices is None:
            self.indices = np.arange(len(source), dtype=np.int32)
        else:
            self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.client_id = client_id
        self.default_use_augmentation = bool(default_use_augmentation)

    def __len__(self) -> int:
        return int(self.indices.size)

    @property
    def num_classes(self) -> int:
        return self.source.num_classes

    @property
    def targets(self) -> np.ndarray:
        """Clean ground-truth labels for samples in this view (shape ``(N,)``)."""
        return np.asarray(self.source.targets[self.indices], dtype=np.int64)

    def subset(self, rel_indices: Sequence[int] | np.ndarray) -> "DataView":
        """Return a zero-copy sub-view indexed relative to ``self``."""
        rel = np.asarray(rel_indices, dtype=np.int64)
        return DataView(
            source=self.source,
            indices=self.indices[rel],
            client_id=self.client_id,
            default_use_augmentation=self.default_use_augmentation,
        )

    def split(
        self,
        val_ratio: float = 0.2,
        *,
        seed: Optional[int] = None,
    ) -> Tuple["DataView", "DataView"]:
        """Deterministically split this view into ``(train_view, val_view)``."""
        if not 0.0 < float(val_ratio) < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        n = len(self)
        if n < 2:
            raise RuntimeError(f"Cannot split DataView with {n} samples (need >= 2).")

        effective_seed = (
            None
            if seed is None
            else derive_stable_seed(seed, "view_split", self.client_id)
        )
        rng = np.random.default_rng(effective_seed)
        perm = rng.permutation(n)
        n_val = min(max(1, int(n * float(val_ratio))), n - 1)
        val_rel = perm[:n_val]
        train_rel = perm[n_val:]

        train_view = self.subset(train_rel)
        val_view = DataView(
            source=self.source,
            indices=self.indices[val_rel],
            client_id=self.client_id,
            default_use_augmentation=False,
        )
        return train_view, val_view

    def loader(
        self,
        batch_size: int = 64,
        *,
        shuffle: bool = True,
        drop_last: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        poisoner: Optional[BatchPoisoner] = None,
        mode: str = "train",
        augment: Optional[bool] = None,
        return_orig_label: bool = False,
        round_idx: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> FastBatchLoader:
        """Build a ``FastBatchLoader`` over this view in microseconds."""
        return FastBatchLoader(
            view=self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            device=device,
            poisoner=poisoner,
            mode=mode,
            augment=augment,
            return_orig_label=return_orig_label,
            round_idx=round_idx,
            seed=seed,
        )
