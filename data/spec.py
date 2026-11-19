"""Lightweight, zero-copy serializable specifications for the v2 data layer.

All dataclasses in this module contain only file paths, scalar metadata, and
compact ``int32`` index arrays, so passing ``FederatedDataSpec`` to Ray workers
never serializes raw image buffers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class AugmentSpec:
    """Batch-level GPU/CPU image augmentation parameters."""

    random_crop_padding: int = 0
    random_horizontal_flip: bool = False

    @property
    def enabled(self) -> bool:
        return self.random_crop_padding > 0 or self.random_horizontal_flip


@dataclass(frozen=True)
class SplitSpec:
    """Descriptor for one persisted CHW uint8 memory-mapped split."""

    name: str
    split: str
    data_path: str
    target_path: str
    data_shape: Tuple[int, int, int, int]  # (N, C, H, W)
    num_classes: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    default_augment: Optional[AugmentSpec] = None

    @property
    def num_samples(self) -> int:
        return int(self.data_shape[0])

    @property
    def input_channels(self) -> int:
        return int(self.data_shape[1])

    @property
    def image_size(self) -> Tuple[int, int]:
        return int(self.data_shape[2]), int(self.data_shape[3])


@dataclass(frozen=True)
class DatasetPairSpec:
    """Train + test split descriptors for a registered dataset."""

    name: str
    train: SplitSpec
    test: SplitSpec

    @property
    def num_classes(self) -> int:
        return self.train.num_classes

    @property
    def input_channels(self) -> int:
        return self.train.input_channels

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.train.image_size


@dataclass
class PartitionMap:
    """Client sample assignment and optional server proxy indices."""

    num_clients: int
    client_indices: Dict[int, np.ndarray] = field(default_factory=dict)
    proxy_indices: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int32)
    )

    def __post_init__(self) -> None:
        self.client_indices = {
            int(cid): np.asarray(idx, dtype=np.int32)
            for cid, idx in self.client_indices.items()
        }
        self.proxy_indices = np.asarray(self.proxy_indices, dtype=np.int32)


@dataclass(frozen=True)
class FederatedDataSpec:
    """Complete serializable data descriptor shared between driver and workers."""

    dataset: DatasetPairSpec
    partition: PartitionMap
    use_augmentation: bool = False
    preload_gpu: bool = False
