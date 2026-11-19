"""FashionMNIST dataset specification for v2."""
from __future__ import annotations

import os

from torchvision.datasets import FashionMNIST

from data.registry import DATASET_REGISTRY
from data.source import ensure_chw_mmap_cache
from data.spec import AugmentSpec, DatasetPairSpec

FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)
FASHIONMNIST_AUGMENT = AugmentSpec(random_crop_padding=2, random_horizontal_flip=True)


@DATASET_REGISTRY.register("fashionmnist")
def build_fashionmnist(root: str) -> DatasetPairSpec:
    train_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: FashionMNIST(root=root, train=True, download=True, transform=None),
        cache_prefix=os.path.join(root, "fashionmnist_train"),
        name="fashionmnist",
        split="train",
        num_classes=10,
        mean=FASHIONMNIST_MEAN,
        std=FASHIONMNIST_STD,
        default_augment=FASHIONMNIST_AUGMENT,
    )
    test_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: FashionMNIST(root=root, train=False, download=True, transform=None),
        cache_prefix=os.path.join(root, "fashionmnist_test"),
        name="fashionmnist",
        split="test",
        num_classes=10,
        mean=FASHIONMNIST_MEAN,
        std=FASHIONMNIST_STD,
        default_augment=None,
    )
    return DatasetPairSpec(name="fashionmnist", train=train_spec, test=test_spec)
