"""CIFAR-10 dataset specification for v2."""
from __future__ import annotations

import os

from torchvision.datasets import CIFAR10

from data.registry import DATASET_REGISTRY
from data.source import ensure_chw_mmap_cache
from data.spec import AugmentSpec, DatasetPairSpec

CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)
CIFAR10_AUGMENT = AugmentSpec(random_crop_padding=4, random_horizontal_flip=True)


@DATASET_REGISTRY.register("cifar10")
def build_cifar10(root: str) -> DatasetPairSpec:
    train_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: CIFAR10(root=root, train=True, download=True, transform=None),
        cache_prefix=os.path.join(root, "cifar10_train"),
        name="cifar10",
        split="train",
        num_classes=10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        default_augment=CIFAR10_AUGMENT,
    )
    test_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: CIFAR10(root=root, train=False, download=True, transform=None),
        cache_prefix=os.path.join(root, "cifar10_test"),
        name="cifar10",
        split="test",
        num_classes=10,
        mean=CIFAR10_MEAN,
        std=CIFAR10_STD,
        default_augment=None,
    )
    return DatasetPairSpec(name="cifar10", train=train_spec, test=test_spec)
