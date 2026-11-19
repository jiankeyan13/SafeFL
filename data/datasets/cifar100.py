"""CIFAR-100 dataset specification for v2."""
from __future__ import annotations

import os

from torchvision.datasets import CIFAR100

from data.registry import DATASET_REGISTRY
from data.source import ensure_chw_mmap_cache
from data.spec import AugmentSpec, DatasetPairSpec

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
CIFAR100_AUGMENT = AugmentSpec(random_crop_padding=4, random_horizontal_flip=True)


@DATASET_REGISTRY.register("cifar100")
def build_cifar100(root: str) -> DatasetPairSpec:
    train_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: CIFAR100(root=root, train=True, download=True, transform=None),
        cache_prefix=os.path.join(root, "cifar100_train"),
        name="cifar100",
        split="train",
        num_classes=100,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        default_augment=CIFAR100_AUGMENT,
    )
    test_spec = ensure_chw_mmap_cache(
        raw_dataset_fn=lambda: CIFAR100(root=root, train=False, download=True, transform=None),
        cache_prefix=os.path.join(root, "cifar100_test"),
        name="cifar100",
        split="test",
        num_classes=100,
        mean=CIFAR100_MEAN,
        std=CIFAR100_STD,
        default_augment=None,
    )
    return DatasetPairSpec(name="cifar100", train=train_spec, test=test_spec)
