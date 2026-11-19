import os

import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST

from core.utils.mmap_dataset import MemoryMappedDataset
from data.dataset_store import DatasetStore
from data.registry import dataset_registry


FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(FASHIONMNIST_MEAN, FASHIONMNIST_STD),
])


def _build_fashionmnist_impl(root: str, is_train: bool) -> DatasetStore:
    real_dataset = FashionMNIST(
        root=root,
        train=is_train,
        download=True,
        transform=None,
    )
    split_name = "train" if is_train else "test"
    cache_path = os.path.join(root, f"fashionmnist_{split_name}_mmap")
    mmap_dataset = MemoryMappedDataset(
        original_dataset=real_dataset,
        cache_path=cache_path,
        transform=_transform,
    )
    inner_split = "train_plain" if is_train else "test"
    return DatasetStore("fashionmnist", inner_split, mmap_dataset)


@dataset_registry.register("fashionmnist_train_plain")
def build_fashionmnist_train_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_fashionmnist_impl(root, is_train=True)


@dataset_registry.register("fashionmnist_test_plain")
def build_fashionmnist_test(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_fashionmnist_impl(root, is_train=False)
