import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from core.utils.mmap_dataset import MemoryMappedDataset
from data.dataset_store import DatasetStore
from data.registry import dataset_registry

# 与官方 BadNets 一致: ToTensor -> [0, 1], Normalize(0.5) -> [-1, 1]
# 触发器 patch_value=1.0 对应 pixel_max=1 的纯白
CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def _build_cifar10_impl(root: str, is_train: bool) -> DatasetStore:
    real_dataset = CIFAR10(root=root, train=is_train, download=True, transform=None)
    split_name = "train" if is_train else "test"
    cache_path = os.path.join(root, f"cifar10_{split_name}_mmap")
    mmap_dataset = MemoryMappedDataset(
        original_dataset=real_dataset,
        cache_path=cache_path,
        transform=_transform,
    )
    inner_split = "train_plain" if is_train else "test"
    return DatasetStore("cifar10", inner_split, mmap_dataset)


@dataset_registry.register("cifar10_train_plain")
def build_cifar10_train_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_cifar10_impl(root, is_train=True)


@dataset_registry.register("cifar10_test_plain")
def build_cifar10_test(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_cifar10_impl(root, is_train=False)
