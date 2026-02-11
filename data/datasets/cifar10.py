import os
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from core.utils.mmap_dataset import MemoryMappedDataset
from data.dataset_store import DatasetStore
from data.registry import register_dataset

# CIFAR10 标准归一化参数，避免在 base/aug 中重复定义
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

_aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


def _build_cifar10_impl(root: str, is_train: bool, use_aug: bool) -> DatasetStore:
    real_dataset = CIFAR10(root=root, train=is_train, download=True, transform=None)
    split_name = "train" if is_train else "test"
    cache_path = os.path.join(root, f"cifar10_{split_name}_mmap")
    final_transform = _aug_transform if (is_train and use_aug) else _base_transform
    mmap_dataset = MemoryMappedDataset(
        original_dataset=real_dataset,
        cache_path=cache_path,
        transform=final_transform,
    )
    split = "train" if use_aug else "train_plain" if is_train else "test"
    return DatasetStore("cifar10", split, mmap_dataset)


@register_dataset("cifar10_train_aug")
def build_cifar10_aug(root: str, is_train: bool) -> DatasetStore:
    del is_train  # 固定为 train+aug
    return _build_cifar10_impl(root, is_train=True, use_aug=True)


@register_dataset("cifar10_train_plain")
def build_cifar10_train_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train  # 固定为 train，无增强
    return _build_cifar10_impl(root, is_train=True, use_aug=False)


@register_dataset("cifar10_test_plain")
def build_cifar10_test(root: str, is_train: bool) -> DatasetStore:
    del is_train  # 固定为 test
    return _build_cifar10_impl(root, is_train=False, use_aug=False)
