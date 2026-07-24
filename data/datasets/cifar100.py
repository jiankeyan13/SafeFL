import os
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms

from core.utils.mmap_dataset import MemoryMappedDataset
from data.dataset_store import DatasetStore
from data.registry import dataset_registry

# CIFAR100 常用归一化参数 (与官方统计一致)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

_aug_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
])


def _build_cifar100_impl(root: str, is_train: bool, use_aug: bool) -> DatasetStore:
    real_dataset = CIFAR100(root=root, train=is_train, download=True, transform=None)
    split_name = "train" if is_train else "test"
    cache_path = os.path.join(root, f"cifar100_{split_name}_mmap")
    final_transform = _aug_transform if (is_train and use_aug) else _base_transform
    mmap_dataset = MemoryMappedDataset(
        original_dataset=real_dataset,
        cache_path=cache_path,
        transform=final_transform,
    )
    split = "train" if use_aug else "train_plain" if is_train else "test"
    return DatasetStore("cifar100", split, mmap_dataset)


@dataset_registry.register("cifar100_train_aug")
def build_cifar100_aug(root: str, is_train: bool) -> DatasetStore:
    del is_train  # 固定为 train+aug，split 已由 tag 名称确定
    return _build_cifar100_impl(root, is_train=True, use_aug=True)


@dataset_registry.register("cifar100_train_plain")
def build_cifar100_train_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train  # 固定为 train，无增强
    return _build_cifar100_impl(root, is_train=True, use_aug=False)


@dataset_registry.register("cifar100_test_plain")
def build_cifar100_test(root: str, is_train: bool) -> DatasetStore:
    del is_train  # 固定为 test
    return _build_cifar100_impl(root, is_train=False, use_aug=False)
