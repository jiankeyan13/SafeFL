import os
from typing import List

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from core.utils.mmap_dataset import MemoryMappedDataset
from data.dataset_store import DatasetStore
from data.registry import dataset_registry

# 常用 ImageNet 统计量, Tiny ImageNet 文献中普遍采用
TINY_IMAGENET_MEAN = (0.485, 0.456, 0.406)
TINY_IMAGENET_STD = (0.229, 0.224, 0.225)

_base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
])

_aug_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
])


def _tiny_imagenet_root(root: str) -> str:
    """解析 Tiny ImageNet 根目录: `root/tiny-imagenet-200` 或 `root` 本身就是该目录."""
    sub = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(sub) and os.path.isfile(os.path.join(sub, "wnids.txt")):
        return sub
    if os.path.isfile(os.path.join(root, "wnids.txt")):
        return root
    raise FileNotFoundError(
        f"未找到 Tiny ImageNet. 请将解压后的 tiny-imagenet-200 放到 {os.path.abspath(root)} 下, "
        "或直接将 tiny-imagenet-200 的路径设为 data.root."
    )


class _TinyImageNetSource(Dataset):
    """原始 RGB 图像, 供 MemoryMappedDataset 建缓存; `split_train` 为 True 用 train/, 否则用 val+标注."""

    def __init__(self, root: str, split_train: bool) -> None:
        self.root = root
        self.paths: List[str] = []
        targets: List[int] = []

        wnids_path = os.path.join(root, "wnids.txt")
        with open(wnids_path, "r", encoding="utf-8") as f:
            wnids = [ln.strip() for ln in f if ln.strip()]
        wnid_to_idx = {w: i for i, w in enumerate(wnids)}

        if split_train:
            train_dir = os.path.join(root, "train")
            for wnid in wnids:
                img_dir = os.path.join(train_dir, wnid, "images")
                if not os.path.isdir(img_dir):
                    continue
                for fn in sorted(os.listdir(img_dir)):
                    if not fn.lower().endswith((".jpeg", ".jpg")):
                        continue
                    self.paths.append(os.path.join(img_dir, fn))
                    targets.append(wnid_to_idx[wnid])
        else:
            val_img_dir = os.path.join(root, "val", "images")
            ann_path = os.path.join(root, "val", "val_annotations.txt")
            with open(ann_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    fname, wnid = parts[0], parts[1]
                    self.paths.append(os.path.join(val_img_dir, fname))
                    targets.append(wnid_to_idx[wnid])

        self.targets = np.array(targets, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        img = Image.open(self.paths[index]).convert("RGB")
        t = int(self.targets[index])
        return img, t


def _build_tiny_impl(root: str, split_train: bool, use_aug: bool) -> DatasetStore:
    tiny_root = _tiny_imagenet_root(root)
    split_name = "train" if split_train else "val"
    cache_path = os.path.join(tiny_root, f"_mmap_{split_name}")
    raw = _TinyImageNetSource(tiny_root, split_train=split_train)
    if len(raw) == 0:
        raise RuntimeError(f"Tiny ImageNet ({split_name}) 在 {tiny_root} 下未读到任何样本.")

    final_transform = _aug_transform if (split_train and use_aug) else _base_transform
    mmap_dataset = MemoryMappedDataset(
        original_dataset=raw,
        cache_path=cache_path,
        transform=final_transform,
    )
    inner_split = "train" if use_aug else "train_plain" if split_train else "test"
    return DatasetStore("tiny_imagenet", inner_split, mmap_dataset)


@dataset_registry.register("tiny_imagenet_train_aug")
def build_tiny_imagenet_train_aug(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_tiny_impl(root, split_train=True, use_aug=True)


@dataset_registry.register("tiny_imagenet_train_plain")
def build_tiny_imagenet_train_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_tiny_impl(root, split_train=True, use_aug=False)


@dataset_registry.register("tiny_imagenet_test_plain")
def build_tiny_imagenet_test_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    return _build_tiny_impl(root, split_train=False, use_aug=False)
