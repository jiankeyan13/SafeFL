from typing import Optional
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

from data.registry import dataset_builders


def _extract_targets_from_dataset(ds) -> Optional[np.ndarray]:
    """从底层 dataset 提取 targets/labels，不依赖 DatasetStore 内部结构。"""
    if hasattr(ds, "targets"):
        arr = ds.targets
        return np.array(arr) if not isinstance(arr, np.ndarray) else arr
    if hasattr(ds, "labels"):
        return np.array(ds.labels)
    return None


def _extract_targets_from_subset(subset) -> Optional[np.ndarray]:
    """从 Subset 的父数据集提取并切片。"""
    parent = subset.dataset
    full = _extract_targets_from_dataset(parent)
    if full is not None:
        idx = subset.indices
        return full[idx] if hasattr(idx, "__getitem__") else full[np.asarray(idx)]
    return None


class DatasetStore(Dataset):
    def __init__(self, name: str, split: str, dataset: Dataset):
        self.name = name
        self.split = split
        self.dataset = dataset
        self._label_cache: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]

    def get_label(self) -> np.ndarray:
        """
        统一获取所有样本标签。
        返回: numpy array of shape (N,)。首次调用后结果会缓存。
        """
        if self._label_cache is not None:
            return self._label_cache

        labels: Optional[np.ndarray] = None
        if isinstance(self.dataset, torch.utils.data.Subset):
            labels = _extract_targets_from_subset(self.dataset)

        if labels is None:
            labels = _extract_targets_from_dataset(self.dataset)

        if labels is None:
            warnings.warn(
                f"DatasetStore '{self.name}' has no targets/labels. "
                "Falling back to full iteration (slow).",
                UserWarning,
                stacklevel=2,
            )
            labels = np.array([self.dataset[i][1] for i in range(len(self.dataset))])

        self._label_cache = labels
        return labels


def build_dataset(name: str, root: str, is_train: bool) -> DatasetStore:
    return dataset_builders[name](root, is_train)