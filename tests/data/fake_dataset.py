"""
测试用假数据集，避免下载真实 CIFAR10。
在运行 data 层测试前注册，覆盖 dataset_builders 中的 cifar10_*。
"""
import numpy as np
from torch.utils.data import Dataset
from data.registry import register_dataset
from data.dataset_store import DatasetStore


class _FakeCIFAR(Dataset):
    """最小可用的假 CIFAR 数据集，满足 Partitioner 与 TaskGenerator 的接口要求。"""
    def __init__(self, n_samples: int, n_classes: int = 10):
        self.n = n_samples
        self.n_classes = n_classes
        self.targets = np.arange(n_samples, dtype=np.int64) % n_classes
        self.data = np.zeros((n_samples, 32, 32, 3), dtype=np.uint8)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.data[i], int(self.targets[i])


def _build_fake(root, is_train, n_train=2000, n_test=500):
    """内部构建逻辑。"""
    n = n_train if is_train else n_test
    ds = _FakeCIFAR(n)
    tag = "cifar10"
    split = "train" if is_train else "test"
    return DatasetStore(tag, split, ds)


@register_dataset("cifar10_train_aug")
def build_fake_train_aug(root, is_train):
    return _build_fake(root, is_train=True)

@register_dataset("cifar10_train_plain")
def build_fake_train_plain(root, is_train):
    return _build_fake(root, is_train=True)

@register_dataset("cifar10_test_plain")
def build_fake_test_plain(root, is_train):
    return _build_fake(root, is_train=False)
