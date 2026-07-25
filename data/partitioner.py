import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, TYPE_CHECKING

from data.constants import client_owner
from data.task import Task, TaskSet
from data.dataset_store import DatasetStore

if TYPE_CHECKING:
    from core.config import PartitionerConfig

class Partitioner(ABC):
    """
    基类，用于将数据集进行划分
    """
    @abstractmethod
    def partition(self, store: DatasetStore, num_clients: int, split: str="train")->TaskSet:
        pass

class IIDPartitioner(Partitioner):
    def __init__(self, seed: int=42):
        self.seed = seed
    def partition(self, store: DatasetStore, num_clients: int, split: str="train")->TaskSet:
        #创建Task集合
        taskset = TaskSet()

        #获取索引列表，打乱
        n = len(store)
        indices = np.arange(n)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)
        
        #获取索引,封装成Task，添加到TaskSet中
        splits = np.array_split(indices, num_clients)
        for i, client_indice in enumerate(splits):
            task = Task(
                owner_id=client_owner(i),
                dataset_tag=store.name,
                split=split,
                indices=client_indice.tolist() # 转成纯 Python list 方便序列化
            )
            taskset.add_task(task)
        return taskset

class DirichletPartitioner(Partitioner):
    def __init__(self, alpha: float = 1.0, seed: int = 42, max_retries: int = 100):
        self.alpha = alpha
        self.seed = seed
        self.max_retries = max_retries

    def partition(self, store: DatasetStore, num_clients: int, split: str = "train") -> TaskSet:
        n_samples = len(store)
        labels = store.get_label()
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if n_samples < num_clients:
            raise ValueError(f"n_samples ({n_samples}) < num_clients ({num_clients})")

        rng = np.random.default_rng(self.seed)
        min_threshold = min(num_classes, 1)
        client_indices: List[List[int]] = []
        attempt = 0

        while attempt < self.max_retries:
            client_indices = [[] for _ in range(num_clients)]
            for label_val in unique_labels:
                idk = np.where(labels == label_val)[0]
                rng.shuffle(idk)
                proportion = rng.dirichlet(np.repeat(self.alpha, num_clients))
                proportion = np.array([
                    p * (len(idx_j) < n_samples / num_clients)
                    for idx_j, p in zip(client_indices, proportion)
                ])
                proportion = proportion / proportion.sum()
                split_points = (np.cumsum(proportion) * len(idk)).astype(int)[:-1]
                split_batch = np.split(idk, split_points)
                for i in range(num_clients):
                    client_indices[i].extend(split_batch[i].tolist())

            min_size = min(len(c_idx) for c_idx in client_indices)
            if min_size >= min_threshold:
                break
            attempt += 1

        if min_size < min_threshold:
            raise RuntimeError(
                f"DirichletPartitioner failed to converge after {self.max_retries} retries. "
                f"min_samples_per_client={min_size}, threshold={min_threshold}. "
                "Try increasing alpha or reducing num_clients."
            )

        taskset = TaskSet()
        for i, client_indice in enumerate(client_indices):
            rng.shuffle(client_indice)
            task = Task(
                owner_id=client_owner(i),
                dataset_tag=store.name,
                split=split,
                indices=client_indice
            )
            taskset.add_task(task)
        return taskset


class QNonIIDPartitioner(Partitioner):
    """
    基于 q 的非 IID 划分 (Cao et al., 2021; Fang et al., 2020).

    将客户端按类别数 X 分成 X 组; 标签为 x 的样本按固定配额分配:
    第 x 组获得 int(n * q) 个, 其余各组各获得 int(n * (1-q)/(X-1)) 个;
    因取整产生的余数随机补入各组. 组内样本均匀分配给该组客户端.
    q 越大, 非 IID 程度越高.
    """

    def __init__(self, q: float = 0.5, seed: int = 42):
        self.q = q
        self.seed = seed

    def _assign_class_to_groups(
        self,
        sample_indices: np.ndarray,
        label_group: int,
        num_classes: int,
        rng: np.random.Generator,
    ) -> List[List[int]]:
        """将同一类别的样本按固定配额分配到各组."""
        num_idx_k = len(sample_indices)
        shuffled = sample_indices.copy()
        rng.shuffle(shuffled)

        main_count = int(num_idx_k * self.q)
        other_count = int(num_idx_k * (1 - self.q) / (num_classes - 1))
        group_counts = np.full(num_classes, other_count, dtype=int)
        group_counts[label_group] = main_count

        remainder = num_idx_k - group_counts.sum()
        if remainder > 0:
            extra_groups = rng.permutation(num_classes)
            for i in range(remainder):
                group_counts[extra_groups[i % num_classes]] += 1

        splits: List[List[int]] = []
        cursor = 0
        for count in group_counts:
            end = cursor + count
            splits.append(shuffled[cursor:end].tolist())
            cursor = end
        return splits

    def partition(self, store: DatasetStore, num_clients: int, split: str = "train") -> TaskSet:
        labels = store.get_label()
        unique_labels = np.sort(np.unique(labels))
        num_classes = len(unique_labels)

        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if not 0 < self.q <= 1:
            raise ValueError("q must be in (0, 1]")
        if num_classes < 2:
            raise ValueError("q_non_iid requires at least 2 classes")
        if num_clients % num_classes != 0:
            raise ValueError(
                f"num_clients ({num_clients}) must be divisible by num_classes ({num_classes})"
            )

        clients_per_group = num_clients // num_classes
        label_to_group = {label: i for i, label in enumerate(unique_labels)}

        rng = np.random.default_rng(self.seed)
        group_indices: List[List[int]] = [[] for _ in range(num_classes)]

        for label_val in unique_labels:
            sample_indices = np.where(labels == label_val)[0]
            label_group = label_to_group[label_val]
            class_splits = self._assign_class_to_groups(
                sample_indices, label_group, num_classes, rng
            )
            for group_id, split_indices in enumerate(class_splits):
                group_indices[group_id].extend(split_indices)

        client_indices: List[List[int]] = [[] for _ in range(num_clients)]
        for group_id, indices in enumerate(group_indices):
            rng.shuffle(indices)
            start_client = group_id * clients_per_group
            splits = np.array_split(indices, clients_per_group)
            for offset, split_indices in enumerate(splits):
                client_indices[start_client + offset].extend(split_indices.tolist())

        taskset = TaskSet()
        for i, client_indice in enumerate(client_indices):
            rng.shuffle(client_indice)
            task = Task(
                owner_id=client_owner(i),
                dataset_tag=store.name,
                split=split,
                indices=client_indice,
            )
            taskset.add_task(task)
        return taskset


def build_partitioner(config: "PartitionerConfig", seed: int) -> Partitioner:
    """
    根据配置构建划分器的工厂函数。

    Args:
        config: 划分器配置 (PartitionerConfig)
        seed: 随机种子

    Returns:
        实例化的划分器

    Raises:
        ValueError: 未知的划分器名称
    """
    if config.name == "dirichlet":
        return DirichletPartitioner(
            alpha=config.alpha,
            seed=seed,
            max_retries=config.max_retries
        )
    elif config.name == "iid":
        return IIDPartitioner(seed=seed)
    elif config.name == "q_non_iid":
        return QNonIIDPartitioner(q=config.q, seed=seed)
    else:
        raise ValueError(f"Unknown partitioner: {config.name}. "
                         f"Supported: 'iid', 'dirichlet', 'q_non_iid'")
