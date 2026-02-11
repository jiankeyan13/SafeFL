import numpy as np
from abc import ABC, abstractmethod
from data.constants import client_owner
from data.task import Task, TaskSet
from data.dataset_store import DatasetStore
from typing import List, Dict

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
