import numpy as np
from abc import ABC, abstractmethod
from task import Task, TaskSet
from dataset_store import DatasetStore
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
        for i,  client_indice in enumerate(splits):
            task = Task(
                owner_id=f"client_{i}",
                dataset_tag=store.name,
                split=split,
                indices=client_indice.tolist() # 转成纯 Python list 方便序列化
            )
            taskset.add_task(task)
        return taskset

class Dirichlet_Partitioner(Partitioner):
    def __init__(self, alpha: float=1.0, seed: int=42):
        self.alpha = alpha
        self.seed = seed
    
    def partition(self, store: DatasetStore, num_clients: int, split: str="train")->TaskSet:
        n_samples = len(store)
        labels = store.get_label()
        num_classes = len(np.unique(labels))

        rng = np.random.default_rng(self.seed)
        client_indices = []

        min_size = 0
        while min_size < num_classes:# 防止客户端数据量过低->num_classes可以考虑替换为batch_size
            client_indices = [[] for _ in range(num_clients)]

            for k in range(num_classes):
                idk_k = np.where(labels == k)[0]
                rng.shuffle(idk_k)

                #处理每个客户端的样本数
                proportion = rng.dirichlet(np.repeat(self.alpha, num_clients))
                proportion = np.array([p * (len(idx_j)<n_samples/num_clients)\
                                       for idx_j, p in zip(client_indices, proportion)])
                proportion = proportion / proportion.sum()

                split_points = (np.cumsum(proportion)*len(idk_k)).astype(int)[:-1]
                split_batch = np.split(idk_k, split_points)
                for i in range(num_clients):
                    client_indices[i].extend(split_batch[i].tolist())

            # min_size = min([len(client_indices[i]) for i in range(num_clients)])
            # 优化：使用生成器表达式
            min_size = min(len(c_idx) for c_idx in client_indices)

        taskset = TaskSet()
        for i, client_indice in enumerate(client_indices):
            rng.shuffle(client_indice)#标签因for有序放置
            task = Task(
                owner_id=f"client_{i}",
                dataset_tag=store.name,
                split=split,
                indices=client_indice
            )
            taskset.add_task(task)
        return taskset

"""
class Balanced_DirichletP(Partitioner):
class Pathological_Partitioner(Partitioner):
"""

if __name__ == '__main__':
    class TestData:
        def __init__(self):
            self.name = "test"
        def __len__(self):
            return 100
    testdata = TestData()
    iid_partitioner = Partitioner()
    taskset = iid_partitioner.partition(testdata, 3)

    t0 = taskset.get_task("client_0", "train")
    print(f"Client_0样本数:{len(t0.indices)}")
    print(f"Client_0样本前5索引:{t0.indices[:5]}")



    