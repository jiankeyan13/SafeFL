import time
import numpy as np
from torch.utils.data import Subset
from typing import Dict, List, Tuple
from data.constants import (
    SPLIT_TRAIN,
    SPLIT_TEST,
    SPLIT_TEST_GLOBAL,
    SPLIT_PROXY,
    SPLIT_TEMP_ALL,
    OWNER_SERVER,
    train_plain_tag,
    train_aug_tag,
    test_plain_tag,
    client_owner,
)
from data.dataset_store import build_dataset, DatasetStore
from data.partitioner import Partitioner
from data.task import Task, TaskSet

class TaskGenerator:
    """
    数据管道的核心控制器。
    职责：
    1. 加载三个数据源 (TrainAug, TrainPlain, TestPlain)。
    2. 调用 Partitioner 进行逻辑划分。
    3. 执行 Train/Val 切分。
    4. 组装最终的 TaskSet。
    """
    def __init__(self, 
                 dataset_name: str, 
                 root: str, 
                 partitioner: Partitioner, 
                 num_clients: int, 
                 val_ratio: float = 0.2,
                 seed: int = 42):
        """
        Args:
            dataset_name: 数据集名称前缀 (如 'cifar10')，会自动拼接后缀找注册表。
            root: 数据存放根目录。
            partitioner: 划分策略实例。
            num_clients: 客户端数量。
            val_ratio: 从客户端分到的数据中切出多少作为本地测试集 (0.0 ~ 1.0)。
            seed: 随机种子，用于 Train/Val 切分。
        """
        self.dataset_name = dataset_name
        self.root = root
        self.partitioner = partitioner
        self.num_clients = num_clients
        self.val_ratio = val_ratio
        self.rng = np.random.default_rng(seed)

        # 预加载数据源容器
        # 结构: { "tag_name": DatasetStore }
        self.stores: Dict[str, DatasetStore] = {}

    def generate(self) -> Tuple[TaskSet, Dict[str, DatasetStore]]:
        """
        执行生成流程。
        Returns:
            task_set: 包含所有任务的集合。
            stores: 数据源字典 (供 Trainer/Tester 使用)。
        """
        print(f"--- 开始生成任务: {self.dataset_name}, Clients: {self.num_clients} ---")

        t0 = time.perf_counter()
        self._load_sources()
        full_train_store = self._get_train_plain_store()
        t_load = time.perf_counter() - t0
        print(f"  [data] source load: {t_load:.2f}s")

        t1 = time.perf_counter()
        proxy_indices, remaining_indices = self._sample_proxy_indices(full_train_store, per_class=10)
        partition_result = self._partition_remaining(remaining_indices, full_train_store)
        t_partition = time.perf_counter() - t1
        print(f"  [data] partition+proxy: {t_partition:.2f}s")

        t2 = time.perf_counter()
        final_task_set = TaskSet()
        self._build_client_tasks(partition_result, remaining_indices, final_task_set)
        self._build_server_tasks(proxy_indices, final_task_set)
        t_compose = time.perf_counter() - t2
        print(f"  [data] task compose: {t_compose:.2f}s")

        print(f"--- 任务生成完毕 (total {time.perf_counter() - t0:.2f}s) ---")
        return final_task_set, self.stores

    def _get_train_plain_store(self) -> DatasetStore:
        """获取训练集（无增强）数据源，缺失时抛出。"""
        tpl_tag = train_plain_tag(self.dataset_name)
        if tpl_tag not in self.stores:
            raise ValueError(
                f"Required data source '{tpl_tag}' not found. "
                "Ensure dataset is registered and loading succeeded."
            )
        return self.stores[tpl_tag]

    def _sample_proxy_indices(
        self,
        store: DatasetStore,
        per_class: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """按类均匀采样 proxy 索引，返回 (proxy_indices, remaining_indices)。"""
        labels = store.get_label()
        unique_classes = np.unique(labels)
        proxy_indices = []
        for c in unique_classes:
            c_indices = np.where(labels == c)[0]
            selected = self.rng.choice(c_indices, min(per_class, len(c_indices)), replace=False)
            proxy_indices.extend(selected)
        proxy_indices = np.array(proxy_indices)
        all_indices_arr = np.arange(len(store))
        remaining_indices = np.setdiff1d(all_indices_arr, proxy_indices)
        return proxy_indices, remaining_indices

    def _partition_remaining(
        self,
        remaining_indices: np.ndarray,
        full_train_store: DatasetStore,
    ) -> TaskSet:
        """将剩余索引划分给各客户端，返回带相对索引的 TaskSet。"""
        remaining_subset = Subset(full_train_store.dataset, remaining_indices)
        remaining_store = DatasetStore("temp_remaining", "train", remaining_subset)
        return self.partitioner.partition(
            remaining_store,
            self.num_clients,
            split=SPLIT_TEMP_ALL,
        )

    def _build_client_tasks(
        self,
        partition_result: TaskSet,
        remaining_indices: np.ndarray,
        final_task_set: TaskSet,
    ) -> None:
        """将划分结果映射为客户端 train/val 任务并加入 final_task_set。"""
        for client_id in range(self.num_clients):
            owner = client_owner(client_id)
            temp_task = partition_result.get_task(owner, SPLIT_TEMP_ALL)
            relative_indices = np.array(temp_task.indices)
            all_indices = remaining_indices[relative_indices]
            self.rng.shuffle(all_indices)
            val_size = int(len(all_indices) * self.val_ratio)
            val_indices = all_indices[:val_size]
            train_indices = all_indices[val_size:]

            if len(train_indices) == 0:
                raise ValueError(
                    f"Client {owner} has no training samples after train/val split "
                    f"(val_ratio={self.val_ratio}). Reduce val_ratio or increase data per client."
                )

            final_task_set.add_task(Task(
                owner_id=owner,
                dataset_tag=train_aug_tag(self.dataset_name),
                split=SPLIT_TRAIN,
                indices=train_indices.tolist(),
            ))
            if len(val_indices) > 0:
                final_task_set.add_task(Task(
                    owner_id=owner,
                    dataset_tag=train_plain_tag(self.dataset_name),
                    split=SPLIT_TEST,
                    indices=val_indices.tolist(),
                ))

    def _build_server_tasks(
        self,
        proxy_indices: np.ndarray,
        final_task_set: TaskSet,
    ) -> None:
        """创建 server 端 global test 与 proxy 任务。"""
        tst_tag = test_plain_tag(self.dataset_name)
        if tst_tag in self.stores:
            test_store = self.stores[tst_tag]
            final_task_set.add_task(Task(
                owner_id=OWNER_SERVER,
                dataset_tag=tst_tag,
                split=SPLIT_TEST_GLOBAL,
                indices=list(range(len(test_store))),
            ))
        final_task_set.add_task(Task(
            owner_id=OWNER_SERVER,
            dataset_tag=train_plain_tag(self.dataset_name),
            split=SPLIT_PROXY,
            indices=proxy_indices.tolist(),
        ))

    def _load_sources(self) -> None:
        """内部方法：根据命名约定加载三个数据源"""
        sources_config = [
            (train_aug_tag(self.dataset_name), True),
            (train_plain_tag(self.dataset_name), True),
            (test_plain_tag(self.dataset_name), False),
        ]
        for full_tag, is_train in sources_config:
            print(f"Loading source: {full_tag} ...")
            try:
                # 调用 dataset_store.py 里的工厂
                store = build_dataset(full_tag, self.root, is_train)
                self.stores[full_tag] = store
            except KeyError:
                print(f"Warning: Dataset '{full_tag}' not found in registry. 跳过。")
            except Exception as e:
                print(f"Error loading '{full_tag}': {e}")
                raise e