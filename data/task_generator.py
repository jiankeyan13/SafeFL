import numpy as np
from torch.utils.data import Subset
from typing import Dict, Tuple
from data.constants import (
    SPLIT_TRAIN,
    SPLIT_TEST_GLOBAL,
    SPLIT_PROXY,
    SPLIT_TEMP_ALL,
    OWNER_SERVER,
    train_plain_tag,
    train_aug_tag,
    test_plain_tag,
    client_owner,
)
from data.dataset_store import DatasetStore
from data.partitioner import Partitioner
from data.task import Task, TaskSet
from data.registry import dataset_registry


class TaskGenerator:
    """
    数据管道的核心控制器。
    职责：
    1. 加载训练/测试数据源 (可选 TrainAug, 以及 TrainPlain, TestPlain)。
    2. 调用 Partitioner 进行逻辑划分。
    3. 组装最终的 TaskSet (客户端全量数据用于训练)。
    """
    def __init__(
        self,
        dataset_name: str,
        root: str,
        partitioner: Partitioner,
        num_clients: int,
        seed: int = 42,
        enable_proxy: bool = False,
        use_aug: bool = False,
    ):
        """
        Args:
            dataset_name: 数据集名称前缀 (如 'cifar10'), 会自动拼接后缀找注册表。
            root: 数据存放根目录。
            partitioner: 划分策略实例。
            num_clients: 客户端数量。
            seed: 随机种子。
            enable_proxy: 为 True 时从训练集按类划出 proxy 样本 (每类 10 条) 并创建 SPLIT_PROXY 任务; 为 False 时不划样本、无 proxy 任务。
            use_aug: 为 True 时客户端训练使用带增强的 train_aug; 默认 False 使用 train_plain。
        """
        self.dataset_name = dataset_name
        self.root = root
        self.partitioner = partitioner
        self.num_clients = num_clients
        self.enable_proxy = enable_proxy
        self.use_aug = bool(use_aug)
        self.rng = np.random.default_rng(seed)

        # 预加载数据源容器
        # 结构: { "tag_name": DatasetStore }
        self.stores: Dict[str, DatasetStore] = {}

    def _train_tag(self) -> str:
        return train_aug_tag(self.dataset_name) if self.use_aug else train_plain_tag(self.dataset_name)

    def generate(self) -> Tuple[TaskSet, Dict[str, DatasetStore]]:
        """
        执行生成流程。
        Returns:
            task_set: 包含所有任务的集合。
            stores: 数据源字典 (供 Trainer/Tester 使用)。
        """
        self._load_sources()
        full_train_store = self._get_train_plain_store()

        if self.enable_proxy:
            proxy_indices, remaining_indices = self._sample_proxy_indices(
                full_train_store, per_class=10
            )
        else:
            proxy_indices = np.array([], dtype=np.intp)
            remaining_indices = np.arange(len(full_train_store), dtype=np.intp)
        partition_result = self._partition_remaining(remaining_indices, full_train_store)

        final_task_set = TaskSet()
        self._build_client_tasks(partition_result, remaining_indices, final_task_set)
        self._build_server_tasks(proxy_indices, final_task_set)
        return final_task_set, self.stores

    def _get_train_plain_store(self) -> DatasetStore:
        return self.stores[train_plain_tag(self.dataset_name)]

    def _sample_proxy_indices(self, store: DatasetStore, per_class: int = 10):
        """按类采样 proxy 索引, 返回 (proxy_indices, remaining_indices)。"""
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

    def _partition_remaining(self, remaining_indices: np.ndarray, full_train_store: DatasetStore) -> TaskSet:
        """将剩余索引划分给各客户端, 返回带相对索引的 TaskSet。"""
        remaining_subset = Subset(full_train_store.dataset, remaining_indices)
        remaining_store = DatasetStore("temp_remaining", "train", remaining_subset)
        return self.partitioner.partition(
            remaining_store,
            self.num_clients,
            split=SPLIT_TEMP_ALL,
        )

    def _build_client_tasks(self, partition_result: TaskSet, remaining_indices: np.ndarray, final_task_set: TaskSet) -> None:
        """将划分结果映射为客户端 train 任务并加入 final_task_set。"""
        for client_id in range(self.num_clients):
            owner = client_owner(client_id)
            temp_task = partition_result.get_task(owner, SPLIT_TEMP_ALL)
            relative_indices = np.array(temp_task.indices)
            all_indices = remaining_indices[relative_indices]
            self.rng.shuffle(all_indices)

            if len(all_indices) == 0:
                raise ValueError(f"Client {owner} has no training samples after partition.")

            final_task_set.add_task(Task(
                owner_id=owner,
                dataset_tag=self._train_tag(),
                split=SPLIT_TRAIN,
                indices=all_indices.tolist(),
            ))

    def _build_server_tasks(self, proxy_indices: np.ndarray, final_task_set: TaskSet) -> None:
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
        if len(proxy_indices) > 0:
            final_task_set.add_task(Task(
                owner_id=OWNER_SERVER,
                dataset_tag=train_plain_tag(self.dataset_name),
                split=SPLIT_PROXY,
                indices=proxy_indices.tolist(),
            ))

    def _load_sources(self) -> None:
        """内部方法: 根据命名约定加载数据源。use_aug=False 时不加载 train_aug。"""
        sources_config = [
            (train_plain_tag(self.dataset_name), True),
            (test_plain_tag(self.dataset_name), False),
        ]
        if self.use_aug:
            sources_config.insert(0, (train_aug_tag(self.dataset_name), True))
        for full_tag, is_train in sources_config:
            self.stores[full_tag] = dataset_registry.build(full_tag, self.root, is_train)
