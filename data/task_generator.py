import numpy as np
from typing import Dict, List, Tuple
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

        # 1. 加载数据源
        self._load_sources()

        # 2. 调用划分器 (使用 Plain 版本进行划分，因为 Partitioner 只看标签，不看 Transform)
        full_train_store = self.stores[f"{self.dataset_name}_train_plain"]
        
        # === 核心修改：提取Server Proxy数据 (每类10条) ===
        labels = full_train_store.get_label()
        unique_classes = np.unique(labels)
        proxy_per_class = 10
        
        proxy_indices = []
        for c in unique_classes:
            c_indices = np.where(labels == c)[0]
            selected = self.rng.choice(c_indices, proxy_per_class, replace=False)
            proxy_indices.extend(selected)
        
        proxy_indices = np.array(proxy_indices)
        all_indices_arr = np.arange(len(full_train_store))
        # 剩余用于FL训练的数据索引
        remaining_indices = np.setdiff1d(all_indices_arr, proxy_indices)
        
        # 构造临时 Store 传给划分器
        from torch.utils.data import Subset
        remaining_subset = Subset(full_train_store.dataset, remaining_indices)
        remaining_store = DatasetStore("temp_remaining", "train", remaining_subset)

        # split_name 暂时叫 'temp_all'，因为马上要被拆掉
        partition_result: TaskSet = self.partitioner.partition(
            remaining_store, 
            self.num_clients, 
            split="temp_all"
        )

        # 3. 组装最终 TaskSet
        final_task_set = TaskSet()

        # 3.1 处理每个客户端的 Train/Val 拆分
        for client_id in range(self.num_clients):
            owner = f"client_{client_id}"
            
            # 从 partition 结果中取出该客户端的所有索引 (相对索引 -> 绝对索引)
            temp_task = partition_result.get_task(owner, "temp_all")
            relative_indices = np.array(temp_task.indices)
            all_indices = remaining_indices[relative_indices]
            
            # 打乱并切分
            self.rng.shuffle(all_indices)
            val_size = int(len(all_indices) * self.val_ratio)
            
            # 切片
            val_indices = all_indices[:val_size]
            train_indices = all_indices[val_size:]

            # 创建 Train 任务 -> 指向 Aug 数据源
            train_task = Task(
                owner_id=owner,
                dataset_tag=f"{self.dataset_name}_train_aug", # 关键：用带增强的数据
                split="train",
                indices=train_indices.tolist()
            )
            final_task_set.add_task(train_task)

            # 创建 Val (Local Test) 任务 -> 指向 Plain 数据源
            if len(val_indices) > 0:
                val_task = Task(
                    owner_id=owner,
                    dataset_tag=f"{self.dataset_name}_train_plain", # 关键：用干净数据
                    split="test",  
                    indices=val_indices.tolist()
                )
                final_task_set.add_task(val_task)

        # 3.2 创建服务器端的 Global Test 任务
        test_store_tag = f"{self.dataset_name}_test_plain"
        if test_store_tag in self.stores:
            test_store = self.stores[test_store_tag]
            global_test_task = Task(
                owner_id="server",
                dataset_tag=test_store_tag,
                split="test_global",
                indices=list(range(len(test_store))) # 全量
            )
            final_task_set.add_task(global_test_task)
            
        # 3.3 注册 Server Proxy 任务
        proxy_task = Task(
            owner_id="server",
            dataset_tag=f"{self.dataset_name}_train_plain", # 用无增强源
            split="proxy",
            indices=proxy_indices.tolist()
        )
        final_task_set.add_task(proxy_task)

        print("--- 任务生成完毕 ---")
        return final_task_set, self.stores

    def _load_sources(self):
        """内部方法：根据命名约定加载三个数据源"""
        # 定义需要加载的后缀和对应的 is_train 参数
        # (注册名后缀, is_train_arg)
        sources_config = [
            ("_train_aug", True),
            ("_train_plain", True),
            ("_test_plain", False)
        ]

        for suffix, is_train in sources_config:
            full_tag = f"{self.dataset_name}{suffix}"
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

# === 简单的测试代码 ===
if __name__ == '__main__':
    # 为了运行这个测试，需要确保 registry 里注册了 cifar10_xxx 
    # 所以必须导入 cifar10 模块
    import datasets.cifar10 
    from partitioner import DirichletPartitioner

    # 1. 准备组件
    partitioner = DirichletPartitioner(seed=114514)
    
    # 2. 初始化 Generator
    generator = TaskGenerator(
        dataset_name="cifar10",
        root="./data_source",
        partitioner=partitioner,
        num_clients=10,
        val_ratio=0.2
    )

    # 3. 运行
    tasks, stores = generator.generate()

    # 4. 验证
    print("\n=== 验证生成结果 ===")
    t_train = tasks.get_task("client_0", "train")
    t_test = tasks.get_task("client_0", "test")
    
    print(f"Client 0 Train: {len(t_train.indices)} samples, Source: {t_train.dataset_tag}")
    print(f"Client 0 Test : {len(t_test.indices)} samples, Source: {t_test.dataset_tag}")
    
    # 验证数据源是否正确分离
    assert "aug" in t_train.dataset_tag
    assert "plain" in t_test.dataset_tag
    
    # 验证 Global Test
    t_global = tasks.get_task("server", "test_global")
    print(f"Server Global Test: {len(t_global.indices)} samples")