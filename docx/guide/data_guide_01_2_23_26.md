# SafeFL 数据层开发指南

> 版本: 2025-02-23  
---

## 1. 架构概览

SafeFL 数据层采用**分层解耦**设计，将数据获取、数据划分、任务生成三大职责分离，每个模块单一职责、接口清晰。

```
┌─────────────────────────────────────────────────────────────┐
│                     TaskGenerator                           │
│           (Process Controller: Orchestrator)                │
└─────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│    DatasetStore   │ │    Partitioner    │ │      TaskSet      │
│ (Data Container)  │ │ Partition Strategy│ │   Task Collection │
└───────────────────┘ └───────────────────┘ └───────────────────┘
            │                 │                 │
            ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                  DatasetRegistry (Registry)                 │
│   ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│   │  CIFAR10 │  │NewDataset│  │ More...  │                  │
│   └──────────┘  └──────────┘  └──────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### 1.1 模块职责矩阵

| 模块 | 核心职责 | 设计模式 |
| :--- | :--- | :--- |
| `DatasetRegistry` | 管理数据集工厂函数 | 注册表模式 (Registry) |
| `DatasetStore` | 统一封装 PyTorch Dataset，提供标签获取 | 适配器模式 (Adapter) |
| `Partitioner` | 将数据划分给各客户端 | 策略模式 (Strategy) |
| `TaskGenerator` | 执行数据管道全流程 | 模板方法模式 (Template Method) |
| `TaskSet/Task` | 描述数据归属与索引 | 数据对象 (Data Transfer Object) |

---

## 2. 快速上手

### 2.1 加载数据集

```python
from data.registry import dataset_registry
import data.datasets.cifar10  # 导入即注册

# 构建无增强训练集
train_store = dataset_registry.build(
    name="cifar10_train_plain",
    root="./datasets",
    is_train=True,
)

print(f"数据集大小: {len(train_store)}")  # 50000
print(f"第一条样本: {train_store[0]}")    # (tensor, label)
```

### 2.2 生成联邦学习任务

```python
from data.task_generator import TaskGenerator
from data.partitioner import IIDPartitioner

# 步骤1: 选择划分策略
partitioner = IIDPartitioner(seed=42)

# 步骤2: 配置任务生成器
generator = TaskGenerator(
    dataset_name="cifar10",
    root="./datasets",
    partitioner=partitioner,
    num_clients=10,      # 10个客户端
    val_ratio=0.2,       # 20%作为验证集
    seed=42,
)

# 步骤3: 执行生成
task_set, stores = generator.generate()

# 步骤4: 查看结果
client_ids = task_set.list_client_ids(exclude_server=True)
print(f"客户端数量: {len(client_ids)}")  # 10

# 获取 client_0 的训练任务
task = task_set.get_task("client_0", "train")
print(f"client_0 训练样本数: {len(task.indices)}")
```

---

## 3. 核心模块详解

### 3.1 DatasetRegistry (注册中心)

**为什么需要它**: 避免全局字典、统一管理数据集创建逻辑、支持快速失败。

```python
from data.registry import dataset_registry

# 注册数据集工厂函数
@dataset_registry.register("my_dataset_train")
def build_my_train(root: str, is_train: bool):
    # 返回 DatasetStore 实例
    return DatasetStore("my", "train", my_dataset)

# 使用注册的数据集
store = dataset_registry.build("my_dataset_train", root="./data", is_train=True)
```

**错误处理**:
- 重复注册 → 立即抛出 `ValueError`
- 未注册名称 → 抛出 `KeyError` 并提示已注册列表

### 3.2 DatasetStore (数据容器)

**职责**: 统一封装底层 Dataset，标准化标签访问。

```python
from data.dataset_store import DatasetStore
from torch.utils.data import Dataset

# 包装自定义 Dataset
class MyDataset(Dataset):
    def __init__(self):
        self.targets = [0, 1, 2, 3, 4]  # 必须有 targets 或 labels
    
    def __len__(self):
        return 5
    
    def __getitem__(self, idx):
        return idx, self.targets[idx]

# 创建 Store
my_ds = MyDataset()
store = DatasetStore(name="my_data", split="train", dataset=my_ds)

# 获取全部标签 (自动缓存)
labels = store.get_label()  # np.array([0, 1, 2, 3, 4])
```

**重要特性**:
- 支持 `torch.utils.data.Subset` 自动切片标签
- 标签缓存避免重复计算
- 无标签时自动回退到全遍历（带警告）

### 3.3 Partitioner (划分策略)

**基类定义**:

```python
from abc import ABC, abstractmethod
from data.dataset_store import DatasetStore
from data.task import TaskSet

class Partitioner(ABC):
    @abstractmethod
    def partition(self, store: DatasetStore, num_clients: int, split: str = "train") -> TaskSet:
        pass
```

**内置策略**:

| 策略类 | 特点 | 适用场景 |
| :--- | :--- | :--- |
| `IIDPartitioner` | 数据均匀随机打乱后均分 | 基准实验、理论分析 |
| `DirichletPartitioner` | 按 Dirichlet 分布分配，控制异构程度 | 模拟真实异构场景 |

**使用 Dirichlet 划分**:

```python
from data.partitioner import DirichletPartitioner

# alpha 越小，异构程度越高 (alpha=0.1 强烈异构, alpha=10 接近 IID)
partitioner = DirichletPartitioner(alpha=0.5, seed=42)

task_set = partitioner.partition(store, num_clients=10, split="train")
```

**自定义划分策略**:

```python
from data.partitioner import Partitioner
from data.task import Task, TaskSet
from data.constants import client_owner

class MyPartitioner(Partitioner):
    def __init__(self, my_param: float):
        self.my_param = my_param
    
    def partition(self, store, num_clients, split="train"):
        task_set = TaskSet()
        n = len(store)
        
        # 实现自定义划分逻辑
        indices_per_client = n // num_clients
        for i in range(num_clients):
            start = i * indices_per_client
            end = start + indices_per_client if i < num_clients - 1 else n
            indices = list(range(start, end))
            
            task = Task(
                owner_id=client_owner(i),
                dataset_tag=store.name,
                split=split,
                indices=indices,
            )
            task_set.add_task(task)
        
        return task_set
```

### 3.4 TaskGenerator (流程控制器)

**数据管道流程**:

```
┌────────────────────────────────────────────────────────────┐
│  Step 1: Load three data sources                           │
│  ├─ {name}_train_aug   (Augmented training data)           │
│  ├─ {name}_train_plain (Plain training data)               │
│  └─ {name}_test_plain  (Global test data)                  │
├────────────────────────────────────────────────────────────┤
│  Step 2: Sample proxy data from train_plain                │
│  (per_class samples for Server pre-train/eval)             │
├────────────────────────────────────────────────────────────┤
│  Step 3: Partition remaining data to clients               │
├────────────────────────────────────────────────────────────┤
│  Step 4: Split client data into train/val                  │
│  (controlled by val_ratio)                                 │
├────────────────────────────────────────────────────────────┤
│  Step 5: Assemble TaskSet                                  │
│  ├─ client_0 ~ client_N: train + val tasks                 │
│  └─ server: global_test + proxy tasks                      │
└────────────────────────────────────────────────────────────┘


原始数据 (CIFAR-10)
    │
    ├─ 加载三个数据源 (train_aug, train_plain, test_plain) → stores
    ├─ 按类均匀采样 proxy 子集
    ├─ 调用 Partitioner (IID / Dirichlet) 分配索引给各客户端
    ├─ 对每个客户端做 train/val 切分
    └─ 创建 server 端任务 (global_test, proxy)
    │
    ▼
输出: (TaskSet, stores)   ← 这才是真正被后续使用的东西


```

**配置参数**:

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `dataset_name` | str | 必填 | 数据集名称前缀，用于拼接 tag |
| `root` | str | 必填 | 数据存放根目录 |
| `partitioner` | Partitioner | 必填 | 划分策略实例 |
| `num_clients` | int | 必填 | 客户端数量 |
| `val_ratio` | float | 0.2 | 验证集占比 (0.0 ~ 1.0) |
| `seed` | int | 42 | 随机种子 |

**参数联动逻辑**:

| 参数组合 | 联动关系 | 风险提示 |
| :--- | :--- | :--- |
| `num_clients` ↑ + `val_ratio` → | 客户端越多，每人分得数据越少；验证比例固定时，训练集会急剧减少 | 当 `num_clients=100`、`val_ratio=0.3` 时，若总数据仅 5000，部分客户端可能分不到训练样本，触发 `ValueError` |
| `partitioner` (Dirichlet) + `num_clients` ↑ | 高异构 (α<0.5) + 多客户端场景，Dirichlet 分布难以均匀分配每类样本 | 可能超过 `max_retries` 限制，抛出 `RuntimeError` |
| `seed` vs `partitioner.seed` | 两者独立：`TaskGenerator.seed` 仅控制 train/val 切分；`Partitioner.seed` 控制数据划分 | 复现实验时需同时固定两个种子，否则结果不一致 |
| `dataset_name` + `dataset_registry` | `dataset_name` 会被拼接为 `{name}_train_aug/plain` 等 tag，必须在注册表中存在 | 未注册时抛出 `KeyError`，提示可用数据集列表 |

**参数配置公式**:

```
总训练样本 = (数据集大小 - proxy_samples) * (1 - val_ratio)
每客户端训练样本 ≈ 总训练样本 / num_clients

安全条件: 每客户端训练样本 >= 1
```

示例计算 (CIFAR10, 50000 样本):
- `num_clients=10`, `val_ratio=0.2` → 每人约 4000 训练样本 (安全)
- `num_clients=1000`, `val_ratio=0.5` → 每人约 25 训练样本 (临界，取决于 proxy 设置)

### 3.5 TaskSet & Task (任务描述)

**Task 结构**:

```python
from data.task import Task

task = Task(
    owner_id="client_0",           # 数据归属者
    dataset_tag="cifar10_train_aug", # 数据源标识
    split="train",                 # 划分类型
    indices=[0, 5, 10, 15],        # 数据索引列表
)
```

**TaskSet 操作**:

```python
from data.task import TaskSet

task_set = TaskSet()

# 添加任务
task_set.add_task(task)

# 获取任务 (不存在时抛出 KeyError)
task = task_set.get_task("client_0", "train")

# 安全获取 (不存在返回 None)
task = task_set.try_get_task("client_0", "train")

# 检查存在性
exists = task_set.has_task("client_0", "train")

# 获取所有客户端 ID (排除 server)
client_ids = task_set.list_client_ids(exclude_server=True)
```

---

## 4. 常量与命名规范

### 4.1 Split 类型

```python
from data.constants import (
    SPLIT_TRAIN,      # "train" - 客户端本地训练集
    SPLIT_TEST,       # "test"  - 客户端本地验证集
    SPLIT_TEST_GLOBAL, # "test_global" - Server 全局测试集
    SPLIT_PROXY,      # "proxy" - Server 代理数据集
    SPLIT_TEMP_ALL,   # "temp_all" - 临时完整划分
)
```

### 4.2 Owner 标识

```python
from data.constants import OWNER_SERVER, client_owner

server_id = OWNER_SERVER           # "server"
client_0_id = client_owner(0)      # "client_0"
client_5_id = client_owner(5)      # "client_5"
```

### 4.3 Tag 命名规则

数据集注册遵循统一命名规范:

```python
from data.constants import train_plain_tag, train_aug_tag, test_plain_tag

tag1 = train_plain_tag("cifar10")  # "cifar10_train_plain"
tag2 = train_aug_tag("cifar10")    # "cifar10_train_aug"
tag3 = test_plain_tag("cifar10")   # "cifar10_test_plain"
```

---

## 5. 添加新数据集

### 5.1 步骤指南

**Step 1**: 在 `data/datasets/` 下创建新文件

```python
# data/datasets/my_dataset.py
from data.dataset_store import DatasetStore
from data.registry import dataset_registry

# 定义三种数据源构建函数

@dataset_registry.register("mydata_train_aug")
def build_train_aug(root: str, is_train: bool) -> DatasetStore:
    del is_train  # 固定为 train
    ds = load_with_augmentation(root)
    return DatasetStore("mydata", "train", ds)

@dataset_registry.register("mydata_train_plain")
def build_train_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    ds = load_without_augmentation(root)
    return DatasetStore("mydata", "train_plain", ds)

@dataset_registry.register("mydata_test_plain")
def build_test_plain(root: str, is_train: bool) -> DatasetStore:
    del is_train
    ds = load_test_set(root)
    return DatasetStore("mydata", "test", ds)
```

**Step 2**: 在 `data/datasets/__init__.py` 中导入

```python
# data/datasets/__init__.py
from . import cifar10
from . import my_dataset  # 新增
```

**Step 3**: 编写测试

参考 `tests/data/fake_dataset.py` 编写假数据集，避免真实数据下载。

### 5.2 最佳实践

- **数据增强分离**: 始终分离 `train_aug` 和 `train_plain`，前者供客户端训练，后者供划分与评估
- **内存映射**: 大数据集使用 `MemoryMappedDataset` 避免全量加载
- **标签标准化**: Dataset 必须提供 `targets` 或 `labels` 属性

---

## 6. 测试指南

### 6.1 运行测试

```bash
# 运行 data 层全部测试
python -m pytest tests/data/ -v

# 运行单个测试文件
python -m pytest tests/data/test_data_pipeline.py -v
```

### 6.2 关键注意事项

**导入顺序** (非常重要):

```python
# 涉及 TaskGenerator 的测试，必须先导入 fake_dataset
import tests.data.fake_dataset  # noqa: F401
from data.task_generator import TaskGenerator
```

不遵守此顺序会导致测试时下载真实 CIFAR10 数据集。

### 6.3 编写测试示例

```python
import unittest
import numpy as np
from data.dataset_store import DatasetStore
from data.partitioner import IIDPartitioner

class TestMyFeature(unittest.TestCase):
    def _make_store(self, n: int, n_classes: int = 10):
        """辅助方法：构造假数据存储。"""
        class _SimpleDataset:
            def __init__(self, data, targets):
                self.data = data
                self.targets = targets
            def __len__(self):
                return len(self.data)
            def __getitem__(self, i):
                return self.data[i], self.targets[i]
        
        targets = np.arange(n, dtype=np.int64) % n_classes
        data = np.zeros((n, 32, 32, 3), dtype=np.uint8)
        ds = _SimpleDataset(data, targets)
        return DatasetStore("test", "train", ds)
    
    def test_partition(self):
        # Arrange
        store = self._make_store(100)
        partitioner = IIDPartitioner(seed=42)
        
        # Act
        task_set = partitioner.partition(store, num_clients=5, split="train")
        
        # Assert
        self.assertEqual(len(task_set.list_client_ids()), 5)
```

---

## 7. 常见问题

### Q1: 如何调整客户端数据异构程度？

使用 `DirichletPartitioner`，通过 `alpha` 参数控制:
- `alpha=0.1` ~ `0.5`: 强异构 (每个客户端只有少数几类)
- `alpha=1.0` ~ `5.0`: 中度异构
- `alpha=10.0+`: 接近 IID

### Q2: 客户端没有训练样本？

增加数据量、减少客户端数量，或降低 `val_ratio`:

```python
generator = TaskGenerator(
    # ...
    num_clients=5,      # 减少客户端数
    val_ratio=0.1,      # 降低验证比例
)
```

### Q3: 如何获取某个客户端的数据？

```python
# 获取 client_0 的训练数据索引
task = task_set.get_task("client_0", "train")
indices = task.indices

# 通过索引访问原始数据
store = stores[task.dataset_tag]
for idx in indices:
    image, label = store[idx]
```

### Q4: 如何添加自定义划分策略？

继承 `Partitioner` 基类，实现 `partition` 方法，返回 `TaskSet`。参考第 3.3 节示例。

---

## 8. API 速查表

### DatasetRegistry

| 方法 | 签名 | 说明 |
| :--- | :--- | :--- |
| `register` | `register(name: str) -> Callable` | 装饰器，注册数据集工厂 |
| `build` | `build(name, root, is_train)` | 构建数据集实例 |

### DatasetStore

| 方法 | 签名 | 说明 |
| :--- | :--- | :--- |
| `__len__` | `__len__() -> int` | 样本数量 |
| `__getitem__` | `__getitem__(index) -> Tuple` | 获取样本 |
| `get_label` | `get_label() -> np.ndarray` | 获取所有标签 |

### TaskGenerator

| 方法 | 签名 | 说明 |
| :--- | :--- | :--- |
| `__init__` | `__init__(dataset_name, root, partitioner, num_clients, val_ratio, seed)` | 初始化 |
| `generate` | `generate() -> Tuple[TaskSet, Dict[str, DatasetStore]]` | 执行生成 |

### TaskSet

| 方法 | 签名 | 说明 |
| :--- | :--- | :--- |
| `add_task` | `add_task(task: Task) -> None` | 添加任务 |
| `get_task` | `get_task(oid, split) -> Task` | 获取任务 |
| `try_get_task` | `try_get_task(oid, split) -> Optional[Task]` | 安全获取 |
| `has_task` | `has_task(oid, split) -> bool` | 检查存在性 |
| `list_client_ids` | `list_client_ids(exclude_server=True) -> List[str]` | 获取客户端列表 |

---


*文档结束。如有问题，请参考源码注释或提交 Issue。*
