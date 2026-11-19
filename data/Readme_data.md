# SafeFL v2 数据层 (`data/`)

面向联邦学习与 Ray 多进程并行的**零拷贝、批向量化**数据层：原始图像持久化为 `CHW uint8` memmap 缓存，跨进程仅传递轻量索引描述（`< 200 KB`），批加载、投毒、增强与归一化全流程向量化执行。

## 核心架构与数据流

```text
Driver: DataHub.build(data_cfg, num_clients, seed)
          ├── DATASET_REGISTRY ──▶ ensure_chw_mmap_cache() ──▶ *_chw_v2_{data,targets}.bin
          └── PARTITIONER_REGISTRY ──▶ create_partition_map() ──▶ PartitionMap (int32 索引)
                                                                      │
                                      hub.spec (FederatedDataSpec) ◀──┘  (仅路径+元数据+索引，无图像拷贝)
                                                │
Worker: DataHub.from_spec(spec, device) ────────┘
          ├── hub.client(cid) / hub.test() / hub.proxy() ──▶ DataView (零拷贝索引切片)
          └── view.loader(batch_size, device, poisoner=...) ──▶ FastBatchLoader
```

### 单 Batch 执行顺序 (`FastBatchLoader`)

1. **Gather**：按索引从 mmap / VRAM 读取 `uint8 [B, C, H, W]` 与 `int64 [B]` → 移至 `device` → 转为 `[0.0, 1.0]` `float32`（每次独立分配，不污染共享底层）。
2. **Poison（可选）**：在真实 `[0, 1]` 像素空间调用 `BatchPoisoner`（位于增强与归一化之前）。
3. **Augment（可选）**：`apply_batch_augment` 对 batch 内各样本独立执行向量化 `HorizontalFlip` / `RandomCrop`。
4. **Normalize**：原地执行通道归一化 `(x - mean) / std`。

## 模块一览

| 文件 | 核心对象 | 职责 |
| :--- | :--- | :--- |
| `hub.py` | `DataHub` | 顶层入口；Driver 端 `build()`、Worker 端 `from_spec()`，提供 `client(cid)` / `test()` / `proxy()` 视图及显存预加载。 |
| `spec.py` | `FederatedDataSpec`, `DatasetPairSpec`, `SplitSpec`, `PartitionMap`, `AugmentSpec` | 可序列化的纯元数据/索引描述，保证 Ray 传输零图像开销。 |
| `source.py` | `ensure_chw_mmap_cache`, `SharedDataSource` | 原子生成与校验 `CHW uint8` 缓存；提供只读 mmap 与可选 VRAM 预加载。 |
| `loader.py` | `DataView`, `FastBatchLoader`, `BatchPoisoner`, `apply_batch_augment` | 零拷贝索引切片（`subset` / `split` / `loader`）、无 worker 向量化迭代器与投毒协议。 |
| `partition.py` | `Partitioner`, `create_partition_map` | 纯索引客户端数据划分器及服务端类别均衡 Proxy 集抽取。 |
| `registry.py` | `DataRegistry`, `DATASET_REGISTRY`, `PARTITIONER_REGISTRY` | 数据层统一注册表（数据集构建函数与划分器注册）。 |
| `datasets/` | `cifar10`, `cifar100`, `fashionmnist`, `tiny_imagenet` | 内置数据集注册与默认增强/归一化参数。 |

## 内置数据集与划分器

### 内置数据集 (`DATASET_REGISTRY`)

| 名称 | 类别数 | 归一化 `(mean / std)` | 默认训练增强 (`AugmentSpec`) |
| :--- | :---: | :--- | :--- |
| `cifar10` | 10 | `(0.5, 0.5, 0.5)` / `(0.5, 0.5, 0.5)` | `pad=4, hflip=True` |
| `cifar100` | 100 | `(0.5071, 0.4867, 0.4408)` / `(0.2675, 0.2565, 0.2761)` | `pad=4, hflip=True` |
| `fashionmnist` | 10 | `(0.2860,)` / `(0.3530,)` | `pad=2, hflip=True` |
| `tiny_imagenet` | 200 | `(0.485, 0.456, 0.406)` / `(0.229, 0.224, 0.225)` | `pad=8, hflip=True` |

> `tiny_imagenet` 首次使用会自动下载并解压；可通过 `SAFEFL_TINY_IMAGENET_URL` 指定自定义镜像，或设 `SAFEFL_SKIP_TINY_IMAGENET_DOWNLOAD=1` 跳过自动下载。

### 内置划分器 (`PARTITIONER_REGISTRY`)

| 名称 | 说明 | 关键参数 |
| :--- | :--- | :--- |
| `iid` | 均匀随机划分 | `seed` |
| `dirichlet` | 带均衡约束的 `Dirichlet(alpha)` Non-IID 划分 | `alpha=1.0`, `max_retries=100`, `seed` |
| `q_non_iid` | 按主类比例 `q` 划分（要求 `num_clients % num_classes == 0`） | `q=0.5`, `seed` |

## 配置项 (`DataConfig`)

| 字段 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `dataset` | `"cifar10"` | 已注册的数据集名称 |
| `root` | `"./data_source"` | 原始数据与 `*_chw_v2_*` 缓存目录 |
| `enable_proxy` / `proxy_per_class` | `False` / `10` | 是否从训练集预先抽取每类 `K` 条样本作为服务端 Proxy 集 |
| `use_augmentation` | `False` | 客户端训练视图是否默认启用数据集的 `default_augment` |
| `preload_gpu` | `False` | Worker 重建时是否将训练集原始 `uint8` 张量预加载到 CUDA 显存 |
| `partitioner` | `q_non_iid` | 划分器配置 `{name, params}` |

## 扩展指南

- **新增数据集**：在 `data/datasets/` 下新建模块，使用 `@DATASET_REGISTRY.register("my_ds")` 注册并返回 `DatasetPairSpec`（借助 `ensure_chw_mmap_cache`），然后在 `data/datasets/__init__.py` 导入。
- **新增划分器**：继承 `Partitioner` 实现 `partition(targets, num_clients) -> Dict[int, np.ndarray]`，并用 `@PARTITIONER_REGISTRY.register("my_part")` 注册。
- **接入投毒算法**：实现 `BatchPoisoner` 协议的 `build_mask`、`apply_train_batch`、`apply_eval_batch` 三个方法即可，无需修改数据层代码。
