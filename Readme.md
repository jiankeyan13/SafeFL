# HFL 联邦学习框架

HFL 是一个面向研究与原型验证的联邦学习框架，提供可插拔的模型、聚合器与防御组件，配合可配置的数据划分与训练流程，便于快速复现实验或验证新想法。本文档基于 `configs/` 目录下的 YAML 配置，帮助你在阅读/修改配置时快速理解每个参数的含义，并梳理框架各模块的耦合关系与调用链，让你更快上手二次开发。

## 项目结构
- `main.py`：入口脚本，读取 YAML 配置后构建 `FederatedRunner` 并启动训练循环。
- `configs/`：示例配置文件，`cifar10_fedavg.yaml` 展示了数据、模型、算法与训练的完整参数。
- `core/`：框架核心，包括 Runner、客户端/服务器基类、聚合/筛选/更新模块以及日志、调度器、指标等工具。
- `algorithms/`：具体算法装配逻辑，如 `fedavg.py` 将聚合器、更新器与客户端/服务器拼装为可运行的算法。
- `data/`：数据管道与划分工具，含数据集注册、Dirichlet/IID 划分、任务生成与数据仓库封装。
- `models/`：模型注册与定义，当前提供 ResNet18 作为默认 CNN 架构。

## 配置参数速查表
以下表格基于 `configs/cifar10_fedavg.yaml` 与 `configs/cifar10_badnets.yaml`，涵盖常用参数及其作用，修改时可快速对照：

### 实验元数据
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `project` | 实验分组名，用于日志与 WandB | `"FL-Test"` |
| `name` | 当前实验名 | `"cifar10_avg_test1_4speed"` |
| `seed` | 全局随机种子（用于划分、采样与模型初始化） | `2025` |
| `device` | 计算设备 | `"cuda:6"` |
| `use_wandb` | 是否启用 Weights & Biases 记录 | `true` |

### 数据划分
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `data.dataset` | 任务数据集注册名 | `"cifar10"` |
| `data.root` | 数据存储根目录 | `"./data_source"` |
| `data.num_clients` | 模拟的总客户端数 | `10` |
| `data.val_ratio` | 本地验证集比例 | `0.1` |
| `data.partitioner.name` | 划分策略（`dirichlet` / `iid`） | `"dirichlet"` |
| `data.partitioner.alpha` | Dirichlet α（越小越非独立同分布） | `1.0` or `0.9` |

### 模型与算法
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `model.name` | 模型注册名（`models/__init__.py`） | `"resnet18"` |
| `model.params.*` | 构造参数 | `num_classes: 10` |
| `algorithm.name` | 算法注册名（`algorithms/`） | `"fedavg"` |

### 服务器组件
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `server.aggregator.name` | 聚合器（平均/中值/Trim-Mean 等） | `"avg"` |
| `server.screener.name` | 筛选器（恶意更新过滤；为空则不筛） | `null` |
| `server.updater.name` | 更新器（控制聚合结果如何写回） | `"standard"` |

### 训练流程
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `training.rounds` | 总通信轮数 | `100` |
| `training.clients_fraction` | 每轮采样比例（`总客户端数 * 比例`） | `0.2` 或 `0.4` |
| `training.attackers_frac` | 在被选中客户端中，攻击者占比 | `0.5`（仅在存在攻击配置时有效） |
| `training.eval_interval` | 本地评估间隔（轮） | `10` |
| `training.scheduler` | 学习率调度器 | `"cosine"` |
| `training.lr` / `min_lr` | 初始与最小学习率 | `0.01` / `0.001` |
| `training.warmup_rounds` | 预热轮数 | `5` |

### 客户端本地训练
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `client.epochs` | 本地 epoch 数 | `1` 或 `3` |
| `client.batch_size` | 批大小 | `32` |
| `client.momentum` | SGD 动量 | `0.9` |
| `client.weight_decay` | 权重衰减 | `0.0005` |
| `client.num_workers` | DataLoader 线程数（可选） | 默认 `0` |

### 攻击/防御（可选）
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `attack.fraction` | 全局攻击者占比 | `0.2` |
| `attack.strategies` | 分组定义：每组含 `fraction`、`strategy`、`evaluation` | `badnets_group: ...` |
| `attack.strategies.*.strategy.name` | 攻击策略名 | `"badnets"` |
| `attack.strategies.*.strategy.params` | 策略参数（如 target label、poison ratio） | 见 `cifar10_badnets.yaml` |
| `attack.strategies.*.evaluation` | 针对该组的指标 | `asr` |

### 评估指标
| 参数 | 说明 | 示例 |
| --- | --- | --- |
| `evaluation.global` | 全局测试集指标列表 | `acc`、`loss` |
| `evaluation.local` | 随机抽样客户端的本地指标 | `acc` |

## 环境准备
1. 安装 Python 3.8+ 以及 PyTorch（含 torchvision）。
2. 安装依赖（示例）：
   ```bash
   pip install torch torchvision pyyaml numpy
   ```

## 快速开始
1. 确认配置文件：默认使用 `configs/cifar10_fedavg.yaml`，可通过 `--config` 指定其它路径。
2. 运行训练：
   ```bash
   python main.py --config configs/cifar10_fedavg.yaml
   ```
   首次运行会自动下载 CIFAR-10 数据集至 `data_source/` 目录。
3. 日志与模型：`core.utils.logger.Logger` 会在 `runs/` 下创建实验目录，保存日志与检查点（如最佳权重 `checkpoint_best.pth`）。

## 模块耦合与调用关系
### Runner 如何串联各组件
1. **读取配置并设定随机性**：`main.py` 解析 YAML 后创建 `FederatedRunner`，设置日志、设备与随机种子。
2. **数据任务生成**：在 Runner 的 `_setup` 中，根据 `data.partitioner` 选择 `DirichletPartitioner` 或 `IIDPartitioner`，交给 `TaskGenerator` 切分训练/验证/测试子集。
3. **模型与算法装配**：Runner 依据 `model.name` 从注册表创建模型，再调用 `ALGORITHM_REGISTRY.build(algorithm.name, ...)` 返回 `(server, client_class)`。FedAvg 例子中，算法工厂组装了 `AvgAggregator`、`BaseUpdater` 与 `BaseClient`，并把全局测试集打包为服务器的 `test_loader`。
4. **学习率调度与攻击者管理**：`core.utils.scheduler.build_scheduler` 读取 `training` 配置动态生成 LR 曲线；若配置了 `attack`，`AttackManager` 会按 `fraction` 与策略分组采样攻击者。
5. **训练主循环**：每轮按 `training.clients_fraction` 选取客户端，服务器 `broadcast` 模型权重，客户端 `execute` 完成本地训练并上报更新，服务器按“筛选器 -> 聚合器 -> 更新器”流水线执行 `step`，随后运行全局/本地评估并记录日志与最佳检查点。

### 服务器侧耦合
- **筛选器 (`server.screener`)**：可选模块，接收 `(updates, global_model)`，丢弃或修正可疑更新后返回净化列表。
- **聚合器 (`server.aggregator`)**：对筛选后的 `weights` 按样本数加权聚合（如均值、中值、Trim-Mean）。
- **更新器 (`server.updater`)**：将聚合结果写回全局模型，支持自定义规则（例如加权步长、裁剪等）。
这三个模块按顺序串联在 `BaseServer.step` 中，任何一个模块都可通过配置替换，而不影响其他模块接口。

### 客户端侧耦合
- **数据加载 (`BaseClient.data_load`)**：结合任务索引与攻击策略生成本地 `DataLoader`，同时统计样本数用于服务器加权。
- **训练 (`BaseClient.train`)**：通过 `StandardTrainer` 完成本地若干 epoch 的前向/反向；攻击策略可注入损失或梯度钩子。
- **更新打包 (`BaseClient.update`)**：计算模型权重差并上报 `{weights, num_samples, metrics}`，供服务器聚合。

### 算法装配示例：FedAvg
`algorithms/fedavg.py` 展示了最基础的算法装配模式：
- 直接使用 `AvgAggregator` 做参数平均。
- 不启用筛选器（`screener=None`），更新器使用默认 `BaseUpdater`。
- 返回的 `BaseServer` 与 `BaseClient` 由 Runner 接管运行；若要自定义算法，只需在同目录新增构建函数并注册到 `ALGORITHM_REGISTRY`，即可在配置里用 `algorithm.name: <your_algo>` 无缝切换。

## 运行流程概览
1. **任务生成**：`TaskGenerator` 加载增强/原始数据，按划分器为每个客户端切分训练/验证集，并为服务器生成全局测试任务。
2. **组件装配**：`FederatedRunner` 通过注册表获取模型、算法、聚合器等组件，并构建服务器与客户端类型。
3. **训练循环**：按轮次选择客户端 -> 下发模型 -> 本地训练并返回更新 -> 服务器聚合与评估 -> 按指标记录与保存最佳模型。

## 扩展指南
- 在 `models/` 中通过 `@MODEL_REGISTRY.register` 注册新模型。
- 在 `algorithms/` 中组合自定义聚合器/筛选器/更新器，注册为新算法。
- 在 `data/datasets/` 中新增数据集构建函数并注册，便可在配置中直接引用。
