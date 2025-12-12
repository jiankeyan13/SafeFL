# HFL 联邦学习框架

HFL 是一个面向研究与原型验证的联邦学习框架，提供可插拔的模型、聚合器与防御组件，配合可配置的数据划分与训练流程，便于快速复现实验或验证新想法。

## 项目结构
- `main.py`：入口脚本，读取 YAML 配置后构建 `FederatedRunner` 并启动训练循环。
- `configs/`：示例配置文件，`cifar10_fedavg.yaml` 展示了数据、模型、算法与训练的完整参数。
- `core/`：框架核心，包括 Runner、客户端/服务器基类、聚合/筛选/更新模块以及日志、调度器、指标等工具。
- `algorithms/`：具体算法装配逻辑，如 `fedavg.py` 将聚合器、更新器与客户端/服务器拼装为可运行的算法。
- `data/`：数据管道与划分工具，含数据集注册、Dirichlet/IID 划分、任务生成与数据仓库封装。
- `models/`：模型注册与定义，当前提供 ResNet18 作为默认 CNN 架构。

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

## 配置说明
- **project/name/seed/device/use_wandb**：实验元数据、随机种子与设备选择。
- **data**：数据集名称（需在 `data/datasets/` 注册）、根目录、客户端数量、本地验证比例与划分器（支持 `dirichlet` 或 `iid`）。
- **model**：模型注册名与构造参数，例如 `resnet18` 的 `num_classes`。
- **algorithm**：算法注册名及其自定义参数。
- **server**：聚合器、筛选器、防御或更新器配置（FedAvg 示例使用 `AvgAggregator` 与标准更新）。
- **training**：总轮次、每轮客户端数、评估间隔、学习率调度器与基础 LR 设置。
- **client**：本地训练的 epoch 数、批大小、动量、权重衰减等优化超参。
- **evaluation**：全局与本地评估时调用的指标列表（需在指标注册表中注册）。

## 运行流程概览
1. **任务生成**：`TaskGenerator` 加载增强/原始数据，按划分器为每个客户端切分训练/验证集，并为服务器生成全局测试任务。
2. **组件装配**：`FederatedRunner` 通过注册表获取模型、算法、聚合器等组件，并构建服务器与客户端类型。
3. **训练循环**：按轮次选择客户端 -> 下发模型 -> 本地训练并返回更新 -> 服务器聚合与评估 -> 按指标记录与保存最佳模型。

## 扩展指南
- 在 `models/` 中通过 `@MODEL_REGISTRY.register` 注册新模型。
- 在 `algorithms/` 中组合自定义聚合器/筛选器/更新器，注册为新算法。
- 在 `data/datasets/` 中新增数据集构建函数并注册，便可在配置中直接引用。
