import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class TrainerConfig:
    """配置模型训练的参数"""
    optimizer_name: str = "SGD"             # 优化器名称
    lr: float = 0.01                        # 学习率
    momentum: float = 0.9                   # 动量
    weight_decay: float = 5e-4              # 权重衰减
    criterion_name: str = "CrossEntropyLoss" # 损失函数名称
    epochs: int = 2                         # 本地训练轮数
    batch_size: int = 32                    # 训练批次大小
    num_workers: int = 0                    # 数据加载线程数
    grad_clip_norm: Optional[float] = 5.0   # 梯度裁剪阈值
    extra_params: Dict[str, Any] = field(default_factory=dict) # 额外参数

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """根据配置动态实例化优化器"""
        kwargs = {}
        if self.optimizer_name == 'SGD':
            kwargs = {'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.weight_decay}
        elif self.optimizer_name == 'Adam' or self.optimizer_name == 'AdamW':
            kwargs = {'lr': self.lr, 'weight_decay': self.weight_decay}
            
        kwargs.update(self.extra_params.get('optimizer_kwargs', {}))
        optimizer_class = getattr(torch.optim, self.optimizer_name)
        return optimizer_class(model.parameters(), **kwargs)
        
    def build_criterion(self) -> torch.nn.Module:
        """根据配置动态实例化损失函数"""
        kwargs = self.extra_params.get('criterion_kwargs', {})
        criterion_class = getattr(torch.nn, self.criterion_name)
        return criterion_class(**kwargs)

@dataclass
class ClientConfig:
    """配置客户端运行的基础参数"""
    bn_calib_batches: int = 5               # BN校准批次数
    num_workers: int = 0                    # 数据加载线程数
    batch_size: int = 32                    # 批次大小
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig) # 训练配置

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClientConfig':
        """从字典解析配置"""
        trainer_kwargs = {}
        for key in ['lr', 'momentum', 'weight_decay', 'optimizer_name', 'criterion_name', 'epochs', 'grad_clip_norm']:
            if key in config_dict:
                trainer_kwargs[key] = config_dict[key]

        for key in ['batch_size', 'num_workers']:
            if key in config_dict:
                trainer_kwargs[key] = config_dict[key]

        trainer_cfg = TrainerConfig(**trainer_kwargs)
        client_kwargs = {'trainer_config': trainer_cfg}
        for key in ['bn_calib_batches', 'num_workers', 'batch_size']:
            if key in config_dict:
                client_kwargs[key] = config_dict[key]

        return cls(**client_kwargs)

@dataclass
class LRScheduleConfig:
    """学习率调度配置 (warmup + cosine)"""
    enabled: bool = True                    # 是否启用
    name: str = "warmup_cosine"             # 调度类型, 目前仅支持 warmup_cosine
    warmup_ratio: float = 0.1               # warmup 占 total_rounds 的比例
    min_lr: float = 0.0                     # cosine 衰减到的最小 lr
    warmup_start_lr: float = 0.0            # warmup 起始 lr

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LRScheduleConfig':
        """从字典解析配置"""
        if not config_dict:
            return cls()
        return cls(
            enabled=config_dict.get("enabled", True),
            name=config_dict.get("name", "warmup_cosine"),
            warmup_ratio=config_dict.get("warmup_ratio", 0.1),
            min_lr=config_dict.get("min_lr", 0.0),
            warmup_start_lr=config_dict.get("warmup_start_lr", 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "enabled": self.enabled,
            "name": self.name,
            "warmup_ratio": self.warmup_ratio,
            "min_lr": self.min_lr,
            "warmup_start_lr": self.warmup_start_lr,
        }


@dataclass
class TrainingConfig:
    """配置联邦学习训练流程的全局参数"""
    num_clients: int = 100                   # 总客户端数量
    rounds: int = 100                        # 总训练轮数
    clients_fraction: float = 0.2           # 每轮参与训练的客户端比例
    eval_interval: int = 5                  # 评估间隔轮数
    local_eval_ratio: float = 0.2           # 本地评估抽样比例
    seed: int = 42                          # 随机种子
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig) # 学习率调度

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典解析配置"""
        kwargs = {}
        for key in ['num_clients', 'rounds', 'clients_fraction', 'eval_interval', 'local_eval_ratio', 'seed']:
            if key in config_dict:
                kwargs[key] = config_dict[key]
        if 'lr_schedule' in config_dict:
            kwargs['lr_schedule'] = LRScheduleConfig.from_dict(config_dict['lr_schedule'])
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式 (供 Runner 使用)"""
        return {
            "num_clients": self.num_clients,
            "rounds": self.rounds,
            "clients_fraction": self.clients_fraction,
            "eval_interval": self.eval_interval,
            "local_eval_ratio": self.local_eval_ratio,
            "seed": self.seed,
            "lr_schedule": self.lr_schedule.to_dict(),
        }

@dataclass
class PartitionerConfig:
    """配置数据划分器的参数"""
    name: str = "dirichlet"                 # 划分器名称
    params: Dict[str, Any] = field(default_factory=dict) # 划分参数

    @property
    def alpha(self) -> float:
        """Dirichlet 划分的 alpha 参数"""
        return self.params.get("alpha", 1.0)

    @property
    def max_retries(self) -> int:
        """Dirichlet 划分的最大重试次数"""
        return self.params.get("max_retries", 100)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PartitionerConfig':
        """从字典解析配置"""
        if not config_dict:
            return cls()
        return cls(
            name=config_dict.get("name", "dirichlet"),
            params=config_dict.get("params", {})
        )

@dataclass
class AttackStrategyConfig:
    """单个攻击策略的配置"""
    name: str = "badnets"                   # 攻击名称
    fraction: float = 1.0                   # 占所有恶意客户端的比例
    params: Dict[str, Any] = field(default_factory=dict) # 攻击参数

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AttackStrategyConfig':
        """从字典解析配置"""
        if not config_dict:
            return cls()
        return cls(
            name=config_dict.get("name", "badnets"),
            fraction=config_dict.get("fraction", 1.0),
            params=config_dict.get("params", {})
        )

@dataclass
class AttackConfig:
    """攻击全局配置"""
    enabled: bool = False                   # 是否启用攻击
    malicious_fraction: float = 0.2         # 恶意客户端占总客户端的比例
    per_round_fraction: float = 0.2         # 每轮选中者中恶意客户端的比例
    strategies: List[AttackStrategyConfig] = field(
        default_factory=lambda: [AttackStrategyConfig()]
    ) # 攻击策略列表

    def __post_init__(self) -> None:
        if self.enabled:
            assert self.per_round_fraction <= self.malicious_fraction, (
                "per_round_fraction 不能超过 malicious_fraction"
            )
            total = sum(s.fraction for s in self.strategies)
            assert abs(total - 1.0) < 1e-6, (
                f"strategies 的 fraction 之和应为 1.0, 当前为 {total}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AttackConfig':
        """从字典解析配置"""
        if not config_dict:
            return cls()
        strategies_raw = config_dict.get("strategies", [])
        strategies = [
            AttackStrategyConfig.from_dict(s) if isinstance(s, dict) else s
            for s in strategies_raw
        ]
        if not strategies:
            strategies = [AttackStrategyConfig()]
        return cls(
            enabled=config_dict.get("enabled", False),
            malicious_fraction=config_dict.get("malicious_fraction", 0.2),
            per_round_fraction=config_dict.get("per_round_fraction", 0.2),
            strategies=strategies
        )

@dataclass
class LoggerConfig:
    """配置 Logger 的参数"""
    project: str = "FL_Project"             # 项目名称
    name: str = "experiment"                # 实验名称
    log_root: str = "./logs"                # 日志根目录
    log_level: str = "INFO"                 # 日志级别
    use_wandb: bool = False                 # 是否使用 wandb
    use_tensorboard: bool = False           # 是否使用 tensorboard
    use_csv: bool = False                   # 是否使用 csv
    console_metrics: bool = False           # 是否在控制台显示指标
    save_interval: int = 10                 # 检查点保存间隔轮数

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoggerConfig":
        """从字典解析配置"""
        if not config_dict:
            return cls()
        logging_section = config_dict.get("logging")
        d = logging_section if isinstance(logging_section, dict) else config_dict
        kwargs = {}
        for key in [
            "project", "name", "log_root", "log_level",
            "use_wandb", "use_tensorboard", "use_csv", "console_metrics", "save_interval"
        ]:
            if key in d:
                kwargs[key] = d[key]
        if "log_dir" in d and "log_root" not in kwargs:
            kwargs["log_root"] = d["log_dir"]
        if "name" not in kwargs and "experiment_name" in config_dict:
            kwargs["name"] = config_dict["experiment_name"]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "project": self.project,
            "name": self.name,
            "log_root": self.log_root,
            "log_level": self.log_level,
            "use_wandb": self.use_wandb,
            "use_tensorboard": self.use_tensorboard,
            "use_csv": self.use_csv,
            "console_metrics": self.console_metrics,
        }

    def to_logging_section(self) -> Dict[str, Any]:
        """转换为 logging 节字典"""
        d = self.to_dict()
        d["save_interval"] = self.save_interval
        return d

@dataclass
class DataConfig:
    """配置数据层的参数"""
    dataset: str = "cifar10"                # 数据集名称
    root: str = "./data_source"             # 数据存储路径
    val_ratio: float = 0.1                  # 验证集比例
    partitioner: PartitionerConfig = field(default_factory=PartitionerConfig) # 划分器配置

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """从字典解析配置"""
        kwargs = {}
        for key in ['dataset', 'root', 'val_ratio']:
            if key in config_dict:
                kwargs[key] = config_dict[key]

        if 'partitioner' in config_dict:
            kwargs['partitioner'] = PartitionerConfig.from_dict(config_dict['partitioner'])

        return cls(**kwargs)

@dataclass
class GlobalConfig:
    """全局配置聚合类"""
    experiment_name: str = "fedavg_cifar10_demo" # 实验名称
    device: str = "cuda" if torch.cuda.is_available() else "cpu" # 运行设备
    seed: int = 42                          # 全局随机种子, 会覆盖所有子配置中的 seed
    data: DataConfig = field(default_factory=DataConfig) # 数据配置
    training: TrainingConfig = field(default_factory=TrainingConfig) # 训练配置
    client: ClientConfig = field(default_factory=ClientConfig) # 客户端配置
    model: Dict[str, Any] = field(default_factory=lambda: {
        "name": "resnet18",
        "params": {"num_classes": 10, "input_channels": 3}
    }) # 模型配置
    algorithm: Dict[str, Any] = field(default_factory=lambda: {
        "name": "fedavg",
        "params": {}
    }) # 算法配置
    logger_config: LoggerConfig = field(default_factory=LoggerConfig) # 日志配置
    attack: AttackConfig = field(default_factory=lambda: AttackConfig(enabled=False)) # 攻击配置

    def __post_init__(self) -> None:
        self._sync_seed()

    def _sync_seed(self) -> None:
        """将全局 seed 传播到所有子配置, 确保全局 seed 具有最高优先级"""
        self.training.seed = self.seed

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GlobalConfig':
        """从字典解析配置"""
        instance = cls()

        if 'data' in config_dict:
            instance.data = DataConfig.from_dict(config_dict['data'])
        if 'training' in config_dict:
            instance.training = TrainingConfig.from_dict(config_dict['training'])
        if 'client' in config_dict:
            instance.client = ClientConfig.from_dict(config_dict['client'])
        if 'model' in config_dict:
            instance.model.update(config_dict['model'])
        if 'algorithm' in config_dict:
            instance.algorithm.update(config_dict['algorithm'])
        if 'logging' in config_dict or any(k in config_dict for k in ['use_wandb', 'log_dir', 'log_root', 'log_level']):
            instance.logger_config = LoggerConfig.from_dict(config_dict)
        if 'attack' in config_dict:
            instance.attack = AttackConfig.from_dict(config_dict['attack'])

        for key, value in config_dict.items():
            if key in ['experiment_name', 'device', 'seed']:
                setattr(instance, key, value)
            
            if key == 'lr':
                instance.client.trainer_config.lr = value
            if key == 'epochs':
                instance.client.trainer_config.epochs = value
            if key == 'batch_size':
                instance.client.batch_size = value
                instance.client.trainer_config.batch_size = value
            if key == 'rounds':
                instance.training.rounds = value
            if key == 'dataset':
                instance.data.dataset = value
            if key == 'num_clients':
                instance.training.num_clients = value
            if key == 'experiment_name':
                instance.logger_config.name = value

        # 全局 seed 具有最高优先级, 最终覆盖所有子配置中的 seed
        instance._sync_seed()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "experiment_name": self.experiment_name,
            "device": self.device,
            "seed": self.seed,
            "data": {
                "dataset": self.data.dataset,
                "root": self.data.root,
                "num_clients": self.training.num_clients,
                "val_ratio": self.data.val_ratio,
                "partitioner": {
                    "name": self.data.partitioner.name,
                    "params": self.data.partitioner.params
                }
            },
            "training": self.training.to_dict(),
            "client": {
                "lr": self.client.trainer_config.lr,
                "momentum": self.client.trainer_config.momentum,
                "weight_decay": self.client.trainer_config.weight_decay,
                "epochs": self.client.trainer_config.epochs,
                "batch_size": self.client.batch_size,
                "num_workers": self.client.num_workers,
                "bn_calib_batches": self.client.bn_calib_batches
            },
            "model": self.model,
            "algorithm": self.algorithm,
            "logging": self.logger_config.to_logging_section(),
            "attack": {
                "enabled": self.attack.enabled,
                "malicious_fraction": self.attack.malicious_fraction,
                "per_round_fraction": self.attack.per_round_fraction,
                "strategies": [
                    {"name": s.name, "fraction": s.fraction, "params": s.params}
                    for s in self.attack.strategies
                ]
            }
        }

    @property
    def training_config(self) -> TrainingConfig:
        return self.training
