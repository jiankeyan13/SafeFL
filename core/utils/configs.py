import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

@dataclass
class TrainerConfig:
    """
    配置模型训练的参数 (取代硬编码和魔法数字)。
    提供默认值，避免“配置地狱”。
    """
    # 优化器配置
    optimizer_name: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    
    # 损失函数配置
    criterion_name: str = "CrossEntropyLoss"
    
    # 训练流程控制
    epochs: int = 1
    batch_size: int = 32
    num_workers: int = 0
    
    # 梯度裁剪 (为 None 则不裁剪)
    grad_clip_norm: Optional[float] = 5.0
    
    # 其他可能的参数字典，作为扩展口
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        """根据配置动态实例化优化器"""
        kwargs = {}
        if self.optimizer_name == 'SGD':
            kwargs = {'lr': self.lr, 'momentum': self.momentum, 'weight_decay': self.weight_decay}
        elif self.optimizer_name == 'Adam' or self.optimizer_name == 'AdamW':
            kwargs = {'lr': self.lr, 'weight_decay': self.weight_decay}
            
        # 允许扩展参数覆盖
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
    """
    配置客户端运行的基础参数。
    """
    bn_calib_batches: int = 5
    num_workers: int = 0
    batch_size: int = 32

    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClientConfig':
        """从字典解析配置，方便向下兼容旧的 Dict 传参"""
        # 提取 trainer 相关的配置
        trainer_kwargs = {}
        for key in ['lr', 'momentum', 'weight_decay', 'optimizer_name', 'criterion_name', 'epochs', 'grad_clip_norm']:
            if key in config_dict:
                trainer_kwargs[key] = config_dict[key]

        # trainer 中也有 batch_size 和 num_workers
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
class TrainingConfig:
    """
    配置联邦学习训练流程的全局参数 (取代 Runner 中的硬编码)。
    """
    rounds: int = 100
    clients_fraction: float = 0.2
    eval_interval: int = 5
    local_eval_ratio: float = 0.2
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """从字典解析配置"""
        kwargs = {}
        for key in ['rounds', 'clients_fraction', 'eval_interval', 'local_eval_ratio', 'seed']:
            if key in config_dict:
                kwargs[key] = config_dict[key]
        return cls(**kwargs)


@dataclass
class PartitionerConfig:
    """
    配置数据划分器的参数。
    """
    name: str = "dirichlet"  # "iid", "dirichlet" 等
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        """Dirichlet 划分的 alpha 参数 (默认 1.0)"""
        return self.params.get("alpha", 1.0)

    @property
    def max_retries(self) -> int:
        """Dirichlet 划分的最大重试次数 (默认 100)"""
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
    """
    单个攻击策略的配置。
    """
    name: str = "badnets"
    fraction: float = 1.0  # 占所有恶意客户端的比例
    params: Dict[str, Any] = field(default_factory=dict)

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
    """
    攻击全局配置。
    """
    enabled: bool = False
    malicious_fraction: float = 0.2  # 恶意客户端占总客户端的比例
    per_round_fraction: float = 0.2   # 每轮选中者中恶意客户端的比例
    strategies: List[AttackStrategyConfig] = field(
        default_factory=lambda: [AttackStrategyConfig()]
    )

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
    """
    配置 Logger 的参数 (取代 Logger 构造时的硬编码默认值).
    与 core.utils.logger.Logger 的 from_config 参数一一对应.
    """
    project: str = "FL_Project"
    name: str = "experiment"
    log_root: str = "./logs"
    log_level: str = "INFO"
    use_wandb: bool = False
    use_tensorboard: bool = False
    use_csv: bool = False
    console_metrics: bool = True
    save_interval: int = 10  # 检查点保存间隔 (轮次)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoggerConfig":
        """从字典解析配置, 支持 logging 嵌套或扁平键."""
        if not config_dict:
            return cls()
        # 优先从 logging 嵌套中解析
        logging_section = config_dict.get("logging")
        d = logging_section if isinstance(logging_section, dict) else config_dict
        kwargs = {}
        for key in [
            "project", "name", "log_root", "log_level",
            "use_wandb", "use_tensorboard", "use_csv", "console_metrics", "save_interval"
        ]:
            if key in d:
                kwargs[key] = d[key]
        # 兼容 log_dir 作为 log_root 的别名
        if "log_dir" in d and "log_root" not in kwargs:
            kwargs["log_root"] = d["log_dir"]
        # 兼容顶层 experiment_name 作为 name
        if "name" not in kwargs and "experiment_name" in config_dict:
            kwargs["name"] = config_dict["experiment_name"]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """转换为 Logger.from_config 可用的扁平字典."""
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
        """转换为 config 中 logging 节的完整字典 (含 save_interval)."""
        d = self.to_dict()
        d["save_interval"] = self.save_interval
        return d


@dataclass
class DataConfig:
    """
    配置数据层的参数 (取代 Runner 中的硬编码)。
    """
    dataset: str = "cifar10"
    root: str = "./data_source"
    num_clients: int = 20
    val_ratio: float = 0.1

    partitioner: PartitionerConfig = field(default_factory=PartitionerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """从 YAML 的 data: 节解析配置"""
        kwargs = {}
        for key in ['dataset', 'root', 'num_clients', 'val_ratio']:
            if key in config_dict:
                kwargs[key] = config_dict[key]

        # 解析 partitioner 配置
        if 'partitioner' in config_dict:
            kwargs['partitioner'] = PartitionerConfig.from_dict(config_dict['partitioner'])

        return cls(**kwargs)


@dataclass
class GlobalConfig:
    """
    全局配置聚合类。
    支持从扁平化字典或嵌套字典中解析配置，并提供完整的默认值。
    """
    experiment_name: str = "fedavg_cifar10_demo"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    
    # 模型配置 (暂不使用 dataclass，保持灵活性)
    model: Dict[str, Any] = field(default_factory=lambda: {
        "name": "resnet18",
        "params": {"num_classes": 10, "input_channels": 3}
    })
    
    # 算法配置
    algorithm: Dict[str, Any] = field(default_factory=lambda: {
        "name": "fedavg",
        "params": {}
    })

    # 日志配置 (LoggerConfig 取代原 logging 字典)
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)

    # 攻击配置
    attack: AttackConfig = field(default_factory=lambda: AttackConfig(enabled=False))

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'GlobalConfig':
        """
        从字典解析配置。支持扁平化覆盖。
        例如：GlobalConfig.from_dict({"lr": 0.001}) 会修改 client.trainer_config.lr
        """
        # 1. 先创建默认实例
        instance = cls()

        # 2. 处理嵌套字典 (向下兼容)
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

        # 3. 处理扁平化键值对 (直接覆盖深层参数)
        # 这种方式允许用户只写 {"lr": 0.001}
        for key, value in config_dict.items():
            # 基础属性
            if key in ['experiment_name', 'device', 'seed']:
                setattr(instance, key, value)
            
            # 快捷映射 (可以根据需要增加更多)
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
                instance.data.num_clients = value
            if key == 'experiment_name':
                instance.logger_config.name = value

        return instance

    def to_dict(self) -> Dict[str, Any]:
        """转换为 Runner 需要的原始字典格式 (向下兼容)"""
        return {
            "experiment_name": self.experiment_name,
            "device": self.device,
            "seed": self.seed,
            "data": {
                "dataset": self.data.dataset,
                "root": self.data.root,
                "num_clients": self.data.num_clients,
                "val_ratio": self.data.val_ratio,
                "partitioner": {
                    "name": self.data.partitioner.name,
                    "params": self.data.partitioner.params
                }
            },
            "training": {
                "rounds": self.training.rounds,
                "clients_fraction": self.training.clients_fraction,
                "eval_interval": self.training_config.eval_interval, # 修正：TrainingConfig 内部属性
                "local_eval_ratio": self.training.local_eval_ratio,
                "seed": self.seed
            },
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
