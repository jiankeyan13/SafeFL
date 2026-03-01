import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

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

    # 日志配置
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "use_wandb": False,
        "log_dir": "./logs",
        "save_interval": 10
    })

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
        if 'logging' in config_dict:
            instance.logging.update(config_dict['logging'])

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
            "logging": self.logging
        }

    @property
    def training_config(self) -> TrainingConfig:
        return self.training
