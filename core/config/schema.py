from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainerConfig:
    """配置模型训练的参数"""

    optimizer_name: str = "SGD"
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    criterion_name: str = "CrossEntropyLoss"
    epochs: int = 2
    batch_size: int = 32
    num_workers: int = 0
    grad_clip_norm: Optional[float] = 5.0
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def build_optimizer(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        kwargs = {}
        if self.optimizer_name == "SGD":
            kwargs = {"lr": self.lr, "momentum": self.momentum, "weight_decay": self.weight_decay}
        elif self.optimizer_name in {"Adam", "AdamW"}:
            kwargs = {"lr": self.lr, "weight_decay": self.weight_decay}

        kwargs.update(self.extra_params.get("optimizer_kwargs", {}))
        optimizer_class = getattr(torch.optim, self.optimizer_name)
        return optimizer_class(model.parameters(), **kwargs)

    def build_criterion(self) -> torch.nn.Module:
        kwargs = self.extra_params.get("criterion_kwargs", {})
        criterion_class = getattr(torch.nn, self.criterion_name)
        return criterion_class(**kwargs)


@dataclass
class ClientConfig:
    """配置客户端运行的基础参数"""

    bn_calib_batches: int = 5
    num_workers: int = 0
    batch_size: int = 32
    trainer_config: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ClientConfig":
        trainer_kwargs = {}
        for key in [
            "lr",
            "momentum",
            "weight_decay",
            "optimizer_name",
            "criterion_name",
            "epochs",
            "grad_clip_norm",
        ]:
            if key in config_dict:
                trainer_kwargs[key] = config_dict[key]

        for key in ["batch_size", "num_workers"]:
            if key in config_dict:
                trainer_kwargs[key] = config_dict[key]

        trainer_cfg = TrainerConfig(**trainer_kwargs)
        client_kwargs = {"trainer_config": trainer_cfg}
        for key in ["bn_calib_batches", "num_workers", "batch_size"]:
            if key in config_dict:
                client_kwargs[key] = config_dict[key]

        return cls(**client_kwargs)


@dataclass
class LRScheduleConfig:
    """学习率调度配置 (warmup + cosine)"""

    enabled: bool = True
    name: str = "warmup_cosine"
    warmup_ratio: float = 0.1
    min_lr: float = 0.0
    warmup_start_lr: float = 0.0

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LRScheduleConfig":
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

    num_clients: int = 100
    rounds: int = 100
    clients_fraction: float = 0.2
    eval_interval: int = 5
    local_eval_ratio: float = 0.2
    seed: int = 42
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        kwargs = {}
        for key in ["num_clients", "rounds", "clients_fraction", "eval_interval", "local_eval_ratio", "seed"]:
            if key in config_dict:
                kwargs[key] = config_dict[key]
        if "lr_schedule" in config_dict:
            kwargs["lr_schedule"] = LRScheduleConfig.from_dict(config_dict["lr_schedule"])
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
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

    name: str = "dirichlet"
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return self.params.get("alpha", 1.0)

    @property
    def max_retries(self) -> int:
        return self.params.get("max_retries", 100)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PartitionerConfig":
        if not config_dict:
            return cls()
        return cls(name=config_dict.get("name", "dirichlet"), params=config_dict.get("params", {}))


@dataclass
class AttackStrategyConfig:
    """单个攻击策略的配置"""

    name: str = "badnets"
    fraction: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AttackStrategyConfig":
        if not config_dict:
            return cls()
        return cls(
            name=config_dict.get("name", "badnets"),
            fraction=config_dict.get("fraction", 1.0),
            params=config_dict.get("params", {}),
        )


@dataclass
class AttackConfig:
    """攻击全局配置"""

    enabled: bool = False
    malicious_fraction: float = 0.2
    per_round_fraction: float = 0.2
    malicious_epochs: Optional[int] = None
    strategies: List[AttackStrategyConfig] = field(default_factory=lambda: [AttackStrategyConfig()])

    def __post_init__(self) -> None:
        if self.enabled:
            assert self.per_round_fraction <= self.malicious_fraction, (
                "per_round_fraction 不能超过 malicious_fraction"
            )
            total = sum(strategy.fraction for strategy in self.strategies)
            assert abs(total - 1.0) < 1e-6, f"strategies 的 fraction 之和应为 1.0, 当前为 {total}"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AttackConfig":
        if not config_dict:
            return cls()
        strategies_raw = config_dict.get("strategies", [])
        strategies = [
            AttackStrategyConfig.from_dict(strategy) if isinstance(strategy, dict) else strategy
            for strategy in strategies_raw
        ]
        if not strategies:
            strategies = [AttackStrategyConfig()]
        malicious_epochs = config_dict.get("malicious_epochs")
        if malicious_epochs is not None:
            malicious_epochs = int(malicious_epochs)
        return cls(
            enabled=config_dict.get("enabled", False),
            malicious_fraction=config_dict.get("malicious_fraction", 0.2),
            per_round_fraction=config_dict.get("per_round_fraction", 0.2),
            malicious_epochs=malicious_epochs,
            strategies=strategies,
        )


def apply_malicious_epochs_override(
    client_config: ClientConfig, malicious_epochs: Optional[int]
) -> ClientConfig:
    if malicious_epochs is None:
        return client_config
    return replace(
        client_config,
        trainer_config=replace(client_config.trainer_config, epochs=int(malicious_epochs)),
    )


@dataclass
class LoggerConfig:
    """配置 Logger 的参数"""

    project: str = "FL_Project"
    name: str = "experiment"
    log_root: str = "./logs"
    log_level: str = "INFO"
    use_wandb: bool = False
    use_tensorboard: bool = False
    use_csv: bool = False
    console_metrics: bool = False
    save_interval: int = 10

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoggerConfig":
        if not config_dict:
            return cls()
        logging_section = config_dict.get("logging")
        d = logging_section if isinstance(logging_section, dict) else config_dict
        kwargs = {}
        for key in [
            "project",
            "name",
            "log_root",
            "log_level",
            "use_wandb",
            "use_tensorboard",
            "use_csv",
            "console_metrics",
            "save_interval",
        ]:
            if key in d:
                kwargs[key] = d[key]
        if "log_dir" in d and "log_root" not in kwargs:
            kwargs["log_root"] = d["log_dir"]
        if "name" not in kwargs and "experiment_name" in config_dict:
            kwargs["name"] = config_dict["experiment_name"]
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
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
        d = self.to_dict()
        d["save_interval"] = self.save_interval
        return d


@dataclass
class DataConfig:
    """配置数据层的参数"""

    dataset: str = "cifar10"
    root: str = "./data_source"
    val_ratio: float = 0.1
    enable_proxy: bool = False
    partitioner: PartitionerConfig = field(default_factory=PartitionerConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DataConfig":
        kwargs = {}
        for key in ["dataset", "root", "val_ratio", "enable_proxy"]:
            if key in config_dict:
                kwargs[key] = config_dict[key]

        if "partitioner" in config_dict:
            kwargs["partitioner"] = PartitionerConfig.from_dict(config_dict["partitioner"])

        return cls(**kwargs)


@dataclass
class GlobalConfig:
    """全局配置聚合类"""

    experiment_name: str = "fedavg_cifar10_demo"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    client: ClientConfig = field(default_factory=ClientConfig)
    model: Dict[str, Any] = field(
        default_factory=lambda: {"name": "resnet18", "params": {"num_classes": 10, "input_channels": 3}}
    )
    algorithm: Dict[str, Any] = field(default_factory=lambda: {"name": "fedavg", "params": {}})
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)
    attack: AttackConfig = field(default_factory=lambda: AttackConfig(enabled=False))

    def __post_init__(self) -> None:
        self._sync_seed()

    def _sync_seed(self) -> None:
        self.training.seed = self.seed

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GlobalConfig":
        instance = cls()

        if "data" in config_dict:
            instance.data = DataConfig.from_dict(config_dict["data"])
        if "training" in config_dict:
            instance.training = TrainingConfig.from_dict(config_dict["training"])
        if "client" in config_dict:
            instance.client = ClientConfig.from_dict(config_dict["client"])
        if "model" in config_dict:
            instance.model.update(config_dict["model"])
        if "algorithm" in config_dict:
            instance.algorithm.update(config_dict["algorithm"])
        if "logging" in config_dict or any(
            key in config_dict for key in ["use_wandb", "log_dir", "log_root", "log_level"]
        ):
            instance.logger_config = LoggerConfig.from_dict(config_dict)
        if "attack" in config_dict:
            instance.attack = AttackConfig.from_dict(config_dict["attack"])

        for key, value in config_dict.items():
            if key in ["experiment_name", "device", "seed"]:
                setattr(instance, key, value)
            if key == "lr":
                instance.client.trainer_config.lr = value
            if key == "epochs":
                instance.client.trainer_config.epochs = value
            if key == "batch_size":
                instance.client.batch_size = value
                instance.client.trainer_config.batch_size = value
            if key == "rounds":
                instance.training.rounds = value
            if key == "dataset":
                instance.data.dataset = value
            if key == "num_clients":
                instance.training.num_clients = value
            if key == "experiment_name":
                instance.logger_config.name = value

        instance._sync_seed()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "device": self.device,
            "seed": self.seed,
            "data": {
                "dataset": self.data.dataset,
                "root": self.data.root,
                "num_clients": self.training.num_clients,
                "val_ratio": self.data.val_ratio,
                "enable_proxy": self.data.enable_proxy,
                "partitioner": {"name": self.data.partitioner.name, "params": self.data.partitioner.params},
            },
            "training": self.training.to_dict(),
            "client": {
                "lr": self.client.trainer_config.lr,
                "momentum": self.client.trainer_config.momentum,
                "weight_decay": self.client.trainer_config.weight_decay,
                "epochs": self.client.trainer_config.epochs,
                "batch_size": self.client.batch_size,
                "num_workers": self.client.num_workers,
                "bn_calib_batches": self.client.bn_calib_batches,
            },
            "model": self.model,
            "algorithm": self.algorithm,
            "logging": self.logger_config.to_logging_section(),
            "attack": {
                "enabled": self.attack.enabled,
                "malicious_fraction": self.attack.malicious_fraction,
                "per_round_fraction": self.attack.per_round_fraction,
                "malicious_epochs": self.attack.malicious_epochs,
                "strategies": [
                    {"name": strategy.name, "fraction": strategy.fraction, "params": strategy.params}
                    for strategy in self.attack.strategies
                ],
            },
        }

    @property
    def training_config(self) -> TrainingConfig:
        return self.training
