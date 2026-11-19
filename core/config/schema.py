"""Configuration schema for SafeFL v2."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch

from core.types import TrainConfig


def parse_train_config(raw: Optional[Dict[str, Any]]) -> TrainConfig:
    """Parse ``client:`` YAML mapping directly into a flat ``TrainConfig``."""
    if not raw:
        return TrainConfig()

    known = {
        "epochs",
        "batch_size",
        "lr",
        "optimizer",
        "optimizer_name",
        "momentum",
        "weight_decay",
        "criterion",
        "criterion_name",
        "grad_clip_norm",
        "num_workers",  # ignored in v2 vectorized loader, excluded from extra
        "extra",
    }
    extra = dict(raw.get("extra") or {})
    for k, v in raw.items():
        if k not in known:
            extra[k] = v

    return TrainConfig(
        epochs=int(raw.get("epochs", 2)),
        batch_size=int(raw.get("batch_size", 64)),
        lr=float(raw.get("lr", 0.1)),
        optimizer=str(raw.get("optimizer") or raw.get("optimizer_name") or "SGD"),
        momentum=float(raw.get("momentum", 0.9)),
        weight_decay=float(raw.get("weight_decay", 5e-4)),
        criterion=str(raw.get("criterion") or raw.get("criterion_name") or "CrossEntropyLoss"),
        grad_clip_norm=(
            None if raw.get("grad_clip_norm") is None else float(raw["grad_clip_norm"])
        ),
        extra=extra,
    )


@dataclass
class ComponentConfig:
    """Generic ``{name, params}`` configuration for models, defenses, or partitioners."""

    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]], default_name: str) -> "ComponentConfig":
        if not raw:
            return cls(name=default_name)
        return cls(
            name=str(raw.get("name", default_name)),
            params=dict(raw.get("params") or {}),
        )


PartitionerConfig = ComponentConfig


@dataclass
class LRScheduleConfig:
    """Round-level learning-rate schedule configuration."""

    enabled: bool = False
    name: str = "warmup_cosine"
    warmup_ratio: float = 0.1
    min_lr: float = 0.0
    warmup_start_lr: float = 0.0

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "LRScheduleConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            name=str(raw.get("name", "warmup_cosine")),
            warmup_ratio=float(raw.get("warmup_ratio", 0.1)),
            min_lr=float(raw.get("min_lr", 0.0)),
            warmup_start_lr=float(raw.get("warmup_start_lr", 0.0)),
        )


@dataclass
class TrainingConfig:
    """Global federated training loop parameters."""

    num_clients: int = 100
    rounds: int = 100
    clients_fraction: float = 0.1
    lr_schedule: LRScheduleConfig = field(default_factory=LRScheduleConfig)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "TrainingConfig":
        if not raw:
            return cls()
        return cls(
            num_clients=int(raw.get("num_clients", 100)),
            rounds=int(raw.get("rounds", 100)),
            clients_fraction=float(raw.get("clients_fraction", 0.1)),
            lr_schedule=LRScheduleConfig.from_dict(raw.get("lr_schedule")),
        )


@dataclass
class DataConfig:
    """Data hub & partitioning configuration."""

    dataset: str = "cifar10"
    root: str = "./data_source"
    enable_proxy: bool = False
    proxy_per_class: int = 10
    use_augmentation: bool = False
    preload_gpu: bool = False
    partitioner: PartitionerConfig = field(
        default_factory=lambda: ComponentConfig(name="q_non_iid")
    )

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "DataConfig":
        if not raw:
            return cls()
        return cls(
            dataset=str(raw.get("dataset", "cifar10")),
            root=str(raw.get("root", "./data_source")),
            enable_proxy=bool(raw.get("enable_proxy", False)),
            proxy_per_class=int(raw.get("proxy_per_class", 10)),
            use_augmentation=bool(raw.get("use_augmentation", False)),
            preload_gpu=bool(raw.get("preload_gpu", False)),
            partitioner=ComponentConfig.from_dict(raw.get("partitioner"), "q_non_iid"),
        )


@dataclass
class AttackStrategyConfig:
    """Single attacker strategy allocation."""

    name: str = "badnets"
    fraction: float = 1.0
    poisoner: Optional[Dict[str, Any]] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "AttackStrategyConfig":
        if not raw:
            return cls()
        return cls(
            name=str(raw.get("name", "badnets")),
            fraction=float(raw.get("fraction", 1.0)),
            poisoner=dict(raw["poisoner"]) if isinstance(raw.get("poisoner"), dict) else None,
            params=dict(raw.get("params") or {}),
        )


@dataclass
class AttackConfig:
    """Global attack & ``AttackCoordinator`` scheduling configuration."""

    enabled: bool = False
    malicious_fraction: float = 0.1
    per_round_fraction: float = 0.1
    sampling_mode: str = "fixed_quota"  # "fixed_quota" | "random"
    start_round: int = 0
    end_round: Optional[int] = None
    attack_interval: int = 1
    malicious_epochs: Optional[int] = None
    strategies: List[AttackStrategyConfig] = field(
        default_factory=lambda: [AttackStrategyConfig()]
    )

    def __post_init__(self) -> None:
        if self.sampling_mode not in {"fixed_quota", "random"}:
            raise ValueError(
                f"attack.sampling_mode must be 'fixed_quota' or 'random', got {self.sampling_mode!r}"
            )
        if self.attack_interval <= 0:
            raise ValueError(f"attack.attack_interval must be >= 1, got {self.attack_interval}")
        if self.enabled:
            if self.sampling_mode == "fixed_quota" and self.per_round_fraction > self.malicious_fraction:
                raise ValueError("per_round_fraction cannot exceed malicious_fraction")
            total = sum(s.fraction for s in self.strategies)
            if abs(total - 1.0) >= 1e-6:
                raise ValueError(f"attack.strategies fractions must sum to 1.0, got {total}")

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "AttackConfig":
        if not raw:
            return cls()
        strategies_raw = raw.get("strategies") or []
        strategies = [
            AttackStrategyConfig.from_dict(s) if isinstance(s, dict) else s
            for s in strategies_raw
        ] or [AttackStrategyConfig()]

        mal_ep = raw.get("malicious_epochs")
        end_rd = raw.get("end_round")
        return cls(
            enabled=bool(raw.get("enabled", False)),
            malicious_fraction=float(raw.get("malicious_fraction", 0.1)),
            per_round_fraction=float(raw.get("per_round_fraction", 0.1)),
            sampling_mode=str(raw.get("sampling_mode", "fixed_quota")),
            start_round=int(raw.get("start_round", 0)),
            end_round=None if end_rd is None else int(end_rd),
            attack_interval=int(raw.get("attack_interval", 1)),
            malicious_epochs=None if mal_ep is None else int(mal_ep),
            strategies=strategies,
        )


@dataclass
class LoggerConfig:
    """Experiment logging configuration."""

    project: str = "SafeFL"
    name: str = "experiment"
    log_root: str = "./logs"
    log_level: str = "INFO"
    use_wandb: bool = False
    use_tensorboard: bool = False
    use_csv: bool = False
    console_metrics: bool = False
    save_interval: int = 10

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]], fallback_name: str = "experiment") -> "LoggerConfig":
        if not raw:
            return cls(name=fallback_name)
        return cls(
            project=str(raw.get("project", "SafeFL")),
            name=str(raw.get("name", fallback_name)),
            log_root=str(raw.get("log_root", raw.get("log_dir", "./logs"))),
            log_level=str(raw.get("log_level", "INFO")),
            use_wandb=bool(raw.get("use_wandb", False)),
            use_tensorboard=bool(raw.get("use_tensorboard", False)),
            use_csv=bool(raw.get("use_csv", False)),
            console_metrics=bool(raw.get("console_metrics", False)),
            save_interval=int(raw.get("save_interval", 10)),
        )


@dataclass
class ParallelConfig:
    """Execution backend configuration (``local`` in-process or ``ray`` actor pool)."""

    backend: str = "ray"
    gpu_ids: List[int] = field(default_factory=list)
    actors_per_gpu: int = 10

    def __post_init__(self) -> None:
        b = str(self.backend).lower().strip()
        if b == "sequential":
            b = "local"
        if b not in {"local", "ray"}:
            raise ValueError(f"parallel.backend must be 'local' or 'ray', got {self.backend!r}")
        self.backend = b
        self.actors_per_gpu = max(1, int(self.actors_per_gpu))
        self.gpu_ids = [int(x) for x in self.gpu_ids]

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "ParallelConfig":
        if not raw:
            return cls()
        return cls(
            backend=str(raw.get("backend", "ray")),
            gpu_ids=list(raw.get("gpu_ids") or []),
            actors_per_gpu=int(raw.get("actors_per_gpu", 10)),
        )


@dataclass
class GlobalConfig:
    """Root experiment configuration for SafeFL v2."""

    experiment_name: str = "fedavg_cifar10"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    client: TrainConfig = field(default_factory=TrainConfig)
    model: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            name="resnet18", params={"num_classes": 10, "input_channels": 3}
        )
    )
    defense: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(name="fedavg")
    )
    attack: AttackConfig = field(default_factory=AttackConfig)
    logger_config: LoggerConfig = field(default_factory=LoggerConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GlobalConfig":
        exp_name = str(raw.get("experiment_name", "fedavg_cifar10"))
        defense_raw = raw.get("defense") or raw.get("algorithm")
        logging_raw = raw.get("logging") if isinstance(raw.get("logging"), dict) else None

        data_cfg = DataConfig.from_dict(raw.get("data"))
        if "dataset" in raw and "data" not in raw:
            data_cfg.dataset = str(raw["dataset"])

        return cls(
            experiment_name=exp_name,
            device=str(raw.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
            seed=int(raw.get("seed", 42)),
            data=data_cfg,
            training=TrainingConfig.from_dict(raw.get("training")),
            client=parse_train_config(raw.get("client")),
            model=ComponentConfig.from_dict(raw.get("model"), "resnet18"),
            defense=ComponentConfig.from_dict(defense_raw, "fedavg"),
            attack=AttackConfig.from_dict(raw.get("attack")),
            logger_config=LoggerConfig.from_dict(logging_raw, fallback_name=exp_name),
            parallel=ParallelConfig.from_dict(raw.get("parallel")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
