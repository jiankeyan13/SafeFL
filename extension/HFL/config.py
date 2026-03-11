"""HFL 异构联邦学习扩展的默认配置."""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.utils.configs import GlobalConfig


@dataclass
class AttackerCapabilityConfig:
    """Malicious client capability allocation config."""

    enabled: bool = False
    sample: str = "uniform"
    p_list: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "sample": self.sample,
            "p_list": list(self.p_list),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AttackerCapabilityConfig":
        if not config_dict:
            return cls()
        return cls(
            enabled=bool(config_dict.get("enabled", False)),
            sample=config_dict.get("sample", "uniform"),
            p_list=list(config_dict.get("p_list", [])),
        )


@dataclass
class HeteroConfig:
    """异构联邦学习的能力分配配置."""

    sample: str = "uniform"
    """采样模式: 'uniform' 均匀配额, 'beta' Beta 分布采样
    """
    p_list: List[float] = field(default_factory=lambda: [0.125, 0.25, 0.5, 1.0])
    """能力值列表. uniform 时从该列表均匀分配; beta 时取 [low, high] 作为区间
    """
    alpha: float = 3.0
    """Beta 采样参数 (仅 sample='beta' 时生效)
    """
    beta: float = 3.0
    """Beta 采样参数 (仅 sample='beta' 时生效)
    """

    attacker: AttackerCapabilityConfig = field(default_factory=AttackerCapabilityConfig)
    """攻击者能力配置. 启用后, 恶意客户端将按给定 p_list 均匀分配能力值.
    """

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample": self.sample,
            "p_list": self.p_list,
            "alpha": self.alpha,
            "beta": self.beta,
            "attacker": self.attacker.to_dict(),
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "HeteroConfig":
        if not config_dict:
            return cls()
        return cls(
            sample=config_dict.get("sample", "uniform"),
            p_list=config_dict.get("p_list", [0.25, 0.5, 1.0]),
            alpha=float(config_dict.get("alpha", 3.0)),
            beta=float(config_dict.get("beta", 3.0)),
            attacker=AttackerCapabilityConfig.from_dict(config_dict.get("attacker", {})),
        )


@dataclass
class HFLConfig(GlobalConfig):
    """HFL 异构联邦学习的默认配置, 继承 GlobalConfig 并补充异构相关字段."""

    hetero: HeteroConfig = field(default_factory=HeteroConfig)
    """异构能力分配配置
    """

    def __post_init__(self) -> None:
        # 确保 hetero 存在后再调用父类
        if not hasattr(self, "hetero") or self.hetero is None:
            object.__setattr__(self, "hetero", HeteroConfig())
        super().__post_init__()

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]] = None) -> "HFLConfig":
        config_dict = copy.deepcopy(config_dict or {})
        # 注入 HFL 默认 model/algorithm (用户未指定时使用)
        defaults = {
            "model": {"name": "hetero_PreResNet18", "params": {"num_classes": 10, "input_channels": 3}},
            "algorithm": {"name": "fedrolex", "params": {"aggregator": {"name": "sub_avg"}}},
        }
        for k, v in defaults.items():
            if k not in config_dict:
                config_dict[k] = v
            elif isinstance(v, dict) and isinstance(config_dict.get(k), dict):
                config_dict[k] = {**v, **config_dict[k]}

        instance = super().from_dict(config_dict)

        if "hetero" in config_dict:
            instance.hetero = HeteroConfig.from_dict(config_dict["hetero"])
        else:
            instance.hetero = HeteroConfig()

        return instance

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["hetero"] = self.hetero.to_dict()
        return d


def get_default_hfl_config(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    获取 HFL 默认配置字典, 供 HeteroRunner 直接使用.

    Args:
        overrides: 用户覆盖的配置项, 会合并到默认配置中

    Returns:
        完整的 HFL 配置字典
    """
    default = HFLConfig.from_dict({})
    config = default.to_dict()
    overrides = overrides or {}

    # 递归合并 overrides
    def merge(base: Dict, override: Dict) -> None:
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                merge(base[k], v)
            else:
                base[k] = v

    merge(config, overrides)
    return config
