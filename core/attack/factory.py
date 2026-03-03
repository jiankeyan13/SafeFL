"""
攻击策略工厂。根据 AttackStrategyConfig 实例化对应的攻击对象。
"""
from __future__ import annotations

from typing import Dict, Any

from core.utils.configs import AttackStrategyConfig
from core.utils.registry import ATTACK_REGISTRY

# 各攻击类型的默认 params, 用户未配置时使用
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "badnets": {
        "target_label": 0,
        "poison_ratio": 0.1,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
    }
}


def build_attack(strategy: AttackStrategyConfig):
    """根据策略配置实例化攻击对象。"""
    defaults = _DEFAULT_PARAMS.get(strategy.name, {})
    params = {**defaults, **strategy.params}
    return ATTACK_REGISTRY.build(strategy.name, **params)
