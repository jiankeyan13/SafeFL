"""
攻击策略工厂。根据 AttackStrategyConfig 实例化对应的攻击对象。
"""
from __future__ import annotations

from typing import Dict, Any

from core.config import AttackStrategyConfig
from core.utils.registry import ATTACK_REGISTRY

# 各攻击类型的默认 params, 用户未配置时使用
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "badnets": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
    },
    "badnets_dual": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "delta_log_interval": 15,
        "delta_log_dir": None,
        "delta_log_enabled": True,
    },
    "batman": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "rank": 4,
        "lamda_reg": 20.0,
        "beta_reg": 15.0,
        "num_selected_layers": 9,
        "selected_layer_ratio": None,
        "stats_top_layers": None,
        "stats_layer_ratio": None,
        "layer_selection": "global_deviation",
        "layer_selection_seed": None,
        "log_selected_layers": True,
        "layer_selection_log_path": None,
    },
    "lga": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
    },
    "neurotoxin": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "mask_ratio": 0.05,
    },
    "pgd": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "epsilon": 5.0,
    },
    "chameleon": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "adaptation_epochs": 10,
        "projection_epochs": 5,
        "fac_scale_weight": 2.0,
        "temperature": 0.07,
        "adaptation_lr": None,
        "projection_lr": None,
    },
    "lp": {
        "target_label": 0,
        "poison_ratio": 0.5,
        "patch_size": 3,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "tau": 0.8,
        "val_ratio": 0.25,
        "log_selected_layers": True,
        "layer_selection_log_path": None,
    },
}


def build_attack(strategy: AttackStrategyConfig):
    """根据策略配置实例化攻击对象。"""
    defaults = _DEFAULT_PARAMS.get(strategy.name, {})
    params = {**defaults, **strategy.params}
    return ATTACK_REGISTRY.build(strategy.name, **params)
