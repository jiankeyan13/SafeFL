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
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
    },
    "dba": {
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "num_blocks": 4,
    },
    "badnets_dual": {
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "delta_log_interval": 15,
        "delta_log_dir": None,
        "delta_log_enabled": True,
    },
    "batman": {
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
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
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "epoch_clean": 0,
    },
    "neurotoxin": {
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "mask_ratio": 0.10,
    },
    "pgd": {
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "dataset": None,
        "cifar10_epsilon_scale": 0.1,
        "default_epsilon_scale": 1.0,
    },
    "chameleon": {
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "adaptation_epochs": 10,
        "projection_epochs": 3,
        "fac_scale_weight": 2.0,
        "temperature": 0.07,
        "adaptation_lr": None,
        "projection_lr": None,
    },
    "lp": {
        "target_label": 5,
        "poison_ratio": 0.5,
        "patch_size": 5,
        "patch_value": 1.0,
        "patch_location": "bottom_right",
        "top_k": 6,
        "min_ndim": 3,
        "val_ratio": 0.2,
        "benign_acc_threshold": 0.8,
        "lsa_max_epoch_mult": 3,
        "lsa_malicious_epoch_mult": 1,
        "lsa_interval": 10,
        "lambda_scale": 1.0,
        "log_selected_layers": True,
        "layer_score_dir": None,
    },
}


def build_attack(strategy: AttackStrategyConfig):
    """根据策略配置实例化攻击对象。"""
    defaults = _DEFAULT_PARAMS.get(strategy.name, {})
    params = {**defaults, **strategy.params}
    return ATTACK_REGISTRY.build(strategy.name, **params)
