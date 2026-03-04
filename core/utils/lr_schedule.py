"""学习率调度工具. 按全局通信轮(round)计算 warmup + cosine 衰减的 lr."""

import math
from typing import Dict, Any


def get_lr_warmup_cosine(
    base_lr: float,
    round_idx: int,
    total_rounds: int,
    warmup_ratio: float = 0.1,
    min_lr: float = 0.0,
    warmup_start_lr: float = 0.0,
) -> float:
    """
    按 round 计算 warmup 后余弦衰减的学习率.

    Args:
        base_lr: 峰值学习率 (warmup 结束后、cosine 起始值).
        round_idx: 当前轮索引 (0-based).
        total_rounds: 总轮数.
        warmup_ratio: warmup 占 total_rounds 的比例, 用于计算 warmup_rounds.
        min_lr: cosine 衰减到的最小学习率.
        warmup_start_lr: warmup 起始学习率.

    Returns:
        当轮学习率.
    """
    warmup_rounds = max(0, min(total_rounds, int(total_rounds * warmup_ratio)))

    if warmup_rounds <= 0:
        # 无 warmup, 直接 cosine
        decay_steps = max(1, total_rounds - 1)
        t = min(1.0, round_idx / decay_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))

    if round_idx < warmup_rounds:
        # warmup 阶段: 线性上升到 base_lr
        return warmup_start_lr + (base_lr - warmup_start_lr) * (round_idx + 1) / warmup_rounds

    # cosine 阶段
    decay_steps = max(1, total_rounds - warmup_rounds - 1)
    t = min(1.0, (round_idx - warmup_rounds) / decay_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * t))


def get_lr_from_schedule(
    schedule_cfg: Dict[str, Any],
    base_lr: float,
    round_idx: int,
    total_rounds: int,
) -> float:
    """
    根据 schedule 配置计算当轮 lr. 若未启用或 name 不支持, 返回 base_lr.

    Args:
        schedule_cfg: 含 enabled, name, warmup_ratio, min_lr, warmup_start_lr 等.
        base_lr: 基准学习率.
        round_idx: 当前轮索引.
        total_rounds: 总轮数.

    Returns:
        当轮学习率.
    """
    if not schedule_cfg.get("enabled", False):
        return base_lr

    name = schedule_cfg.get("name", "warmup_cosine")
    if name != "warmup_cosine":
        return base_lr

    return get_lr_warmup_cosine(
        base_lr=base_lr,
        round_idx=round_idx,
        total_rounds=total_rounds,
        warmup_ratio=schedule_cfg.get("warmup_ratio", 0.1),
        min_lr=schedule_cfg.get("min_lr", 0.0),
        warmup_start_lr=schedule_cfg.get("warmup_start_lr", 0.0),
    )
