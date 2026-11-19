"""Round-level warmup + cosine learning-rate schedule for SafeFL v2."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Mapping, Union

if TYPE_CHECKING:
    from core.config import LRScheduleConfig


def get_lr_warmup_cosine(
    base_lr: float,
    round_idx: int,
    total_rounds: int,
    warmup_ratio: float = 0.1,
    min_lr: float = 0.0,
    warmup_start_lr: float = 0.0,
) -> float:
    """Compute round-level learning rate with linear warmup + cosine decay."""
    warmup_rounds = max(0, min(total_rounds, int(total_rounds * warmup_ratio)))

    if warmup_rounds <= 0:
        decay_steps = max(1, total_rounds - 1)
        t = min(1.0, round_idx / decay_steps)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    if round_idx < warmup_rounds:
        return warmup_start_lr + (base_lr - warmup_start_lr) * (round_idx + 1) / warmup_rounds

    decay_steps = max(1, total_rounds - warmup_rounds - 1)
    t = min(1.0, (round_idx - warmup_rounds) / decay_steps)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def get_lr_from_schedule(
    schedule_cfg: Union["LRScheduleConfig", Mapping[str, Any]],
    base_lr: float,
    round_idx: int,
    total_rounds: int,
) -> float:
    """Resolve current round's learning rate from ``LRScheduleConfig`` or dict."""
    if isinstance(schedule_cfg, Mapping):
        enabled = bool(schedule_cfg.get("enabled", False))
        name = str(schedule_cfg.get("name", "warmup_cosine"))
        warmup_ratio = float(schedule_cfg.get("warmup_ratio", 0.1))
        min_lr = float(schedule_cfg.get("min_lr", 0.0))
        warmup_start_lr = float(schedule_cfg.get("warmup_start_lr", 0.0))
    else:
        enabled = bool(schedule_cfg.enabled)
        name = str(schedule_cfg.name)
        warmup_ratio = float(schedule_cfg.warmup_ratio)
        min_lr = float(schedule_cfg.min_lr)
        warmup_start_lr = float(schedule_cfg.warmup_start_lr)

    if not enabled or name != "warmup_cosine":
        return float(base_lr)

    return get_lr_warmup_cosine(
        base_lr=float(base_lr),
        round_idx=int(round_idx),
        total_rounds=int(total_rounds),
        warmup_ratio=warmup_ratio,
        min_lr=min_lr,
        warmup_start_lr=warmup_start_lr,
    )
