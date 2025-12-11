import numpy as np
from abc import ABC, abstractmethod

class LRScheduler(ABC):
    """
    学习率调度器基类。
    职责：根据当前的 round_idx 计算这一轮的 global_lr。
    特点：纯数学计算，不依赖 PyTorch Optimizer。
    """
    def __init__(self, base_lr: float):
        self.base_lr = base_lr

    @abstractmethod
    def get_lr(self, round_idx: int) -> float:
        pass

class ConstantLR(LRScheduler):
    """固定学习率 (Baseline)"""
    def get_lr(self, round_idx: int) -> float:
        return self.base_lr

class StepLR(LRScheduler):
    """
    阶梯衰减。
    在指定的 milestones 轮次，将 lr 乘以 gamma。
    """
    def __init__(self, base_lr: float, milestones: list, gamma: float = 0.1):
        super().__init__(base_lr)
        self.milestones = sorted(milestones)
        self.gamma = gamma

    def get_lr(self, round_idx: int) -> float:
        # 计算当前轮次过了几个 milestone
        decay_count = sum(1 for m in self.milestones if round_idx >= m)
        return self.base_lr * (self.gamma ** decay_count)

class CosineAnnealingLR(LRScheduler):
    """
    余弦退火 (最常用)。
    从 base_lr 平滑衰减到 min_lr。
    """
    def __init__(self, base_lr: float, total_rounds: int, min_lr: float = 0.0, warmup_rounds: int = 0):
        super().__init__(base_lr)
        self.total_rounds = total_rounds
        self.min_lr = min_lr
        self.warmup_rounds = warmup_rounds

    def get_lr(self, round_idx: int) -> float:
        # 1. Warmup 阶段
        if round_idx < self.warmup_rounds:
            # 线性增长: 0 -> base_lr
            # 避免除以0
            if self.warmup_rounds == 0: return self.base_lr
            alpha = (round_idx + 1) / self.warmup_rounds
            return self.base_lr * alpha

        # 2. Cosine 阶段
        # 调整 round_idx，使其相对于 warmup 结束后的进度
        curr_round = round_idx - self.warmup_rounds
        total_round = self.total_rounds - self.warmup_rounds
        
        # 防止越界
        if curr_round >= total_round:
            return self.min_lr

        # Cosine 公式: 0.5 * (1 + cos(pi * t / T))
        coeff = 0.5 * (1 + np.cos(np.pi * curr_round / total_round))
        
        return self.min_lr + (self.base_lr - self.min_lr) * coeff

# --- 工厂函数 (方便在 main.py 里调用) ---

def build_scheduler(config: dict) -> LRScheduler:
    name = config.get('scheduler', 'constant')
    lr = config.get('lr', 0.01)
    rounds = config.get('rounds', 100)
    
    if name == 'constant':
        return ConstantLR(lr)
    
    elif name == 'step':
        return StepLR(lr, 
                      milestones=config.get('milestones', [50, 75]), 
                      gamma=config.get('gamma', 0.1))
    
    elif name == 'cosine':
        return CosineAnnealingLR(lr, 
                                 total_rounds=rounds, 
                                 min_lr=config.get('min_lr', 0.0),
                                 warmup_rounds=config.get('warmup_rounds', 0))
    else:
        raise ValueError(f"Unknown scheduler: {name}")