from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch

class BaseScreener(ABC):
    """
    防御/筛选器基类 (Sanitizer/Screener).
    职责：对客户端更新进行筛选/降权，返回模型差值和加权分数。
    """
    
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: 具体的防御参数 (如 krum 的 k, trim 的 ratio)
        """
        pass

    def screen(self, 
               client_deltas: List[Dict[str, torch.Tensor]], 
               num_samples: List[float],
               global_model: torch.nn.Module = None) -> List[float]:
        """
        核心筛选逻辑。
        
        Args:
            client_deltas: 客户端模型差值列表
            num_samples: 客户端样本数列表
            global_model: 全局模型（某些防御如 FLTrust 需要）
                            
        Returns:
            screen_scores: 每个客户端的筛选分数 (0-1)
                - 0 表示完全剔除
                - 1 表示完全保留
                - 0-1 之间表示软降权
        
        Note:
            默认实现不做任何筛选，返回全 1.0 的分数。
            子类可以覆盖此方法实现具体的防御策略。
            重要: 保持 deltas 列表不变，仅通过分数进行控制。
        """
        return [1.0] * len(client_deltas)

    def __call__(self, client_deltas, num_samples, global_model=None):
        """使得 Screener 可以像函数一样被调用"""
        return self.screen(client_deltas, num_samples, global_model)