from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import torch

class BaseScreener(ABC):
    """
    防御/筛选器基类 (Sanitizer/Screener).
    职责：对客户端更新进行筛选/降权，返回筛选分数和上下文信息。
    """
    
    def __init__(self, **kwargs):
        pass

    def screen(self, 
               client_deltas: List[Dict[str, torch.Tensor]], 
               num_samples: List[float],
               global_model: torch.nn.Module = None,
               context: Dict[str, Any] = None) -> Tuple[List[float], Dict[str, Any]]:
        """
        核心筛选逻辑。
        
        Args:
            client_deltas: 客户端模型差值列表
            num_samples: 客户端样本数列表
            global_model: 全局模型
            context: 上下文信息（可由上游传入或修改）
                            
        Returns:
            (screen_scores, context): 
                - screen_scores: 每个客户端的筛选分数 (0-1)
                - context: 更新后的上下文，传递给后续阶段
        """
        return [1.0] * len(client_deltas), context or {}

    def __call__(self, client_deltas, num_samples, global_model=None, context=None):
        return self.screen(client_deltas, num_samples, global_model, context)