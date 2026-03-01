import torch
import numpy as np
from typing import List, Dict, Any, Tuple

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener

@SCREENER_REGISTRY.register("krum")
class KrumScreener(BaseScreener):
    def __init__(self, f: int = 0, m: int = 1, **kwargs):
        """
        Args:
            f: 假设的攻击者数量 (Byzantine Tolerance)。
            m: Multi-Krum 最终保留的客户端数量。如果 m=1，就是标准 Krum。
        """
        super().__init__()
        self.f = f
        self.m = m

    def screen(self, 
               client_deltas: List[Dict[str, torch.Tensor]], 
               num_samples: List[float],
               global_model: torch.nn.Module = None,
               context: Dict[str, Any] = None) -> Tuple[List[float], Dict[str, Any]]:
        """
        执行 Multi-Krum 筛选，返回筛选分数。
        只使用可学习参数计算距离，忽略 BN 统计量。
        """
        context = context or {}
        n = len(client_deltas)
        
        if n <= 2 * self.f + 2:
            return [1.0] * n, context

        # 获取可学习参数名称
        if global_model is not None:
            learnable_params = set(name for name, p in global_model.named_parameters() if p.requires_grad)
        else:
            learnable_params = set(client_deltas[0].keys())

        # 展平可学习参数
        vectors = []
        for delta_dict in client_deltas:
            flat_params = [tensor.view(-1) for name, tensor in delta_dict.items() if name in learnable_params]
            vectors.append(torch.cat(flat_params) if flat_params else torch.tensor([]))
        
        vec_stack = torch.stack(vectors)
        dists = torch.cdist(vec_stack, vec_stack, p=2)
        
        # 计算 Krum Score
        k = max(n - self.f - 2, 1)
        krum_scores = []
        for i in range(n):
            sorted_dists, _ = torch.sort(dists[i])
            krum_scores.append(torch.sum(sorted_dists[1:k+1]).item())
            
        sorted_indices = np.argsort(krum_scores)
        selected_indices = set(sorted_indices[:self.m].tolist())
        
        screen_scores = [1.0 if i in selected_indices else 0.0 for i in range(n)]
        return screen_scores, context