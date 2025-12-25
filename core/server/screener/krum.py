import torch
import numpy as np
from typing import List, Dict

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener

@SCREENER_REGISTRY.register("krum")
class KrumScreener(BaseScreener):
    def __init__(self, f: int = 0, m: int = 1, **kwargs):
        """
        Args:
            f: 假设的攻击者数量 (Byzantine Tolerance)。
            m: Multi-Krum 最终保留的客户端数量。
               如果 m=1, 就是标准的 Krum。
        """
        super().__init__()
        self.f = f
        self.m = m

    def screen(self, 
               client_deltas: List[Dict[str, torch.Tensor]], 
               num_samples: List[float],
               global_model: torch.nn.Module = None) -> List[float]:
        """
        执行 Multi-Krum 筛选，返回筛选分数。
        只使用可学习参数计算距离，忽略 BN 统计量。
        """
        n = len(client_deltas)
        
        # 边界检查
        if n <= 2 * self.f + 2:
            # 如果客户端数量不足，返回全 1.0（不筛选）
            return [1.0] * n

        # 1. 获取可学习参数的名称集合
        if global_model is not None:
            learnable_params = set([name for name, param in global_model.named_parameters() if param.requires_grad])
        else:
            # 如果没有提供 global_model，假设所有参数都是可学习的
            learnable_params = set(client_deltas[0].keys())

        # 2. 展平可学习参数 (Flatten Learnable Parameters Only)
        vectors = []
        for delta_dict in client_deltas:
            # 只拼接可学习参数，忽略 BN 统计量
            flat_params = []
            for name, tensor in delta_dict.items():
                if name in learnable_params:
                    flat_params.append(tensor.view(-1))
            
            if len(flat_params) > 0:
                flat = torch.cat(flat_params)
            else:
                # 如果没有可学习参数，创建一个空向量
                flat = torch.tensor([])
            vectors.append(flat)
        
        # 堆叠成矩阵 [n, d]
        vec_stack = torch.stack(vectors)
        
        # 3. 计算两两距离矩阵 (Pairwise Distance)
        dists = torch.cdist(vec_stack, vec_stack, p=2)
        
        # 4. 计算 Krum Score
        k = n - self.f - 2
        if k <= 0: k = 1
        
        krum_scores = []
        for i in range(n):
            sorted_dists, _ = torch.sort(dists[i])
            score = torch.sum(sorted_dists[1 : k+1])
            krum_scores.append(score.item())
            
        # 5. 选择 Score 最小的 m 个索引
        sorted_indices = np.argsort(krum_scores)
        selected_indices = set(sorted_indices[:self.m].tolist())
        
        # 6. 返回筛选分数：被选中的为 1.0，未被选中的为 0.0
        screen_scores = [1.0 if i in selected_indices else 0.0 for i in range(n)]     
        return screen_scores