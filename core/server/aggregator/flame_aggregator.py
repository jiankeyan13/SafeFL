import torch
from typing import List, Dict, Any, Optional, Tuple

from core.utils.registry import AGGREGATOR_REGISTRY
from .base_aggregator import BaseAggregator


# 22usenix-FLAME & 22ndss-DeepSight 使用此聚合器
@AGGREGATOR_REGISTRY.register("flame_aggregator")
class FlameAggregator(BaseAggregator):
    """
    FLAME 聚合器：对良性客户端进行范数裁剪后聚合。
    裁剪值从 context['clip_value'] 获取（由 FlameScreener 计算）。
    返回聚合后的纯 delta（不含全局模型权重）。
    """
    
    def __init__(self, device='cuda', **kwargs):
        super().__init__(device)

    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  sample_weights: Optional[List[float]] = None,
                  screen_scores: Optional[List[float]] = None,
                  global_model: torch.nn.Module = None,
                  context: Dict[str, Any] = None,
                  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        对通过筛选的客户端进行范数裁剪和聚合，返回纯 delta。
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
        context = context or {}
        n = len(updates)
        
        # 获取裁剪值和范数列表
        clip_value = context.get('clip_value', None)
        norm_list = context.get('norm_list', None)
        
        # 如果没有从 context 获取，则自行计算
        if clip_value is None or norm_list is None:
            learnable_keys = self._get_learnable_keys(global_model)
            norm_list = [torch.norm(self._flatten_update(u, learnable_keys)).item() for u in updates]
            clip_value = float(torch.median(torch.tensor(norm_list)).item())
            context['clip_value'] = clip_value
            context['norm_list'] = norm_list
        
        # 融合权重
        if sample_weights is None:
            sample_weights = [1.0] * n
        if screen_scores is None:
            screen_scores = [1.0] * n
        
        # 对每个客户端进行范数裁剪
        clipped_updates = []
        for i, update in enumerate(updates):
            if screen_scores[i] == 0:
                # 被筛选掉的客户端，直接跳过（权重为0）
                clipped_updates.append(update)
                continue
                
            client_norm = norm_list[i]
            gamma = min(clip_value / (client_norm + 1e-9), 1.0)
            
            if gamma < 1.0:
                # 需要裁剪
                clipped_update = {k: v * gamma for k, v in update.items()}
                clipped_updates.append(clipped_update)
            else:
                clipped_updates.append(update)
        
        # 计算最终权重：使用简单平均 (Simple Average) 而非样本数加权
        combined_weights = screen_scores
        
        # 检查是否有有效权重
        if sum(combined_weights) == 0:
            # 如果所有客户端都被筛选掉，返回零 delta
            zero_delta = {k: torch.zeros_like(v) for k, v in updates[0].items()}
            return zero_delta, context
        
        self._check_inputs(clipped_updates, combined_weights)
        norm_weights = self._normalize_weights(combined_weights)

        aggregated_deltas = self._weighted_aggregate(clipped_updates, norm_weights)
                    
        return aggregated_deltas, context

