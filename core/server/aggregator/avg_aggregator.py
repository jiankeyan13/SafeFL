import torch
from typing import List, Dict, Optional, Any, Tuple
from .base_aggregator import BaseAggregator
from core.utils.registry import AGGREGATOR_REGISTRY

@AGGREGATOR_REGISTRY.register("avg")
class AvgAggregator(BaseAggregator):
    """
    通用线性聚合器 (Linear Aggregator)。
    返回聚合后的纯 delta（不含全局模型权重）。
    """

    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  sample_weights: Optional[List[float]] = None,
                  screen_scores: Optional[List[float]] = None,
                  global_model: torch.nn.Module = None,
                  context: Dict[str, Any] = None,
                  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        聚合客户端 Deltas，返回纯 delta。
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
        context = context or {}
        num_clients = len(updates)

        # 融合 sample_weights 和 screen_scores
        if sample_weights is None:
            sample_weights = [1.0] * num_clients
        if screen_scores is None:
            screen_scores = [1.0] * num_clients
            
        combined_weights = [s * sc for s, sc in zip(sample_weights, screen_scores)]
        
        self._check_inputs(updates, combined_weights)
        norm_weights = self._normalize_weights(combined_weights)

        aggregated_deltas = self._weighted_aggregate(updates, norm_weights)
                    
        return aggregated_deltas, context