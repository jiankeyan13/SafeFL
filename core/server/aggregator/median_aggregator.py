import torch
from typing import List, Dict, Optional, Any, Tuple
from .base_aggregator import BaseAggregator
from core.utils.registry import AGGREGATOR_REGISTRY

@AGGREGATOR_REGISTRY.register("median")
class MedianAggregator(BaseAggregator):
    """
    坐标轴中位数聚合器 (Coordinate-wise Median Aggregator)。
    
    对于参数矩阵的每一个位置 (Coordinate)，取所有客户端上传值的"中位数"。
    
    忽略 `sample_weights` 和 `screen_scores`。
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
        Args:
            updates: 客户端 delta 列表。
            sample_weights: 权重列表 (被忽略)。
            screen_scores: 筛选分数 (被忽略)。
            global_model: 全局模型 (未使用)。
            context: 上下文信息。
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
        context = context or {}

        # 使用第一个客户端作为 Key 的模板
        template_update = updates[0]
        aggregated_params = {}

        # 逐层遍历 (Layer-wise)
        # 这种方式比把整个模型 Flatten 更加节省显存
        for name in template_update.keys():
            
            # shape: [N, d1, d2, ...]
            client_tensors = []
            for client_idx, update in enumerate(updates):
                param = update[name]
                # 确保数据在正确的设备上，准备进行 Stack
                if param.device != self.device:
                    param = param.to(self.device)
                client_tensors.append(param)

            # 此时 stacked_tensor 的维度是 (Client_Num, Param_Dim_1, Param_Dim_2, ...)
            stacked_tensor = torch.stack(client_tensors, dim=0)

            # torch.median 返回一个元组 (values, indices)，我们只需要 values
            # dim=0 表示沿着"客户端数量"这一维求中位数
            median_result = torch.median(stacked_tensor, dim=0).values

            aggregated_params[name] = median_result

        return aggregated_params, context