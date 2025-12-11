import torch
from typing import List, Dict, Optional
from .base_aggregator import BaseAggregator

class MedianAggregator(BaseAggregator):
    """
    坐标轴中位数聚合器 (Coordinate-wise Median Aggregator)。
    
    对于参数矩阵的每一个位置 (Coordinate)，取所有客户端上传值的"中位数"。
    
    忽略 `weights` (权重)。
    """

    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  weights: Optional[List[float]] = None, 
                  **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            updates: 客户端参数列表。
            weights: 权重列表 (被忽略)。
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
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

            # 6. 存入结果字典
            aggregated_params[name] = median_result

        return aggregated_params