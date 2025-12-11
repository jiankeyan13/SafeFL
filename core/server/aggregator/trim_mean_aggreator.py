import torch
from typing import List, Dict, Optional
from .base_aggregator import BaseAggregator

class TrimmedMeanAggregator(BaseAggregator):
    """
    截断平均聚合器 (Trimmed Mean Aggregator).
    
    逻辑:
    对于每一个参数维度(dimension-wise):
    1. 收集所有客户端在该维度的值。
    2. 进行排序。
    3. 去除最大和最小的 k 个值 (k based on trim_ratio)。
    4. 对剩余的值求算术平均。
    
    """

    def __init__(self, trim_ratio: float = 0.1, device='cpu'):
        """
        Args:
            trim_ratio: 截断比例 (0.0 <= trim_ratio < 0.5)。
                        例如 0.1 表示去掉最大的 10% 和最小的 10%。
            device: 计算设备。
        """
        super().__init__(device)
        
        if not (0.0 <= trim_ratio < 0.5):
            raise ValueError("trim_ratio must be in [0.0, 0.5). If 0.5, use MedianAggregator instead.")
        
        self.trim_ratio = trim_ratio

    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  weights: Optional[List[float]] = None, 
                  **kwargs) -> Dict[str, torch.Tensor]:

        if not updates:
            raise ValueError("Updates list is empty")
        
        num_clients = len(updates)
        
        # 计算需要截断的数量
        # int(10 * 0.1) = 1 -> 去掉头部1个，尾部1个
        num_cut = int(num_clients * self.trim_ratio)
        
        # 边界保护：如果客户端太少，或者比例导致所有都被切掉了
        if num_clients - 2 * num_cut <= 0:
            raise ValueError(
                f"Too many clients trimmed! Total: {num_clients}, "
                f"Trim Ratio: {self.trim_ratio}, Cut per side: {num_cut}. "
                "Remaining clients would be <= 0."
            )

        # 初始化结果容器
        aggregated_params = {}
        template_update = updates[0]

        # 逐层处理 (Layer-wise Processing)
        for name, _ in template_update.items():
            # Shape: [num_clients, param_shape...]
            client_tensors = []
            for upd in updates:
                t = upd[name].to(self.device)
                client_tensors.append(t)
            
            stacked_tensors = torch.stack(client_tensors, dim=0)

            # sorted_tensors Shape: [num_clients, param_shape...]
            sorted_tensors, _ = torch.sort(stacked_tensors, dim=0)

            # Trimming
            keep_start = num_cut
            keep_end = num_clients - num_cut
            
            trimmed_tensors = sorted_tensors[keep_start:keep_end]

            aggregated_params[name] = torch.mean(trimmed_tensors, dim=0)

        return aggregated_params