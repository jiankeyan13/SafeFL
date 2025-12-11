import torch
from typing import List, Dict, Optional
from .base_aggregator import BaseAggregator

class AvgAggregator(BaseAggregator):
    """
    通用线性聚合器 (Linear Aggregator)。
    
    兼容两种模式：
    1. 加权平均 (Weighted Avg / FedAvg): 传入具体的 weights 列表。
    2. 算术平均 (Arithmetic Mean / Simple Avg): 传入 weights=None。
    
    """

    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  weights: Optional[List[float]] = None, 
                  **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            updates: 客户端参数列表
            weights: 权重列表。
                     - 如果为 None, 则执行"算术平均" (所有权重 = 1/N)。
                     - 如果有值，则执行"加权平均"。
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
        num_clients = len(updates)

        if weights is None:
            norm_weights = [1.0 / num_clients] * num_clients
        else:
            self._check_inputs(updates, weights)
            norm_weights = self._normalize_weights(weights)

        # 使用第一个客户端作为 Shape 模板
        template_update = updates[0]
        aggregated_params = {}

        # 逐层遍历 (Layer-wise)
        for name, param_template in template_update.items():
            
            # 初始化Float64累加器
            accumulator = torch.zeros_like(
                param_template, 
                device=self.device, 
                dtype=torch.float64
            )

            # 遍历客户端累加
            for client_idx, client_update in enumerate(updates):
                weight = norm_weights[client_idx]
                client_param = client_update[name]

                # 设备与精度对齐
                if client_param.device != self.device or client_param.dtype != torch.float64:
                    client_param = client_param.to(device=self.device, dtype=torch.float64)
                
                # 核心计算：accum = sum(w_i * p_i)
                # 当 w_i 都是 1/N 时，这等价于 sum(p_i) / N
                accumulator += client_param * weight

            # 还原为 Float32
            aggregated_params[name] = accumulator.to(dtype=torch.float32)

        return aggregated_params