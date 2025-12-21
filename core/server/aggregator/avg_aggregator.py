import torch
from typing import List, Dict, Optional
from .base_aggregator import BaseAggregator
from core.utils.registry import AGGREGATOR_REGISTRY
@AGGREGATOR_REGISTRY.register("avg")
class AvgAggregator(BaseAggregator):
    """
    通用线性聚合器 (Linear Aggregator)。
    优化点：采用向量化运算 (Vectorization) 代替客户端循环，消除 PCIe 传输瓶颈。
    """

    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  weights: Optional[List[float]] = None, 
                  **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            updates: 客户端参数列表 (每个元素为一个 state_dict)
            weights: 权重列表。如果为 None 则执行算术平均。
        """
        if not updates:
            raise ValueError("Updates list is empty")
        
        num_clients = len(updates)

        if weights is None:
            norm_weights = [1.0 / num_clients] * num_clients
        else:
            self._check_inputs(updates, weights)
            norm_weights = self._normalize_weights(weights)

        w_tensor = torch.tensor(norm_weights, dtype=torch.float32, device=self.device)

        aggregated_params = {}
        # 逐层遍历 (Layer-wise)
        layer_names = updates[0].keys() # 获取第一层的名称作为模板

        for name in layer_names:
            # 在 CPU 上堆叠所有客户端的这一层参数，产生 (num_clients, ...) 的形状
            # 然后一次性推送到目标设备 (GPU)，极大减少 PCIe 握手次数
            layer_stack = torch.stack([u[name].to(torch.float32) for u in updates]).to(self.device)

            # 加权求和
            # 将 w_tensor 的形状从 (num_clients,) 调整为 (num_clients, 1, 1, 1...)
            # 以匹配当前层 (layer_stack) 的维度
            w_view_shape = [num_clients] + [1] * (layer_stack.dim() - 1)
            w_view = w_tensor.view(*w_view_shape)

            aggregated_params[name] = torch.sum(layer_stack * w_view, dim=0)

        return aggregated_params