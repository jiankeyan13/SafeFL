import torch
from typing import List, Dict, Optional, Any, Tuple
from core.server.aggregator.base_aggregator import BaseAggregator
from core.utils.registry import AGGREGATOR_REGISTRY


@AGGREGATOR_REGISTRY.register("sub_avg")
class SubAvgAggregator(BaseAggregator):
    """
    子模型聚合器 (Sub-model Aggregator)，采用"基于零值推断的动态掩码聚合"策略。
    
    适用场景：
        - 模型异构联邦学习：客户端通过结构化剪枝仅训练全局模型的子集
        - 稀疏更新:Delta 矩阵在未训练的参数位置被 Zero-Padding
        - 防御机制共存：系统包含后门/投毒防御，为客户端计算软信任分数
    
    核心区别于 AvgAggregator:
        - 采用 逐元素动态归一化 (Element-wise Dynamic Normalization)
        - 分母仅包含更新了该参数的客户端权重之和，消除稀释效应
    
    实现逻辑：
        1. 零值推断 (Zero Inference): 利用 Delta 中的 0 值生成二进制掩码
        2. 软加权结合 (Soft-Weighted Integration): 分子加权求和，分母累加有效权重
        3. 按位还原 (Element-wise Restoration): 分子除以分母得到加权平均
    """

    def __init__(self, device='cpu', eps: float = 1e-10):
        """
        初始化子模型聚合器。
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
            eps: 防止除零的小量，默认 1e-10
        """
        super().__init__(device)
        self.eps = eps

    def aggregate(self,
                  updates: List[Dict[str, torch.Tensor]],
                  sample_weights: Optional[List[float]] = None,
                  screen_scores: Optional[List[float]] = None,
                  global_model: torch.nn.Module = None,
                  context: Dict[str, Any] = None,
                  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        聚合客户端 Deltas 并返回完整模型权重。
        
        采用掩码感知 (Mask-Aware) 的加权平均，通过动态调整分母解决
        异构联邦学习中参数更新频率不一致导致的梯度尺度失真问题。
        
        Args:
            updates: 客户端 Delta 列表，每个 Delta 是 {layer_name: tensor} 的字典
            sample_weights: 样本权重列表（如客户端样本数）
            screen_scores: 防御机制计算的软信任分数
            global_model: 全局模型，用于获取基础权重
            context: 上下文信息字典
            **kwargs: 其他参数
            
        Returns:
            (final_weights, context): 聚合后的完整模型权重和更新后的上下文
        """
        if not updates:
            raise ValueError("Updates list is empty")

        context = context or {}
        num_clients = len(updates)

        # ===== Step 0: 融合 sample_weights 和 screen_scores =====
        if sample_weights is None:
            sample_weights = [1.0] * num_clients
        if screen_scores is None:
            screen_scores = [1.0] * num_clients

        combined_weights = [s * sc for s, sc in zip(sample_weights, screen_scores)]
        self._check_inputs(updates, combined_weights)
        
        # 注意：这里不做全局归一化，因为我们需要逐元素动态归一化
        w_tensor = torch.tensor(combined_weights, dtype=torch.float32, device=self.device)

        aggregated_deltas = {}
        layer_names = updates[0].keys()

        for name in layer_names:
            # ===== Step 1: 堆叠所有客户端的 Delta =====
            # layer_stack shape: [num_clients, *param_shape]
            layer_stack = torch.stack(
                [u[name].to(torch.float32) for u in updates]
            ).to(self.device)

            # ===== Step 2: 零值推断生成掩码 =====
            # mask shape: [num_clients, *param_shape]，非零位置为 1，零位置为 0
            mask = (layer_stack != 0).float()

            # ===== Step 3: 计算逐元素的有效权重和（分母）=====
            # 将权重 tensor 广播到与 layer_stack 相同的形状
            w_view_shape = [num_clients] + [1] * (layer_stack.dim() - 1)
            w_view = w_tensor.view(*w_view_shape)  # shape: [num_clients, 1, 1, ...]

            # 有效权重 = 掩码 * 权重，然后沿客户端维度求和
            # denominator shape: [*param_shape]
            denominator = torch.sum(mask * w_view, dim=0)

            # ===== Step 4: 计算加权 Delta 和（分子）=====
            # numerator shape: [*param_shape]
            numerator = torch.sum(layer_stack * w_view, dim=0)

            # ===== Step 5: 按位还原（逐元素除法）=====
            # 对于被剪枝区域: 0 / eps ≈ 0（保持未更新状态）
            # 对于重叠区域: 得到参与该参数训练的客户端的加权平均值
            aggregated_deltas[name] = numerator / (denominator + self.eps)

        # ===== Step 6: 构建完整的 state_dict: Base + Delta =====
        final_weights = {}
        global_state = global_model.state_dict()

        for key, value in global_state.items():
            final_weights[key] = value.clone()
            if key in aggregated_deltas:
                delta = aggregated_deltas[key].to(device=value.device, dtype=value.dtype)
                final_weights[key] += delta

        return final_weights, context
