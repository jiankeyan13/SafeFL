from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import torch

class BaseAggregator(ABC):
    def __init__(self, device='cuda'):
        self.device = device

    @abstractmethod
    def aggregate(self, 
                  updates: List[Dict[str, torch.Tensor]], 
                  sample_weights: Optional[List[float]] = None,
                  screen_scores: Optional[List[float]] = None,
                  global_model: torch.nn.Module = None,
                  context: Dict[str, Any] = None,
                  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        核心聚合逻辑，子类必须实现。
        
        Returns:
            (aggregated_weights, context): 聚合后的权重和更新后的上下文
        """
        pass

    def _check_inputs(self, updates, weights):
        if not updates or not weights:
            raise ValueError("Empty updates or weights")
        if len(updates) != len(weights):
            raise ValueError("Updates and weights length mismatch")

    def _normalize_weights(self, weights):
        total = sum(weights)
        if total == 0:
            return [0.0] * len(weights)
        return [w / total for w in weights]

    def _flatten_update(self, update: Dict[str, torch.Tensor], learnable_keys: set = None) -> torch.Tensor:
        """将单个 update dict 展平为一维向量"""
        flat_params = []
        for name, tensor in update.items():
            if learnable_keys is None or name in learnable_keys:
                flat_params.append(tensor.view(-1))
        return torch.cat(flat_params) if flat_params else torch.tensor([])

    def _get_learnable_keys(self, global_model: torch.nn.Module) -> set:
        """获取可学习参数的名称集合"""
        if global_model is not None:
            return set(name for name, param in global_model.named_parameters() if param.requires_grad)
        return None

    def _weighted_aggregate(self, 
                            updates: List[Dict[str, torch.Tensor]], 
                            norm_weights: List[float]) -> Dict[str, torch.Tensor]:
        """
        向量化加权聚合公共逻辑。
        输入 N 个 delta + 归一化权重 → 输出 1 个聚合后的 delta。
        """
        num_clients = len(updates)
        w_tensor = torch.tensor(norm_weights, dtype=torch.float32, device=self.device)
        
        aggregated_deltas = {}
        layer_names = updates[0].keys()

        for name in layer_names:
            layer_stack = torch.stack([u[name].to(torch.float32) for u in updates]).to(self.device)
            w_view_shape = [num_clients] + [1] * (layer_stack.dim() - 1)
            w_view = w_tensor.view(*w_view_shape)
            aggregated_deltas[name] = torch.sum(layer_stack * w_view, dim=0)

        return aggregated_deltas