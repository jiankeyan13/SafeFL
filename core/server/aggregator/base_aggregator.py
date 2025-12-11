from abc import ABC, abstractmethod
import torch

class BaseAggregator(ABC):
    def __init__(self, device='cpu'):
        self.device = device

    @abstractmethod
    def aggregate(self, updates, weights, **kwargs):
        """
        核心聚合逻辑，子类必须实现。
        """
        pass

    def _check_inputs(self, updates, weights):
        """检查逻辑"""
        if not updates or not weights:
            raise ValueError("Empty updates or weights")
        if len(updates) != len(weights):
            raise ValueError("Updates and weights length mismatch")
        # 还可以加 check shape 等

    def _normalize_weights(self, weights):
        """归一化权重"""
        total = sum(weights)
        if total == 0:
            return [0.0] * len(weights)
        return [w / total for w in weights]

    def _flatten_updates(self, updates):
        """
        工具函数：将 List[Dict] 转换为 Tensor(N, D)
        这对 Krum/Median 等算法是必须的，
        对 FedAvg 也可以用来做向量化加速。
        """
        # 实现逻辑：torch.cat([p.flatten() for p in client_params])
        pass

    def _reconstruct_updates(self, flattened_tensor, template):
        """工具函数：将 Tensor(D) 还原为 Dict 结构"""
        # 实现逻辑：根据 template 的 shape 切分 tensor
        pass