import math
from typing import Dict, List

from extension.HFL.hetero_server import HeteroServer
from .cap_manager import CapManager


class FedrolexServer(HeteroServer):
    """Fedrolex异构联邦学习服务器: 使用滚动选取替代随机选取"""

    def __init__(self, model, device, cap_manager: CapManager, \
                 rolex_to_next: bool = True, **kwargs):
        super().__init__(model=model, device=device, cap_manager=cap_manager, **kwargs)
        self.client_layer_start: Dict[str, Dict[str, int]] = {}
        self.rolex_to_next = rolex_to_next

    def _get_layer_indices(self, num_channels: int, p: float, \
                          layer_key: str, client_id: str) -> List[int]:
        """滚动选取神经元索引，从上次位置开始连续选取"""
        if client_id not in self.client_layer_start:
            self.client_layer_start[client_id] = {}
        
        layer_sid = self.client_layer_start[client_id].get(layer_key, None)
        if layer_sid is None:
            layer_sid = 0
        else:
            if self.rolex_to_next:
                layer_sid = (layer_sid + 1) % num_channels
        self.client_layer_start[client_id][layer_key] = layer_sid
        
        n_keep = max(1, math.ceil(num_channels * p))
        order = [(layer_sid + ni) % num_channels for ni in range(n_keep)]
        return sorted(order)
