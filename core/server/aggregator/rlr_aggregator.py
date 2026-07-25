import torch
from typing import List, Dict, Optional, Any, Tuple

from .base_aggregator import BaseAggregator
from core.utils.registry import AGGREGATOR_REGISTRY


@AGGREGATOR_REGISTRY.register("rlr")
class RLRAggregator(BaseAggregator):
    """
    Robust Learning Rate (RLR) 聚合器.

    先对客户端 delta 做加权 FedAvg, 再按逐维符号投票结果乘以 +/-1 掩码.
    论文: Ozdayi et al., AAAI 2021.
    """

    def __init__(self, robustLR_threshold: int = 4, device: str = "cuda"):
        super().__init__(device=device)
        if robustLR_threshold < 1:
            raise ValueError("robustLR_threshold must be >= 1")
        self.robustLR_threshold = robustLR_threshold

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_weights: Optional[List[float]] = None,
        screen_scores: Optional[List[float]] = None,
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if not updates:
            raise ValueError("Updates list is empty")

        context = context or {}
        num_clients = len(updates)

        if sample_weights is None:
            sample_weights = [1.0] * num_clients
        if screen_scores is None:
            screen_scores = [1.0] * num_clients

        combined_weights = [s * sc for s, sc in zip(sample_weights, screen_scores)]
        self._check_inputs(updates, combined_weights)
        norm_weights = self._normalize_weights(combined_weights)

        w_tensor = torch.tensor(norm_weights, dtype=torch.float32, device=self.device)
        aggregated_params: Dict[str, torch.Tensor] = {}
        template_update = updates[0]

        for name in template_update.keys():
            client_tensors = [
                upd[name].to(device=self.device, dtype=torch.float32) for upd in updates
            ]
            stacked = torch.stack(client_tensors, dim=0)

            w_view_shape = [num_clients] + [1] * (stacked.dim() - 1)
            w_view = w_tensor.view(*w_view_shape)
            avg_delta = torch.sum(stacked * w_view, dim=0)

            sum_signs = torch.sign(stacked).sum(dim=0)
            mask = torch.where(
                sum_signs.abs() >= self.robustLR_threshold,
                torch.ones_like(sum_signs),
                -torch.ones_like(sum_signs),
            )
            aggregated_params[name] = mask * avg_delta

        return aggregated_params, context
