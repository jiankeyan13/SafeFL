from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from core.utils.registry import AGGREGATOR_REGISTRY
from .base_aggregator import BaseAggregator


@AGGREGATOR_REGISTRY.register("alignins")
class AlignInsAggregator(BaseAggregator):
    """
    AlignIns aggregation stage:
    1. Use benign clients to determine the clipping threshold.
    2. Clip all client updates by that threshold.
    3. Average clipped benign updates.
    """

    def __init__(self, device: str = "cuda", eps: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(device=device)
        self.eps = float(eps)

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_weights: Optional[List[float]] = None,
        screen_scores: Optional[List[float]] = None,
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        del sample_weights, kwargs
        if not updates:
            raise ValueError("Updates list is empty")

        context = context or {}
        num_clients = len(updates)
        benign_indices = context.get("alignins_benign_indices")
        if benign_indices is None:
            benign_indices = [
                idx for idx, score in enumerate(screen_scores or [1.0] * num_clients)
                if score > 0
            ]
        benign_indices = [int(idx) for idx in benign_indices]
        if not benign_indices:
            benign_indices = list(range(num_clients))

        learnable_keys = self._get_learnable_keys(global_model)
        norms = [
            torch.norm(self._flatten_update(update, learnable_keys), p=2).item()
            for update in updates
        ]

        benign_norms = torch.tensor([norms[idx] for idx in benign_indices], dtype=torch.float32)
        clip_value = float(torch.median(benign_norms).item()) if benign_norms.numel() > 0 else 0.0

        clipped_updates: List[Dict[str, torch.Tensor]] = []
        for norm, update in zip(norms, updates):
            scale = min(1.0, clip_value / max(norm, self.eps)) if clip_value > 0.0 else 1.0
            if scale >= 1.0:
                clipped_updates.append(update)
                continue
            clipped_updates.append({name: tensor * scale for name, tensor in update.items()})

        selected_updates = [clipped_updates[idx] for idx in benign_indices]
        norm_weights = [1.0 / len(selected_updates)] * len(selected_updates)
        aggregated = self._weighted_aggregate(selected_updates, norm_weights)

        context["alignins_benign_indices"] = benign_indices
        return aggregated, context
