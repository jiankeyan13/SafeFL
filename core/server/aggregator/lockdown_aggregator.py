from typing import Any, Dict, List, Optional, Tuple

import torch

from core.client.lockdown_client import LOCKDOWN_MASK_PREFIX
from core.utils.registry import AGGREGATOR_REGISTRY
from .base_aggregator import BaseAggregator


@AGGREGATOR_REGISTRY.register("lockdown")
class LockdownAggregator(BaseAggregator):
    """
    FedAvg over sparse Lockdown deltas plus consensus-mask construction.
    """

    def __init__(self, device="cuda", theta: Optional[int] = None, theta_ratio: Optional[float] = None):
        super().__init__(device=device)
        self.theta = theta
        self.theta_ratio = theta_ratio

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

        cleaned_updates = [
            {
                name: tensor
                for name, tensor in update.items()
                if not name.startswith(LOCKDOWN_MASK_PREFIX)
            }
            for update in updates
        ]
        aggregated_delta = self._weighted_aggregate(cleaned_updates, norm_weights)

        masks = context.get("lockdown_masks")
        if masks is None:
            masks = [self._extract_masks(update) for update in updates]

        consensus_mask = self._build_consensus_mask(masks, screen_scores)
        context["lockdown_consensus_mask"] = consensus_mask
        return aggregated_delta, context

    def _extract_masks(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        for key, value in update.items():
            if key.startswith(LOCKDOWN_MASK_PREFIX):
                masks[key[len(LOCKDOWN_MASK_PREFIX):]] = value.detach().to(dtype=torch.bool, device="cpu")
        return masks

    def _resolve_theta(self, num_active_clients: int) -> int:
        if self.theta is not None:
            return int(self.theta)
        if self.theta_ratio is not None:
            return max(1, int(torch.ceil(torch.tensor(num_active_clients * self.theta_ratio)).item()))
        return max(1, int(torch.ceil(torch.tensor(num_active_clients / 2.0)).item()))

    def _build_consensus_mask(
        self,
        masks: List[Dict[str, torch.Tensor]],
        screen_scores: Optional[List[float]],
    ) -> Dict[str, torch.Tensor]:
        active_indices = [
            idx for idx in range(len(masks))
            if screen_scores is None or float(screen_scores[idx]) > 0.0
        ]
        theta = self._resolve_theta(len(active_indices))
        consensus: Dict[str, torch.Tensor] = {}
        if not active_indices:
            return consensus

        names = set()
        for idx in active_indices:
            names.update(masks[idx].keys())

        for name in names:
            vote = None
            for idx in active_indices:
                if name not in masks[idx]:
                    continue
                mask = masks[idx][name].to(dtype=torch.int16)
                vote = mask if vote is None else vote + mask
            if vote is not None:
                consensus[name] = (vote >= theta).cpu()

        return consensus
