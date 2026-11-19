"""
FLTrust 聚合器: 幅度归一化 (将 delta_i 缩放到与 delta_0 等范数) + 以 TS 为权重的加权平均.
delta_0 不参与加权求和, 仅作方向锚点.
"""
from typing import List, Dict, Any, Optional, Tuple, Set

import torch

from core.utils.registry import AGGREGATOR_REGISTRY
from .base_aggregator import BaseAggregator


def _get_delta_keys(state_dict: Dict[str, torch.Tensor]) -> Set[str]:
    """与 Screener 一致的 delta 键过滤: 排除 BN running stats 和 num_batches_tracked."""
    return {
        k for k in state_dict.keys()
        if "num_batches_tracked" not in k
        and not k.endswith("running_mean")
        and not k.endswith("running_var")
    }


@AGGREGATOR_REGISTRY.register("fltrust")
class FLTrustAggregator(BaseAggregator):
    """
    FLTrust 聚合器: 同范数归一化 + TS 加权平均. delta_0 不参与聚合.
    """

    def __init__(self, device: str = "cuda", eps: float = 1e-12, **kwargs):
        super().__init__(device=device)
        self.eps = eps

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_weights: Optional[List[float]] = None,
        screen_scores: Optional[List[float]] = None,
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        context = context or {}
        if not updates:
            raise ValueError("Updates list is empty")

        trust_scores = context.get("trust_scores")
        ref_norm_raw = context.get("reference_norm")

        if trust_scores is None or ref_norm_raw is None:
            combined = [
                s * (sc if sc is not None else 1.0)
                for s, sc in zip(sample_weights or [1.0] * len(updates), screen_scores or [1.0] * len(updates))
            ]
            total = sum(combined)
            if total <= 0:
                zero_delta = {k: torch.zeros_like(v) for k, v in updates[0].items()}
                return zero_delta, context
            norm_weights = [w / total for w in combined]
            return self._weighted_aggregate(updates, norm_weights), context

        device = next(global_model.parameters()).device
        ref_norm = float(ref_norm_raw) + self.eps

        delta_keys = context.get("delta_keys")
        if delta_keys is None:
            delta_keys = _get_delta_keys(context.get("reference_delta", updates[0]))
        client_norms = context.get("client_norms")

        normalized_updates = []
        for i, delta in enumerate(updates):
            if trust_scores[i] <= 0:
                continue
            if client_norms is not None and i < len(client_norms):
                norm_i = float(client_norms[i]) + self.eps
            else:
                flat = self._flatten_update(delta, delta_keys)
                norm_i = torch.norm(flat).item() + self.eps
            scale = ref_norm / norm_i
            scaled_delta = {}
            for k, v in delta.items():
                v_f = v.to(device, dtype=torch.float32)
                if k in delta_keys:
                    scaled_delta[k] = v_f * scale
                else:
                    scaled_delta[k] = v_f
            normalized_updates.append(scaled_delta)

        if not normalized_updates:
            zero_delta = {k: torch.zeros_like(v).to(device) for k, v in updates[0].items()}
            return zero_delta, context

        weights = [trust_scores[i] for i in range(len(updates)) if trust_scores[i] > 0]
        total = sum(weights)
        if total <= 0:
            zero_delta = {k: torch.zeros_like(v).to(device) for k, v in updates[0].items()}
            return zero_delta, context

        norm_weights = [w / total for w in weights]
        aggregated = self._weighted_aggregate(normalized_updates, norm_weights)
        return aggregated, context
