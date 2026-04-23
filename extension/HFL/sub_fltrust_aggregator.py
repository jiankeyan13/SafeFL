import torch
from typing import List, Dict, Optional, Any, Tuple, Set

from core.utils.registry import AGGREGATOR_REGISTRY
from .sub_aggregator import SubAvgAggregator


def _get_delta_keys(state_dict: Dict[str, torch.Tensor]) -> Set[str]:
    return {
        k for k in state_dict.keys()
        if "num_batches_tracked" not in k
        and not k.endswith("running_mean")
        and not k.endswith("running_var")
    }


@AGGREGATOR_REGISTRY.register("sub_fltrust")
class SubFLTrustAggregator(SubAvgAggregator):
    """HFL-compatible FLTrust: norm alignment + trust-score weighted sub-model aggregation."""

    def __init__(self, device: str = "cuda", eps: float = 1e-12, **kwargs):
        super().__init__(device=device, eps=eps)
        self.norm_eps = eps

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_weights: Optional[List[float]] = None,
        screen_scores: Optional[List[float]] = None,
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        del kwargs
        context = context or {}
        if not updates:
            raise ValueError("Updates list is empty")

        trust_scores = context.get("trust_scores")
        ref_norm_raw = context.get("reference_norm")
        if trust_scores is None or ref_norm_raw is None:
            return super().aggregate(
                updates,
                sample_weights=sample_weights or [1.0] * len(updates),
                screen_scores=screen_scores or [1.0] * len(updates),
                global_model=global_model,
                context=context,
            )

        ref_norm = float(ref_norm_raw) + self.norm_eps
        delta_keys = context.get("delta_keys")
        if delta_keys is None:
            delta_keys = _get_delta_keys(context.get("reference_delta", updates[0]))
        client_norms = context.get("client_norms")

        normalized_updates: List[Dict[str, torch.Tensor]] = []
        for idx, update in enumerate(updates):
            trust_score = float(trust_scores[idx]) if idx < len(trust_scores) else 0.0
            if trust_score <= 0:
                normalized_updates.append(update)
                continue

            if client_norms is not None and idx < len(client_norms):
                norm_i = float(client_norms[idx]) + self.norm_eps
            else:
                flat = self._flatten_update(update, delta_keys)
                norm_i = torch.norm(flat).item() + self.norm_eps

            scale = ref_norm / norm_i
            scaled_update = {}
            for key, value in update.items():
                value_f = value.to(torch.float32)
                scaled_update[key] = value_f * scale if key in delta_keys else value_f
            normalized_updates.append(scaled_update)

        return super().aggregate(
            normalized_updates,
            sample_weights=[1.0] * len(normalized_updates),
            screen_scores=trust_scores,
            global_model=global_model,
            context=context,
        )
