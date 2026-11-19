from typing import Any, Dict, List, Tuple

import torch

from core.client.lockdown_client import LOCKDOWN_MASK_PREFIX
from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("lockdown")
class LockdownScreener(BaseScreener):
    """
    Lockdown does not reject clients here; this component extracts client masks
    from payloads and passes them through context for consensus fusion.
    """

    def screen(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        num_samples: List[float],
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        context = context or {}
        masks: List[Dict[str, torch.Tensor]] = []
        densities: List[float] = []

        for delta in client_deltas:
            client_mask: Dict[str, torch.Tensor] = {}
            active = 0.0
            total = 0.0
            for key, value in delta.items():
                if not key.startswith(LOCKDOWN_MASK_PREFIX):
                    continue
                name = key[len(LOCKDOWN_MASK_PREFIX):]
                mask = value.detach().to(dtype=torch.bool, device="cpu")
                client_mask[name] = mask
                active += float(mask.sum().item())
                total += float(mask.numel())
            masks.append(client_mask)
            densities.append(active / total if total > 0 else 0.0)

        context["lockdown_masks"] = masks
        context["lockdown_mask_density"] = densities
        return [1.0] * len(client_deltas), context
