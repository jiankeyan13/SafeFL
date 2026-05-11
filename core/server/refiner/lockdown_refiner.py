from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from core.utils.registry import REFINER_REGISTRY
from .base_refiner import BaseRefiner


@REFINER_REGISTRY.register("lockdown")
class LockdownRefiner(BaseRefiner):
    """
    Applies Lockdown consensus fusion after BaseServer has built new_state.
    """

    def __init__(self, config=None, apply_consensus: bool = True):
        super().__init__(config)
        self.apply_consensus = bool(apply_consensus)

    def process(
        self,
        global_model: torch.nn.Module,
        new_state: Dict[str, torch.Tensor],
        calibration_loader: Optional[DataLoader] = None,
        device: torch.device = None,
        context: Dict[str, Any] = None,
    ):
        context = context or {}
        refined_state = {name: value.clone() for name, value in new_state.items()}

        if self.apply_consensus:
            consensus_mask = context.get("lockdown_consensus_mask", {})
            for name, mask in consensus_mask.items():
                if name not in refined_state:
                    continue
                value = refined_state[name]
                if not torch.is_floating_point(value):
                    continue
                mask = mask.to(device=value.device, dtype=torch.bool)
                refined_state[name] = torch.where(mask, value, torch.zeros_like(value))

        global_model.load_state_dict(refined_state)

        if calibration_loader:
            self.calibrate_bn(global_model, calibration_loader, device)
