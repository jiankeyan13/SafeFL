"""Neurotoxin: durable backdoors in federated learning.

The attack avoids coordinates that are most active in the previous global
update. Backdoor training is therefore pushed into parameters that benign
training is less likely to overwrite in following rounds.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY


@torch.no_grad()
def build_neurotoxin_masks(
    model: nn.Module,
    reference_update: Optional[Mapping[str, torch.Tensor]],
    mask_ratio: float,
) -> Dict[str, torch.Tensor]:
    """Build global top-k masks keyed by ``model.named_parameters()``.

    A zero suppresses that gradient coordinate. Only trainable parameters
    present in ``reference_update`` participate in the global ranking;
    buffers and missing/incompatible entries are ignored.
    """
    if not 0.0 <= mask_ratio <= 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
    if reference_update is None or mask_ratio == 0.0:
        return {}

    entries = []
    flat_scores = []
    for name, parameter in model.named_parameters():
        reference = reference_update.get(name)
        if (
            not parameter.requires_grad
            or not torch.is_tensor(reference)
            or reference.numel() != parameter.numel()
        ):
            continue
        scores = reference.detach().to(device="cpu", dtype=torch.float32).abs().reshape(-1)
        entries.append((name, parameter, scores.numel()))
        flat_scores.append(scores)

    if not flat_scores:
        return {}

    scores = torch.cat(flat_scores)
    num_masked = min(scores.numel(), int(scores.numel() * mask_ratio))
    if num_masked == 0:
        return {}

    blocked = torch.zeros(scores.numel(), dtype=torch.bool)
    blocked[torch.topk(scores, k=num_masked, largest=True, sorted=False).indices] = True

    masks: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, parameter, numel in entries:
        keep = ~blocked[offset:offset + numel]
        masks[name] = keep.reshape(parameter.shape).to(
            device=parameter.device, dtype=parameter.dtype
        )
        offset += numel
    return masks


@ATTACK_REGISTRY.register("neurotoxin")
class NeurotoxinAttack:
    """Training-stage Neurotoxin attack with BadNets trigger poisoning."""

    client_class = None  # late-bound below to avoid a client/attack import cycle

    def __init__(
        self,
        target_label: int = 0,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        mask_ratio: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")
        self._badnets = BadNetsAttack(
            target_label=target_label,
            poison_ratio=poison_ratio,
            patch_size=patch_size,
            patch_value=patch_value,
            patch_location=patch_location,
            seed=seed,
        )
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.mask_ratio = mask_ratio
        self.seed = seed
        self._gradient_masks: Dict[str, torch.Tensor] = {}

    def poison_dataset(
        self,
        dataset: Dataset,
        mode: str,
        split: str = "",
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        return self._badnets.poison_dataset(
            dataset,
            mode=mode,
            split=split,
            client_id=client_id,
            round_idx=round_idx,
            **kwargs,
        )

    def prepare_gradient_mask(
        self,
        model: nn.Module,
        previous_global_delta: Optional[Mapping[str, torch.Tensor]],
    ) -> None:
        """Prepare one fixed mask for all malicious local steps this round."""
        self._gradient_masks = build_neurotoxin_masks(
            model=model,
            reference_update=previous_global_delta,
            mask_ratio=self.mask_ratio,
        )

    @torch.no_grad()
    def poison_train(
        self,
        model: nn.Module,
        hook_point: str,
        **kwargs,
    ) -> Dict[str, object]:
        """Suppress high-activity coordinates after backpropagation."""
        if hook_point != "after_backward" or not self._gradient_masks:
            return {}
        for name, parameter in model.named_parameters():
            mask = self._gradient_masks.get(name)
            if parameter.grad is not None and mask is not None:
                parameter.grad.mul_(mask)
        return {}


from core.client.neurotoxin_client import NeurotoxinClient  # noqa: E402

NeurotoxinAttack.client_class = NeurotoxinClient
