"""Chameleon: durable FL backdoors via peer-image contrastive adaptation (ICML 2023).

Local malicious training has two stages on a BadNets-poisoned dataset:
1) Adaptation: supervised contrastive loss pulls poisoned embeddings toward
   facilitators (target label) and away from interferers (original class).
2) Projection: freeze the encoder and train only the classifier with CE.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY

_CLASSIFIER_NAMES = ("linear", "fc", "classifier")


class SupConLoss(nn.Module):
    """Supervised contrastive loss with facilitator up-weighting (beta)."""

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if base_temperature <= 0.0:
            raise ValueError(f"base_temperature must be > 0, got {base_temperature}")
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        scale_weight: float = 1.0,
        fac_label: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: L2-normalized embeddings ``[B, D]``.
            labels: class labels ``[B]`` (poisoned samples already use target label).
            scale_weight: beta multiplier for anchors whose label equals ``fac_label``.
            fac_label: facilitator / backdoor target label.
        """
        if features.ndim != 2:
            raise ValueError(f"features must be [B, D], got shape {tuple(features.shape)}")
        if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
            raise ValueError(
                f"labels must be [B] matching features, got {tuple(labels.shape)} "
                f"vs batch {features.shape[0]}"
            )

        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            return features.new_zeros(())

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).to(device=device, dtype=features.dtype)
        mask_scale = mask.clone()
        if fac_label is not None and scale_weight != 1.0:
            fac = labels.view(-1) == int(fac_label)
            if fac.any():
                mask_scale[fac] = mask[fac] * float(scale_weight)

        logits = torch.div(features @ features.T, self.temperature)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        pos_counts = mask.sum(dim=1)
        # Skip anchors with no positives (single-sample classes in the batch).
        valid = pos_counts > 0
        if not bool(valid.any()):
            return features.new_zeros(())

        mean_log_prob_pos = (mask_scale * log_prob).sum(dim=1) / pos_counts.clamp_min(1.0)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss[valid].mean()


def resolve_classifier_name(model: nn.Module) -> str:
    """Locate the classification head used by SafeFL vision backbones."""
    for name in _CLASSIFIER_NAMES:
        module = getattr(model, name, None)
        if isinstance(module, nn.Linear):
            return name

    linear_names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name
    ]
    if not linear_names:
        raise AttributeError(
            f"model {type(model).__name__} has no Linear classifier head "
            f"(expected one of {_CLASSIFIER_NAMES})"
        )
    return linear_names[-1]


def classifier_param_names(model: nn.Module, classifier_name: str) -> List[str]:
    prefix = f"{classifier_name}."
    names = [
        name for name, _ in model.named_parameters()
        if name == classifier_name or name.startswith(prefix)
    ]
    if not names:
        raise AttributeError(f"no parameters found for classifier '{classifier_name}'")
    return names


def split_encoder_classifier_params(
    model: nn.Module,
    classifier_name: Optional[str] = None,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], str]:
    """Return (encoder_params, classifier_params, classifier_name)."""
    head_name = classifier_name or resolve_classifier_name(model)
    head_names = set(classifier_param_names(model, head_name))
    encoder_params: List[nn.Parameter] = []
    classifier_params: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in head_names:
            classifier_params.append(parameter)
        else:
            encoder_params.append(parameter)
    if not encoder_params:
        raise RuntimeError("Chameleon adaptation stage found no trainable encoder parameters")
    if not classifier_params:
        raise RuntimeError("Chameleon projection stage found no trainable classifier parameters")
    return encoder_params, classifier_params, head_name


def extract_features(
    model: nn.Module,
    inputs: torch.Tensor,
    classifier_name: Optional[str] = None,
) -> torch.Tensor:
    """Forward backbone features and L2-normalize them for SupCon."""
    head_name = classifier_name or resolve_classifier_name(model)
    parts = head_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    head = getattr(parent, leaf)
    setattr(parent, leaf, nn.Identity())
    try:
        features = model(inputs)
    finally:
        setattr(parent, leaf, head)

    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    return F.normalize(features, dim=1)


def set_requires_grad(parameters: Iterable[nn.Parameter], requires_grad: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = bool(requires_grad)


@ATTACK_REGISTRY.register("chameleon")
class ChameleonAttack:
    """Training-stage Chameleon attack with BadNets trigger poisoning."""

    client_class = None  # late-bound below to avoid a client/attack import cycle

    def __init__(
        self,
        target_label: int = 0,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        adaptation_epochs: int = 10,
        projection_epochs: Optional[int] = 5,
        fac_scale_weight: float = 2.0,
        temperature: float = 0.07,
        adaptation_lr: Optional[float] = None,
        projection_lr: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        if adaptation_epochs < 0:
            raise ValueError(f"adaptation_epochs must be >= 0, got {adaptation_epochs}")
        if projection_epochs is not None and projection_epochs < 0:
            raise ValueError(f"projection_epochs must be >= 0, got {projection_epochs}")
        if fac_scale_weight <= 0.0:
            raise ValueError(f"fac_scale_weight must be > 0, got {fac_scale_weight}")

        self._badnets = BadNetsAttack(
            target_label=target_label,
            poison_ratio=poison_ratio,
            patch_size=patch_size,
            patch_value=patch_value,
            patch_location=patch_location,
            seed=seed,
        )
        self.target_label = int(target_label)
        self.poison_ratio = poison_ratio
        self.adaptation_epochs = int(adaptation_epochs)
        self.projection_epochs = (
            None if projection_epochs is None else int(projection_epochs)
        )
        self.fac_scale_weight = float(fac_scale_weight)
        self.temperature = float(temperature)
        self.adaptation_lr = adaptation_lr
        self.projection_lr = projection_lr
        self.seed = seed
        self.supcon_loss = SupConLoss(
            temperature=self.temperature,
            base_temperature=self.temperature,
        )

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

    def contrastive_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.supcon_loss(
            features,
            labels,
            scale_weight=self.fac_scale_weight,
            fac_label=self.target_label,
        )

    def resolve_projection_epochs(self, default_epochs: int) -> int:
        if self.projection_epochs is not None:
            return self.projection_epochs
        return int(default_epochs)


from core.client.chameleon_client import ChameleonClient  # noqa: E402

ChameleonAttack.client_class = ChameleonClient
