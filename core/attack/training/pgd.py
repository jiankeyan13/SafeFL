"""PGD-constrained model-poisoning attack for federated learning.

The malicious objective is learned from a trigger-poisoned local dataset. PGD
is the stage-2 intervention: after the last batch of each local epoch, model
parameters are projected into an L2 ball around the round-start global model.

Threshold (epsilon):

    epsilon = 0.1 * ||w_0||_2   for CIFAR-10
    epsilon = ||w_0||_2         for other datasets

If ||w_local - w_0||_2 > epsilon:

    w_proj = w_0 + epsilon * (w_local - w_0) / ||w_local - w_0||_2
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY


def _is_cifar10_dataset(dataset: Optional[str]) -> bool:
    if dataset is None:
        return False
    return dataset.lower().strip() == "cifar10"


def compute_parameter_l2_norm(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
) -> float:
    """Return the L2 norm of trainable parameters in ``state``."""
    squared_norm: Optional[torch.Tensor] = None
    for name, parameter in model.named_parameters():
        reference = state.get(name)
        if reference is None or not torch.is_tensor(reference):
            raise KeyError(f"state is missing trainable parameter '{name}'")
        if reference.shape != parameter.shape:
            raise ValueError(
                f"shape mismatch for '{name}': model={tuple(parameter.shape)}, "
                f"state={tuple(reference.shape)}"
            )
        value = reference.detach().to(device=parameter.device, dtype=torch.float64)
        term = value.square().sum()
        squared_norm = term if squared_norm is None else squared_norm + term
    if squared_norm is None:
        return 0.0
    return float(squared_norm.sqrt().item())


def resolve_pgd_epsilon(
    model: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    dataset: Optional[str] = None,
    cifar10_epsilon_scale: float = 0.1,
    default_epsilon_scale: float = 1.0,
) -> float:
    """Compute PGD radius from ||w_0||_2 and dataset-specific scaling."""
    if not math.isfinite(cifar10_epsilon_scale) or cifar10_epsilon_scale < 0.0:
        raise ValueError(
            f"cifar10_epsilon_scale must be finite and >= 0, got {cifar10_epsilon_scale}"
        )
    if not math.isfinite(default_epsilon_scale) or default_epsilon_scale < 0.0:
        raise ValueError(
            f"default_epsilon_scale must be finite and >= 0, got {default_epsilon_scale}"
        )
    w0_norm = compute_parameter_l2_norm(model, global_state)
    scale = cifar10_epsilon_scale if _is_cifar10_dataset(dataset) else default_epsilon_scale
    return scale * w0_norm


@torch.no_grad()
def project_model_l2_(
    model: nn.Module,
    global_state: Mapping[str, torch.Tensor],
    epsilon: float,
) -> Tuple[float, float]:
    """Project trainable parameters in-place around ``global_state``.

    Returns the pre-projection L2 distance and the applied scale. Model buffers
    (for example BatchNorm running statistics) are intentionally excluded,
    matching the usual PGD constraint over optimization variables.
    """
    if not math.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError(f"epsilon must be finite and >= 0, got {epsilon}")

    parameter_deltas = []
    for name, parameter in model.named_parameters():
        reference = global_state.get(name)
        if reference is None or not torch.is_tensor(reference):
            raise KeyError(f"round-start global state is missing parameter '{name}'")
        if reference.shape != parameter.shape:
            raise ValueError(
                f"shape mismatch for '{name}': model={tuple(parameter.shape)}, "
                f"global={tuple(reference.shape)}"
            )
        reference = reference.to(device=parameter.device, dtype=parameter.dtype)
        parameter_deltas.append((parameter, reference, parameter - reference))

    if not parameter_deltas:
        return 0.0, 1.0

    squared_norm = torch.zeros((), device=parameter_deltas[0][0].device, dtype=torch.float64)
    for _, _, delta in parameter_deltas:
        squared_norm.add_(delta.detach().double().square().sum())
    distance = float(squared_norm.sqrt().item())
    scale = min(1.0, epsilon / distance) if distance > 0.0 else 1.0

    if scale < 1.0:
        for parameter, reference, delta in parameter_deltas:
            parameter.copy_(reference + delta * scale)
    return distance, scale


@ATTACK_REGISTRY.register("pgd")
class PGDAttack:
    """Backdoor training constrained by an L2 PGD projection.

    ``poison_dataset`` supplies the trigger objective. ``PGDClient`` performs
    the actual stage-2 projection. No upload-stage tampering is implemented.
    """

    client_class = None

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        dataset: Optional[str] = None,
        cifar10_epsilon_scale: float = 0.1,
        default_epsilon_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
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
        self.dataset = dataset
        self.cifar10_epsilon_scale = float(cifar10_epsilon_scale)
        self.default_epsilon_scale = float(default_epsilon_scale)
        self.seed = seed

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

    def project_model_(
        self,
        model: nn.Module,
        global_state: Mapping[str, torch.Tensor],
    ) -> Tuple[float, float]:
        epsilon = resolve_pgd_epsilon(
            model,
            global_state,
            dataset=self.dataset,
            cifar10_epsilon_scale=self.cifar10_epsilon_scale,
            default_epsilon_scale=self.default_epsilon_scale,
        )
        return project_model_l2_(model, global_state, epsilon)


from core.client.pgd_client import PGDClient  # noqa: E402

PGDAttack.client_class = PGDClient
