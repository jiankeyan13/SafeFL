"""Chameleon two-stage contrastive backdoor attack (ICML 2023, single-file v2)."""
from __future__ import annotations

from typing import Any, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import build_optimizer, train_model

_CLASSIFIER_NAMES = ("linear", "fc", "classifier")


class SupConLoss(nn.Module):
    """Supervised contrastive loss with facilitator up-weighting."""

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0.0 or base_temperature <= 0.0:
            raise ValueError("temperature and base_temperature must be > 0")
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        scale_weight: float = 1.0,
        fac_label: Optional[int] = None,
    ) -> torch.Tensor:
        if features.shape[0] < 2:
            return features.new_zeros(())

        device = features.device
        labels_col = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels_col, labels_col.T).to(device=device, dtype=features.dtype)
        mask_scale = mask.clone()
        if fac_label is not None and scale_weight != 1.0:
            fac = labels.view(-1) == int(fac_label)
            if bool(fac.any()):
                mask_scale[fac] = mask[fac] * float(scale_weight)

        logits = torch.div(features @ features.T, self.temperature)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask
        mask_scale = mask_scale * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))

        pos_counts = mask.sum(dim=1)
        valid = pos_counts > 0
        if not bool(valid.any()):
            return features.new_zeros(())

        mean_log_prob_pos = (mask_scale * log_prob).sum(dim=1) / pos_counts.clamp_min(1.0)
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss[valid].mean()


def resolve_classifier_name(model: nn.Module) -> str:
    for name in _CLASSIFIER_NAMES:
        if isinstance(getattr(model, name, None), nn.Linear):
            return name
    linear_names = [
        name for name, mod in model.named_modules() if isinstance(mod, nn.Linear) and name
    ]
    if not linear_names:
        raise AttributeError(f"Model {type(model).__name__} has no Linear classifier head.")
    return linear_names[-1]


def split_encoder_classifier_params(
    model: nn.Module,
) -> Tuple[List[nn.Parameter], List[nn.Parameter], str]:
    head_name = resolve_classifier_name(model)
    prefix = f"{head_name}."
    encoder_params: List[nn.Parameter] = []
    classifier_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == head_name or name.startswith(prefix):
            classifier_params.append(param)
        else:
            encoder_params.append(param)
    if not encoder_params or not classifier_params:
        raise RuntimeError("Chameleon requires non-empty encoder and classifier parameters.")
    return encoder_params, classifier_params, head_name


def extract_features(model: nn.Module, inputs: torch.Tensor, head_name: str) -> torch.Tensor:
    parts = head_name.split(".")
    parent: nn.Module = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    leaf = parts[-1]
    head = getattr(parent, leaf)
    setattr(parent, leaf, nn.Identity())
    try:
        feats = model(inputs)
    finally:
        setattr(parent, leaf, head)
    if feats.ndim != 2:
        feats = feats.reshape(feats.shape[0], -1)
    return F.normalize(feats, dim=1)


def _set_requires_grad(params: Iterable[nn.Parameter], flag: bool) -> None:
    for p in params:
        p.requires_grad = flag


@ATTACK_REGISTRY.register("chameleon")
class ChameleonAttack(BaseAttacker):
    """Two-stage SupCon encoder adaptation + frozen-encoder CE projection."""

    def __init__(
        self,
        adaptation_epochs: int = 10,
        projection_epochs: Optional[int] = 3,
        fac_scale_weight: float = 2.0,
        temperature: float = 0.07,
        adaptation_lr: Optional[float] = None,
        projection_lr: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        if adaptation_epochs < 0:
            raise ValueError(f"adaptation_epochs must be >= 0, got {adaptation_epochs}")
        if projection_epochs is not None and projection_epochs < 0:
            raise ValueError(f"projection_epochs must be >= 0, got {projection_epochs}")
        if fac_scale_weight <= 0.0:
            raise ValueError(f"fac_scale_weight must be > 0, got {fac_scale_weight}")
        super().__init__(**kwargs)
        self.adaptation_epochs = int(adaptation_epochs)
        self.projection_epochs = None if projection_epochs is None else int(projection_epochs)
        self.fac_scale_weight = float(fac_scale_weight)
        self.temperature = float(temperature)
        self.adaptation_lr = adaptation_lr
        self.projection_lr = projection_lr
        self.supcon_loss = SupConLoss(temperature=self.temperature, base_temperature=self.temperature)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        encoder_params, classifier_params, head_name = split_encoder_classifier_params(ctx.model)
        poison_loader = ctx.loader(poisoner=self.poisoner, seed=self.seed)
        proj_epochs = (
            self.projection_epochs if self.projection_epochs is not None else ctx.epochs
        )
        max_norm = ctx.train_cfg.grad_clip_norm

        total_loss = 0.0
        total_samples = 0

        try:
            # Stage 1: Contrastive adaptation on encoder features.
            if self.adaptation_epochs > 0:
                ctx.model.train()
                _set_requires_grad(encoder_params, True)
                _set_requires_grad(classifier_params, False)
                opt = build_optimizer(encoder_params, ctx.train_cfg, lr=self.adaptation_lr)

                for _ in range(self.adaptation_epochs):
                    for bx, by in poison_loader:
                        if bx.size(0) < 2:
                            continue
                        opt.zero_grad(set_to_none=True)
                        feats = extract_features(ctx.model, bx, head_name)
                        loss_val = self.supcon_loss(
                            feats,
                            by,
                            scale_weight=self.fac_scale_weight,
                            fac_label=self.target_label,
                        )
                        if not loss_val.requires_grad:
                            continue
                        loss_val.backward()
                        if max_norm is not None and max_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(encoder_params, float(max_norm))
                        opt.step()
                        bsz = int(by.size(0))
                        total_loss += float(loss_val.detach().item()) * bsz
                        total_samples += bsz

            # Stage 2: Freeze encoder, train linear classifier with CE.
            if proj_epochs > 0:
                _set_requires_grad(encoder_params, False)
                _set_requires_grad(classifier_params, True)
                stats = train_model(
                    ctx.model,
                    poison_loader,
                    ctx.train_cfg,
                    epochs=proj_epochs,
                    lr=self.projection_lr,
                )
                total_loss += stats.loss * stats.num_samples
                total_samples += stats.num_samples
        finally:
            _set_requires_grad(encoder_params, True)
            _set_requires_grad(classifier_params, True)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return ctx.pack_update(metrics={"loss": avg_loss})
