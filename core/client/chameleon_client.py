"""Malicious client for Chameleon two-stage local training."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from core.client.malicious_client import MaliciousClient


class ChameleonClient(MaliciousClient):
    """Adaptation (SupCon) then projection (frozen-encoder CE) local training."""

    def _build_stage_optimizer(
        self,
        parameters: Iterable[nn.Parameter],
        lr: Optional[float],
    ) -> torch.optim.Optimizer:
        trainer = self.config.trainer_config
        param_list = [parameter for parameter in parameters if parameter.requires_grad]
        if not param_list:
            raise RuntimeError("Chameleon stage optimizer received no trainable parameters")

        learning_rate = float(trainer.lr if lr is None else lr)
        kwargs: Dict[str, Any] = {"lr": learning_rate}
        if trainer.optimizer_name == "SGD":
            kwargs["momentum"] = trainer.momentum
            kwargs["weight_decay"] = trainer.weight_decay
        elif trainer.optimizer_name in {"Adam", "AdamW"}:
            kwargs["weight_decay"] = trainer.weight_decay
        kwargs.update(trainer.extra_params.get("optimizer_kwargs", {}))
        optimizer_class = getattr(torch.optim, trainer.optimizer_name)
        return optimizer_class(param_list, **kwargs)

    def _maybe_clip_grad(self, parameters: List[nn.Parameter]) -> None:
        max_norm = self.config.trainer_config.grad_clip_norm
        if max_norm is None or max_norm <= 0:
            return
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def train(self) -> Dict[str, Any]:
        # Late import avoids the attack<->client registration cycle.
        from core.attack.training.chameleon import (
            extract_features,
            set_requires_grad,
            split_encoder_classifier_params,
        )

        attack = self.attack_profile
        if attack is None:
            return super().train()

        self.model.train()
        initial_state = {key: value.clone() for key, value in self.model.state_dict().items()}
        self._last_initial_state = initial_state

        encoder_params, classifier_params, classifier_name = split_encoder_classifier_params(
            self.model
        )
        adaptation_epochs = int(getattr(attack, "adaptation_epochs", 0))
        if hasattr(attack, "resolve_projection_epochs"):
            projection_epochs = int(
                attack.resolve_projection_epochs(self.config.trainer_config.epochs)
            )
        else:
            projection_epochs = int(self.config.trainer_config.epochs)
        adaptation_lr = getattr(attack, "adaptation_lr", None)
        projection_lr = getattr(attack, "projection_lr", None)

        total_loss = 0.0
        total_samples = 0

        # Stage 1: adaptation with supervised contrastive loss on encoder features.
        if adaptation_epochs > 0:
            set_requires_grad(encoder_params, True)
            set_requires_grad(classifier_params, False)
            optimizer = self._build_stage_optimizer(encoder_params, adaptation_lr)

            for _ in range(adaptation_epochs):
                for data, target in self.train_loader:
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)
                    if data.size(0) < 2:
                        continue

                    optimizer.zero_grad(set_to_none=True)
                    features = extract_features(self.model, data, classifier_name)
                    loss_val = attack.contrastive_loss(features, target)
                    if not torch.is_tensor(loss_val) or not loss_val.requires_grad:
                        continue
                    loss_val.backward()
                    self._maybe_clip_grad(encoder_params)
                    optimizer.step()

                    total_loss += float(loss_val.detach().item()) * target.size(0)
                    total_samples += target.size(0)

        # Stage 2: freeze encoder and train classifier with CE.
        set_requires_grad(encoder_params, False)
        set_requires_grad(classifier_params, True)
        if projection_epochs > 0:
            optimizer = self._build_stage_optimizer(classifier_params, projection_lr)
            for _ in range(projection_epochs):
                for data, target in self.train_loader:
                    data = data.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)
                    output = self.model(data)
                    loss_val = self.loss(output, target)
                    loss_val.backward()
                    self._maybe_clip_grad(classifier_params)
                    optimizer.step()

                    total_loss += float(loss_val.detach().item()) * target.size(0)
                    total_samples += target.size(0)

        # Restore trainability for subsequent rounds / packaging.
        set_requires_grad(encoder_params, True)
        set_requires_grad(classifier_params, True)

        current_state = self.model.state_dict()
        delta = {
            key: current_state[key] - initial_state[key]
            for key in initial_state
            if key in current_state
        }
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}
