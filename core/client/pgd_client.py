"""Malicious client for PGD-constrained backdoor training."""
from __future__ import annotations

from typing import Any, Dict

from core.client.malicious_client import MaliciousClient


class PGDClient(MaliciousClient):
    """Project malicious local parameters after every optimizer update."""

    def train(self) -> Dict[str, Any]:
        self.model.train()
        initial_state = {key: value.clone() for key, value in self.model.state_dict().items()}
        self._last_initial_state = initial_state

        optimizer = self.config.trainer_config.build_optimizer(self.model)
        local_epochs = self.config.trainer_config.epochs
        project_fn = getattr(self.attack_profile, "project_model_", None)
        if not callable(project_fn):
            raise RuntimeError("PGDClient requires attack_profile.project_model_")

        total_loss = 0.0
        total_samples = 0
        for epoch_idx in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                data, target, _, skip_backward, skip_step = self._apply_train_hook(
                    "before_forward", optimizer, data, target, epoch_idx, batch_idx
                )
                output = self.model(data)
                loss_val = self.loss(output, target)
                data, target, loss_val, skip_backward, skip_step = self._apply_train_hook(
                    "before_backward", optimizer, data, target, epoch_idx, batch_idx,
                    output, loss_val, skip_backward, skip_step,
                )

                if not skip_backward and loss_val.requires_grad:
                    loss_val.backward()
                else:
                    skip_step = True
                _, _, _, _, skip_step = self._apply_train_hook(
                    "after_backward", optimizer, data, target, epoch_idx, batch_idx,
                    output, loss_val, skip_backward, skip_step,
                )

                if not skip_step:
                    optimizer.step()
                    project_fn(self.model, initial_state)

                total_loss += float(loss_val.detach().item()) * target.size(0)
                total_samples += target.size(0)

        current_state = self.model.state_dict()
        delta = {
            key: current_state[key] - initial_state[key]
            for key in initial_state
            if key in current_state
        }
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}
