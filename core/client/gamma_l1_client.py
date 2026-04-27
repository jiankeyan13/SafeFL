from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from core.client.base_client import BaseClient
from core.utils.configs import ClientConfig
from core.utils.evaluator import Evaluator
from data.dataset_store import DatasetStore
from data.task import TaskSet


class GammaL1Client(BaseClient):
    """
    在任务损失上加入 BN gamma 的 L1 正则:
        L_client = L_task + lambda_i * sum(|gamma_i|)
    """

    def __init__(
        self,
        client_id: Union[int, str],
        task_set: TaskSet,
        stores: Dict[str, DatasetStore],
        model: nn.Module,
        device: torch.device,
        config: ClientConfig,
        evaluator: Optional[Evaluator] = None,
        gamma_l1_lambda: float = 1e-4,
        gamma_l1_lambda_by_client: Optional[Mapping[str, float]] = None,
        gamma_l1_lambda_by_width: Optional[Mapping[Union[str, float], float]] = None,
    ):
        self._default_gamma_l1_lambda = float(gamma_l1_lambda)
        self._gamma_l1_lambda_by_client = dict(gamma_l1_lambda_by_client or {})
        self._gamma_l1_lambda_by_width = {
            float(width): float(value)
            for width, value in (gamma_l1_lambda_by_width or {}).items()
        }
        super().__init__(
            client_id=client_id,
            task_set=task_set,
            stores=stores,
            model=model,
            device=device,
            config=config,
            evaluator=evaluator,
        )
        self.gamma_l1_lambda = self._resolve_gamma_l1_lambda()

    def _resolve_gamma_l1_lambda(self) -> float:
        if self.owner_id in self._gamma_l1_lambda_by_client:
            return float(self._gamma_l1_lambda_by_client[self.owner_id])

        width_ratio = getattr(self.model, "width_ratio", None)
        if width_ratio is not None:
            width = float(width_ratio)
            for configured_width, value in self._gamma_l1_lambda_by_width.items():
                if abs(configured_width - width) < 1e-12:
                    return float(value)

        return self._default_gamma_l1_lambda

    def _gamma_l1_penalty(self) -> torch.Tensor:
        penalty: Optional[torch.Tensor] = None
        for module in self.model.modules():
            if not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                continue
            if module.weight is None:
                continue
            term = module.weight.abs().sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return torch.zeros((), device=self.device)
        return penalty

    def _client_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        task_loss = self.loss(output, target)
        if self.gamma_l1_lambda == 0.0:
            return task_loss
        return task_loss + self.gamma_l1_lambda * self._gamma_l1_penalty()

    def train(self) -> Dict[str, Any]:
        self.model.train()

        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        optimizer = self.config.trainer_config.build_optimizer(self.model)
        local_epochs = self.config.trainer_config.epochs

        total_loss = 0.0
        total_samples = 0

        for _ in range(local_epochs):
            for data, target in self.train_loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = self.model(data)
                loss_val = self._client_loss(output, target)
                loss_val.backward()
                optimizer.step()

                total_loss += loss_val.item() * target.size(0)
                total_samples += target.size(0)

        current_state = self.model.state_dict()
        delta = {
            k: current_state[k] - initial_state[k]
            for k in initial_state
            if k in current_state
        }
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}
