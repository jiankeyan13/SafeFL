"""
LGA malicious client: poison train with per-epoch layer-wise update projection.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from core.client.malicious_client import MaliciousClient
from core.config import ClientConfig
from core.utils.evaluator import Evaluator
from data.dataset_store import DatasetStore
from data.task import TaskSet


class LGAClient(MaliciousClient):
    """
    MaliciousClient that applies LGA getNewW after every local epoch.

    Runner may set ``prev_global_delta`` (previous round aggregated delta) before
    ``step()``; when absent (e.g. round 0), projection is skipped.
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
        attack_profile: Optional[Any] = None,
        round_idx: Optional[int] = None,
    ):
        # 由 Runner / Ray worker 在创建后按需注入; 默认 None
        self.prev_global_delta: Optional[Dict[str, torch.Tensor]] = None
        super().__init__(
            client_id=client_id,
            task_set=task_set,
            stores=stores,
            model=model,
            device=device,
            config=config,
            evaluator=evaluator,
            attack_profile=attack_profile,
            round_idx=round_idx,
        )

    def train(self) -> Dict[str, Any]:
        self.model.train()

        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._last_initial_state = initial_state

        optimizer = self.config.trainer_config.build_optimizer(self.model)
        local_epochs = self.config.trainer_config.epochs

        total_loss = 0.0
        total_samples = 0
        prev_delta = self.prev_global_delta
        project_fn = getattr(self.attack_profile, "project_layerwise", None)

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
                    "before_backward",
                    optimizer,
                    data,
                    target,
                    epoch_idx,
                    batch_idx,
                    output,
                    loss_val,
                    skip_backward,
                    skip_step,
                )

                if not skip_backward and loss_val.requires_grad:
                    loss_val.backward()
                else:
                    skip_step = True

                _, _, _, _, skip_step = self._apply_train_hook(
                    "after_backward",
                    optimizer,
                    data,
                    target,
                    epoch_idx,
                    batch_idx,
                    output,
                    loss_val,
                    skip_backward,
                    skip_step,
                )

                if not skip_step:
                    optimizer.step()

                total_loss += float(loss_val.detach().item()) * target.size(0)
                total_samples += target.size(0)

            # 每个 local epoch 结束后做 layer-wise 缩放 (对齐官方 train_ours)
            if prev_delta is not None and callable(project_fn):
                projected = project_fn(
                    initial_state,
                    self.model.state_dict(),
                    prev_delta,
                )
                self.model.load_state_dict(projected, strict=False)

        current_state = self.model.state_dict()
        delta = {
            k: current_state[k] - initial_state[k]
            for k in initial_state
            if k in current_state
        }
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}
