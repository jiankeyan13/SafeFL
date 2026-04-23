from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from core.client.base_client import BaseClient
from core.utils.configs import ClientConfig
from core.utils.evaluator import Evaluator
from data.dataset_store import DatasetStore
from data.task import TaskSet


class MaliciousClient(BaseClient):
    """
    Composable malicious client with 3 attack stages:
    1) data poisoning via poison_dataset
    2) training intervention via poison_train
    3) upload tampering via poison_upload
    """

    def __init__(self, client_id: Union[int, str], task_set: TaskSet, stores: Dict[str, DatasetStore],
                 model: nn.Module, device: torch.device, config: ClientConfig,
                 evaluator: Optional[Evaluator] = None, attack_profile: Optional[Any] = None,
                 round_idx: Optional[int] = None):
        self.attack_profile = attack_profile
        self.round_idx = round_idx
        self._last_initial_state: Dict[str, torch.Tensor] = {}
        super().__init__(client_id=client_id, task_set=task_set, stores=stores, model=model,
                         device=device, config=config, evaluator=evaluator)

    def _call_attack(self, method: str, default: Any, *args: Any, **kwargs: Any) -> Any:
        """Call attack method if available, otherwise return default."""
        if self.attack_profile is None:
            return default
        attack_method = getattr(self.attack_profile, method, None)
        if not callable(attack_method):
            return default
        return attack_method(*args, **kwargs) or default

    def _build_dataset(self, split: str) -> Optional[DatasetStore]:
        ds_store = super()._build_dataset(split)
        if ds_store is None:
            return None

        poisoned = self._call_attack(
            "poison_dataset", ds_store.dataset, ds_store.dataset,
            mode=split, split=split, client_id=self.owner_id, round_idx=self.round_idx
        )

        return DatasetStore(name=ds_store.name, split=ds_store.split, dataset=poisoned)

    def _run_train_hook(self, *, hook_point: str, optimizer: torch.optim.Optimizer,
                        data: torch.Tensor, target: torch.Tensor, local_epoch: int, batch_idx: int,
                        output: Optional[torch.Tensor] = None, loss_val: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        result = self._call_attack(
            "poison_train", {}, model=self.model, optimizer=optimizer,
            loss_fn=self.loss, data=data, target=target, output=output,
            loss=loss_val, hook_point=hook_point, local_epoch=local_epoch,
            batch_idx=batch_idx, client_id=self.owner_id,
            round_idx=self.round_idx, device=self.device
        )

        return {
            "data": result.get("data", data),
            "target": result.get("target", target),
            "loss": result.get("loss", loss_val),
            "skip_backward": result.get("skip_backward", False),
            "skip_step": result.get("skip_step", False),
        }

    def _apply_train_hook(self, hook_point: str, optimizer: torch.optim.Optimizer,
                          data: torch.Tensor, target: torch.Tensor, local_epoch: int, batch_idx: int,
                          output: Optional[torch.Tensor] = None, loss_val: Optional[torch.Tensor] = None,
                          skip_backward: bool = False, skip_step: bool = False) -> tuple:
        """Apply training hook and return updated state."""
        state = self._run_train_hook(
            hook_point=hook_point, optimizer=optimizer,
            data=data, target=target, local_epoch=local_epoch,
            batch_idx=batch_idx, output=output, loss_val=loss_val
        )
        loss = state["loss"] if state["loss"] is not None else loss_val
        return (
            state["data"], state["target"], loss,
            skip_backward or state["skip_backward"],
            skip_step or state["skip_step"]
        )

    def train(self) -> Dict[str, Any]:
        self.model.train()

        # S2 优化: 保留在 GPU 上 clone, 避免 CPU 搬运（含 BN 统计量，与 BaseClient 一致）
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._last_initial_state = initial_state

        optimizer = self.config.trainer_config.build_optimizer(self.model)
        local_epochs = self.config.trainer_config.epochs

        total_loss = 0.0
        total_samples = 0

        for epoch_idx in range(local_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                data, target, _, skip_backward, skip_step = self._apply_train_hook(
                    "before_forward", optimizer, data, target, epoch_idx, batch_idx)

                output = self.model(data)
                loss_val = self.loss(output, target)

                data, target, loss_val, skip_backward, skip_step = self._apply_train_hook(
                    "before_backward", optimizer, data, target, epoch_idx, batch_idx,
                    output, loss_val, skip_backward, skip_step)

                if not skip_backward and loss_val.requires_grad:
                    loss_val.backward()
                else:
                    skip_step = True

                _, _, _, _, skip_step = self._apply_train_hook(
                    "after_backward", optimizer, data, target, epoch_idx, batch_idx,
                    output, loss_val, skip_backward, skip_step)

                if not skip_step:
                    optimizer.step()

                total_loss += float(loss_val.detach().item()) * target.size(0)
                total_samples += target.size(0)

        # S2 优化: GPU 内直接计算 delta
        current_state = self.model.state_dict()
        delta = {
            k: current_state[k] - initial_state[k]
            for k in initial_state if k in current_state
        }
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}

    def package(self, train_metrics: Dict[str, Any]) -> Dict[str, Any]:
        upload_delta = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in train_metrics["delta"].items()
        }

        final_delta = self._call_attack(
            "poison_upload", upload_delta,
            update=upload_delta, initial_weights=self._last_initial_state,
            client_id=self.owner_id, round_idx=self.round_idx,
            num_samples=len(self.train_loader.dataset)
        )

        return {
            "client_id": self.owner_id,
            "delta": final_delta,
            "metrics": train_metrics["train_loss"],
        }
