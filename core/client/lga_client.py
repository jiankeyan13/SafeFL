"""
LGA malicious client: clean pretrain, then poison train with per-epoch projection.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.client.base_client import BaseClient
from core.client.malicious_client import MaliciousClient
from core.config import ClientConfig
from core.utils.evaluator import Evaluator
from data.constants import SPLIT_TRAIN
from data.dataset_store import DatasetStore
from data.task import TaskSet


class LGAClient(MaliciousClient):
    """
    MaliciousClient with two-phase local training:

    1. Clean phase: train on unpoisoned data for ``epoch_clean`` epochs (no LGA).
    2. Poison phase: continue on poisoned data for ``epochs`` (LGA after each epoch).

    Runner may set ``prev_global_delta`` before ``step()``; when absent, LGA projection is skipped.
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
        self.prev_global_delta: Optional[Dict[str, torch.Tensor]] = None
        self.clean_loader: Optional[DataLoader] = None
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
        self.clean_loader = self._create_clean_train_dataloader()

    def _create_clean_train_dataloader(self) -> DataLoader:
        """Unpoisoned local train data for the clean pretrain phase."""
        clean_store = BaseClient._build_dataset(self, SPLIT_TRAIN)
        if clean_store is None:
            raise RuntimeError(f"Client {self.owner_id} has no clean training data.")
        return DataLoader(
            clean_store.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def _train_epochs(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        *,
        apply_hooks: bool,
        apply_lga: bool,
        global_state: Mapping[str, torch.Tensor],
        prev_delta: Optional[Mapping[str, torch.Tensor]],
        project_fn: Optional[Callable[..., Dict[str, torch.Tensor]]],
    ) -> tuple[float, int]:
        """Run ``num_epochs`` on ``loader``; optionally apply LGA after each epoch."""
        total_loss = 0.0
        total_samples = 0

        for epoch_idx in range(num_epochs):
            for batch_idx, (data, target) in enumerate(loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                skip_backward = False
                skip_step = False

                if apply_hooks:
                    data, target, _, skip_backward, skip_step = self._apply_train_hook(
                        "before_forward", optimizer, data, target, epoch_idx, batch_idx
                    )

                output = self.model(data)
                loss_val = self.loss(output, target)

                if apply_hooks:
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

                if apply_hooks:
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

            if apply_lga and prev_delta is not None and callable(project_fn):
                projected = project_fn(
                    global_state,
                    self.model.state_dict(),
                    prev_delta,
                )
                self.model.load_state_dict(projected, strict=False)

        return total_loss, total_samples

    def train(self) -> Dict[str, Any]:
        self.model.train()

        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._last_initial_state = initial_state

        if self.clean_loader is None:
            raise RuntimeError("LGAClient.clean_loader is not initialized")
        if self.attack_profile is None:
            raise RuntimeError("LGAClient requires attack_profile")

        epoch_clean = int(getattr(self.attack_profile, "epoch_clean", 0))
        if epoch_clean < 0:
            raise ValueError(f"epoch_clean must be >= 0, got {epoch_clean}")

        poison_epochs = self.config.trainer_config.epochs
        optimizer = self.config.trainer_config.build_optimizer(self.model)
        prev_delta = self.prev_global_delta
        project_fn = getattr(self.attack_profile, "project_layerwise", None)

        total_loss = 0.0
        total_samples = 0

        if epoch_clean > 0:
            clean_loss, clean_samples = self._train_epochs(
                self.clean_loader,
                optimizer,
                epoch_clean,
                apply_hooks=False,
                apply_lga=False,
                global_state=initial_state,
                prev_delta=prev_delta,
                project_fn=project_fn,
            )
            total_loss += clean_loss
            total_samples += clean_samples

        poison_loss, poison_samples = self._train_epochs(
            self.train_loader,
            optimizer,
            poison_epochs,
            apply_hooks=True,
            apply_lga=True,
            global_state=initial_state,
            prev_delta=prev_delta,
            project_fn=project_fn,
        )
        total_loss += poison_loss
        total_samples += poison_samples

        current_state = self.model.state_dict()
        delta = {
            k: current_state[k] - initial_state[k]
            for k in initial_state
            if k in current_state
        }
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}
