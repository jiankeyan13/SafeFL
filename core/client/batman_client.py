"""
Batman malicious client: parallel clean/poison local training.

Both branches start from the same global weights Wg:
  - clean branch trains on unpoisoned data -> Wc
  - poison branch trains independently on triggered data -> Wp
Upload alignment is handled by BatmanAttack.poison_upload.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Union

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


class BatmanClient(MaliciousClient):
    """MaliciousClient subclass with clean-reference training for Batman."""

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

    def _build_dataset(self, split: str) -> Optional[DatasetStore]:
        """Poison train data for the malicious branch; leave other splits to parent."""
        return super()._build_dataset(split)

    def _create_clean_train_dataloader(self) -> DataLoader:
        """Unpoisoned local train data for the clean reference branch."""
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

    def _clone_model_from_state(self, state: Dict[str, torch.Tensor]) -> nn.Module:
        """Build a same-architecture clone at ``state`` without sharing storage."""
        clone = copy.deepcopy(self.model)
        clone.load_state_dict(state)
        clone.to(self.device)
        return clone

    def _train_one_model(
        self,
        model: nn.Module,
        loader: DataLoader,
        apply_hooks: bool,
    ) -> float:
        """Run local epochs on the given model/loader. Returns average loss."""
        model.train()
        optimizer = self.config.trainer_config.build_optimizer(model)
        local_epochs = self.config.trainer_config.epochs
        total_loss = 0.0
        total_samples = 0

        for epoch_idx in range(local_epochs):
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

                output = model(data)
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

        return total_loss / total_samples if total_samples > 0 else 0.0

    def train(self) -> Dict[str, Any]:
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._last_initial_state = initial_state

        if self.clean_loader is None:
            raise RuntimeError("BatmanClient.clean_loader is not initialized")
        if self.attack_profile is None or not hasattr(self.attack_profile, "cache_clean_state"):
            raise RuntimeError("BatmanClient requires attack_profile.cache_clean_state")

        # 1) Clean branch: Wg -> Wc on unpoisoned data (independent clone)
        clean_model = self._clone_model_from_state(initial_state)
        self._train_one_model(clean_model, self.clean_loader, apply_hooks=False)
        self.attack_profile.cache_clean_state(self.owner_id, clean_model.state_dict())

        # 2) Poison branch: Wg -> Wp independently (self.model remains at Wg)
        avg_loss = self._train_one_model(self.model, self.train_loader, apply_hooks=True)

        current_state = self.model.state_dict()
        delta = {
            k: current_state[k] - initial_state[k]
            for k in initial_state
            if k in current_state
        }
        return {"train_loss": avg_loss, "delta": delta}

    def package(self, train_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Run poison_upload alignment and package the client update."""
        upload_delta = {
            k: v.clone() if torch.is_tensor(v) else v
            for k, v in train_metrics["delta"].items()
        }

        final_delta = self._call_attack(
            "poison_upload",
            upload_delta,
            update=upload_delta,
            initial_weights=self._last_initial_state,
            client_id=self.owner_id,
            round_idx=self.round_idx,
            num_samples=len(self.train_loader.dataset),
        )

        return {
            "client_id": self.owner_id,
            "delta": final_delta,
            "metrics": train_metrics["train_loss"],
            "num_samples": 1,
        }
