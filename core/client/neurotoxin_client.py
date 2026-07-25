"""Malicious client integration for the Neurotoxin training attack."""
from __future__ import annotations

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


class NeurotoxinClient(MaliciousClient):
    """Build a clean-batch gradient mask before poisoned local training."""

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

    def _create_clean_train_dataloader(self) -> DataLoader:
        """Unpoisoned local train data for clean gradient estimation."""
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

    def _compute_clean_batch_gradients(self) -> Dict[str, torch.Tensor]:
        # Lazy import to avoid neurotoxin <-> neurotoxin_client cycle.
        from core.attack.training.neurotoxin import compute_clean_batch_gradients

        if self.clean_loader is None:
            return {}
        data, target = next(iter(self.clean_loader))
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        return compute_clean_batch_gradients(
            model=self.model,
            data=data,
            target=target,
            loss_fn=self.loss,
        )

    def train(self) -> Dict[str, Any]:
        prepare = getattr(self.attack_profile, "prepare_gradient_mask", None)
        if callable(prepare):
            prepare(self.model, self._compute_clean_batch_gradients())
        return super().train()
