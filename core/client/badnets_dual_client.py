"""
BadNets dual-model malicious client.

Trains two independent models from the same global weights Wg:
  - Wc on fully clean local data
  - Wb on BadNets-poisoned local data

Upload delta is the standard BadNets update (Wb - Wg). Every N rounds, each
malicious client also persists Wb-Wc, Wc-Wg, and Wg-Wg_1 under a round folder.
"""
from __future__ import annotations

import logging
import os
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

_logger = logging.getLogger(__name__)


def _state_diff(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Return left - right for shared keys, detached on CPU."""
    out: Dict[str, torch.Tensor] = {}
    for key, left_t in left.items():
        if key not in right or not torch.is_tensor(left_t) or not torch.is_tensor(right[key]):
            continue
        out[key] = (left_t.detach() - right[key].detach()).to("cpu")
    return out


def _clone_state_cpu(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        k: v.detach().to("cpu").clone() if torch.is_tensor(v) else v
        for k, v in state.items()
    }


class BadNetsDualClient(MaliciousClient):
    """MaliciousClient that trains clean and poison branches independently from Wg."""

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
        # Injected by Runner / Ray worker; equals Wg - Wg_1 when available.
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
        self.clean_loader = self._create_clean_train_dataloader()

    def _create_clean_train_dataloader(self) -> DataLoader:
        """Unpoisoned local train data for the clean branch Wc."""
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

    def _should_log_deltas(self) -> bool:
        profile = self.attack_profile
        if profile is None or not getattr(profile, "delta_log_enabled", True):
            return False
        interval = int(getattr(profile, "delta_log_interval", 15))
        if interval <= 0 or self.round_idx is None:
            return False
        # 跳过 round 0 (无 Wg_1); 在 15, 30, 45, ... 落盘
        return self.round_idx > 0 and self.round_idx % interval == 0

    def _resolve_delta_log_dir(self) -> str:
        profile = self.attack_profile
        root = getattr(profile, "delta_log_dir", None) if profile is not None else None
        if not root:
            from core.attack.data.badnets_dual import resolve_delta_log_root

            root = resolve_delta_log_root(None)
        return os.path.join(str(root), f"round_{int(self.round_idx):03d}", str(self.owner_id))

    def _save_round_deltas(
        self,
        wb_state: Dict[str, torch.Tensor],
        wc_state: Dict[str, torch.Tensor],
        wg_state: Dict[str, torch.Tensor],
    ) -> None:
        """Persist Wb-Wc, Wc-Wg, Wg-Wg_1 for this malicious client."""
        save_dir = self._resolve_delta_log_dir()
        os.makedirs(save_dir, exist_ok=True)

        malicious_delta = _state_diff(wb_state, wc_state)
        benign_delta = _state_diff(wc_state, wg_state)
        if self.prev_global_delta is not None:
            global_delta = _clone_state_cpu(self.prev_global_delta)
        else:
            global_delta = {}

        torch.save(malicious_delta, os.path.join(save_dir, "wb_minus_wc.pt"))
        torch.save(benign_delta, os.path.join(save_dir, "wc_minus_wg.pt"))
        torch.save(global_delta, os.path.join(save_dir, "wg_minus_wg1.pt"))
        _logger.info(
            "Saved BadNets dual deltas: round=%s client=%s dir=%s",
            self.round_idx,
            self.owner_id,
            save_dir,
        )

    def train(self) -> Dict[str, Any]:
        if self.clean_loader is None:
            raise RuntimeError("BadNetsDualClient.clean_loader is not initialized")

        # Wg: broadcast global weights (training start)
        wg_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._last_initial_state = wg_state

        # 1) Independent clean train from Wg -> Wc
        self._train_one_model(self.model, self.clean_loader, apply_hooks=False)
        wc_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        # 2) Reload Wg, independent poison train -> Wb
        self.model.load_state_dict(wg_state, strict=True)
        avg_loss = self._train_one_model(self.model, self.train_loader, apply_hooks=True)
        wb_state = self.model.state_dict()

        # Standard BadNets upload delta: Wb - Wg
        delta = {
            k: wb_state[k] - wg_state[k]
            for k in wg_state
            if k in wb_state
        }

        if self._should_log_deltas():
            self._save_round_deltas(wb_state, wc_state, wg_state)

        # Free Wc snapshot ASAP
        del wc_state

        return {"train_loss": avg_loss, "delta": delta}
