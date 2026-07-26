"""
LP malicious client aligned with method_1-style LP flow.

Each selected round:
1) Train local_ep benign/malicious pair from Wg.
2) FLS + BLS(tau) on that pair to pick BC layers (no LSA long-train, no cache).
3) Craft upload in poison_upload (lambda=1 by default).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core.client.base_client import BaseClient
from core.client.malicious_client import MaliciousClient
from core.config import ClientConfig
from core.utils.evaluator import Evaluator
from data.constants import SPLIT_TRAIN
from data.dataset_store import DatasetStore
from data.task import TaskSet


class LPClient(MaliciousClient):
    """MaliciousClient with per-round FLS+BLS selection and local_ep craft."""

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
        self.clean_train_loader: Optional[DataLoader] = None
        self.clean_val_loader: Optional[DataLoader] = None
        self.poison_train_loader: Optional[DataLoader] = None
        self.poison_val_loader: Optional[DataLoader] = None
        self._layer_selection_record: Optional[Dict[str, Any]] = None
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
        self._setup_lp_loaders()

    def _attr(self, name: str, default: Any) -> Any:
        if self.attack_profile is None:
            return default
        return getattr(self.attack_profile, name, default)

    def _val_ratio(self) -> float:
        return float(self._attr("val_ratio", 0.2))

    def _setup_lp_loaders(self) -> None:
        """Split local train into train/val; build clean + poison loaders."""
        clean_store = BaseClient._build_dataset(self, SPLIT_TRAIN)
        if clean_store is None:
            raise RuntimeError(f"Client {self.owner_id} has no clean training data.")

        full_ds = clean_store.dataset
        n = len(full_ds)
        if n < 2:
            raise RuntimeError(
                f"Client {self.owner_id} needs at least 2 samples for LP train/val split."
            )

        val_ratio = self._val_ratio()
        if not 0.0 < val_ratio < 1.0:
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

        rng = np.random.default_rng(
            None if self.attack_profile is None else getattr(self.attack_profile, "seed", None)
        )
        perm = rng.permutation(n)
        n_val = max(1, int(n * val_ratio))
        n_val = min(n_val, n - 1)
        val_indices = perm[:n_val].tolist()
        train_indices = perm[n_val:].tolist()

        clean_train_ds = Subset(full_ds, train_indices)
        clean_val_ds = Subset(full_ds, val_indices)

        if self.attack_profile is None or not hasattr(self.attack_profile, "poison_dataset"):
            raise RuntimeError("LPClient requires attack_profile.poison_dataset")

        poison_train_ds = self.attack_profile.poison_dataset(
            clean_train_ds,
            mode="train",
            split=SPLIT_TRAIN,
            client_id=self.owner_id,
            round_idx=self.round_idx,
        )
        poison_val_ds = self.attack_profile.poison_dataset(
            clean_val_ds,
            mode="test",
            split="val",
            client_id=self.owner_id,
            round_idx=self.round_idx,
        )

        loader_kwargs = dict(
            batch_size=self.config.batch_size,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )
        self.clean_train_loader = DataLoader(
            clean_train_ds, shuffle=True, **loader_kwargs
        )
        self.clean_val_loader = DataLoader(
            clean_val_ds, shuffle=False, **loader_kwargs
        )
        self.poison_train_loader = DataLoader(
            poison_train_ds, shuffle=True, **loader_kwargs
        )
        self.poison_val_loader = DataLoader(
            poison_val_ds, shuffle=False, **loader_kwargs
        )
        self.train_loader = self.poison_train_loader

    def _train_one_model(
        self,
        model: nn.Module,
        loader: DataLoader,
        num_epochs: int,
        apply_hooks: bool,
    ) -> float:
        """Run num_epochs on the given model/loader. Returns average loss."""
        if num_epochs <= 0:
            return 0.0

        model.train()
        optimizer = self.config.trainer_config.build_optimizer(model)
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

    def _clone_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            k: v.detach().clone() if torch.is_tensor(v) else v
            for k, v in model.state_dict().items()
        }

    def train(self) -> Dict[str, Any]:
        if self.clean_train_loader is None or self.poison_train_loader is None:
            raise RuntimeError("LPClient loaders are not initialized")
        if self.clean_val_loader is None or self.poison_val_loader is None:
            raise RuntimeError("LPClient val loaders are not initialized")
        if self.attack_profile is None:
            raise RuntimeError("LPClient requires attack_profile")
        for required in (
            "cache_benign_state",
            "cache_malicious_state",
            "identify_bc_layers",
        ):
            if not hasattr(self.attack_profile, required):
                raise RuntimeError(f"LPClient requires attack_profile.{required}")

        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self._last_initial_state = initial_state
        local_epochs = int(self.config.trainer_config.epochs)

        # 1) local_ep benign / malicious pair from the same Wg.
        self.model.load_state_dict(initial_state, strict=False)
        self._train_one_model(
            self.model,
            self.clean_train_loader,
            num_epochs=local_epochs,
            apply_hooks=False,
        )
        benign_state = self._clone_state(self.model)
        self.attack_profile.cache_benign_state(self.owner_id, benign_state)

        self.model.load_state_dict(initial_state, strict=False)
        avg_loss = self._train_one_model(
            self.model,
            self.poison_train_loader,
            num_epochs=local_epochs,
            apply_hooks=True,
        )
        malicious_state = self._clone_state(self.model)
        self.attack_profile.cache_malicious_state(self.owner_id, malicious_state)

        # 2) Every selected round: re-select BC layers via FLS + BLS(tau).
        _, self._layer_selection_record = self.attack_profile.identify_bc_layers(
            self.model,
            client_id=self.owner_id,
            benign_state=benign_state,
            malicious_state=malicious_state,
            val_loader=self.poison_val_loader,
            device=self.device,
            round_idx=self.round_idx,
        )

        current_state = self.model.state_dict()
        delta = {
            k: current_state[k] - initial_state[k]
            for k in initial_state
            if k in current_state
        }
        return {"train_loss": avg_loss, "delta": delta}

    def package(self, train_metrics: Dict[str, Any]) -> Dict[str, Any]:
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
            num_samples=len(self.poison_train_loader.dataset)
            if self.poison_train_loader is not None
            else 0,
        )

        payload = {
            "client_id": self.owner_id,
            "delta": final_delta,
            "metrics": train_metrics["train_loss"],
            "num_samples": 1,
        }
        if self._layer_selection_record is not None:
            payload["layer_selection"] = self._layer_selection_record
        return payload
