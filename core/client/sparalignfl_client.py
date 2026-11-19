"""SparAlignFL malicious client: probe and concentrate critical layers."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

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


class SparAlignFLClient(MaliciousClient):
    """Two-stage local attack: contrastive locate and masked concentrate."""

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
        self.probe_poison_loader: Optional[DataLoader] = None
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
        self._setup_probe_loaders()

    def _loader_kwargs(self) -> Dict[str, Any]:
        return dict(
            batch_size=self.config.batch_size,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def _setup_probe_loaders(self) -> None:
        clean_store = BaseClient._build_dataset(self, SPLIT_TRAIN)
        if clean_store is None:
            raise RuntimeError(f"Client {self.owner_id} has no clean training data.")
        if self.attack_profile is None or not hasattr(self.attack_profile, "build_probe_poison_dataset"):
            raise RuntimeError("SparAlignFLClient requires attack_profile.build_probe_poison_dataset")

        loader_kwargs = self._loader_kwargs()
        self.clean_loader = DataLoader(clean_store.dataset, shuffle=True, **loader_kwargs)
        probe_dataset = self.attack_profile.build_probe_poison_dataset(
            clean_store.dataset,
            client_id=self.owner_id,
            round_idx=self.round_idx,
        )
        self.probe_poison_loader = DataLoader(probe_dataset, shuffle=True, **loader_kwargs)

    def _clone_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in model.state_dict().items()
        }

    def _set_trainable_parameters(self, trainable_names: Optional[Sequence[str]]) -> None:
        allowed = None if trainable_names is None else set(trainable_names)
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = allowed is None or name in allowed

    def _apply_grad_masks(self, masks: Mapping[str, torch.Tensor]) -> None:
        named = dict(self.model.named_parameters())
        for name, mask in masks.items():
            parameter = named.get(name)
            if parameter is None or parameter.grad is None:
                continue
            parameter.grad.mul_(mask.to(device=parameter.grad.device, dtype=parameter.grad.dtype))

    def _train_epochs(
        self,
        loader: DataLoader,
        num_epochs: int,
        *,
        trainable_names: Optional[Sequence[str]] = None,
        grad_masks: Optional[Mapping[str, torch.Tensor]] = None,
        align_layerwise: bool = False,
        global_state: Optional[Mapping[str, torch.Tensor]] = None,
        prev_global_delta: Optional[Mapping[str, torch.Tensor]] = None,
        project_fn: Optional[Callable[..., Dict[str, torch.Tensor]]] = None,
        project_skip_keys: Optional[Sequence[str]] = None,
    ) -> float:
        if num_epochs <= 0:
            return 0.0

        self.model.train()
        self._set_trainable_parameters(trainable_names)
        optimizer = self.config.trainer_config.build_optimizer(self.model)
        total_loss = 0.0
        total_samples = 0

        for _ in range(num_epochs):
            for data, target in loader:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = self.model(data)
                loss_val = self.loss(output, target)
                if not loss_val.requires_grad:
                    continue
                loss_val.backward()
                if grad_masks:
                    self._apply_grad_masks(grad_masks)
                optimizer.step()
                total_loss += float(loss_val.detach().item()) * target.size(0)
                total_samples += target.size(0)

            if (
                align_layerwise
                and global_state is not None
                and prev_global_delta is not None
                and callable(project_fn)
            ):
                projected = project_fn(
                    global_state,
                    self.model.state_dict(),
                    prev_global_delta,
                    skip_keys=project_skip_keys,
                )
                self.model.load_state_dict(projected, strict=False)

        self._set_trainable_parameters(None)
        return total_loss / total_samples if total_samples > 0 else 0.0

    def train(self) -> Dict[str, Any]:
        if self.clean_loader is None or self.probe_poison_loader is None:
            raise RuntimeError("SparAlignFLClient probe loaders are not initialized")
        if self.attack_profile is None:
            raise RuntimeError("SparAlignFLClient requires attack_profile")
        required = ["locate_critical_layers", "build_masks"]
        for name in required:
            if not hasattr(self.attack_profile, name):
                raise RuntimeError(f"SparAlignFLClient requires attack_profile.{name}")

        initial_state = self._clone_state(self.model)
        self._last_initial_state = initial_state
        probe_epochs = int(getattr(self.attack_profile, "probe_epochs", self.config.trainer_config.epochs))
        local_epochs = int(self.config.trainer_config.epochs)
        prev_global_delta = self.prev_global_delta
        project_fn = getattr(self.attack_profile, "project_layerwise", None)
        process_critical_layers = bool(
            getattr(self.attack_profile, "process_critical_layers", True)
        )
        process_non_critical_layers = bool(
            getattr(self.attack_profile, "process_non_critical_layers", True)
        )

        # Stage I: clean probe from the same global initialization.
        self.model.load_state_dict(initial_state, strict=False)
        self._train_epochs(self.clean_loader, probe_epochs)
        benign_state = self._clone_state(self.model)

        # Stage I: 100% poison probe with the same training configuration.
        self.model.load_state_dict(initial_state, strict=False)
        self._train_epochs(self.probe_poison_loader, probe_epochs)
        malicious_state = self._clone_state(self.model)

        critical_layers, _ = self.attack_profile.locate_critical_layers(
            initial_state,
            benign_state,
            malicious_state,
            client_id=self.owner_id,
            round_idx=self.round_idx,
        )
        grad_masks = (
            self.attack_profile.build_masks(
                initial_state, benign_state, critical_layers
            )
            if process_critical_layers
            else {}
        )

        # Stage II: retrain from global on the configured poison-ratio data.
        # Do not freeze any layer. Optionally mask coordinates in L_c and/or
        # project non-critical tensors for the layer-processing ablations.
        self.model.load_state_dict(initial_state, strict=False)
        avg_loss = self._train_epochs(
            self.train_loader,
            local_epochs,
            trainable_names=None,
            grad_masks=grad_masks,
            align_layerwise=process_non_critical_layers,
            global_state=initial_state,
            prev_global_delta=prev_global_delta,
            project_fn=project_fn,
            project_skip_keys=critical_layers if process_non_critical_layers else None,
        )
        concentrated_state = self._clone_state(self.model)

        delta = {
            key: concentrated_state[key] - initial_state[key]
            for key in initial_state
            if key in concentrated_state
        }
        return {
            "train_loss": avg_loss,
            "delta": delta,
            "concentrated_state": concentrated_state,
        }

    def package(self, train_metrics: Dict[str, Any]) -> Dict[str, Any]:
        concentrated_state = train_metrics.get("concentrated_state")
        if not isinstance(concentrated_state, dict):
            raise RuntimeError("SparAlignFLClient.package requires concentrated_state")

        final_delta = self._call_attack(
            "poison_upload",
            train_metrics["delta"],
            update=train_metrics["delta"],
            initial_weights=self._last_initial_state,
            concentrated_state=concentrated_state,
            client_id=self.owner_id,
            round_idx=self.round_idx,
            num_samples=len(self.train_loader.dataset),
        )
        payload = {
            "client_id": self.owner_id,
            "delta": final_delta,
            "metrics": train_metrics["train_loss"],
            "num_samples": 1,
        }
        score_stats = getattr(self.attack_profile, "last_layer_score_stats", None)
        if isinstance(score_stats, dict) and score_stats:
            payload["layer_score_stats"] = score_stats
        stats = getattr(self.attack_profile, "last_layer_mask_stats", None)
        if isinstance(stats, dict) and stats:
            payload["layer_mask_stats"] = stats
        return payload
