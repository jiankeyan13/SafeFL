from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.nn as nn

from core.client.base_client import BaseClient
from core.config import ClientConfig
from core.utils.evaluator import Evaluator
from data.dataset_store import DatasetStore
from data.task import TaskSet


LOCKDOWN_MASK_PREFIX = "__lockdown_mask__."


class LockdownClient(BaseClient):
    """
    Lockdown client with isolated subspace training and dynamic sparse masks.

    Runner creates client objects per round, so masks are stored at class scope
    and keyed by owner_id.
    """

    _mask_store: Dict[str, Dict[str, torch.Tensor]] = {}
    _num_remove_store: Dict[str, Dict[str, int]] = {}
    _step_store: Dict[str, int] = {}
    _shared_initial_mask: Optional[Dict[str, torch.Tensor]] = None

    def __init__(
        self,
        client_id: Union[int, str],
        task_set: TaskSet,
        stores: Dict[str, DatasetStore],
        model: nn.Module,
        device: torch.device,
        config: ClientConfig,
        evaluator: Optional[Evaluator] = None,
        dense_ratio: float = 0.25,
        mask_init: str = "ERK",
        same_mask: bool = True,
        anneal_factor: float = 1e-4,
        total_rounds: int = 200,
        dis_check_gradient: bool = False,
        mask_seed: int = 0,
    ):
        self.dense_ratio = float(dense_ratio)
        self.mask_init = mask_init
        self.same_mask = bool(same_mask)
        self.anneal_factor = float(anneal_factor)
        self.total_rounds = max(int(total_rounds), 1)
        self.dis_check_gradient = bool(dis_check_gradient)
        self.mask_seed = int(mask_seed)
        super().__init__(
            client_id=client_id,
            task_set=task_set,
            stores=stores,
            model=model,
            device=device,
            config=config,
            evaluator=evaluator,
        )
        self._ensure_mask_initialized()

    @classmethod
    def reset_lockdown_state(cls) -> None:
        cls._mask_store.clear()
        cls._num_remove_store.clear()
        cls._step_store.clear()
        cls._shared_initial_mask = None

    def _parameter_state(self) -> Dict[str, torch.Tensor]:
        return {name: p.detach() for name, p in self.model.named_parameters() if p.requires_grad}

    def _ensure_mask_initialized(self) -> None:
        if self.owner_id in self._mask_store:
            return

        params = self._parameter_state()
        if self.same_mask:
            if LockdownClient._shared_initial_mask is None:
                LockdownClient._shared_initial_mask = self._init_masks(params, self.mask_seed)
            mask = {name: value.clone() for name, value in LockdownClient._shared_initial_mask.items()}
        else:
            seed = self.mask_seed + self.client_id
            mask = self._init_masks(params, seed)

        self._mask_store[self.owner_id] = mask
        self._num_remove_store[self.owner_id] = {}
        self._step_store[self.owner_id] = 0

    def _init_masks(self, params: Mapping[str, torch.Tensor], seed: int) -> Dict[str, torch.Tensor]:
        sparsities = self._calculate_sparsities(params)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        masks: Dict[str, torch.Tensor] = {}
        for name, tensor in params.items():
            mask = torch.zeros_like(tensor, dtype=torch.bool, device="cpu")
            dense_numel = int((1.0 - sparsities[name]) * mask.numel())
            if dense_numel > 0:
                perm = torch.randperm(mask.numel(), generator=generator)[:dense_numel]
                mask.view(-1)[perm] = True
            masks[name] = mask
        return masks

    def _calculate_sparsities(self, params: Mapping[str, torch.Tensor]) -> Dict[str, float]:
        if self.mask_init.lower() == "uniform":
            return {name: 1.0 - self.dense_ratio for name in params}
        if self.mask_init.upper() != "ERK":
            raise ValueError(f"Unsupported Lockdown mask_init: {self.mask_init}")

        dense_layers = set()
        raw_probabilities: Dict[str, float] = {}
        epsilon = 0.0

        while True:
            divisor = 0.0
            rhs = 0.0
            raw_probabilities = {}

            for name, tensor in params.items():
                n_param = tensor.numel()
                if n_param == 0:
                    dense_layers.add(name)
                    continue

                n_zeros = n_param * (1.0 - self.dense_ratio)
                n_ones = n_param * self.dense_ratio

                if name in dense_layers:
                    rhs -= n_zeros
                    continue

                raw_probability = sum(tensor.shape) / float(n_param)
                raw_probabilities[name] = raw_probability
                rhs += n_ones
                divisor += raw_probability * n_param

            if divisor == 0.0 or not raw_probabilities:
                break

            epsilon = rhs / divisor
            max_prob = max(raw_probabilities.values())
            if max_prob * epsilon <= 1.0:
                break

            for name, raw_probability in raw_probabilities.items():
                if raw_probability == max_prob:
                    dense_layers.add(name)

        sparsities: Dict[str, float] = {}
        for name in params:
            if name in dense_layers or name not in raw_probabilities:
                sparsities[name] = 0.0
            else:
                sparsities[name] = 1.0 - epsilon * raw_probabilities[name]
        return sparsities

    def _mask_for_device(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        return self._mask_store[self.owner_id][name].to(device=tensor.device)

    def _apply_mask_to_parameters(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._mask_store[self.owner_id]:
                    param.mul_(self._mask_for_device(name, param))

    def _screen_gradients(self) -> Dict[str, torch.Tensor]:
        self.model.train()
        gradients = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if name in self._mask_store[self.owner_id]
        }

        for data, target in self.train_loader:
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            self.model.zero_grad(set_to_none=True)
            output = self.model(data)
            loss_val = self.loss(output, target)
            loss_val.backward()

            for name, param in self.model.named_parameters():
                if name in gradients and param.grad is not None:
                    gradients[name].add_(param.grad.detach())

        return gradients

    def _grow_mask(self, num_remove: Mapping[str, int], gradients: Mapping[str, torch.Tensor]) -> None:
        mask_store = self._mask_store[self.owner_id]
        for name, remove_count in num_remove.items():
            if remove_count <= 0 or name not in gradients:
                continue

            mask = mask_store[name].to(self.device)
            inactive = ~mask
            inactive_count = int(inactive.sum().item())
            grow_count = min(int(remove_count), inactive_count)
            if grow_count <= 0:
                continue

            if self.dis_check_gradient:
                candidates = inactive.flatten().nonzero(as_tuple=False).flatten()
                choice = torch.randperm(candidates.numel(), device=self.device)[:grow_count]
                idx = candidates[choice]
            else:
                scores = torch.where(
                    inactive,
                    gradients[name].abs(),
                    torch.full_like(gradients[name], -1.0),
                )
                idx = torch.topk(scores.flatten(), grow_count).indices

            mask.view(-1)[idx] = True
            mask_store[name] = mask.cpu()

    def _fire_mask(self, step: int) -> Dict[str, int]:
        mask_store = self._mask_store[self.owner_id]
        drop_ratio = self.anneal_factor / 2.0 * (
            1.0 + math.cos((step * math.pi) / self.total_rounds)
        )
        num_remove: Dict[str, int] = {}

        for name, param in self.model.named_parameters():
            if name not in mask_store:
                continue
            mask = mask_store[name].to(param.device)
            remove_count = int(math.ceil(drop_ratio * int(mask.sum().item())))
            if remove_count <= 0:
                num_remove[name] = 0
                continue

            active_count = int(mask.sum().item())
            remove_count = min(remove_count, active_count)
            num_remove[name] = remove_count
            if remove_count <= 0:
                continue

            scores = torch.where(
                mask,
                param.detach().abs(),
                torch.full_like(param.detach(), float("inf")),
            )
            idx = torch.topk(scores.flatten(), remove_count, largest=False).indices
            mask.view(-1)[idx] = False
            mask_store[name] = mask.cpu()

        return num_remove

    def train(self) -> Dict[str, Any]:
        self._ensure_mask_initialized()
        step = self._step_store.get(self.owner_id, 0)

        self.model.train()
        self._apply_mask_to_parameters()

        pending_remove = self._num_remove_store.get(self.owner_id, {})
        if pending_remove:
            gradients = self._screen_gradients()
            self._grow_mask(pending_remove, gradients)
            self._apply_mask_to_parameters()

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
                loss_val = self.loss(output, target)
                loss_val.backward()

                for name, param in self.model.named_parameters():
                    if param.grad is not None and name in self._mask_store[self.owner_id]:
                        param.grad.mul_(self._mask_for_device(name, param.grad))

                optimizer.step()
                self._apply_mask_to_parameters()

                total_loss += loss_val.item() * target.size(0)
                total_samples += target.size(0)

        self._num_remove_store[self.owner_id] = self._fire_mask(step)
        self._step_store[self.owner_id] = step + 1

        current_state = self.model.state_dict()
        delta = {}
        for name, initial_value in initial_state.items():
            if name not in current_state:
                continue
            value_delta = current_state[name] - initial_value
            if name in self._mask_store[self.owner_id] and torch.is_floating_point(value_delta):
                value_delta = value_delta * self._mask_for_device(name, value_delta)
            delta[name] = value_delta

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}

    def package(self, train_metrics: Dict[str, Any]) -> Dict[str, Any]:
        delta = dict(train_metrics["delta"])
        for name, mask in self._mask_store[self.owner_id].items():
            delta[f"{LOCKDOWN_MASK_PREFIX}{name}"] = mask.to(dtype=torch.float32)

        return {
            "client_id": self.owner_id,
            "delta": delta,
            "metrics": train_metrics["train_loss"],
            "num_samples": len(self.train_loader.dataset),
        }
