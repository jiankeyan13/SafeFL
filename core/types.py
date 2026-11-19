"""Core data contracts ("boxes") and model parameter/buffer helpers for SafeFL v2.

This module defines the three primary data structures exchanged across layers:
  1. ``ServerBroadcast`` : Server -> ClientWorker (downlink per round)
  2. ``ClientContext``   : ClientWorker -> BenignClient / Attacker (local workspace)
  3. ``ClientUpdate``    : BenignClient / Attacker -> Server / Defense (uplink per round)

Plus ``TrainConfig`` and ``ServerContext`` / ``DefenseOutput`` for server-side defense docking,
and helper functions that keep learnable ``params`` physically separated from BN ``buffers``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from data.loader import BatchPoisoner, DataView, FastBatchLoader

StateDict = Dict[str, torch.Tensor]


# =============================================================================
# 1. Model Parameter & Buffer Split / Merge Helpers
# =============================================================================


@torch.no_grad()
def split_model_state(
    model: nn.Module,
    *,
    trainable_only: bool = True,
) -> Tuple[StateDict, StateDict]:
    """Split ``model`` into cloned ``(params, buffers)`` dicts.

    - ``params`` contains learnable parameters from ``model.named_parameters()``
      (filtered by ``requires_grad`` when ``trainable_only=True``).
    - ``buffers`` contains non-parameter state from ``model.named_buffers()``
      (e.g., BatchNorm ``running_mean``, ``running_var``, ``num_batches_tracked``).
    """
    params: StateDict = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if (param.requires_grad or not trainable_only)
    }
    buffers: StateDict = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
    }
    return params, buffers


@torch.no_grad()
def load_model_state(
    model: nn.Module,
    params: Mapping[str, torch.Tensor],
    buffers: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    """Copy ``params`` and optional ``buffers`` into ``model`` in-place."""
    for name, param in model.named_parameters():
        if name in params:
            src = params[name]
            param.copy_(src.to(device=param.device, dtype=param.dtype))

    if buffers is not None:
        for name, buf in model.named_buffers():
            if name in buffers:
                src = buffers[name]
                buf.copy_(src.to(device=buf.device, dtype=buf.dtype))


@torch.no_grad()
def compute_param_delta(
    model: nn.Module,
    initial_params: Mapping[str, torch.Tensor],
) -> Tuple[StateDict, StateDict]:
    """Compute ``(param_delta, current_buffers)`` after local training.

    ``param_delta`` strictly contains ``current_param - initial_param`` for keys
    in ``initial_params``; BatchNorm statistics are returned separately in
    ``current_buffers``.
    """
    named_params = dict(model.named_parameters())
    param_delta: StateDict = {}
    for name, init_val in initial_params.items():
        if name in named_params:
            cur_val = named_params[name].detach()
            param_delta[name] = cur_val - init_val.to(device=cur_val.device, dtype=cur_val.dtype)

    current_buffers: StateDict = {
        name: buf.detach().clone()
        for name, buf in model.named_buffers()
    }
    return param_delta, current_buffers


@torch.no_grad()
def apply_update_to_model(
    model: nn.Module,
    param_delta: Mapping[str, torch.Tensor],
    buffers: Optional[Mapping[str, torch.Tensor]] = None,
) -> None:
    """Apply aggregated ``param_delta`` (and optional ``buffers``) to ``model``."""
    for name, param in model.named_parameters():
        if name in param_delta:
            delta_t = param_delta[name].to(device=param.device, dtype=param.dtype)
            param.add_(delta_t)

    if buffers is not None:
        for name, buf in model.named_buffers():
            if name not in buffers:
                continue
            src = buffers[name].to(device=buf.device)
            if torch.is_floating_point(buf):
                buf.copy_(src.to(dtype=buf.dtype))
            else:
                buf.copy_(torch.round(src.to(dtype=torch.float64)).to(dtype=buf.dtype))


def to_cpu_state(state: Optional[Mapping[str, torch.Tensor]]) -> Optional[StateDict]:
    """Detach and move a tensor dict onto CPU for safe cross-process transport."""
    if state is None:
        return None
    return {
        k: (v.detach().cpu() if torch.is_tensor(v) else v)
        for k, v in state.items()
    }


# =============================================================================
# 2. Training Hyperparameter Config
# =============================================================================


@dataclass(frozen=True)
class TrainConfig:
    """Flat, immutable local training hyperparameters."""

    epochs: int = 2
    batch_size: int = 64
    lr: float = 0.1
    optimizer: str = "SGD"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    criterion: str = "CrossEntropyLoss"
    grad_clip_norm: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_overrides(self, **kwargs: Any) -> "TrainConfig":
        """Return a new ``TrainConfig`` with specified fields updated."""
        known_fields = {
            "epochs",
            "batch_size",
            "lr",
            "optimizer",
            "momentum",
            "weight_decay",
            "criterion",
            "grad_clip_norm",
            "extra",
        }
        direct = {k: v for k, v in kwargs.items() if k in known_fields and v is not None}
        unknown = {k: v for k, v in kwargs.items() if k not in known_fields}
        if unknown:
            merged_extra = {**self.extra, **unknown}
            direct["extra"] = merged_extra
        return replace(self, **direct)


# =============================================================================
# 3. The Three Core Communication Boxes (+ Server Defense Context)
# =============================================================================


@dataclass
class ServerBroadcast:
    """Box 1 (Downlink): Sent from ``Server`` to ``ClientWorker`` each round."""

    round_idx: int
    params: StateDict
    buffers: StateDict
    lr: float
    prev_delta: Optional[StateDict] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_cpu(self) -> "ServerBroadcast":
        return ServerBroadcast(
            round_idx=int(self.round_idx),
            params=to_cpu_state(self.params) or {},
            buffers=to_cpu_state(self.buffers) or {},
            lr=float(self.lr),
            prev_delta=to_cpu_state(self.prev_delta),
            extra=dict(self.extra),
        )


@dataclass
class ClientUpdate:
    """Box 2 (Uplink): Returned by ``BenignClient`` / ``Attacker`` to ``Server``."""

    client_id: int
    param_delta: StateDict
    buffers: StateDict
    num_samples: int
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_cpu(self) -> "ClientUpdate":
        return ClientUpdate(
            client_id=int(self.client_id),
            param_delta=to_cpu_state(self.param_delta) or {},
            buffers=to_cpu_state(self.buffers) or {},
            num_samples=int(self.num_samples),
            metrics={str(k): float(v) for k, v in self.metrics.items()},
        )


@dataclass
class ClientContext:
    """Box 3 (Workspace): Handed to ``client.step(ctx)`` or ``attacker.step(ctx)``.

    ``model`` is already loaded with ``broadcast.params + buffers`` on ``device``,
    and ``initial_params`` / ``initial_buffers`` hold round-start snapshots on ``device``.
    """

    client_id: int
    device: torch.device
    model: nn.Module
    initial_params: StateDict
    initial_buffers: StateDict
    data: "DataView"
    train_cfg: TrainConfig
    broadcast: ServerBroadcast
    state: Dict[str, Any] = field(default_factory=dict)
    run_dir: Optional[str] = None

    @property
    def round_idx(self) -> int:
        return self.broadcast.round_idx

    @property
    def lr(self) -> float:
        return self.train_cfg.lr

    @property
    def epochs(self) -> int:
        return self.train_cfg.epochs

    @property
    def batch_size(self) -> int:
        return self.train_cfg.batch_size

    @property
    def prev_global_delta(self) -> Optional[StateDict]:
        return self.broadcast.prev_delta

    def reset_model(self) -> None:
        """Reset ``self.model`` back to the round-start global weights ``W_g``."""
        load_model_state(self.model, self.initial_params, self.initial_buffers)

    def loader(
        self,
        view: Optional["DataView"] = None,
        *,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        poisoner: Optional["BatchPoisoner"] = None,
        mode: str = "train",
        augment: Optional[bool] = None,
        return_orig_label: bool = False,
        seed: Optional[int] = None,
    ) -> "FastBatchLoader":
        """Build a ``FastBatchLoader`` pre-configured with ``device``, ``batch_size``, and ``round_idx``."""
        target_view = self.data if view is None else view
        return target_view.loader(
            batch_size=self.train_cfg.batch_size if batch_size is None else int(batch_size),
            shuffle=shuffle,
            drop_last=drop_last,
            device=self.device,
            poisoner=poisoner,
            mode=mode,
            augment=augment,
            return_orig_label=return_orig_label,
            round_idx=self.round_idx,
            seed=seed,
        )

    def pack_update(
        self,
        param_delta: Optional[StateDict] = None,
        *,
        buffers: Optional[StateDict] = None,
        num_samples: Optional[int] = None,
        metrics: Optional[Mapping[str, float]] = None,
    ) -> ClientUpdate:
        """Package local results into a ``ClientUpdate`` box.

        When ``param_delta`` is omitted, automatically computes
        ``self.model - self.initial_params`` and extracts current BN ``buffers``.
        """
        if param_delta is None or buffers is None:
            auto_delta, auto_buffers = compute_param_delta(self.model, self.initial_params)
            if param_delta is None:
                param_delta = auto_delta
            if buffers is None:
                buffers = auto_buffers

        return ClientUpdate(
            client_id=int(self.client_id),
            param_delta=param_delta,
            buffers=buffers,
            num_samples=len(self.data) if num_samples is None else int(num_samples),
            metrics={str(k): float(v) for k, v in (metrics or {}).items()},
        )


@dataclass
class ServerContext:
    """Context provided to ``Defense.defend(...)`` each round."""

    round_idx: int
    client_lr: float
    device: torch.device
    train_cfg: TrainConfig
    proxy_data: Optional["DataView"] = None
    run_dir: Optional[str] = None


@dataclass
class DefenseOutput:
    """Result returned by ``Defense.defend(...)`` to update the global model."""

    param_delta: StateDict
    buffers: Optional[StateDict] = None
    metrics: Dict[str, float] = field(default_factory=dict)
