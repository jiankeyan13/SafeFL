"""Homomorphic execution layer: ONE ``ClientWorker`` class powering both Local and Ray."""
from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from core.attack.base import build_attacker
from core.simulation.client import BenignClient
from core.simulation.coordinator import ParticipantTask
from core.types import (
    ClientContext,
    ClientUpdate,
    ServerBroadcast,
    StateDict,
    load_model_state,
    to_cpu_tree,
)
from core.utils.config import ComponentConfig, ParallelConfig
from core.utils.registry import MODEL_REGISTRY, bootstrap_registries
from data.hub import DataHub
from data.spec import FederatedDataSpec

_logger = logging.getLogger(__name__)

TaskResult = Tuple[ClientUpdate, Optional[Dict[str, Any]]]

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None  # type: ignore[assignment]


def resolve_gpu_ids(parallel_cfg: ParallelConfig) -> List[int]:
    if parallel_cfg.gpu_ids:
        return list(parallel_cfg.gpu_ids)
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


class ClientWorker:
    """Plain Python worker executing participant tasks sequentially on one device.

    - ``LocalExecutor`` instantiates ``ClientWorker`` directly in the main process
      (zero IPC overhead, native IDE breakpoint debugging, CPU/GPU compatible).
    - ``RayExecutor`` wraps this exact same class via ``ray.remote(ClientWorker)``
      to run a multi-GPU actor pool with 100% identical execution semantics.
    """

    def __init__(
        self,
        data_spec: FederatedDataSpec,
        model_cfg: ComponentConfig,
        device: Union[str, torch.device],
        *,
        seed: int = 42,
        run_dir: Optional[str] = None,
        is_remote: bool = False,
    ) -> None:
        if is_remote:
            bootstrap_registries()
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.seed = int(seed)
        self.run_dir = run_dir
        self.hub = DataHub.from_spec(data_spec, device=self.device)
        self.model = MODEL_REGISTRY.build(model_cfg.name, **model_cfg.params).to(self.device)
        self.benign_client = BenignClient()

    def run_batch(
        self,
        tasks: Sequence[ParticipantTask],
        broadcast: ServerBroadcast,
        *,
        export_cpu: bool = False,
    ) -> List[TaskResult]:
        if not tasks:
            return []

        # Stage global params & buffers onto self.device ONCE per batch rather than per client.
        dev_params: StateDict = {
            k: (v if v.device == self.device else v.to(self.device, non_blocking=True)).detach()
            for k, v in broadcast.params.items()
        }
        dev_buffers: StateDict = {
            k: (v if v.device == self.device else v.to(self.device, non_blocking=True)).detach()
            for k, v in broadcast.buffers.items()
        }
        dev_prev_delta: Optional[StateDict] = (
            {
                k: (v if v.device == self.device else v.to(self.device, non_blocking=True)).detach()
                for k, v in broadcast.prev_delta.items()
            }
            if broadcast.prev_delta is not None
            else None
        )
        dev_broadcast = ServerBroadcast(
            round_idx=broadcast.round_idx,
            params=dev_params,
            buffers=dev_buffers,
            lr=broadcast.lr,
            prev_delta=dev_prev_delta,
            extra=broadcast.extra,
        )

        results: List[TaskResult] = []
        for task in tasks:
            for p in self.model.parameters():
                p.requires_grad = True
            load_model_state(self.model, dev_params, dev_buffers)

            ctx = ClientContext(
                client_id=task.client_id,
                device=self.device,
                model=self.model,
                initial_params=dev_params,
                initial_buffers=dev_buffers,
                data=self.hub.client(task.client_id),
                train_cfg=task.train_cfg,
                broadcast=dev_broadcast,
                seed=self.seed,
                attacker_rank=task.attacker_rank,
                state=task.state,
                run_dir=self.run_dir,
            )

            if task.is_malicious:
                if task.strategy is None:
                    raise ValueError(f"Malicious task for client {task.client_id} missing strategy.")
                attacker = build_attacker(
                    task.strategy,
                    seed=self.seed,
                    attacker_rank=task.attacker_rank,
                )
                update = attacker.step(ctx)
                next_state: Optional[Dict[str, Any]] = (
                    (to_cpu_tree(ctx.state) if export_cpu else ctx.state)
                    if ctx.state
                    else None
                )
            else:
                update = self.benign_client.step(ctx)
                next_state = None

            self.model.zero_grad(set_to_none=True)
            if export_cpu:
                update = update.to_cpu()
            results.append((update, next_state))

        return results


class BaseExecutor(ABC):
    """Unified round execution backend interface."""

    @abstractmethod
    def run_round(
        self,
        tasks: Sequence[ParticipantTask],
        broadcast: ServerBroadcast,
    ) -> List[TaskResult]:
        """Execute all ``tasks`` for one FL round and return results in matching order."""

    def shutdown(self) -> None:
        pass


class LocalExecutor(BaseExecutor):
    """In-process executor ideal for breakpoint debugging, CPU dev, or single-stream runs."""

    def __init__(
        self,
        data_spec: FederatedDataSpec,
        model_cfg: ComponentConfig,
        device: torch.device,
        *,
        seed: int = 42,
        run_dir: Optional[str] = None,
    ) -> None:
        self.worker = ClientWorker(
            data_spec=data_spec,
            model_cfg=model_cfg,
            device=device,
            seed=seed,
            run_dir=run_dir,
            is_remote=False,
        )

    def run_round(
        self,
        tasks: Sequence[ParticipantTask],
        broadcast: ServerBroadcast,
    ) -> List[TaskResult]:
        return self.worker.run_batch(tasks, broadcast, export_cpu=False)

    def shutdown(self) -> None:
        self.worker.hub.evict_device()


class RayExecutor(BaseExecutor):
    """Multi-GPU Ray actor-pool executor wrapping ``ClientWorker``."""

    def __init__(
        self,
        parallel_cfg: ParallelConfig,
        data_spec: FederatedDataSpec,
        model_cfg: ComponentConfig,
        *,
        seed: int = 42,
        run_dir: Optional[str] = None,
    ) -> None:
        if ray is None:
            raise RuntimeError("Ray is not installed (`pip install 'ray>=2.9.0'`).")

        self.gpu_ids = resolve_gpu_ids(parallel_cfg)
        if not self.gpu_ids:
            raise RuntimeError("RayExecutor requires at least one visible CUDA GPU.")

        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        self._owns_ray = False
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level=logging.ERROR,
            )
            self._owns_ray = True

        spec_ref = ray.put(data_spec)
        remote_worker_cls = ray.remote(ClientWorker)
        self.actors: List[Any] = []

        for gpu_id in self.gpu_ids:
            for _ in range(parallel_cfg.actors_per_gpu):
                actor = remote_worker_cls.options(
                    num_cpus=1,
                    num_gpus=0,
                    runtime_env={
                        "env_vars": {
                            "CUDA_VISIBLE_DEVICES": str(gpu_id),
                            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                        }
                    },
                ).remote(
                    spec_ref,
                    model_cfg,
                    "cuda:0",
                    seed=seed,
                    run_dir=run_dir,
                    is_remote=True,
                )
                self.actors.append(actor)

    @property
    def num_actors(self) -> int:
        return len(self.actors)

    def run_round(
        self,
        tasks: Sequence[ParticipantTask],
        broadcast: ServerBroadcast,
    ) -> List[TaskResult]:
        if not tasks:
            return []

        n_actors = len(self.actors)
        shards: List[List[ParticipantTask]] = [[] for _ in range(n_actors)]
        for idx, t in enumerate(tasks):
            shards[idx % n_actors].append(t)

        broadcast_ref = ray.put(broadcast.to_cpu())
        futures = [
            actor.run_batch.remote(shard, broadcast_ref, export_cpu=True)
            for actor, shard in zip(self.actors, shards)
            if shard
        ]
        batches: List[List[TaskResult]] = ray.get(futures) if futures else []

        by_cid: Dict[int, TaskResult] = {}
        for batch in batches:
            for item in batch:
                by_cid[int(item[0].client_id)] = item
        return [by_cid[int(t.client_id)] for t in tasks]

    def shutdown(self) -> None:
        for actor in self.actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self.actors.clear()
        if self._owns_ray and ray is not None and ray.is_initialized():
            ray.shutdown()
            self._owns_ray = False


def create_executor(
    parallel_cfg: ParallelConfig,
    *,
    data_spec: FederatedDataSpec,
    model_cfg: ComponentConfig,
    default_device: torch.device,
    seed: int = 42,
    run_dir: Optional[str] = None,
    info_fn: Optional[Any] = None,
) -> BaseExecutor:
    """Create ``RayExecutor`` when configured & available; seamlessly fall back to ``LocalExecutor``."""
    log = info_fn if callable(info_fn) else _logger.info

    if parallel_cfg.backend == "ray":
        if ray is None:
            log("Ray not installed; falling back to local in-process executor.")
        elif not resolve_gpu_ids(parallel_cfg):
            log("No CUDA GPUs visible for Ray; falling back to local in-process executor.")
        else:
            try:
                executor = RayExecutor(
                    parallel_cfg=parallel_cfg,
                    data_spec=data_spec,
                    model_cfg=model_cfg,
                    seed=seed,
                    run_dir=run_dir,
                )
                log(
                    f"Execution backend: ray | gpu_ids={executor.gpu_ids} | "
                    f"actors={executor.num_actors}"
                )
                return executor
            except Exception as exc:
                log(f"Ray initialization failed ({exc}); falling back to local executor.")

    log(f"Execution backend: local ({default_device})")
    return LocalExecutor(
        data_spec=data_spec,
        model_cfg=model_cfg,
        device=default_device,
        seed=seed,
        run_dir=run_dir,
    )
