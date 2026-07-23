"""Ray-based parallel client training executor."""

from __future__ import annotations

import importlib
import logging
import os
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from core.config import (
    AttackStrategyConfig,
    ClientConfig,
    ParallelConfig,
    apply_malicious_epochs_override,
)
from core.utils.evaluator import Accuracy, AverageLoss, Evaluator
from core.utils.registry import MODEL_REGISTRY

logger = logging.getLogger(__name__)

ClientClassSpec = Dict[str, Any]


def resolve_gpu_ids(parallel_cfg: ParallelConfig) -> List[int]:
    """Resolve training GPU ids; empty config means all visible CUDA devices."""
    if parallel_cfg.gpu_ids:
        return list(parallel_cfg.gpu_ids)
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []


def client_config_to_dict(config: ClientConfig) -> Dict[str, Any]:
    """Serialize ClientConfig into the dict shape accepted by ClientConfig.from_dict."""
    tc = config.trainer_config
    return {
        "lr": tc.lr,
        "momentum": tc.momentum,
        "weight_decay": tc.weight_decay,
        "epochs": tc.epochs,
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "bn_calib_batches": config.bn_calib_batches,
        "optimizer_name": tc.optimizer_name,
        "criterion_name": tc.criterion_name,
        "grad_clip_norm": tc.grad_clip_norm,
    }


def qualified_name(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def serialize_client_class(client_class: Any) -> ClientClassSpec:
    """Serialize a client class or functools.partial into a worker-side spec."""
    if isinstance(client_class, partial):
        func = client_class.func
        return {
            "class": qualified_name(func),
            "kwargs": dict(client_class.keywords or {}),
        }
    return {"class": qualified_name(client_class), "kwargs": {}}


def load_class(path: str) -> type:
    module_name, _, class_name = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid class path: {path}")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def resolve_client_class(spec: Union[str, ClientClassSpec]) -> Tuple[type, Dict[str, Any]]:
    if isinstance(spec, str):
        return load_class(spec), {}
    return load_class(spec["class"]), dict(spec.get("kwargs") or {})


def split_round_robin(items: Sequence[str], num_shards: int) -> List[List[str]]:
    if num_shards <= 0:
        return [list(items)]
    shards: List[List[str]] = [[] for _ in range(num_shards)]
    for idx, item in enumerate(items):
        shards[idx % num_shards].append(item)
    return shards


def _move_payload_to_cpu(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    delta = out.get("delta")
    if isinstance(delta, dict):
        out["delta"] = {
            k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in delta.items()
        }
    return out


def _export_cpu_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in state_dict.items()}


try:
    import ray
except ImportError:  # pragma: no cover - optional until installed
    ray = None  # type: ignore


if ray is not None:

    @ray.remote
    class ClientWorker:
        """常驻 Ray Actor: 绑定单卡, 顺序训练分到的客户端."""

        def __init__(
            self,
            gpu_id: Optional[int],
            task_set: Any,
            stores: Any,
            model_name: str,
            model_params: Dict[str, Any],
            default_client_class: Union[str, ClientClassSpec],
        ) -> None:
            from core.bootstrap import bootstrap_registries

            bootstrap_registries()

            self.gpu_id = gpu_id
            if torch.cuda.is_available():
                # runtime_env 已将物理 GPU 映射为本地 cuda:0
                self.device = torch.device("cuda:0")
                torch.cuda.set_device(self.device)
            else:
                self.device = torch.device("cpu")
            logger.info(
                "ClientWorker ready: requested_gpu=%s device=%s cuda_available=%s visible=%s",
                gpu_id,
                self.device,
                torch.cuda.is_available(),
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            )

            self.task_set = task_set
            self.stores = stores
            self.model_name = model_name
            self.model_params = dict(model_params or {})
            self.default_client_class = default_client_class
            self.evaluator = Evaluator(metrics={"accuracy": Accuracy(), "loss": AverageLoss()})

        def train_batch(
            self,
            client_ids: List[str],
            weights: Dict[str, torch.Tensor],
            client_config_dict: Dict[str, Any],
            round_idx: int,
            job_specs: Dict[str, Dict[str, Any]],
            prev_global_delta: Optional[Dict[str, torch.Tensor]] = None,
        ) -> List[Dict[str, Any]]:
            updates: List[Dict[str, Any]] = []
            for cid in client_ids:
                spec = job_specs.get(cid, {"kind": "benign"})
                client = self._create_client(cid, client_config_dict, round_idx, spec)
                # 始终注入可选属性; 不需要的 Client 忽略即可
                client.prev_global_delta = prev_global_delta
                payload = client.step(weights)
                updates.append(_move_payload_to_cpu(payload))
                del client
            return updates

        def _build_model(self, spec: Dict[str, Any]) -> torch.nn.Module:
            model_cls = MODEL_REGISTRY.get(self.model_name)
            params = {**self.model_params, **(spec.get("model_params") or {})}
            return model_cls(**params)

        def _create_client(
            self,
            cid: str,
            client_config_dict: Dict[str, Any],
            round_idx: int,
            spec: Dict[str, Any],
        ) -> Any:
            from core.attack import build_attack
            from core.client.malicious_client import MaliciousClient

            config = ClientConfig.from_dict(client_config_dict)
            model = self._build_model(spec)
            kind = spec.get("kind", "benign")

            if kind == "malicious":
                strategy = AttackStrategyConfig.from_dict(spec.get("strategy") or {})
                attack_profile = build_attack(strategy)
                malicious_epochs = spec.get("malicious_epochs")
                if malicious_epochs is not None:
                    config = apply_malicious_epochs_override(config, int(malicious_epochs))
                client_cls = getattr(attack_profile, "client_class", None) or MaliciousClient
                return client_cls(
                    client_id=cid,
                    task_set=self.task_set,
                    stores=self.stores,
                    model=model,
                    device=self.device,
                    config=config,
                    evaluator=self.evaluator,
                    attack_profile=attack_profile,
                    round_idx=spec.get("round_idx", round_idx),
                )

            client_cls_spec = spec.get("client_class") or self.default_client_class
            client_cls, extra_kwargs = resolve_client_class(client_cls_spec)
            return client_cls(
                client_id=cid,
                task_set=self.task_set,
                stores=self.stores,
                model=model,
                device=self.device,
                config=config,
                evaluator=self.evaluator,
                **extra_kwargs,
            )

else:

    class ClientWorker:  # type: ignore[no-redef]
        """Placeholder when ray is not installed."""

        pass


class RayTrainingExecutor:
    """Manage a pool of ClientWorker actors for one experiment."""

    def __init__(self, parallel_cfg: ParallelConfig) -> None:
        self.parallel_cfg = parallel_cfg
        self.gpu_ids: List[int] = []
        self.actors: List[Any] = []
        self._started = False
        self._owns_ray = False

    @property
    def num_actors(self) -> int:
        return len(self.actors)

    def start(
        self,
        task_set: Any,
        stores: Any,
        model_name: str,
        model_params: Dict[str, Any],
        default_client_class: type,
    ) -> None:
        if ray is None:
            raise RuntimeError("ray 未安装, 请先 pip install 'ray>=2.9.0'")

        self.gpu_ids = resolve_gpu_ids(self.parallel_cfg)
        if not self.gpu_ids:
            raise RuntimeError("Ray 并行训练需要至少一张可见 GPU, 或改用 parallel.backend=sequential")

        # 避免 num_gpus=0 时 Ray 清空 CUDA_VISIBLE_DEVICES, 导致 Worker 落到 CPU
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                logging_level=logging.ERROR,
            )
            self._owns_ray = True

        task_set_ref = ray.put(task_set)
        stores_ref = ray.put(stores)
        client_cls_spec = serialize_client_class(default_client_class)
        actors_per_gpu = self.parallel_cfg.actors_per_gpu

        self.actors = []
        for gpu_id in self.gpu_ids:
            for _ in range(actors_per_gpu):
                # 用 runtime_env 固定物理 GPU; 进程内统一使用 cuda:0
                # num_gpus=0 + RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0, 避免 Ray 清空可见设备
                actor = ClientWorker.options(
                    num_cpus=1,
                    num_gpus=0,
                    runtime_env={
                        "env_vars": {
                            "CUDA_VISIBLE_DEVICES": str(gpu_id),
                            "RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0",
                        }
                    },
                ).remote(
                    0,
                    task_set_ref,
                    stores_ref,
                    model_name,
                    model_params,
                    client_cls_spec,
                )
                self.actors.append(actor)

        self._started = True
        logger.info(
            "RayTrainingExecutor started: gpu_ids=%s actors_per_gpu=%d total_actors=%d",
            self.gpu_ids,
            actors_per_gpu,
            len(self.actors),
        )

    def train_round(
        self,
        selected_ids: List[str],
        global_state: Dict[str, torch.Tensor],
        client_config: ClientConfig,
        round_idx: int,
        job_specs: Dict[str, Dict[str, Any]],
        prev_global_delta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> List[Dict[str, Any]]:
        if not self._started or not self.actors:
            raise RuntimeError("RayTrainingExecutor 尚未 start()")

        weights = _export_cpu_state_dict(global_state)
        weights_ref = ray.put(weights)
        prev_ref = ray.put(prev_global_delta) if prev_global_delta is not None else None
        config_dict = client_config_to_dict(client_config)
        shards = split_round_robin(selected_ids, len(self.actors))

        futures = []
        for actor, shard in zip(self.actors, shards):
            if not shard:
                continue
            shard_specs = {cid: job_specs[cid] for cid in shard if cid in job_specs}
            futures.append(
                actor.train_batch.remote(
                    shard, weights_ref, config_dict, round_idx, shard_specs, prev_ref,
                )
            )

        batches = ray.get(futures) if futures else []
        updates: List[Dict[str, Any]] = []
        for batch in batches:
            updates.extend(batch)
        return updates

    def shutdown(self) -> None:
        if not self._started:
            return
        for actor in self.actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self.actors = []
        self._started = False
        if self._owns_ray and ray is not None and ray.is_initialized():
            ray.shutdown()
            self._owns_ray = False
        logger.info("RayTrainingExecutor shutdown complete")


def try_create_ray_executor(
    parallel_cfg: ParallelConfig,
) -> Tuple[Optional[RayTrainingExecutor], Optional[str]]:
    """
    Create a Ray executor when backend=ray and GPUs are available.
    Returns (executor_or_none, fallback_reason_or_none).
    """
    if parallel_cfg.backend != "ray":
        return None, None
    if ray is None:
        return None, "ray 未安装, 回退到 sequential"
    gpu_ids = resolve_gpu_ids(parallel_cfg)
    if not gpu_ids:
        return None, "无可见 CUDA GPU, 回退到 sequential"
    return RayTrainingExecutor(parallel_cfg), None
