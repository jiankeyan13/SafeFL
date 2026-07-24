from __future__ import annotations

import os
import copy
import random
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core.utils.logger import Logger
from core.utils.evaluator import Accuracy, AverageLoss, Evaluator
from core.utils.registry import ALGORITHM_REGISTRY, MODEL_REGISTRY
from core.config import ClientConfig, TrainingConfig, DataConfig, GlobalConfig, ParallelConfig
from core.utils.lr_schedule import get_lr_from_schedule
from core.client.base_client import BaseClient
from core.server.base_server import BaseServer
from core.simulation.ray_executor import (
    serialize_client_class,
    try_create_ray_executor,
)
from data.constants import OWNER_SERVER, SPLIT_PROXY, SPLIT_TEST_GLOBAL
from data.task_generator import TaskGenerator
from data.partitioner import build_partitioner


def _clone_state_dict_cpu(
    state: Optional[Dict[str, torch.Tensor]],
) -> Optional[Dict[str, torch.Tensor]]:
    """Clone a state/delta dict onto CPU for client injection / Ray transport."""
    if state is None:
        return None
    return {
        k: v.detach().cpu().clone() if torch.is_tensor(v) else v
        for k, v in state.items()
    }


class BaseRunner:
    """
    联邦学习基础胶水层 (Simulator).

    职责:
      1. 初始化: 数据管道, 模型, 服务器, 评估器, Logger
      2. 驱动 Round 循环: 选人 -> 广播 -> 本地训练 -> 服务端聚合 -> 全局评估 -> 本地评估 -> 保存检查点
      3. 管理实验状态: 轮次计数, Checkpoint, Logger 生命周期
    """

    def __init__(self, config: Optional[Dict[str, Any] | GlobalConfig] = None) -> None:
        # 支持传入字典、GlobalConfig 或直接使用默认 GlobalConfig
        if config is None:
            self.global_config = GlobalConfig()
        elif isinstance(config, GlobalConfig):
            self.global_config = config
        else:
            self.global_config = GlobalConfig.from_dict(config)
        
        # 保持向下兼容，将 config 字典化
        self.config = self.global_config.to_dict()
        
        self.device = torch.device(self.global_config.device)
        self.training_config = self.global_config.training
        self.seed = self.global_config.seed
        self._set_seed(self.seed)

        self.task_set:        Any = None
        self.dataset_stores:  Any = None
        self.model_fn:        Any = None
        self.server:          Optional[BaseServer] = None
        self.client_class:    Any = None
        self.client_ids:      List[str] = []
        self.evaluator:       Optional[Evaluator] = None
        self.criterion:       Optional[nn.Module] = None
        self.eval_loader:     Optional[DataLoader] = None
        self.proxy_loader:    Optional[DataLoader] = None
        self._train_executor: Any = None
        self.parallel_config: ParallelConfig = self.global_config.parallel

        # Logger 从 GlobalConfig.logger_config 构建, _setup 末尾才 start() (唯一触发 IO 的时机)
        self.logger: Logger = Logger.from_logger_config(
            self.global_config.logger_config,
            self.config,
        )

        self._setup()

    def _setup(self) -> None:
        """编排初始化顺序; 子类可重写 (记得调 super()._setup())."""
        self._setup_data_pipeline()
        self._setup_model()
        self._setup_algorithm()
        self._setup_evaluator()
        self._setup_eval_loaders()
        # 所有组件就绪后, 统一触发 Logger IO (创建目录, 写 config.json)
        self.logger.start()
        self.logger.info(f"Device: {self.device} | Clients: {len(self.client_ids)}")
        self.logger.info(
            f"Server: {type(self.server).__name__} | "
            f"Client: {self._callable_name(self.client_class)}"
        )
        self._setup_parallel_training()

    @staticmethod
    def _callable_name(obj: Any) -> str:
        if isinstance(obj, partial):
            return BaseRunner._callable_name(obj.func)
        return getattr(obj, "__name__", type(obj).__name__)

    def _setup_data_pipeline(self) -> None:
        """构建 TaskSet 与 DatasetStores."""
        self.data_config = DataConfig.from_dict(self.config.get("data", {}))

        # 使用工厂函数构建划分器
        partitioner = build_partitioner(self.data_config.partitioner, self.seed)

        generator = TaskGenerator(
            dataset_name=self.data_config.dataset,
            root=self.data_config.root,
            partitioner=partitioner,
            num_clients=self.training_config.num_clients,
            seed=self.seed,
            enable_proxy=self.data_config.enable_proxy,
            use_aug=self.data_config.use_aug,
        )
        self.task_set, self.dataset_stores = generator.generate()

    def _setup_model(self) -> None:
        """构建模型工厂函数 (model_fn), 供构建客户端时重复调用."""
        model_conf     = self.config["model"]
        model_cls      = MODEL_REGISTRY.get(model_conf["name"])
        self.model_name = model_conf["name"]
        self.model_params = dict(model_conf.get("params", {}))
        self.model_fn  = partial(model_cls, **self.model_params)

    def _setup_parallel_training(self) -> None:
        """按 parallel 配置初始化 Ray 执行器; 失败时回退串行."""
        self.parallel_config = ParallelConfig.from_dict(self.config.get("parallel", {}))
        self.config["parallel"] = self.parallel_config.to_dict()
        executor, reason = try_create_ray_executor(self.parallel_config)
        if reason:
            self.logger.info(reason)
        if executor is None:
            self._train_executor = None
            self.logger.info("Local training backend: sequential")
            return
        try:
            executor.start(
                task_set=self.task_set,
                stores=self.dataset_stores,
                model_name=self.model_name,
                model_params=self.model_params,
                default_client_class=self.client_class,
            )
            self._train_executor = executor
            self.logger.info(
                f"Local training backend: ray | gpu_ids={executor.gpu_ids} | "
                f"actors={executor.num_actors}"
            )
        except Exception as exc:
            self.logger.info(f"Ray 启动失败 ({exc}), 回退到 sequential")
            try:
                executor.shutdown()
            except Exception:
                pass
            self._train_executor = None

    def _shutdown_parallel_training(self) -> None:
        if self._train_executor is not None:
            self._train_executor.shutdown()
            self._train_executor = None

    def _setup_algorithm(self) -> None:
        """通过 ALGORITHM_REGISTRY 构建 (server, client_class), 并列举客户端 ID."""
        algo_conf = self.config["algorithm"]
        self.server, self.client_class = ALGORITHM_REGISTRY.build(
            algo_conf["name"],
            model=self.model_fn(),
            device=self.device,
            config=self.config,
            seed=self.seed,
            **algo_conf.get("params", {}),
        )
        self.client_ids = self.task_set.list_client_ids(exclude_server=True)

    def _setup_evaluator(self) -> None:
        """构建默认 Evaluator (accuracy + loss) 与损失函数."""
        self.evaluator = Evaluator(metrics={"accuracy": Accuracy(), "loss": AverageLoss()})
        self.criterion = nn.CrossEntropyLoss()

    def _setup_eval_loaders(self) -> None:
        """
        一次性构建全程不变的评估 DataLoader:
          - eval_loader   : 全局干净测试集
          - proxy_loader  : Server 端 BN 校准代理集 (若 TaskSet 内无 proxy 任务则为 None)
        """
        client_config = ClientConfig.from_dict(self.config.get("client", {}))

        # 1. 干净测试集
        test_task  = self.task_set.get_task(OWNER_SERVER, SPLIT_TEST_GLOBAL)
        test_store = self.dataset_stores[test_task.dataset_tag]
        self.eval_loader = DataLoader(
            Subset(test_store.dataset, test_task.indices),
            batch_size=client_config.batch_size, shuffle=False,
            num_workers=client_config.num_workers,
        )

        # 2. Proxy 数据集（可选）
        proxy_task = self.task_set.try_get_task(OWNER_SERVER, SPLIT_PROXY)
        if proxy_task is not None:
            proxy_store = self.dataset_stores[proxy_task.dataset_tag]
            self.proxy_loader = DataLoader(
                Subset(proxy_store.dataset, proxy_task.indices),
                batch_size=client_config.batch_size, shuffle=True,
                num_workers=client_config.num_workers,
            )
        else:
            self.proxy_loader = None

    def run(self) -> None:
        """联邦学习主控循环. 使用 Logger Context Manager 保证 close() 必被调用."""
        training_conf = self.config["training"]
        total_rounds  = training_conf["rounds"]

        self.logger.info(">>> Start Training")
        # best_acc = 0.0

        try:
            for round_idx in range(total_rounds):
                self.logger.info(f"--- Round {round_idx} / {total_rounds - 1} ---")

                # 1. 选人
                selected_ids = self._select_clients(round_idx)

                # 2. 广播全局模型
                server_payloads = self.server.broadcast(selected_ids)

                # 3. 本地训练
                updates, round_lr = self._run_local_training(selected_ids, server_payloads, round_idx)

                # 4. 服务端聚合
                self.server.step(updates, proxy_loader=self.proxy_loader)

                # 5. 训练指标汇总并记录
                train_metrics = self._aggregate_train_metrics(updates)
                train_metrics["train/lr"] = round_lr
                self.logger.log_metrics(train_metrics, step=round_idx)

                # 6. 全局评估（每轮）
                test_metrics = self._run_global_eval(round_idx)

                # # 7. 最优 Checkpoint
                # if test_metrics.get("accuracy", 0.0) > best_acc:
                #     best_acc = test_metrics["accuracy"]
                #     self._save_checkpoint(round_idx)

            # self.logger.info(f"Training Finished. Best Accuracy: {best_acc:.4f}")
            self.logger.info("Training Finished.")

        finally:
            self._shutdown_parallel_training()
            self.logger.close()

    def _get_prev_global_delta_cpu(self) -> Optional[Dict[str, torch.Tensor]]:
        """上一轮聚合 delta 的 CPU 副本; 无则 None. 不需要的策略可忽略该属性."""
        prev = getattr(self.server, "last_aggregated_delta", None)
        return _clone_state_dict_cpu(prev)

    def _inject_prev_global_delta(
        self,
        client: BaseClient,
        prev_global_delta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """始终写入可选属性; BadNets/Batman 等不读取即可."""
        if prev_global_delta is None:
            prev_global_delta = self._get_prev_global_delta_cpu()
        client.prev_global_delta = prev_global_delta

    def _create_client(self, cid: str, round_config: ClientConfig) -> BaseClient:
        return self.client_class(
            client_id=cid, task_set=self.task_set, stores=self.dataset_stores,
            model=self.model_fn(), device=self.device, config=round_config,
            evaluator=self.evaluator,
        )

    def _default_model_params_for_client(self, cid: str) -> Dict[str, Any]:
        """子类可覆盖以提供 per-client 模型超参 (如 HFL capability p)."""
        return {}

    def _build_client_job_specs(
        self,
        selected_ids: List[str],
        client_config: ClientConfig,
        round_idx: int,
    ) -> Dict[str, Dict[str, Any]]:
        """构建可序列化的客户端任务规格, 供 Ray Worker 重建 Client."""
        client_cls_spec = serialize_client_class(self.client_class)
        specs: Dict[str, Dict[str, Any]] = {}
        for cid in selected_ids:
            spec: Dict[str, Any] = {
                "kind": "benign",
                "client_class": client_cls_spec,
                "round_idx": round_idx,
            }
            model_params = self._default_model_params_for_client(cid)
            if model_params:
                spec["model_params"] = model_params
            specs[cid] = spec
        return specs

    def _prepare_local_training(
        self, round_idx: int,
    ) -> tuple[ClientConfig, float]:
        client_config = ClientConfig.from_dict(self.config.get("client", {}))
        training_conf = self.config["training"]
        total_rounds = training_conf["rounds"]
        schedule_cfg = training_conf.get("lr_schedule", {})

        lr = get_lr_from_schedule(
            schedule_cfg, base_lr=client_config.trainer_config.lr,
            round_idx=round_idx, total_rounds=total_rounds,
        )
        client_config.trainer_config.lr = lr
        return client_config, lr

    def _run_local_training(
        self, selected_ids: List[str], server_payloads: Dict[str, Any], round_idx: int,
    ) -> List[Dict[str, Any]]:
        """本地训练: Ray 并行或串行回退."""
        client_config, lr = self._prepare_local_training(round_idx)
        prev_global_delta = self._get_prev_global_delta_cpu()

        if self._train_executor is not None:
            job_specs = self._build_client_job_specs(selected_ids, client_config, round_idx)
            updates = self._train_executor.train_round(
                selected_ids=selected_ids,
                global_state=self.server.global_model.state_dict(),
                client_config=client_config,
                round_idx=round_idx,
                job_specs=job_specs,
                prev_global_delta=prev_global_delta,
            )
            return updates, lr

        updates: List[Dict[str, Any]] = []
        for cid in selected_ids:
            client = self._create_client(cid, client_config)
            self._inject_prev_global_delta(client, prev_global_delta)
            payload = client.step(server_payloads[cid])
            updates.append(payload)
            del client

        return updates, lr

    def _select_clients(self, round_idx: int) -> List[str]:
        """从全量客户端中随机选取本轮参与者, 委托给 server.select_clients."""
        clients_frac = self.training_config.clients_fraction
        num_select   = max(1, int(len(self.client_ids) * clients_frac))
        selected     = self.server.select_clients(self.client_ids, num_select)
        self.logger.info(f"Selected {len(selected)} clients: {selected}")
        return selected

    def _run_global_eval(self, round_idx: int) -> Dict[str, float]:
        """在全局干净测试集上评估服务端全局模型, 由 Evaluator 执行."""
        results = self.server.eval(
            evaluator=self.evaluator, dataloader=self.eval_loader,
            criterion=self.criterion, device=self.device,
        )
        prefixed = {f"global/{k}": v for k, v in results.items()}
        self.logger.log_metrics(prefixed, step=round_idx)
        summary = " | ".join(f"{k}: {v:.4f}" for k, v in results.items())
        self.logger.info(f"[Global Eval] {summary}")
        return results

    def _aggregate_train_metrics(self, updates: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        按样本数加权平均客户端上传的训练指标.

        兼容 BaseClient.package() 中 metrics 字段为标量 (train_loss float) 的情况.
        """
        total_samples = sum(p["num_samples"] for p in updates)
        if total_samples == 0:
            return {}

        aggregated: Dict[str, float] = {}
        first_metrics = updates[0].get("metrics", {})

        if isinstance(first_metrics, dict):
            for key in first_metrics.keys():
                weighted = sum(p["metrics"][key] * p["num_samples"] for p in updates)
                aggregated[f"train/avg_{key}"] = weighted / total_samples
        else:
            # 标量 (BaseClient 默认返回 train_loss float)
            weighted = sum(p["metrics"] * p["num_samples"] for p in updates)
            aggregated["train/avg_loss"] = weighted / total_samples

        return aggregated

    def _save_checkpoint(self, round_idx: int, tag: str = "best") -> None:
        """将全局模型状态与实验配置持久化到 run_dir."""
        filename = f"checkpoint_{tag}.pth"
        path = os.path.join(self.logger.run_dir, filename)
        torch.save(
            {"round": round_idx, "model_state_dict": self.server.global_model.state_dict(), "config": self.config},
            path,
        )
        self.logger.info(f"Checkpoint saved -> {filename} (round {round_idx})")

    def _set_seed(self, seed: int) -> None:
        """设置全局随机种子, 保证实验可复现."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # 确定性慢开关 - 开启将准确复现结果, 但会降低性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
