from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Dict

import torch

from core.client.base_client import BaseClient
from core.client.malicious_client import MaliciousClient
from core.simulation.runner import Runner
from core.utils.configs import ClientConfig
from core.utils.registry import ALGORITHM_REGISTRY, MODEL_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.config import HFLConfig
import extension.HFL.algorithms  # noqa: F401


class HeteroRunner(Runner):
    """HFL runner with attack support inherited from Runner."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        config = config or {}
        hfl_config = HFLConfig.from_dict(config)
        full_config = hfl_config.to_dict()
        super().__init__(full_config)
        # 显式确保 self.config 中包含 hetero 键，即使它是默认值
        if "hetero" not in self.config:
            self.config["hetero"] = hfl_config.hetero.to_dict()
        # 同样确保 global_config 包含 hetero 属性
        if not hasattr(self.global_config, "hetero"):
            setattr(self.global_config, "hetero", hfl_config.hetero)

    def _setup(self) -> None:
        super()._setup()
        self._setup_attacker_capabilities()

    def _setup_model(self) -> None:
        model_conf = self.config["model"]
        self.model_cls = MODEL_REGISTRY.get(model_conf["name"])

        self._model_base_params = dict(model_conf.get("params", {}))
        self._model_supports_p = "p" in inspect.signature(self.model_cls).parameters
        if self._model_supports_p:
            self._model_base_params.pop("p", None)

        global_params = dict(self._model_base_params)
        if self._model_supports_p:
            global_params["p"] = 1.0
        self.model_fn = partial(self.model_cls, **global_params)

    def _setup_algorithm(self) -> None:
        algo_conf = self.config["algorithm"]
        self.client_ids = self.task_set.list_client_ids(exclude_server=True)

        hetero_conf = self._get_hetero_config()
        if not hetero_conf:
            raise ValueError("HeteroRunner requires top-level config['hetero'].")
        self.cap_manager = CapManager(hetero_conf, seed=self.seed)
        self.cap_manager.register_clients(self.client_ids)
        self.server, self.client_class = ALGORITHM_REGISTRY.build(
            algo_conf["name"],
            model=self.model_fn(),
            device=self.device,
            config=self.config,
            seed=self.seed,
            cap_manager=self.cap_manager,
            **algo_conf.get("params", {}),
        )
        self.logger.info(f"CapManager summary: {self.cap_manager.summary()}")

    def _create_client(self, cid: str, round_config: ClientConfig) -> BaseClient:
        model = self._build_client_model(cid)
        if self.attack_config.enabled and cid in self.malicious_client_ids:
            attack_profile = self.client_attack_map[cid]
            return MaliciousClient(
                client_id=cid, task_set=self.task_set, stores=self.dataset_stores,
                model=model, device=self.device, config=round_config, evaluator=self.evaluator,
                attack_profile=attack_profile, round_idx=self.current_round,
            )
        return self.client_class(
            client_id=cid, task_set=self.task_set, stores=self.dataset_stores,
            model=model, device=self.device, config=round_config, evaluator=self.evaluator,
        )

    def _build_client_model(self, client_id: str) -> torch.nn.Module:
        if self._model_supports_p:
            params = {**self._model_base_params, "p": self.cap_manager.get_bucketed_capability(client_id)}
            return self.model_cls(**params)
        return self.model_cls(**self._model_base_params)

    def _get_hetero_config(self) -> Dict[str, Any]:
        if "hetero" not in self.config:
            # 防御性处理：如果 self.config 仍然没有 hetero，则使用默认值
            return HFLConfig.from_dict({}).hetero.to_dict()
        return self.config["hetero"]

    def _setup_attacker_capabilities(self) -> None:
        attacker_conf = self._get_hetero_config().get("attacker", {})
        if not attacker_conf.get("enabled", False):
            return
        if not self.attack_config.enabled or not self.malicious_client_ids:
            return

        p_list = attacker_conf.get("p_list") or self.cap_manager.p_list
        self.cap_manager.assign_uniform_capabilities(
            list(self.malicious_client_ids), p_list=p_list
        )
        malicious_caps = {
            cid: self.cap_manager.get_bucketed_capability(cid)
            for cid in self.malicious_client_ids
        }
        self.logger.info(
            f"Attacker capability summary: p_list={list(p_list)}, assignments={malicious_caps}"
        )
