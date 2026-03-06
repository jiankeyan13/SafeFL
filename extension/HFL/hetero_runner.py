from __future__ import annotations

import inspect
from functools import partial
from typing import Any, Dict, Tuple

import torch

from core.client.base_client import BaseClient
from core.server.refiner.base_refiner import BaseRefiner
from core.simulation.base_runner import BaseRunner
from core.utils.configs import ClientConfig
from core.utils.registry import (
    AGGREGATOR_REGISTRY,
    MODEL_REGISTRY,
    REFINER_REGISTRY,
    SCREENER_REGISTRY,
)
from extension.HFL.cap_manager import CapManager
from extension.HFL.config import HFLConfig
from extension.HFL.fedrolex_server import FedrolexServer
from extension.HFL.hetero_server import HeteroServer
import extension.HFL.sub_aggregator  # noqa: F401


class HeteroRunner(BaseRunner):
    """Runner adapted to current BaseRunner/BaseServer/BaseClient APIs."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        config = config or {}
        self._raw_user_config = config.copy()
        full_config = HFLConfig.from_dict(config).to_dict()
        super().__init__(full_config)
        # GlobalConfig 不包含 hetero, 需合并回 self.config
        self.config["hetero"] = full_config["hetero"]

    def _setup_model(self) -> None:
        model_conf = self.config["model"]
        self.model_cls = MODEL_REGISTRY.get(model_conf["name"])

        self._model_base_params = dict(model_conf.get("params", {}))
        self._model_supports_p = self._supports_kwarg(self.model_cls, "p")
        if self._model_supports_p:
            self._model_base_params.pop("p", None)

        global_params = dict(self._model_base_params)
        if self._model_supports_p:
            global_params["p"] = 1.0
        self.model_fn = partial(self.model_cls, **global_params)

    def _setup_algorithm(self) -> None:
        self.client_ids = self.task_set.list_client_ids(exclude_server=True)

        hetero_conf = self._get_hetero_config()
        if not hetero_conf:
            raise ValueError("HeteroRunner requires top-level config['hetero'].")
        self.cap_manager = CapManager(hetero_conf, seed=self.seed)
        self.cap_manager.register_clients(self.client_ids)
        self.logger.info(f"CapManager summary: {self.cap_manager.summary()}")

        aggregator_conf, screener_conf, refiner_conf = self._resolve_server_configs()

        aggregator_name = aggregator_conf.get("name", "sub_avg")
        aggregator = AGGREGATOR_REGISTRY.build(
            aggregator_name, **aggregator_conf.get("params", {})
        )

        screener = None
        if screener_conf.get("name"):
            screener = SCREENER_REGISTRY.build(
                screener_conf["name"], **screener_conf.get("params", {})
            )

        if refiner_conf.get("name"):
            refiner = REFINER_REGISTRY.build(
                refiner_conf["name"], **refiner_conf.get("params", {})
            )
        else:
            refiner = BaseRefiner(config=refiner_conf.get("params", {}))

        self.server = FedrolexServer(
            model=self.model_fn(),
            device=self.device,
            cap_manager=self.cap_manager,
            screener=screener,
            aggregator=aggregator,
            refiner=refiner,
            seed=self.seed,
        )
        # Keep the same client workflow as BaseRunner/BaseClient:
        # receive -> train -> package(delta) -> server aggregate.
        self.client_class = BaseClient

    def _create_client(self, cid: str, round_config: ClientConfig) -> BaseClient:
        model = self._build_client_model(cid)
        return self.client_class(
            client_id=cid,
            task_set=self.task_set,
            stores=self.dataset_stores,
            model=model,
            device=self.device,
            config=round_config,
            evaluator=self.evaluator,
        )

    def _build_client_model(self, client_id: str) -> torch.nn.Module:
        params = dict(self._model_base_params)
        if self._model_supports_p:
            params["p"] = self.cap_manager.get_bucketed_capability(client_id)
        return self.model_cls(**params)

    def _resolve_server_configs(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        algo_params = self.config.get("algorithm", {}).get("params", {})
        legacy_server = self.config.get("server", {})

        def pick(name: str) -> Dict[str, Any]:
            conf = algo_params.get(name)
            if conf is None:
                conf = legacy_server.get(name, {})
            return conf or {}

        return pick("aggregator"), pick("screener"), pick("refiner")

    def _get_hetero_config(self) -> Dict[str, Any]:
        if "hetero" in self.config:
            return self.config["hetero"]
        if isinstance(self._raw_user_config, dict):
            return self._raw_user_config.get("hetero", {})
        return {}

    @staticmethod
    def _supports_kwarg(callable_obj: Any, kwarg: str) -> bool:
        try:
            sig = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False
        return kwarg in sig.parameters
