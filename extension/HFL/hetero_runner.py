import copy
import random
from functools import partial
from typing import Dict, Any, List

import numpy as np

from core.runner import FederatedRunner
from core.utils.registry import MODEL_REGISTRY


class HeteroRunner(FederatedRunner):
    """
    Runner 子类：为每个 client 构建带 p 的模型工厂，保证子网权重能正确 load。
    """

    def _setup(self):
        super()._setup()
        model_conf = self.config["model"]
        self._model_cls = MODEL_REGISTRY.get(model_conf["name"])
        self._base_params = dict(model_conf.get("params", {}))
        self._base_params.pop("p", None)
        self.pruning_mode = self.config.get("server", {}).get("pruning", {}).get("mode", "structured")
        self.dispatch_policy = getattr(self.server, "dispatch_policy", None)

    def _client_model_params(self, cid: str) -> Dict[str, Any]:
        # 优先使用 Server 对外的稳定接口
        if hasattr(self.server, "get_client_model_kwargs"):
            return self.server.get_client_model_kwargs(cid)
        # 兜底（旧逻辑）
        if self.pruning_mode == "unstructured":
            return {"p": 1.0}
        return {"p": self.server._resolve_p_for_client(cid)}

    def _client_model_fn(self, cid: str):
        params = {**self._base_params, **self._client_model_params(cid)}
        return partial(self._model_cls, **params)

    def _run_local_training(self, client_ids, client_models, config):
        updates = []
        for cid in client_ids:
            attack_strategy = None
            if self.attack_manager:
                attack_strategy = self.attack_manager.get_strategy(cid)

            client = self.client_class(cid, self.device, self._client_model_fn(cid))
            task = self.task_set.get_task(cid, "train")
            store = self.dataset_stores[task.dataset_tag]

            payload = client.execute(
                global_state_dict=client_models[cid].get("state_dict", client_models[cid]),
                task=task,
                dataset_store=store,
                config=config,
                attack_profile=attack_strategy,
                payload=client_models[cid] if isinstance(client_models[cid], dict) else None,
                apply_fn=getattr(self.dispatch_policy, "apply_payload", None),
            )
            updates.append(payload)
            del client
        return updates

    def _run_local_evaluation(self, round_idx, config, metrics):
        all_clients = list(self.task_set._tasks.keys())
        client_candidates = [c for c in all_clients if c != "server"]
        eval_ids = random.sample(client_candidates, k=max(1, int(len(client_candidates) * 0.2)))

        results_collector: Dict[str, List[float]] = {m.name: [] for m in metrics}

        for cid in eval_ids:
            client = self.client_class(cid, self.device, self._client_model_fn(cid))
            task = self.task_set.get_task(cid, "test")
            store = self.dataset_stores[task.dataset_tag]

            model_payload = self.server.broadcast([cid])[cid]

            res = client.evaluate(
                global_state_dict=model_payload.get("state_dict", model_payload),
                task=task,
                dataset_store=store,
                config=config,
                metrics=metrics,
            )

            for key, val in res.items():
                if key in results_collector:
                    results_collector[key].append(val)
            del client

        log_dict = {}
        for metric_name, values in results_collector.items():
            if len(values) > 0:
                log_dict[f"local/avg_{metric_name}"] = np.mean(values)
                log_dict[f"local/std_{metric_name}"] = np.std(values)
        self.logger.log_metrics(log_dict, step=round_idx)

