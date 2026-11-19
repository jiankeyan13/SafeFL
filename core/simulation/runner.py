"""Single, algorithm-agnostic Federated Learning Runner for SafeFL v2."""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from core.attack.coordinator import AttackCoordinator
from core.config import GlobalConfig
from core.defense import build_defense
from core.server import Server
from core.simulation.executor import BaseExecutor, create_executor
from core.types import ClientUpdate, ServerContext
from core.utils.evaluator import Evaluator
from core.utils.logger import Logger
from core.utils.lr_schedule import get_lr_from_schedule
from core.utils.registry import MODEL_REGISTRY
from core.utils.trainer import build_criterion
from data.hub import DataHub


class Runner:
    """Drives the standard 6-step FL round loop without any algorithm-specific branches."""

    def __init__(self, config: Optional[Dict[str, Any] | GlobalConfig] = None) -> None:
        if config is None:
            self.config = GlobalConfig()
        elif isinstance(config, GlobalConfig):
            self.config = config
        else:
            self.config = GlobalConfig.from_dict(config)

        self.device = torch.device(self.config.device)
        self.seed = int(self.config.seed)
        self._set_seed(self.seed)

        self.logger = Logger.from_logger_config(
            self.config.logger_config,
            self.config.to_dict(),
        )
        self.logger.start()

        # 1. Data Hub
        self.hub = DataHub.build(
            self.config.data,
            num_clients=self.config.training.num_clients,
            seed=self.seed,
        )

        # 2. Server + Defense
        global_model = MODEL_REGISTRY.build(
            self.config.model.name,
            **self.config.model.params,
        )
        defense = build_defense(self.config.defense, seed=self.seed)
        self.server = Server(
            model=global_model,
            device=self.device,
            defense=defense,
            seed=self.seed,
        )

        # 3. Attack Coordinator (inactive no-op when attack.enabled=False)
        self.coordinator = AttackCoordinator(
            self.config.attack,
            self.hub.client_ids,
            seed=self.seed,
        )

        # 4. Evaluation Loaders
        self.evaluator = Evaluator()
        self.criterion = build_criterion(self.config.client)
        eval_bs = max(self.config.client.batch_size, 128)
        self.clean_eval_loader = self.hub.test().loader(
            batch_size=eval_bs,
            shuffle=False,
            device=self.device,
        )
        self.poisoned_eval_loader = (
            self.hub.test().loader(
                batch_size=eval_bs,
                shuffle=False,
                device=self.device,
                poisoner=self.coordinator.eval_poisoner,
                mode="eval",
                return_orig_label=True,
            )
            if self.coordinator.eval_poisoner is not None
            else None
        )

        # 5. Homomorphic Executor (Local or Ray)
        self.executor: BaseExecutor = create_executor(
            self.config.parallel,
            data_spec=self.hub.spec,
            model_cfg=self.config.model,
            default_device=self.device,
            seed=self.seed,
            run_dir=self.logger.run_dir,
            info_fn=self.logger.info,
        )

        self.logger.info(
            f"Initialized SafeFL v2 | device={self.device} | clients={self.hub.num_clients} | "
            f"defense={type(self.server.defense).__name__} | "
            f"attack_enabled={self.coordinator.enabled}"
        )
        if self.coordinator.enabled:
            strategies = [s.name for s in self.config.attack.strategies]
            self.logger.info(
                f"Compromised clients ({len(self.coordinator.malicious_ids)}): "
                f"{sorted(self.coordinator.malicious_ids)} | strategies={strategies}"
            )

    def run(self) -> None:
        total_rounds = int(self.config.training.rounds)
        num_select = max(1, int(self.hub.num_clients * self.config.training.clients_fraction))
        self.logger.info(f">>> Start Training ({total_rounds} rounds)")

        try:
            for round_idx in range(total_rounds):
                self.logger.info(f"--- Round {round_idx} / {total_rounds - 1} ---")

                # Step 1: Resolve LR, select participants, and plan tasks
                round_lr = get_lr_from_schedule(
                    self.config.training.lr_schedule,
                    base_lr=self.config.client.lr,
                    round_idx=round_idx,
                    total_rounds=total_rounds,
                )
                round_train_cfg = self.config.client.with_overrides(lr=round_lr)
                selected_ids = self.coordinator.select_clients(
                    round_idx, num_select, self.server.select_clients
                )
                tasks = self.coordinator.plan_round(round_idx, selected_ids, round_train_cfg)

                active_mal = [t.client_id for t in tasks if t.is_malicious]
                if self.coordinator.enabled:
                    self.logger.info(
                        f"Selected {len(selected_ids)} clients: {selected_ids} "
                        f"(active attackers: {active_mal})"
                    )
                else:
                    self.logger.info(f"Selected {len(selected_ids)} clients: {selected_ids}")

                # Step 2: Broadcast downlink packet
                packet = self.server.broadcast(round_idx=round_idx, lr=round_lr)

                # Step 3: Execute local/attack steps via LocalExecutor or RayExecutor
                raw_results = self.executor.run_round(tasks, packet)

                # Step 4: Persist cross-round attacker states & run optional collusion hook
                updates = self.coordinator.finalize_round(round_idx, raw_results)

                # Step 5: Server defense & global model update
                server_ctx = ServerContext(
                    round_idx=round_idx,
                    client_lr=round_lr,
                    device=self.device,
                    train_cfg=round_train_cfg,
                    proxy_data=self.hub.proxy(),
                    run_dir=self.logger.run_dir,
                )
                defense_metrics = self.server.step(updates, server_ctx)

                # Step 6: Aggregate round metrics & run global evaluation
                round_metrics = self._aggregate_client_metrics(updates)
                round_metrics["train/lr"] = round_lr
                round_metrics.update(defense_metrics)
                self.logger.log_metrics(round_metrics, step=round_idx)

                self._run_global_eval(round_idx)

            self.logger.info("Training Finished.")
        finally:
            self.executor.shutdown()
            self.hub.evict_device()
            self.logger.close()

    def _run_global_eval(self, round_idx: int) -> Dict[str, float]:
        results = self.server.evaluate(self.evaluator, self.clean_eval_loader, self.criterion)
        if self.poisoned_eval_loader is not None:
            bd = self.evaluator.evaluate_backdoor(
                self.server.global_model,
                self.poisoned_eval_loader,
                self.criterion,
                self.device,
            )
            results.update(bd)

        prefixed = {f"global/{k}": v for k, v in results.items()}
        self.logger.log_metrics(prefixed, step=round_idx)

        summary = f"accuracy: {results['accuracy']:.4f} | loss: {results.get('loss', 0.0):.4f}"
        if "asr" in results:
            summary += f" | asr: {results['asr']:.4f} | backdoor_acc: {results['backdoor_acc']:.4f}"
        self.logger.info(f"[Global Eval] {summary}")
        return results

    @staticmethod
    def _aggregate_client_metrics(updates: List[ClientUpdate]) -> Dict[str, float]:
        if not updates:
            return {}

        all_keys = {k for u in updates for k in u.metrics.keys()}
        aggregated: Dict[str, float] = {}
        for key in sorted(all_keys):
            pairs = [(u.metrics[key], max(u.num_samples, 1)) for u in updates if key in u.metrics]
            tot_w = sum(w for _, w in pairs)
            if tot_w <= 0:
                continue
            val = sum(v * w for v, w in pairs) / tot_w
            metric_name = f"train/avg_{key}" if "/" not in key else f"train/{key}"
            aggregated[metric_name] = val
        return aggregated

    def save_checkpoint(self, round_idx: int, tag: str = "best") -> str:
        filename = f"checkpoint_{tag}.pth"
        path = os.path.join(self.logger.run_dir, filename)
        torch.save(
            {
                "round": round_idx,
                "model_state_dict": self.server.global_model.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )
        self.logger.info(f"Checkpoint saved -> {filename} (round {round_idx})")
        return path

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
