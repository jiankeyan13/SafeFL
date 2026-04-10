from __future__ import annotations

import random
from typing import Any, Dict, List, Set, Optional

import torch
from torch.utils.data import DataLoader, Subset

from core.attack import build_attack
from core.client.malicious_client import MaliciousClient
from core.simulation.base_runner import BaseRunner
from core.utils.configs import ClientConfig, apply_malicious_epochs_override
from data.constants import OWNER_SERVER, SPLIT_TEST_GLOBAL


class Runner(BaseRunner):
    """
    支持攻击的联邦学习 Runner.
    根据 AttackConfig 创建恶意客户端, 支持多种攻击策略 (通过 fraction 分配).
    """

    def _setup(self) -> None:
        super()._setup()
        self.attack_config = self.global_config.attack
        self.current_round = 0
        self.malicious_client_ids: Set[str] = set()
        self.client_attack_map: Dict[str, Any] = {}
        self.poisoned_eval_loader: Optional[DataLoader] = None
        if self.attack_config.enabled:
            self._setup_attack()
            self._setup_poisoned_eval_loader()

    def _setup_poisoned_eval_loader(self) -> None:
        """构建用于 ASR 评估的投毒测试集加载器。"""
        if not self.attack_config.enabled or not self.attack_config.strategies:
            return

        # 默认取第一个攻击策略进行 ASR 评估 (通常实验只开启一种攻击)
        strategy = self.attack_config.strategies[0]
        attack_profile = build_attack(strategy)

        # 获取全局干净测试集
        test_task = self.task_set.get_task(OWNER_SERVER, SPLIT_TEST_GLOBAL)
        test_store = self.dataset_stores[test_task.dataset_tag]
        clean_test_dataset = Subset(test_store.dataset, test_task.indices)

        # 包装为投毒测试集 (mode='test' 表示全部投毒)
        poisoned_test_dataset = attack_profile.poison_dataset(clean_test_dataset, mode="test")

        client_config = ClientConfig.from_dict(self.config.get("client", {}))
        self.poisoned_eval_loader = DataLoader(
            poisoned_test_dataset,
            batch_size=client_config.batch_size,
            shuffle=False,
            num_workers=client_config.num_workers,
        )
        self.logger.info(f"Poisoned eval loader initialized for ASR (strategy: {strategy.name})")

    def _setup_attack(self) -> None:
        """初始化攻击策略, 按 fraction 将恶意客户端分配到各策略."""
        num_malicious = max(0, int(len(self.client_ids) * self.attack_config.malicious_fraction))
        if num_malicious == 0:
            return

        # 使用固定种子进行采样，确保每次运行选中的恶意客户端一致
        rng = random.Random(self.seed)
        malicious_list = rng.sample(self.client_ids, num_malicious)
        self.malicious_client_ids = set(malicious_list)

        start_idx = 0
        for strategy in self.attack_config.strategies:
            count = int(num_malicious * strategy.fraction)
            end_idx = min(start_idx + count, len(malicious_list))

            attack_profile = build_attack(strategy)
            for cid in malicious_list[start_idx:end_idx]:
                self.client_attack_map[cid] = attack_profile

            start_idx = end_idx

        self.logger.info(
            f"Attack enabled: {len(self.malicious_client_ids)} malicious clients, "
            f"strategies: {[s.name for s in self.attack_config.strategies]}, "
            f"malicious ids: {sorted(list(self.malicious_client_ids))}"
        )

    def _create_client(self, cid: str, round_config: ClientConfig):
        if self.attack_config.enabled and cid in self.malicious_client_ids:
            attack_profile = self.client_attack_map.get(cid)
            rc = apply_malicious_epochs_override(
                round_config, self.attack_config.malicious_epochs
            )
            return MaliciousClient(
                client_id=cid, task_set=self.task_set, stores=self.dataset_stores,
                model=self.model_fn(), device=self.device, config=rc,
                evaluator=self.evaluator, attack_profile=attack_profile,
                round_idx=self.current_round,
            )
        return super()._create_client(cid, round_config)

    def _select_clients(self, round_idx: int) -> List[str]:
        """选取客户端, 攻击启用时确保恶意客户端比例符合 per_round_fraction。
        使用固定种子确保每轮选择的结果是可复现且固定的。
        """
        clients_frac = self.training_config.clients_fraction
        num_select = max(1, int(len(self.client_ids) * clients_frac))

        if not self.attack_config.enabled or not self.malicious_client_ids:
            return self.server.select_clients(self.client_ids, num_select)

        # 使用基于 round_idx 的固定种子，确保每轮选择固定
        rng = random.Random(self.seed + round_idx)

        benign_ids = sorted([c for c in self.client_ids if c not in self.malicious_client_ids])
        malicious_ids = sorted(list(self.malicious_client_ids))

        # 修正：确保 num_malicious 至少为 1，如果 per_round_fraction > 0 且有恶意客户端可用
        target_malicious = int(num_select * self.attack_config.per_round_fraction)
        if self.attack_config.per_round_fraction > 0 and target_malicious == 0 and len(malicious_ids) > 0:
            target_malicious = 1
            
        num_malicious = max(0, min(target_malicious, len(malicious_ids)))
        num_benign = num_select - num_malicious
        num_benign = max(0, min(num_benign, len(benign_ids)))

        selected_malicious = rng.sample(malicious_ids, num_malicious) if num_malicious > 0 else []
        selected_benign = rng.sample(benign_ids, num_benign) if num_benign > 0 else []
        selected = selected_malicious + selected_benign
        rng.shuffle(selected)

        self.logger.info(
            f"Selected {len(selected)} clients: {selected} "
            f"(malicious: {len(selected_malicious)}, ids: {selected_malicious})"
        )
        return selected

    def _run_global_eval(self, round_idx: int) -> Dict[str, float]:
        """评估全局模型，包括 CDA (干净数据准确率) 和 ASR (攻击成功率)。"""
        # 1. CDA 评估
        results = self.server.eval(
            evaluator=self.evaluator, dataloader=self.eval_loader,
            criterion=self.criterion, device=self.device,
        )
        prefixed = {f"global/{k}": v for k, v in results.items()}

        # 2. ASR 评估
        if self.attack_config.enabled and self.poisoned_eval_loader is not None:
            asr_res = self.server.eval(self.evaluator, self.poisoned_eval_loader, self.criterion, self.device)
            results["asr"] = asr_res.get("accuracy", 0.0)
            prefixed["global/asr"] = results["asr"]

        # 统一记录并打印一行摘要
        self.logger.log_metrics(prefixed, step=round_idx)
        summary = f"accuracy: {results['accuracy']:.4f} | loss: {results['loss']:.4f}"
        if "asr" in results: summary += f" | asr: {results['asr']:.4f}"
        self.logger.info(f"[Global Eval] {summary}")

        return results

    def run(self) -> None:
        """联邦学习主控循环, 每轮更新 current_round 供 MaliciousClient 使用."""
        training_conf = self.config["training"]
        total_rounds = training_conf["rounds"]
        eval_interval = self.training_config.eval_interval

        self.logger.info(">>> Start Training")
        best_acc = 0.0

        for round_idx in range(total_rounds):
            self.current_round = round_idx
            self.logger.info(f"--- Round {round_idx} / {total_rounds - 1} ---")

            selected_ids = self._select_clients(round_idx)
            server_payloads = self.server.broadcast(selected_ids)
            updates, round_lr = self._run_local_training(selected_ids, server_payloads, round_idx)
            self.server.step(updates, proxy_loader=self.proxy_loader)

            train_metrics = self._aggregate_train_metrics(updates)
            train_metrics["train/lr"] = round_lr
            self.logger.log_metrics(train_metrics, step=round_idx)

            test_metrics = self._run_global_eval(round_idx)

            if round_idx % eval_interval == 0:
                self._run_local_eval(round_idx)

            if test_metrics.get("accuracy", 0.0) > best_acc:
                best_acc = test_metrics["accuracy"]
                self._save_checkpoint(round_idx)

        self.logger.info(f"Training Finished. Best Accuracy: {best_acc:.4f}")
        self.logger.close()
