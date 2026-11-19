from __future__ import annotations

import csv
import os
import random
from typing import Any, Dict, List, Set, Optional

import torch
from torch.utils.data import DataLoader, Subset

from core.attack import build_attack
from core.client.malicious_client import MaliciousClient
from core.simulation.base_runner import BaseRunner
from core.config import AttackStrategyConfig, ClientConfig, apply_malicious_epochs_override
from data.constants import OWNER_SERVER, SPLIT_TEST_GLOBAL, client_owner


class Runner(BaseRunner):
    """
    支持攻击的联邦学习 Runner.
    根据 AttackConfig 创建恶意客户端, 支持多种攻击策略 (通过 fraction 分配).
    """

    def _setup(self) -> None:
        self.attack_config = self.global_config.attack
        self.current_round = 0
        self.malicious_client_ids: Set[str] = set()
        self.client_attack_map: Dict[str, Any] = {}
        self.client_strategy_map: Dict[str, AttackStrategyConfig] = {}
        self.poisoned_eval_loader: Optional[DataLoader] = None
        self._lp_layer_score_dir: Optional[str] = None
        super()._setup()
        if self.attack_config.enabled:
            self._setup_attack()
            self._setup_poisoned_eval_loader()

    def _setup_poisoned_eval_loader(self) -> None:
        """构建用于 ASR 评估的投毒测试集加载器。"""
        if not self.attack_config.enabled or not self.attack_config.strategies:
            return

        # 默认取第一个攻击策略进行 ASR 评估 (通常实验只开启一种攻击)
        strategy = self.attack_config.strategies[0]
        attack_profile = build_attack(strategy, seed=self.seed)

        # 获取全局干净测试集
        test_task = self.task_set.get_task(OWNER_SERVER, SPLIT_TEST_GLOBAL)
        test_store = self.dataset_stores[test_task.dataset_tag]
        clean_test_dataset = Subset(test_store.dataset, test_task.indices)

        # 包装为投毒测试集 (mode='test' 表示全部投毒, 同时返回目标标签与原始标签)
        poisoned_test_dataset = attack_profile.poison_dataset(
            clean_test_dataset, mode="test", return_original_label=True
        )

        client_config = ClientConfig.from_dict(self.config.get("client", {}))
        self.poisoned_eval_loader = DataLoader(
            poisoned_test_dataset,
            batch_size=client_config.batch_size,
            shuffle=False,
            num_workers=client_config.num_workers,
        )
        self.logger.info(
            f"Poisoned eval loader initialized for ASR & Backdoor Acc (strategy: {strategy.name})"
        )

    def _setup_attack(self) -> None:
        """初始化攻击策略, 按 fraction 将恶意客户端分配到各策略."""
        num_malicious = max(0, int(len(self.client_ids) * self.attack_config.malicious_fraction))
        if num_malicious == 0:
            return

        # 与官方一致: 固定第 0 组 client_0..client_{num_malicious-1} 为恶意客户端
        malicious_list = [client_owner(i) for i in range(num_malicious)]
        self.malicious_client_ids = set(malicious_list)

        start_idx = 0
        for strategy in self.attack_config.strategies:
            count = int(num_malicious * strategy.fraction)
            end_idx = min(start_idx + count, len(malicious_list))

            attack_profile = build_attack(strategy, seed=self.seed)
            if strategy.name == "pgd" and getattr(attack_profile, "dataset", None) is None:
                attack_profile.dataset = self.data_config.dataset
            for cid in malicious_list[start_idx:end_idx]:
                self.client_attack_map[cid] = attack_profile
                self.client_strategy_map[cid] = strategy

            start_idx = end_idx

        self.logger.info(
            f"Attack enabled: {len(self.malicious_client_ids)} malicious clients, "
            f"strategies: {[s.name for s in self.attack_config.strategies]}, "
            f"malicious ids: {sorted(list(self.malicious_client_ids))}"
        )
        self._setup_lp_layer_logging()

    def _setup_lp_layer_logging(self) -> None:
        """Centralize LP layer-score dumps under the experiment run directory."""
        if not any(s.name == "lp" for s in self.attack_config.strategies):
            self._lp_layer_score_dir = None
            return
        self._lp_layer_score_dir = os.path.join(self.logger.run_dir, "lp_layer_scores")
        os.makedirs(self._lp_layer_score_dir, exist_ok=True)
        # Ensure local + Ray workers write into the same run directory.
        for cid, attack_profile in self.client_attack_map.items():
            strategy = self.client_strategy_map.get(cid)
            if strategy is None or strategy.name != "lp":
                continue
            if hasattr(attack_profile, "layer_score_dir"):
                attack_profile.layer_score_dir = self._lp_layer_score_dir

    def _build_client_job_specs(
        self,
        selected_ids: List[str],
        client_config: ClientConfig,
        round_idx: int,
    ) -> Dict[str, Dict[str, Any]]:
        specs = super()._build_client_job_specs(selected_ids, client_config, round_idx)
        if not self.attack_config.enabled:
            return specs

        for cid in selected_ids:
            if cid not in self.malicious_client_ids:
                continue
            strategy = self.client_strategy_map.get(cid)
            if strategy is None:
                continue
            params = dict(strategy.params)
            params.setdefault("seed", self.seed)
            if strategy.name == "pgd":
                params.setdefault("dataset", self.data_config.dataset)
            # 将 driver 上已解析的绝对 delta_log_dir 传给 Ray worker
            attack_profile = self.client_attack_map.get(cid)
            resolved_dir = getattr(attack_profile, "delta_log_dir", None)
            if resolved_dir:
                params.setdefault("delta_log_dir", resolved_dir)
            if strategy.name == "lp" and self._lp_layer_score_dir:
                params.setdefault("layer_score_dir", self._lp_layer_score_dir)
            spec: Dict[str, Any] = {
                "kind": "malicious",
                "strategy": {
                    "name": strategy.name,
                    "fraction": strategy.fraction,
                    "params": params,
                },
                "malicious_epochs": self.attack_config.malicious_epochs,
                "round_idx": round_idx,
            }
            model_params = self._default_model_params_for_client(cid)
            if model_params:
                spec["model_params"] = model_params
            specs[cid] = spec
        return specs

    def _create_client(self, cid: str, round_config: ClientConfig):
        if self.attack_config.enabled and cid in self.malicious_client_ids:
            attack_profile = self.client_attack_map.get(cid)
            rc = apply_malicious_epochs_override(
                round_config, self.attack_config.malicious_epochs
            )
            client_cls = getattr(attack_profile, "client_class", None) or MaliciousClient
            return client_cls(
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
        """评估全局模型，包括 CDA (干净数据准确率), ASR 与 Backdoor Acc."""
        # 1. CDA 评估
        results = self.server.eval(
            evaluator=self.evaluator, dataloader=self.eval_loader,
            criterion=self.criterion, device=self.device,
        )
        prefixed = {f"global/{k}": v for k, v in results.items()}

        # 2. ASR 与 Backdoor Acc (同一次前向)
        if self.attack_config.enabled and self.poisoned_eval_loader is not None:
            bd_res = self.evaluator.evaluate_backdoor(
                self.server.global_model,
                self.poisoned_eval_loader,
                self.criterion,
                self.device,
            )
            results["asr"] = bd_res["asr"]
            results["backdoor_acc"] = bd_res["backdoor_acc"]
            prefixed["global/asr"] = results["asr"]
            prefixed["global/backdoor_acc"] = results["backdoor_acc"]
            if "backdoor_loss" in bd_res:
                results["backdoor_loss"] = bd_res["backdoor_loss"]
                prefixed["global/backdoor_loss"] = results["backdoor_loss"]

        # 统一记录并打印一行摘要
        self.logger.log_metrics(prefixed, step=round_idx)
        summary = f"accuracy: {results['accuracy']:.4f} | loss: {results['loss']:.4f}"
        if "asr" in results:
            summary += f" | asr: {results['asr']:.4f} | backdoor_acc: {results['backdoor_acc']:.4f}"
        self.logger.info(f"[Global Eval] {summary}")

        return results

    def run(self) -> None:
        """联邦学习主控循环, 每轮更新 current_round 供 MaliciousClient 使用."""
        training_conf = self.config["training"]
        total_rounds = training_conf["rounds"]

        self.logger.info(">>> Start Training")
        # best_acc = 0.0

        try:
            for round_idx in range(total_rounds):
                self.current_round = round_idx
                self.logger.info(f"--- Round {round_idx} / {total_rounds - 1} ---")

                selected_ids = self._select_clients(round_idx)
                server_payloads = self.server.broadcast(selected_ids)
                updates, round_lr = self._run_local_training(selected_ids, server_payloads, round_idx)
                self.server.step(updates, proxy_loader=self.proxy_loader, client_lr=round_lr)

                train_metrics = self._aggregate_train_metrics(updates)
                train_metrics["train/lr"] = round_lr
                self.logger.log_metrics(train_metrics, step=round_idx)
                self._log_layer_mask_stats(updates, round_idx)
                self._log_layer_score_stats(updates, round_idx)

                test_metrics = self._run_global_eval(round_idx)

                # if test_metrics.get("accuracy", 0.0) > best_acc:
                #     best_acc = test_metrics["accuracy"]
                #     self._save_checkpoint(round_idx)

            # self.logger.info(f"Training Finished. Best Accuracy: {best_acc:.4f}")
            self.logger.info("Training Finished.")
        finally:
            self._shutdown_parallel_training()
            self.logger.close()

    def _log_layer_mask_stats(self, updates: List[Dict[str, Any]], round_idx: int) -> None:
        """Record attacker per-layer mask counts to wandb and CSV."""
        records = []
        for payload in updates:
            stats = payload.get("layer_mask_stats")
            if not isinstance(stats, dict) or not stats.get("layers"):
                continue
            client_id = str(payload.get("client_id", ""))
            records.append((client_id, stats))
        if not records:
            return

        wandb_metrics: Dict[str, float] = {}
        grouped: Dict[str, List[tuple]] = {}
        for client_id, stats in records:
            attack_name = str(stats.get("attack", "neurotoxin"))
            grouped.setdefault(attack_name, []).append((client_id, stats))

        label_map = {"neurotoxin": "Neurotoxin", "sparalignfl": "SparAlignFL"}
        prefix_map = {"neurotoxin": "nt", "sparalignfl": "safl"}
        for attack_name, attack_records in grouped.items():
            label = label_map.get(attack_name, attack_name)
            prefix = prefix_map.get(attack_name, attack_name)
            csv_path = os.path.join(self.logger.run_dir, f"{attack_name}_layer_masks.csv")
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["round", "client_id", "layer", "n_total", "n_masked", "ratio"],
                )
                if write_header:
                    writer.writeheader()
                for client_id, stats in attack_records:
                    total_ratio = float(stats.get("ratio", 0.0))
                    total_masked = int(stats.get("n_masked", 0))
                    total_n = int(stats.get("n_total", 0))
                    self.logger.info(
                        f"[{label} mask] round={round_idx} client={client_id} "
                        f"global {total_masked}/{total_n} ({total_ratio:.4f})"
                    )
                    wandb_metrics[f"{prefix}/mask_n_masked"] = float(total_masked)
                    wandb_metrics[f"{prefix}/mask_n_total"] = float(total_n)
                    wandb_metrics[f"{prefix}/mask_ratio"] = total_ratio
                    lines = []
                    for layer in stats["layers"]:
                        name = str(layer["name"])
                        n_total = int(layer["n_total"])
                        n_masked = int(layer["n_masked"])
                        ratio = float(layer["ratio"])
                        writer.writerow(
                            {
                                "round": round_idx,
                                "client_id": client_id,
                                "layer": name,
                                "n_total": n_total,
                                "n_masked": n_masked,
                                "ratio": f"{ratio:.6f}",
                            }
                        )
                        wandb_metrics[f"{prefix}/mask_ratio/{name}"] = ratio
                        wandb_metrics[f"{prefix}/mask_n/{name}"] = float(n_masked)
                        lines.append(f"{name}={n_masked}/{n_total}({ratio:.3f})")
                    self.logger.info(f"[{label} mask layers] " + " ".join(lines))

        wandb_run = self.logger.wandb_run
        if wandb_run is not None and wandb_metrics:
            wandb_run.log(wandb_metrics, step=round_idx)

    def _log_layer_score_stats(self, updates: List[Dict[str, Any]], round_idx: int) -> None:
        """Record every SparAlignFL contrastive layer score to CSV and W&B."""
        records = []
        for payload in updates:
            stats = payload.get("layer_score_stats")
            if not isinstance(stats, dict) or not stats.get("layers"):
                continue
            client_id = str(payload.get("client_id", ""))
            records.append((client_id, stats))
        if not records:
            return

        wandb_metrics: Dict[str, float] = {}
        for client_id, stats in records:
            attack_name = str(stats.get("attack", "sparalignfl"))
            prefix = "safl" if attack_name == "sparalignfl" else attack_name
            csv_path = os.path.join(self.logger.run_dir, f"{attack_name}_layer_scores.csv")
            write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["round", "client_id", "rank", "layer", "score", "selected"],
                )
                if write_header:
                    writer.writeheader()

                score_lines = []
                for layer in stats["layers"]:
                    name = str(layer["name"])
                    rank = int(layer["rank"])
                    score = float(layer["score"])
                    selected = bool(layer.get("selected", False))
                    writer.writerow(
                        {
                            "round": round_idx,
                            "client_id": client_id,
                            "rank": rank,
                            "layer": name,
                            "score": f"{score:.9g}",
                            "selected": int(selected),
                        }
                    )
                    # The CSV preserves client identity; these scalar series
                    # make the per-layer ranking directly visible in W&B.
                    wandb_metrics[f"{prefix}/layer_score/{name}"] = score
                    wandb_metrics[f"{prefix}/layer_rank/{name}"] = float(rank)
                    wandb_metrics[f"{prefix}/layer_selected/{name}"] = float(selected)
                    score_lines.append(
                        f"{name}={score:.6f}"
                        + ("*" if selected else "")
                    )

            self.logger.info(
                f"[{attack_name} layer scores] round={round_idx} client={client_id} "
                + " ".join(score_lines)
            )

        wandb_run = self.logger.wandb_run
        if wandb_run is not None and wandb_metrics:
            wandb_run.log(wandb_metrics, step=round_idx)
