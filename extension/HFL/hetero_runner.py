from __future__ import annotations

import inspect
import random
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

from core.client.base_client import BaseClient
from core.client.malicious_client import MaliciousClient
from core.simulation.runner import Runner
from core.utils.configs import ClientConfig
from core.utils.registry import ALGORITHM_REGISTRY, MODEL_REGISTRY
from data.constants import SPLIT_TRAIN, train_plain_tag
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
        self._setup_global_proxy_loader()

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

    def _setup_global_proxy_loader(self) -> None:
        """
        用含毒全局训练集替换 proxy_loader，使 BN 校准统计量反映真实训练分布。
        恶意客户端持有的数据会重放投毒变换，确保统计量包含毒化影响。
        """
        dataset_name = self.data_config.dataset
        tag = train_plain_tag(dataset_name)
        full_dataset = self.dataset_stores[tag].dataset
        client_config = ClientConfig.from_dict(self.config.get("client", {}))

        datasets = []
        for cid in self.client_ids:
            task = self.task_set.try_get_task(cid, SPLIT_TRAIN)
            if task is None:
                continue
            subset = Subset(full_dataset, task.indices)
            if cid in self.malicious_client_ids:
                attack_profile = self.client_attack_map.get(cid)
                if attack_profile is not None:
                    subset = attack_profile.poison_dataset(
                        subset, mode="train",
                        client_id=cid, round_idx=0,
                    )
            datasets.append(subset)

        combined = ConcatDataset(datasets)
        self.proxy_loader = DataLoader(
            combined,
            batch_size=client_config.batch_size,
            shuffle=True,
            num_workers=client_config.num_workers,
        )
        self.logger.info(
            f"Global proxy loader rebuilt: {len(combined)} samples "
            f"(poisoned clients: {sorted(self.malicious_client_ids)})"
        )

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

    def _group_clients_by_tier(self, client_ids: List[str]) -> Dict[float, List[str]]:
        tier_map: Dict[float, List[str]] = defaultdict(list)
        for cid in client_ids:
            tier = float(self.cap_manager.get_bucketed_capability(cid))
            tier_map[tier].append(cid)
        for tier in tier_map:
            tier_map[tier] = sorted(tier_map[tier])
        return dict(tier_map)

    @staticmethod
    def _build_balanced_quota(total_count: int, tiers: List[float]) -> Dict[float, int]:
        if total_count <= 0 or not tiers:
            return {tier: 0 for tier in tiers}
        base = total_count // len(tiers)
        remainder = total_count % len(tiers)
        quota = {tier: base for tier in tiers}
        for tier in tiers[:remainder]:
            quota[tier] += 1
        return quota

    @staticmethod
    def _normalize_quota_total(
        quota: Dict[float, int], expected_total: int, tiers: List[float]
    ) -> Dict[float, int]:
        adjusted = {tier: max(0, int(quota.get(tier, 0))) for tier in tiers}
        current = sum(adjusted.values())
        if not tiers:
            return adjusted

        if current < expected_total:
            gap = expected_total - current
            for i in range(gap):
                tier = tiers[i % len(tiers)]
                adjusted[tier] += 1
        elif current > expected_total:
            overflow = current - expected_total
            idx = 0
            reversed_tiers = list(reversed(tiers))
            while overflow > 0 and idx < overflow + len(reversed_tiers) * 4:
                tier = reversed_tiers[idx % len(reversed_tiers)]
                if adjusted[tier] > 0:
                    adjusted[tier] -= 1
                    overflow -= 1
                idx += 1
        return adjusted

    def _sample_with_tier_quota(
        self,
        tiers: List[float],
        tier_to_ids: Dict[float, List[str]],
        quota: Dict[float, int],
        rng: random.Random,
    ) -> Tuple[List[str], Dict[str, Any]]:
        pools: Dict[float, List[str]] = {
            tier: list(sorted(tier_to_ids.get(tier, [])))
            for tier in tiers
        }
        selected_by_tier: Dict[float, List[str]] = {tier: [] for tier in tiers}
        shortfalls: Dict[float, int] = {}

        for tier in tiers:
            target = max(0, int(quota.get(tier, 0)))
            if target == 0:
                continue
            available = pools[tier]
            take = min(target, len(available))
            if take > 0:
                picked = rng.sample(available, take)
                selected_by_tier[tier].extend(picked)
                picked_set = set(picked)
                pools[tier] = [cid for cid in available if cid not in picked_set]
            if take < target:
                shortfalls[tier] = target - take

        shortfall_total = sum(shortfalls.values())
        fallback_picks = 0
        if shortfall_total > 0:
            fallback_pool: List[Tuple[float, str]] = []
            for tier in tiers:
                fallback_pool.extend([(tier, cid) for cid in pools[tier]])

            take_extra = min(shortfall_total, len(fallback_pool))
            if take_extra > 0:
                extra = rng.sample(fallback_pool, take_extra)
                for src_tier, cid in extra:
                    selected_by_tier[src_tier].append(cid)
                fallback_picks = take_extra

        selected_ids: List[str] = []
        selected_count_by_tier: Dict[float, int] = {}
        for tier in tiers:
            selected_count_by_tier[tier] = len(selected_by_tier[tier])
            selected_ids.extend(selected_by_tier[tier])

        details = {
            "quota_by_tier": {tier: int(quota.get(tier, 0)) for tier in tiers},
            "selected_by_tier": selected_count_by_tier,
            "shortfall_by_tier": shortfalls,
            "fallback_picks": fallback_picks,
            "unresolved_shortfall": max(0, shortfall_total - fallback_picks),
        }
        return selected_ids, details

    def _select_clients(self, round_idx: int) -> List[str]:
        clients_frac = self.training_config.clients_fraction
        num_select = max(1, int(len(self.client_ids) * clients_frac))
        rng = random.Random(self.seed + round_idx)

        all_tier_map = self._group_clients_by_tier(self.client_ids)
        tiers = sorted(all_tier_map.keys())
        if not tiers:
            return []

        malicious_enabled = self.attack_config.enabled and bool(self.malicious_client_ids)
        if malicious_enabled:
            malicious_ids = sorted(cid for cid in self.client_ids if cid in self.malicious_client_ids)
            benign_ids = sorted(cid for cid in self.client_ids if cid not in self.malicious_client_ids)

            target_malicious = int(num_select * self.attack_config.per_round_fraction)
            if self.attack_config.per_round_fraction > 0 and target_malicious == 0 and malicious_ids:
                target_malicious = 1
            num_malicious = max(0, min(target_malicious, len(malicious_ids), num_select))

            malicious_tier_map = self._group_clients_by_tier(malicious_ids)
            benign_tier_map = self._group_clients_by_tier(benign_ids)

            malicious_quota = self._build_balanced_quota(num_malicious, tiers)
            selected_malicious, malicious_details = self._sample_with_tier_quota(
                tiers, malicious_tier_map, malicious_quota, rng
            )

            total_quota = self._build_balanced_quota(num_select, tiers)
            benign_target_total = max(0, num_select - len(selected_malicious))
            benign_quota = {
                tier: max(0, total_quota[tier] - malicious_details["selected_by_tier"].get(tier, 0))
                for tier in tiers
            }
            benign_quota = self._normalize_quota_total(benign_quota, benign_target_total, tiers)
            selected_benign, benign_details = self._sample_with_tier_quota(
                tiers, benign_tier_map, benign_quota, rng
            )

            selected = selected_malicious + selected_benign
        else:
            total_quota = self._build_balanced_quota(num_select, tiers)
            selected, all_details = self._sample_with_tier_quota(
                tiers, all_tier_map, total_quota, rng
            )
            selected_malicious = []
            selected_benign = selected
            malicious_details = {
                "quota_by_tier": {tier: 0 for tier in tiers},
                "selected_by_tier": {tier: 0 for tier in tiers},
                "shortfall_by_tier": {},
                "fallback_picks": 0,
                "unresolved_shortfall": 0,
            }
            benign_details = all_details

        if len(selected) < num_select:
            selected_set = set(selected)
            remaining_pool = sorted([cid for cid in self.client_ids if cid not in selected_set])
            need = min(num_select - len(selected), len(remaining_pool))
            if need > 0:
                selected.extend(rng.sample(remaining_pool, need))

        rng.shuffle(selected)

        selected_set = set(selected)
        tier_stats: Dict[float, Dict[str, int]] = {}
        for tier in tiers:
            tier_members = set(all_tier_map.get(tier, []))
            selected_in_tier = tier_members & selected_set
            malicious_in_tier = selected_in_tier & self.malicious_client_ids
            tier_stats[tier] = {
                "total": len(selected_in_tier),
                "malicious": len(malicious_in_tier),
                "benign": len(selected_in_tier) - len(malicious_in_tier),
            }

        if malicious_details.get("unresolved_shortfall", 0) > 0 or benign_details.get("unresolved_shortfall", 0) > 0:
            self.logger.warning(
                "Tier-balanced sampling had unresolved shortfall. "
                f"malicious={malicious_details.get('unresolved_shortfall', 0)}, "
                f"benign={benign_details.get('unresolved_shortfall', 0)}"
            )

        self.logger.info(
            "Selected "
            f"{len(selected)} clients: {selected} "
            f"(malicious: {len(selected_malicious)}, ids: {sorted(selected_malicious)}) "
            f"| tier_stats={tier_stats} "
            f"| malicious_quota={malicious_details.get('quota_by_tier', {})} "
            f"| benign_quota={benign_details.get('quota_by_tier', {})} "
            f"| fallback(m={malicious_details.get('fallback_picks', 0)}, "
            f"b={benign_details.get('fallback_picks', 0)})"
        )
        return selected
