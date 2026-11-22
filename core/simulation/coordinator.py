"""Algorithm-agnostic threat-model scheduler & state coordinator for SafeFL v2."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from core.attack.base import BaseAttacker, build_attacker
from core.attack.poisoner import BasePoisoner
from core.types import ClientUpdate, TrainConfig
from core.utils.config import AttackConfig, AttackStrategyConfig
from core.utils.registry import ATTACK_REGISTRY


@dataclass
class ParticipantTask:
    """Serializable per-client round execution descriptor for ``ClientWorker``."""

    client_id: int
    is_malicious: bool
    train_cfg: TrainConfig
    strategy: Optional[AttackStrategyConfig] = None
    attacker_rank: Optional[int] = None
    state: Dict[str, Any] = field(default_factory=dict)


class AttackCoordinator:
    """Orchestrates compromised client identities, active round windows, and cross-round state.

    - Purely configuration-driven and completely agnostic to individual attack implementations.
    - Compromised clients automatically execute as normal ``BenignClient`` outside the active
      ``[start_round, end_round]`` / ``attack_interval`` window.
    - Assigns sequential ``attacker_rank = 0, 1, ...`` to active attackers each round so
      distributed triggers (e.g., DBA) partition cleanly without coordinator special-casing.
    """

    def __init__(
        self,
        attack_cfg: AttackConfig,
        all_client_ids: Sequence[int],
        *,
        seed: int = 42,
    ) -> None:
        self.cfg = attack_cfg
        self.seed = int(seed)
        self.all_client_ids: List[int] = sorted(int(c) for c in all_client_ids)

        self.malicious_ids: Set[int] = set()
        self.client_strategy_map: Dict[int, AttackStrategyConfig] = {}
        self._attacker_states: Dict[int, Dict[str, Any]] = {}
        self.eval_poisoner: Optional[BasePoisoner] = None
        self.excluded_train_indices: List[int] = []
        self._primary_strategy: Optional[AttackStrategyConfig] = None
        self.uses_prev_global_delta: bool = False

        if self.cfg.enabled:
            self._assign_compromised_clients()

    def _assign_compromised_clients(self) -> None:
        num_malicious = max(0, int(len(self.all_client_ids) * self.cfg.malicious_fraction))
        if num_malicious == 0 or not self.cfg.strategies:
            return

        compromised = self.all_client_ids[:num_malicious]
        self.malicious_ids = set(compromised)

        cursor = 0
        for idx, strategy in enumerate(self.cfg.strategies):
            if idx == len(self.cfg.strategies) - 1:
                end = num_malicious
            else:
                end = min(cursor + int(round(num_malicious * strategy.fraction)), num_malicious)
            for cid in compromised[cursor:end]:
                self.client_strategy_map[cid] = strategy
            cursor = end

        protos = [build_attacker(s, seed=self.seed) for s in self.cfg.strategies]
        self._primary_strategy = self.cfg.strategies[0]
        self.eval_poisoner = protos[0].poisoner
        self.excluded_train_indices = sorted(
            {int(i) for a in protos for i in a.poisoner.excluded_train_indices()}
        )
        self.uses_prev_global_delta = any(
            bool(getattr(ATTACK_REGISTRY.get(s.name), "needs_prev_global_delta", False))
            for s in self.cfg.strategies
        )

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.malicious_ids)

    def is_round_active(self, round_idx: int) -> bool:
        if not self.enabled:
            return False
        if round_idx < self.cfg.start_round:
            return False
        if self.cfg.end_round is not None and round_idx > self.cfg.end_round:
            return False
        return (round_idx - self.cfg.start_round) % self.cfg.attack_interval == 0

    def is_client_active_attacker(self, round_idx: int, client_id: int) -> bool:
        return int(client_id) in self.malicious_ids and self.is_round_active(round_idx)

    def select_clients(
        self,
        round_idx: int,
        num_select: int,
        server_select_fn: Callable[[List[int], int], List[int]],
    ) -> List[int]:
        """Select round participants under ``random`` or ``fixed_quota`` threat models."""
        num_select = max(1, min(int(num_select), len(self.all_client_ids)))

        if not self.enabled or self.cfg.sampling_mode == "random":
            return server_select_fn(self.all_client_ids, num_select)

        rng = random.Random(self.seed + int(round_idx))
        mal_pool = sorted(self.malicious_ids)
        ben_pool = [c for c in self.all_client_ids if c not in self.malicious_ids]

        target_mal = int(num_select * self.cfg.per_round_fraction)
        if self.cfg.per_round_fraction > 0.0 and target_mal == 0 and mal_pool:
            target_mal = 1

        n_mal = max(0, min(target_mal, len(mal_pool), num_select))
        n_ben = max(0, min(num_select - n_mal, len(ben_pool)))

        chosen_mal = rng.sample(mal_pool, n_mal) if n_mal > 0 else []
        chosen_ben = rng.sample(ben_pool, n_ben) if n_ben > 0 else []
        selected = chosen_mal + chosen_ben
        rng.shuffle(selected)
        return selected

    def plan_round(
        self,
        round_idx: int,
        selected_ids: Sequence[int],
        base_train_cfg: TrainConfig,
    ) -> List[ParticipantTask]:
        """Build per-participant execution tasks for ``round_idx``."""
        tasks: List[ParticipantTask] = []
        mal_rank = 0

        for raw_cid in selected_ids:
            cid = int(raw_cid)
            if self.is_client_active_attacker(round_idx, cid):
                strategy = self.client_strategy_map[cid]
                t_cfg = (
                    base_train_cfg.with_overrides(epochs=self.cfg.malicious_epochs)
                    if self.cfg.malicious_epochs is not None
                    else base_train_cfg
                )
                saved_state = self._attacker_states.get(cid)
                tasks.append(
                    ParticipantTask(
                        client_id=cid,
                        is_malicious=True,
                        train_cfg=t_cfg,
                        strategy=strategy,
                        attacker_rank=mal_rank,
                        state=copy.deepcopy(saved_state) if saved_state else {},
                    )
                )
                mal_rank += 1
            else:
                tasks.append(
                    ParticipantTask(
                        client_id=cid,
                        is_malicious=False,
                        train_cfg=base_train_cfg,
                    )
                )
        return tasks

    def finalize_round(
        self,
        round_idx: int,
        results: Sequence[Tuple[ClientUpdate, Optional[Dict[str, Any]]]],
    ) -> List[ClientUpdate]:
        """Persist updated attacker states and invoke optional collusion hook."""
        updates: List[ClientUpdate] = []
        mal_indices: List[int] = []

        synced_eval = False
        for idx, (update, new_state) in enumerate(results):
            cid = int(update.client_id)
            if new_state is not None and cid in self.malicious_ids:
                self._attacker_states[cid] = new_state
                if (
                    not synced_eval
                    and self.eval_poisoner is not None
                    and self.client_strategy_map.get(cid) == self._primary_strategy
                ):
                    self.eval_poisoner.load_state(new_state)
                    synced_eval = True
            if self.is_client_active_attacker(round_idx, cid):
                mal_indices.append(idx)
            updates.append(update)

        if mal_indices:
            mal_updates = [updates[i] for i in mal_indices]
            coordinated = self.coordinate_updates(round_idx, mal_updates, updates)
            for idx, new_up in zip(mal_indices, coordinated):
                updates[idx] = new_up

        return updates

    def coordinate_updates(
        self,
        round_idx: int,
        malicious_updates: List[ClientUpdate],
        all_updates: Sequence[ClientUpdate],
    ) -> List[ClientUpdate]:
        """Extension hook for multi-attacker same-round collusion (no-op by default)."""
        del round_idx, all_updates
        return malicious_updates
