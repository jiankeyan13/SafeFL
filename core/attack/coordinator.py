"""Built-in AttackCoordinator managing scheduling, sampling, state, and collusion hooks."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from core.attack.base import BaseAttacker, build_attacker
from core.attack.data.poisoner import BasePoisoner, DBAPoisoner
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
    state: Dict[str, Any] = field(default_factory=dict)
    dba_block_map: Optional[Dict[int, int]] = None


class AttackCoordinator:
    """Orchestrates compromised client identities, round windows, and cross-round state.

    - Purely configuration-driven for 95% of experiments (no subclassing required).
    - Compromised clients automatically execute as normal ``BenignClient`` outside
      the active ``[start_round, end_round]`` / ``attack_interval`` window (e.g.,
      during pre-attack warmup or post-attack lifespan evaluation).
    - Persists per-attacker ``state`` dicts on the driver across rounds so Ray
      workers remain stateless while stateful attacks retain full memory.
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

        # Build global evaluation poisoner & check whether any strategy needs prev_global_delta.
        primary_attacker: BaseAttacker = build_attacker(self.cfg.strategies[0], seed=self.seed)
        self.eval_poisoner = primary_attacker.poisoner
        self.uses_prev_global_delta = any(
            bool(getattr(ATTACK_REGISTRY.get(s.name), "needs_prev_global_delta", False))
            for s in self.cfg.strategies
        )

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enabled and self.malicious_ids)

    def is_round_active(self, round_idx: int) -> bool:
        """Check whether the attack is active in ``round_idx``."""
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

        # Fixed-quota sampling: reproducible per-round malicious count.
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
        active_mal_ids = [
            int(cid)
            for cid in selected_ids
            if self.is_client_active_attacker(round_idx, int(cid))
        ]
        # Dynamically assign sequential sub-block indices (0, 1, 2, ...) to active attackers
        # in this round so DBA never suffers block collisions when sampling arbitrary client IDs.
        dba_block_map: Optional[Dict[int, int]] = (
            {cid: idx for idx, cid in enumerate(active_mal_ids)}
            if active_mal_ids
            else None
        )

        tasks: List[ParticipantTask] = []
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
                        state=copy.deepcopy(saved_state) if saved_state else {},
                        dba_block_map=dba_block_map,
                    )
                )
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

        for idx, (update, new_state) in enumerate(results):
            cid = int(update.client_id)
            if new_state is not None and cid in self.malicious_ids:
                self._attacker_states[cid] = new_state
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


def instantiate_task_attacker(
    task: ParticipantTask,
    *,
    seed: int,
) -> BaseAttacker:
    """Helper used by ``ClientWorker`` to instantiate a task's ``BaseAttacker``."""
    if task.strategy is None:
        raise ValueError(f"Malicious task for client {task.client_id} is missing strategy.")
    attacker = build_attacker(task.strategy, seed=seed)
    if task.dba_block_map and isinstance(attacker.poisoner, DBAPoisoner):
        attacker.poisoner.set_client_block_map(task.dba_block_map)
    return attacker
