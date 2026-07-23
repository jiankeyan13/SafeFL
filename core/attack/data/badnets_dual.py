"""
BadNets dual-model attack: standard BadNets poisoning + dual-branch client.

Uses BadNets triggers for the poison branch. The paired BadNetsDualClient trains
Wc (clean) and Wb (poison) independently from Wg, uploads Wb-Wg, and periodically
logs Wb-Wc / Wc-Wg / Wg-Wg_1.
"""
from __future__ import annotations

import os
from typing import Optional

from torch.utils.data import Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY


def resolve_delta_log_root(delta_log_dir: Optional[str] = None) -> str:
    """Resolve absolute delta_logs root (prefer Hydra output_dir on driver)."""
    if delta_log_dir:
        return os.path.abspath(delta_log_dir)
    try:
        from hydra.core.hydra_config import HydraConfig

        out_dir = HydraConfig.get().runtime.output_dir
        if out_dir:
            return os.path.abspath(os.path.join(out_dir, "delta_logs"))
    except Exception:
        pass
    return os.path.abspath("delta_logs")


@ATTACK_REGISTRY.register("badnets_dual")
class BadNetsDualAttack:
    """
    BadNets 数据投毒 + 双模型客户端.

    数据侧与 BadNets 相同; 训练侧由 BadNetsDualClient 完成双分支独立训练与增量落盘.
    """

    def __init__(
        self,
        target_label: int,
        poison_ratio: float = 0.1,
        patch_size: int = 3,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = None,
        delta_log_interval: int = 15,
        delta_log_dir: Optional[str] = None,
        delta_log_enabled: bool = True,
    ):
        self._badnets = BadNetsAttack(
            target_label=target_label,
            poison_ratio=poison_ratio,
            patch_size=patch_size,
            patch_value=patch_value,
            patch_location=patch_location,
            seed=seed,
        )
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.patch_location = patch_location
        self.seed = seed
        self.delta_log_interval = int(delta_log_interval)
        # 在 driver 上解析为绝对路径, 便于经 job spec 传给 Ray worker
        self.delta_log_dir = resolve_delta_log_root(delta_log_dir)
        self.delta_log_enabled = bool(delta_log_enabled)

    def poison_dataset(
        self,
        dataset: Dataset,
        mode: str,
        split: str = "",
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        return self._badnets.poison_dataset(
            dataset,
            mode=mode,
            split=split,
            client_id=client_id,
            round_idx=round_idx,
            **kwargs,
        )


# Late-bind client_class to avoid import cycles.
BadNetsDualAttack.client_class = None  # type: ignore[attr-defined]


def _bind_client_class() -> None:
    from core.client.badnets_dual_client import BadNetsDualClient

    BadNetsDualAttack.client_class = BadNetsDualClient  # type: ignore[attr-defined]


_bind_client_class()
