from __future__ import annotations

from typing import Optional

from torch.utils.data import Dataset

from core.attack.data.poison_dataset import PoisonedDatasetWrapper
from core.attack.data.triggers import PartialPatchTrigger, PatchTrigger
from core.utils.registry import ATTACK_REGISTRY


def parse_client_id(client_id: Optional[str]) -> int:
    """Parse numeric client id from owner string like 'client_7'."""
    if client_id is None:
        raise ValueError("client_id is required for DBA train poisoning")
    return int(str(client_id).split("_")[-1])


@ATTACK_REGISTRY.register("dba")
class DBAAttack:
    """
    DBA (Distributed Backdoor Attack) 数据投毒.

    训练时每个恶意客户端只贴 patch 的一个子块 (由 client_id % num_blocks 决定).
    评估时使用完整 patch 测量 ASR.
    """

    def __init__(
        self,
        target_label: int,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        num_blocks: int = 4,
        seed: Optional[int] = None,
    ):
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.patch_location = patch_location
        self.num_blocks = num_blocks
        self.seed = seed

    def poison_dataset(
        self,
        dataset: Dataset,
        mode: str,
        split: str = "",
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        return_original_label = bool(kwargs.get("return_original_label", False))

        if mode == "test" or client_id is None:
            trigger = PatchTrigger(
                patch_size=self.patch_size,
                patch_value=self.patch_value,
                location=self.patch_location,
            )
        else:
            cid_num = parse_client_id(client_id)
            block_id = cid_num % self.num_blocks
            trigger = PartialPatchTrigger(
                patch_size=self.patch_size,
                patch_value=self.patch_value,
                location=self.patch_location,
                block_id=block_id,
                num_blocks=self.num_blocks,
            )

        return PoisonedDatasetWrapper(
            original_dataset=dataset,
            trigger_transform=trigger,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            mode=mode,
            seed=self.seed,
            return_original_label=return_original_label,
        )
