from __future__ import annotations

from typing import Optional

from torch.utils.data import Dataset

from core.attack.data.poison_dataset import PoisonedDatasetWrapper
from core.attack.data.triggers import PatchTrigger
from core.utils.registry import ATTACK_REGISTRY


@ATTACK_REGISTRY.register("badnets")
class BadNetsAttack:
    """
    BadNets 数据投毒攻击配置。
    纯数据端攻击,仅实现 poison_dataset,不干预训练和上传。
    """

    def __init__(self, target_label: int, poison_ratio: float = 0.1, patch_size: int = 3,
                 patch_value: float = 1.0, patch_location: str = "bottom_right",
                 seed: Optional[int] = None):
        """
        Args:
            target_label: 后门目标类别。
            poison_ratio: 训练集投毒比例 (0.0~1.0)。
            patch_size: 触发器方块边长 (像素)。
            patch_value: 触发器像素值 (0.0 黑, 1.0 白)。
            patch_location: 触发器位置, 可选 'bottom_right', 'bottom_left', 'top_right', 'top_left'。
            seed: 随机种子, 用于确定投毒样本子集, 保证可复现性。
        """
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.patch_location = patch_location
        self.seed = seed

    def poison_dataset(self, dataset: Dataset, mode: str, split: str = "",
                       client_id: Optional[str] = None, round_idx: Optional[int] = None,
                       **kwargs) -> Dataset:
        """
        将原始数据集包装为投毒数据集。

        Args:
            dataset: 原始数据集。
            mode: 数据模式 ('train' 按比例投毒, 'test' 全部投毒用于 ASR 评估)。
            split: 数据划分标识 (透传, 未使用)。
            client_id: 客户端 ID (透传, 未使用)。
            round_idx: 轮次索引 (透传, 未使用).
            **kwargs: 其他透传参数。

        Returns:
            投毒后的数据集。
        """
        trigger = PatchTrigger(patch_size=self.patch_size, patch_value=self.patch_value,
                               location=self.patch_location)
        return PoisonedDatasetWrapper(
            original_dataset=dataset, trigger_transform=trigger,
            target_label=self.target_label, poison_ratio=self.poison_ratio,
            mode=mode, seed=self.seed
        )
