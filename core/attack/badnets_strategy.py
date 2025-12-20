from typing import Dict, Any, Optional

from core.utils.registry import ATTACK_REGISTRY
from core.utils.triggers import PatchTrigger
from core.utils.poison_dataset import PoisonedDatasetWrapper
from .base_strategy import AttackStrategy

@ATTACK_REGISTRY.register("badnets")
class BadNetsStrategy(AttackStrategy):
    """
    特定的 Patch (触发器) 并修改其标签，使模型建立触发器与目标标签之间的映射。
    """
    def __init__(self, 
                 target_label: int, 
                 poison_ratio: float = 0.5, 
                 patch_size: int = 3, 
                 patch_value: float = 1.0,
                 location: str = 'bottom_right',
                 seed: Optional[int] = None,
                 **kwargs):
        """
        Args:
            target_label: 攻击的目标类别标签。
            poison_ratio: 训练数据中投毒样本的比例 (0.0 ~ 1.0)。
            patch_size: 触发器方块的边长。
            patch_value: 触发器像素的值 (如 1.0 代表白色)。
            location: 触发器在图片上的位置。
            seed: 随机种子，用于确定投毒样本的子集，保证可复现性。
        """
        super().__init__(name="badnets", **kwargs)
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.seed = seed
        
        # 1. 实例化触发器工具
        self.trigger = PatchTrigger(
            patch_size=patch_size, 
            patch_value=patch_value, 
            location=location
        )

    def poison_dataset(self, dataset, mode='train'):
        """
        [阶段一：数据准备]
        使用 PoisonedDatasetWrapper 包装原始数据集，实现动态投毒。
        """
        return PoisonedDatasetWrapper(
            original_dataset=dataset,
            trigger_transform=self.trigger,
            target_label=self.target_label,
            poison_ratio=self.poison_ratio,
            mode=mode,
            seed=self.seed
        )
