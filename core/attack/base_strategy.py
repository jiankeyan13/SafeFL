from typing import Dict, Any, List
import torch
from torch.utils.data import Dataset

from core.utils.registry import ATTACK_REGISTRY

class AttackStrategy:
    """
    定义了攻击者可以在 Client 训练流程的三个核心阶段
    注入恶意行为的切入点/钩子。
    """
    def __init__(self, name: str, **kwargs):
        """
        接收来自 config 的所有攻击参数。
        例如: target_label, trigger_alpha, etc.
        """
        self.name = name
        pass

    def poison_dataset(self, 
                       dataset: Dataset, 
                       mode: str = 'train') -> Dataset:
        """
        [钩子1: 数据投毒]
        在数据加载时修改数据集。
        """
        return dataset

    def poison_train(self, 
                     model: torch.nn.Module, 
                     # ... 可以根据需要增加 optimizer, loss, data, target 等参数 ...
                     ):
        """
        [钩子2: 过程投毒]
        在每个训练步骤中执行，用于操纵梯度或训练过程。
        """
        pass

    def poison_update(self, 
                      update: Dict[str, torch.Tensor], 
                      initial_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        [钩子3: 模型投毒]
        在 Client 准备上传更新包之前，对其进行修改。
        """
        return update
