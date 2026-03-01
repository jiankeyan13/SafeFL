import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Callable, Optional, Set

class PoisonedDatasetWrapper(Dataset):
    """
    初始化时就确定好哪些样本会被投毒。
    """
    def __init__(self, 
                 original_dataset: Dataset, 
                 trigger_transform: Callable,
                 target_label: int,
                 poison_ratio: float = 0.1,
                 mode: str = 'train',
                 seed: Optional[int] = None):
        """
        Args:
            seed: 随机种子，用于确定投毒的样本子集，保证可复现性。
        """
        self.original_dataset = original_dataset
        self.trigger_transform = trigger_transform
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.mode = mode
        

        self.poisoned_indices: Set[int] = set()
        
        if self.mode == 'train' and self.poison_ratio > 0:
            dataset_size = len(self.original_dataset)
            num_poison = int(dataset_size * self.poison_ratio)
            
            # 使用 numpy 的 RNG，它比 Python 的 random 更适合科学计算
            rng = np.random.default_rng(seed)
            
            # 随机选择不重复的索引进行投毒
            indices_to_poison = rng.choice(dataset_size, size=num_poison, replace=False)
            self.poisoned_indices = set(indices_to_poison)

    def __len__(self) -> int:
        return len(self.original_dataset)

    def __getitem__(self, index: int) -> tuple:
        """
        核心投毒逻辑现在是确定性的查询。
        """
        img, label = self.original_dataset[index]

        if self.mode == 'train':
            # 训练模式：查询预计算的索引集
            if index in self.poisoned_indices:
                poisoned_img = self.trigger_transform(img)
                return poisoned_img, self.target_label
            else:
                return img, label
        
        elif self.mode == 'test':
            # 测试模式：全部投毒
            poisoned_img = self.trigger_transform(img)
            return poisoned_img, label
            
        else: # 'val' or other modes
            return img, label