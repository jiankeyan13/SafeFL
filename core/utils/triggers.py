import torch
import torch.nn as nn
from typing import Tuple, Union

class PatchTrigger(nn.Module):
    """
    一个可配置的 Patch Trigger (贴片触发器)。
    
    在图片的指定角落贴上一个纯色方块。
    继承 nn.Module, 与 torchvision.transforms 组合使用。
    """
    def __init__(self, 
                 patch_size: int = 3, 
                 patch_value: Union[float, Tuple[float, float, float]] = 1.0, 
                 location: str = 'bottom_right'):
        """
        Args:
            patch_size: 方块的边长 (像素)。
            patch_value: 像素值 (0.0 for black, 1.0 for white)。
                         对于多通道图像，可以是一个元组。
            location: 'bottom_right', 'bottom_left', 'top_right', 'top_left'。
        """
        super().__init__()
        if location not in ['bottom_right', 'bottom_left', 'top_right', 'top_left']:
            raise ValueError("Invalid location. Must be one of 'bottom_right', 'bottom_left', 'top_right', 'top_left'.")
            
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.location = location

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        将 Trigger 应用到输入的图片 Tensor 上。
        Args:
            img: 输入的图片 Tensor, 形状应为 [C, H, W]。
            
        Returns:
            被修改后的图片 Tensor。
        """
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
            
        # 复制一份以避免原地修改
        img = img.clone()
        
        c, h, w = img.shape
        
        if self.location == 'bottom_right':
            img[:, h - self.patch_size:, w - self.patch_size:] = self.patch_value
        elif self.location == 'bottom_left':
            img[:, h - self.patch_size:, :self.patch_size] = self.patch_value
        elif self.location == 'top_right':
            img[:, :self.patch_size, w - self.patch_size:] = self.patch_value
        elif self.location == 'top_left':
            img[:, :self.patch_size, :self.patch_size] = self.patch_value
            
        return img

# --- 其他 Trigger ---
# class BlendedTrigger(nn.Module): ...
# class SemanticTrigger(nn.Module): ...