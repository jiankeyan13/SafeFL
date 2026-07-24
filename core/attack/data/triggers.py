import torch
import torch.nn as nn
from typing import Tuple, Union

_VALID_LOCATIONS = ("bottom_right", "bottom_left", "top_right", "top_left")


def _patch_origin(location: str, h: int, w: int, patch_size: int) -> Tuple[int, int]:
    """Return top-left (row, col) of the patch region in image coordinates."""
    if location == "bottom_right":
        return h - patch_size, w - patch_size
    if location == "bottom_left":
        return h - patch_size, 0
    if location == "top_right":
        return 0, w - patch_size
    if location == "top_left":
        return 0, 0
    raise ValueError(
        "Invalid location. Must be one of 'bottom_right', 'bottom_left', 'top_right', 'top_left'."
    )


def block_slices(patch_size: int, block_id: int, num_blocks: int = 4) -> Tuple[int, int, int, int]:
    """
    Return relative (row_start, row_end, col_start, col_end) within the patch for a DBA block.

    Splits the patch at mid = patch_size // 2 into four quadrants (2x2, 2x3, 3x2, 3x3 for size 5).
    """
    if num_blocks != 4:
        raise ValueError(f"Only num_blocks=4 is supported, got {num_blocks}")
    if block_id not in (0, 1, 2, 3):
        raise ValueError(f"block_id must be in [0, 3], got {block_id}")

    mid = patch_size // 2
    if block_id == 0:
        return 0, mid, 0, mid
    if block_id == 1:
        return 0, mid, mid, patch_size
    if block_id == 2:
        return mid, patch_size, 0, mid
    return mid, patch_size, mid, patch_size


class PatchTrigger(nn.Module):
    """
    一个可配置的 Patch Trigger (贴片触发器)。
    
    在图片的指定角落贴上一个纯色方块。
    继承 nn.Module, 与 torchvision.transforms 组合使用。
    """
    def __init__(self, 
                 patch_size: int = 5, 
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
        if location not in _VALID_LOCATIONS:
            raise ValueError(
                "Invalid location. Must be one of 'bottom_right', 'bottom_left', 'top_right', 'top_left'."
            )
            
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


class PartialPatchTrigger(nn.Module):
    """
    DBA 分布式触发器: 只在 patch 的一个子矩形块内填色.

    多个恶意客户端各自训练不同 block, 聚合后模型对完整 patch 产生后门响应.
    """

    def __init__(
        self,
        patch_size: int = 5,
        patch_value: Union[float, Tuple[float, float, float]] = 1.0,
        location: str = "bottom_right",
        block_id: int = 0,
        num_blocks: int = 4,
    ):
        super().__init__()
        if location not in _VALID_LOCATIONS:
            raise ValueError(
                "Invalid location. Must be one of 'bottom_right', 'bottom_left', 'top_right', 'top_left'."
            )
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.location = location
        self.block_id = block_id
        self.num_blocks = num_blocks

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if not isinstance(img, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        img = img.clone()
        _, h, w = img.shape
        r0, c0 = _patch_origin(self.location, h, w, self.patch_size)
        rs, re, cs, ce = block_slices(self.patch_size, self.block_id, self.num_blocks)
        img[:, r0 + rs : r0 + re, c0 + cs : c0 + ce] = self.patch_value
        return img


# --- 其他 Trigger ---
# class BlendedTrigger(nn.Module): ...
# class SemanticTrigger(nn.Module): ...