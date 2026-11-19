import torch
import torch.nn as nn


class Scaler(nn.Module):
    """恒定缩放层：输出 = x * scale"""

    def __init__(self, scale: float):
        super().__init__()
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        # buffer 确保随 state_dict 同步，但不参与梯度
        self.register_buffer("scale", torch.tensor(float(scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

