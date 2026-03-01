import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from core.utils.registry import REFINER_REGISTRY
from .base_refiner import BaseRefiner


@REFINER_REGISTRY.register("noise")
class NoiseRefiner(BaseRefiner):
    """
    噪声精炼器：在更新全局模型后添加高斯噪声。
    噪声标准差 = noise * clip_value(从 context 获取)。
    """
    
    def __init__(self, noise: float = 0.001, config=None):
        # paper：Accordingly, λ = 0.001 for IC and NLP, and λ = 0.01 for the NIDS scenario.
        """
        Args:
            noise: 噪声系数，实际噪声 std = noise * clip_value
        """
        super().__init__(config)
        self.noise_factor = noise

    def process(self, 
                global_model: torch.nn.Module, 
                new_state: Dict[str, torch.Tensor], 
                calibration_loader: Optional[DataLoader] = None, 
                device: torch.device = None,
                context: Dict[str, Any] = None):
        """
        对 new_state 添加噪声后加载到全局模型。
        """
        context = context or {}
        clip_value = context.get('clip_value', 1.0)
        noise_std = self.noise_factor * clip_value
        
        # 获取可学习参数名称（用于只对可学习参数加噪声）
        learnable_params = set(name for name, p in global_model.named_parameters() if p.requires_grad)
        
        # 对聚合后的权重添加噪声
        noisy_state = {}
        for key, value in new_state.items():
            noisy_state[key] = value.clone()
            
            # 只对可学习参数添加噪声，跳过 BN 统计量
            if key in learnable_params and noise_std > 0:
                noise_tensor = torch.randn_like(value) * noise_std
                noisy_state[key] += noise_tensor

        
        # 加载权重
        global_model.load_state_dict(noisy_state)
        
        # BN 校准
        if calibration_loader:
            self.calibrate_bn(global_model, calibration_loader, device)
