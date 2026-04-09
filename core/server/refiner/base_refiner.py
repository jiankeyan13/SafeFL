import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

class BaseRefiner:
    """
    精炼器基类。
    职责：将聚合后的 new_state 应用到全局模型，并执行可选的后处理（如 BN 校准）。
    """
    def __init__(self, config=None):
        pass

    def process(self, 
                global_model: torch.nn.Module, 
                new_state: Dict[str, torch.Tensor], 
                calibration_loader: Optional[DataLoader] = None, 
                device: torch.device = None,
                context: Dict[str, Any] = None):
        """
        将 new_state 应用到全局模型。
        
        Args:
            global_model: 全局模型
            new_state: 聚合后的完整模型权重 (Base + Delta)
            calibration_loader: BN 校准数据加载器
            device: 设备
            context: 上下文信息
        """
        global_model.load_state_dict(new_state)
        
        if calibration_loader:
            self.calibrate_bn(global_model, calibration_loader, device)

    def calibrate_bn(self, model, loader, device):
        """代理数据BN校准"""
        model.train()
        if device: 
            model.to(device)
        with torch.no_grad():
            for data, _ in loader:
                if device: 
                    data = data.to(device)
                model(data)
