import torch
from torch.utils.data import DataLoader
from typing import Optional

class BaseUpdater:
    def __init__(self, config=None):
        pass

    def update(self, global_model, aggregated_update, calibration_loader: Optional[DataLoader] = None, device=None):
        """
        更新全局模型。如果提供了 calibration_loader，则在更新后执行BN校准。
        """
        global_model.load_state_dict(aggregated_update)
        
        if calibration_loader:
             self.calibrate_bn(global_model, calibration_loader, device)

    def calibrate_bn(self, model, loader, device):
        """代理数据BN校准"""
        model.train()
        if device: model.to(device)
        with torch.no_grad():
            for data, _ in loader:
                if device: data = data.to(device)
                model(data)