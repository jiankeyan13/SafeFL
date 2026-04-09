import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from core.server.refiner.base_refiner import BaseRefiner
from core.server.refiner.noise_refiner import NoiseRefiner
from models.modules.sbn import set_sbn_eval, reset_bn_running_stats


def _calibrate_bn_hfl(model, loader, device):
    """
    HFL BN 校准核心逻辑，供多个 refiner 复用。
    针对异构模型 BN 层默认 track_running_stats=False 的情况：
    1. reset: running_mean=0, running_var=1, num_batches_tracked=0
    2. set_sbn_eval: track_running_stats=True (开启累积)
    3. model.train(): BN 层使用 batch stats 并更新 running stats
       momentum=None 时为 cumulative moving average (1/N)，校准完得到精确统计量
    4. model.eval(): 后续推理使用校准后的 running stats
    """
    reset_bn_running_stats(model)
    set_sbn_eval(model)
    model.train()
    if device:
        model.to(device)
    with torch.no_grad():
        for data, _ in loader:
            if device:
                data = data.to(device)
            model(data)
    model.eval()


class HFLRefiner(BaseRefiner):
    """HFL 专用精炼器。"""

    def calibrate_bn(self, model, loader, device):
        _calibrate_bn_hfl(model, loader, device)


class HFLNoiseRefiner(NoiseRefiner):
    """HFL 专用噪声精炼器 (FLAME 用)。"""

    def calibrate_bn(self, model, loader, device):
        _calibrate_bn_hfl(model, loader, device)
