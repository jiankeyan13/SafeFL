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


def _load_hfl_state(global_model: torch.nn.Module, new_state: Dict[str, torch.Tensor]) -> None:
    current_state = global_model.state_dict()
    compatible_state = {}
    for key, current_value in current_state.items():
        loaded_value = new_state.get(key, current_value)
        if not torch.is_tensor(loaded_value):
            compatible_state[key] = loaded_value
            continue
        if loaded_value.shape != current_value.shape:
            compatible_state[key] = current_value
            continue
        compatible_state[key] = loaded_value.to(
            device=current_value.device,
            dtype=current_value.dtype,
        )
    global_model.load_state_dict(compatible_state, strict=False)


class HFLRefiner(BaseRefiner):
    """HFL 专用精炼器。"""

    def process(self,
                global_model: torch.nn.Module,
                new_state: Dict[str, torch.Tensor],
                calibration_loader: Optional[DataLoader] = None,
                device: torch.device = None,
                context: Dict[str, Any] = None):
        _load_hfl_state(global_model, new_state)

        if calibration_loader:
            self.calibrate_bn(global_model, calibration_loader, device)

    def calibrate_bn(self, model, loader, device):
        _calibrate_bn_hfl(model, loader, device)


class HFLNoiseRefiner(NoiseRefiner):
    """HFL 专用噪声精炼器 (FLAME 用)。"""

    def process(self,
                global_model: torch.nn.Module,
                new_state: Dict[str, torch.Tensor],
                calibration_loader: Optional[DataLoader] = None,
                device: torch.device = None,
                context: Dict[str, Any] = None):
        context = context or {}
        clip_value = context.get('clip_value', 1.0)
        noise_std = self.noise_factor * clip_value
        learnable_params = set(name for name, p in global_model.named_parameters() if p.requires_grad)

        noisy_state = {}
        for key, value in new_state.items():
            if not torch.is_tensor(value):
                noisy_state[key] = value
                continue
            noisy_value = value.clone()
            if key in learnable_params and noise_std > 0:
                noisy_value = noisy_value + torch.randn_like(noisy_value) * noise_std
            noisy_state[key] = noisy_value

        _load_hfl_state(global_model, noisy_state)

        if calibration_loader:
            self.calibrate_bn(global_model, calibration_loader, device)

    def calibrate_bn(self, model, loader, device):
        _calibrate_bn_hfl(model, loader, device)
