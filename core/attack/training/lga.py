"""
LGA: Adaptive Layer-wise Gradient Alignment (ICCV 2025).

Core action: after each local poison epoch, scale the per-layer update so that
||Delta_m^l|| <= ||Delta_g^{t-1,l}||, matching official getNewW.
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch.utils.data import Dataset

from core.attack.data.badnets import BadNetsAttack
from core.utils.registry import ATTACK_REGISTRY


@torch.no_grad()
def project_layerwise(
    global_state: Mapping[str, torch.Tensor],
    current_state: Mapping[str, torch.Tensor],
    prev_global_delta: Optional[Mapping[str, torch.Tensor]],
    eps: float = 1e-8,
) -> Dict[str, torch.Tensor]:
    """
    Official LGA getNewW: scale each layer's delta vs round-start global.

    S^l = min(1, ||prev_global_delta^l|| / ||current - global||^l)
    w^l <- global^l + S^l * (current^l - global^l)
    """
    projected: Dict[str, torch.Tensor] = {}
    for key, cur in current_state.items():
        if not torch.is_tensor(cur):
            projected[key] = cur
            continue

        # 对齐官方: BN num_batches_tracked 直接回退到全局值
        if key.endswith("num_batches_tracked"):
            if key in global_state and torch.is_tensor(global_state[key]):
                projected[key] = global_state[key].clone()
            else:
                projected[key] = cur.clone()
            continue

        if (
            prev_global_delta is None
            or key not in prev_global_delta
            or key not in global_state
            or not torch.is_tensor(global_state[key])
            or not torch.is_tensor(prev_global_delta[key])
        ):
            projected[key] = cur.clone()
            continue

        base = global_state[key]
        # 与 base 同 device / dtype 计算, 避免跨设备 norm
        cur_aligned = cur.to(device=base.device, dtype=base.dtype)
        delta = cur_aligned - base
        ref = prev_global_delta[key].to(device=base.device, dtype=torch.float32)
        delta_norm = torch.norm(delta.float())
        scale = min(1.0, float(torch.norm(ref).item()) / (float(delta_norm.item()) + eps))
        projected[key] = (delta * scale + base).to(dtype=cur.dtype)

    return projected


@ATTACK_REGISTRY.register("lga")
class LGAAttack:
    """
    Training-stage LGA backdoor attack.

    Local training is two-phase (see LGAClient):
    1. ``epoch_clean`` epochs on unpoisoned data (no LGA).
    2. ``epochs`` (client trainer config) on poisoned data with LGA after each epoch.

    - poison_dataset: BadNets trigger poisoning
    - per-epoch projection: LGAClient.train via project_layerwise (poison phase only)
    """

    client_class = None  # late-bound to LGAClient

    def __init__(
        self,
        target_label: int = 5,
        poison_ratio: float = 0.5,
        patch_size: int = 5,
        patch_value: float = 1.0,
        patch_location: str = "bottom_right",
        seed: Optional[int] = None,
        epoch_clean: int = 2,
    ):
        if epoch_clean < 0:
            raise ValueError(f"epoch_clean must be >= 0, got {epoch_clean}")
        self._badnets = BadNetsAttack(
            target_label=target_label,
            poison_ratio=poison_ratio,
            patch_size=patch_size,
            patch_value=patch_value,
            patch_location=patch_location,
            seed=seed,
        )
        self.target_label = target_label
        self.poison_ratio = poison_ratio
        self.seed = seed
        self.epoch_clean = int(epoch_clean)

    def poison_dataset(
        self,
        dataset: Dataset,
        mode: str,
        split: str = "",
        client_id: Optional[str] = None,
        round_idx: Optional[int] = None,
        **kwargs,
    ) -> Dataset:
        return self._badnets.poison_dataset(
            dataset,
            mode=mode,
            split=split,
            client_id=client_id,
            round_idx=round_idx,
            **kwargs,
        )

    def project_layerwise(
        self,
        global_state: Mapping[str, torch.Tensor],
        current_state: Mapping[str, torch.Tensor],
        prev_global_delta: Optional[Mapping[str, torch.Tensor]],
        eps: float = 1e-8,
    ) -> Dict[str, torch.Tensor]:
        """Instance wrapper so LGAClient need not import this module at top-level."""
        return project_layerwise(
            global_state=global_state,
            current_state=current_state,
            prev_global_delta=prev_global_delta,
            eps=eps,
        )


# Late bind client class to avoid import cycles with LGAClient.
from core.client.lga_client import LGAClient  # noqa: E402

LGAAttack.client_class = LGAClient
