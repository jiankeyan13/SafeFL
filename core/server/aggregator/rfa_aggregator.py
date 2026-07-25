import torch
from typing import List, Dict, Optional, Any, Tuple, Sequence

from core.utils.registry import AGGREGATOR_REGISTRY
from .base_aggregator import BaseAggregator


@AGGREGATOR_REGISTRY.register("rfa")
class RFAAggregator(BaseAggregator):
    """
    Robust Federated Aggregation (RFA).

    对客户端更新向量求加权几何中位数, 用平滑 Weiszfeld 迭代近似:
        y* = argmin_y  sum_i α_i * ||y - Δ_i||
        β_i^(t) = α_i / max(ν, ||y^(t) - Δ_i||)
        y^(t+1) = sum_i β_i^(t) Δ_i / sum_i β_i^(t)

    参考: Pillutla et al., "Robust Aggregation for Federated Learning".
    """

    def __init__(
        self,
        num_iters: int = 4,
        nu: float = 1e-6,
        device: str = "cuda",
    ):
        """
        Args:
            num_iters: Weiszfeld 迭代次数 T.
            nu: 平滑参数 ν, 避免距离为 0 时权重爆炸.
            device: 计算设备.
        """
        super().__init__(device)
        if num_iters < 1:
            raise ValueError("num_iters must be >= 1")
        if nu <= 0:
            raise ValueError("nu must be > 0")
        self.num_iters = num_iters
        self.nu = nu

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_weights: Optional[List[float]] = None,
        screen_scores: Optional[List[float]] = None,
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        if not updates:
            raise ValueError("Updates list is empty")

        context = context or {}
        n = len(updates)
        template = updates[0]

        if sample_weights is None:
            sample_weights = [1.0] * n
        if screen_scores is None:
            screen_scores = [1.0] * n

        alphas = [
            float(s) * float(sc) for s, sc in zip(sample_weights, screen_scores)
        ]
        if sum(alphas) <= 0:
            zero_delta = {
                k: torch.zeros_like(v, device=self.device) for k, v in template.items()
            }
            return zero_delta, context

        learnable_keys = self._get_learnable_keys(global_model)
        gm_keys = self._get_gm_keys(template, learnable_keys)

        flat_updates = torch.stack(
            [self._flatten_by_keys(u, gm_keys) for u in updates], dim=0
        )  # [N, D]
        alpha = torch.tensor(alphas, dtype=torch.float32, device=self.device)

        median_flat = self._smoothed_weiszfeld(flat_updates, alpha)

        aggregated = self._unflatten_by_keys(median_flat, template, gm_keys)

        # 非几何中位数覆盖的键 (如整数 BN buffer): 坐标中位数
        for name, tensor in template.items():
            if name in aggregated:
                continue
            stacked = torch.stack(
                [
                    u[name].to(device=self.device, dtype=torch.float32)
                    for u in updates
                ],
                dim=0,
            )
            aggregated[name] = torch.median(stacked, dim=0).values

        return aggregated, context

    def _get_gm_keys(
        self,
        template: Dict[str, torch.Tensor],
        learnable_keys: Optional[set],
    ) -> Sequence[str]:
        keys = []
        for name, tensor in template.items():
            if not torch.is_floating_point(tensor):
                continue
            if learnable_keys is None or name in learnable_keys:
                keys.append(name)
        if not keys:
            raise ValueError("No floating-point parameters available for RFA")
        return keys

    def _flatten_by_keys(
        self,
        update: Dict[str, torch.Tensor],
        keys: Sequence[str],
    ) -> torch.Tensor:
        parts = [
            update[name].to(device=self.device, dtype=torch.float32).reshape(-1)
            for name in keys
        ]
        return torch.cat(parts)

    def _unflatten_by_keys(
        self,
        flat: torch.Tensor,
        template: Dict[str, torch.Tensor],
        keys: Sequence[str],
    ) -> Dict[str, torch.Tensor]:
        aggregated: Dict[str, torch.Tensor] = {}
        offset = 0
        for name in keys:
            shape = template[name].shape
            numel = template[name].numel()
            aggregated[name] = flat[offset : offset + numel].view(shape)
            offset += numel
        return aggregated

    def _smoothed_weiszfeld(
        self,
        points: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        平滑 Weiszfeld 求加权几何中位数.

        Args:
            points: [N, D] 客户端更新向量.
            alpha: [N] 非负权重.
        Returns:
            [D] 几何中位数估计.
        """
        # 以加权均值初始化, 迭代更稳
        y = (points * alpha.unsqueeze(1)).sum(dim=0) / alpha.sum().clamp_min(1e-12)

        for _ in range(self.num_iters):
            dists = torch.norm(points - y.unsqueeze(0), dim=1)  # [N]
            beta = alpha / dists.clamp_min(self.nu)
            y = (points * beta.unsqueeze(1)).sum(dim=0) / beta.sum().clamp_min(1e-12)

        return y
