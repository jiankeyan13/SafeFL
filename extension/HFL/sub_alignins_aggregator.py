import torch
from typing import Any, Dict, List, Optional, Tuple

from core.utils.registry import AGGREGATOR_REGISTRY

from .sub_aggregator import SubAvgAggregator


@AGGREGATOR_REGISTRY.register("sub_alignins")
class SubAlignInsAggregator(SubAvgAggregator):
    """
    HFL 版 AlignIns 聚合: 与 vanilla AlignInsAggregator 一致的中位数范数裁剪,
    再仅对 benign 客户端做等权 mask-aware 平均 (忽略 sample_weights).
    依赖筛选阶段写入的 alignins_client_norms (dense 对齐后的 L2 范数).
    """

    def __init__(self, device: str = "cuda", eps: float = 1e-12):
        super().__init__(device=device, eps=eps)
        self.norm_eps = float(eps)

    def aggregate(
        self,
        updates: List[Dict[str, torch.Tensor]],
        sample_weights: Optional[List[float]] = None,
        screen_scores: Optional[List[float]] = None,
        global_model=None,
        context: Dict[str, Any] = None,
        **kwargs,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        del sample_weights, screen_scores, kwargs
        if not updates:
            raise ValueError("Updates list is empty")

        context = context or {}
        num_clients = len(updates)
        benign_indices = context.get("alignins_benign_indices")
        if benign_indices is None:
            benign_indices = list(range(num_clients))
        benign_indices = [int(i) for i in benign_indices]
        if not benign_indices:
            benign_indices = list(range(num_clients))

        norms = context.get("alignins_client_norms")
        if norms is None or len(norms) != num_clients:
            learnable_keys = self._get_learnable_keys(global_model)
            norms = [
                torch.norm(self._flatten_update(u, learnable_keys), p=2).item()
                for u in updates
            ]

        benign_norms = torch.tensor(
            [norms[i] for i in benign_indices], dtype=torch.float32
        )
        clip_value = (
            float(torch.median(benign_norms).item())
            if benign_norms.numel() > 0
            else 0.0
        )

        clipped: List[Dict[str, torch.Tensor]] = []
        for i, u in enumerate(updates):
            norm_i = float(norms[i])
            scale = (
                min(1.0, clip_value / max(norm_i, self.norm_eps))
                if clip_value > 0.0
                else 1.0
            )
            if scale >= 1.0:
                clipped.append(u)
            else:
                clipped.append({k: t * scale for k, t in u.items()})

        n_b = len(benign_indices)
        benign_set = set(benign_indices)
        eff_scores = [
            (1.0 / n_b) if idx in benign_set else 0.0 for idx in range(num_clients)
        ]

        return super().aggregate(
            clipped,
            sample_weights=[1.0] * num_clients,
            screen_scores=eff_scores,
            global_model=global_model,
            context=context,
        )
