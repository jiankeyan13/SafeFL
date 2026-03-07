"""HFL 异构联邦学习扩展 - SubFlame 聚合器."""

import torch
from typing import List, Dict, Optional, Any, Tuple

from core.utils.registry import AGGREGATOR_REGISTRY
from extension.HFL.sub_aggregator import SubAvgAggregator


@AGGREGATOR_REGISTRY.register("sub_flame")
class SubFlameAggregator(SubAvgAggregator):
    """
    子模型 FLAME 聚合器 (Sub-model FLAME Aggregator), 继承 SubAvgAggregator,
    在掩码感知聚合前增加 FLAME 范数裁剪防御, 适配异构联邦学习场景.

    适用场景:
        - 模型异构联邦学习 + 后门/投毒防御
        - 客户端通过结构化剪枝仅训练全局模型的子集
        - 需同时抵御恶意更新 (范数裁剪) 与稀疏更新稀释 (掩码感知聚合)

    核心流程:
        1. 范数裁剪: 按 FLAME 策略对每个客户端 delta 进行范数裁剪
        2. 掩码感知聚合: 调用父类 SubAvgAggregator 的逐元素动态归一化聚合
    """

    def __init__(self, device='cuda', eps: float = 1e-10, **kwargs):
        super().__init__(device, eps)

    def aggregate(self,
                  updates: List[Dict[str, torch.Tensor]],
                  sample_weights: Optional[List[float]] = None,
                  screen_scores: Optional[List[float]] = None,
                  global_model: torch.nn.Module = None,
                  context: Dict[str, Any] = None,
                  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        先对客户端 updates 进行范数裁剪, 再调用父类掩码感知聚合.
        """
        if not updates:
            raise ValueError("Updates list is empty")

        context = context or {}
        n = len(updates)

        # 获取或计算裁剪值与范数列表 (FLAME 逻辑)
        clip_value = context.get('clip_value', None)
        norm_list = context.get('norm_list', None)

        if clip_value is None or norm_list is None:
            learnable_keys = self._get_learnable_keys(global_model)
            norm_list = [
                torch.norm(self._flatten_update(u, learnable_keys)).item()
                for u in updates
            ]
            clip_value = float(torch.median(torch.tensor(norm_list)).item())
            context['clip_value'] = clip_value
            context['norm_list'] = norm_list

        if sample_weights is None:
            sample_weights = [1.0] * n
        if screen_scores is None:
            screen_scores = [1.0] * n

        # 对每个客户端进行范数裁剪
        clipped_updates = []
        for i, update in enumerate(updates):
            if screen_scores[i] == 0:
                clipped_updates.append(update)
                continue

            client_norm = norm_list[i]
            gamma = min(clip_value / (client_norm + 1e-9), 1.0)

            if gamma < 1.0:
                clipped_update = {k: v * gamma for k, v in update.items()}
                clipped_updates.append(clipped_update)
            else:
                clipped_updates.append(update)

        # 调用父类 SubAvgAggregator 的掩码感知聚合
        return super().aggregate(
            clipped_updates,
            sample_weights=sample_weights,
            screen_scores=screen_scores,
            global_model=global_model,
            context=context,
            **kwargs
        )
