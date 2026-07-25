"""
FoolsGold 筛选器: 通过历史梯度相似性动态分配聚合权重, 抵御 Sybil 攻击.
# 论文官方仓库默认关闭重要性加权

算法流程:
1. 将完整模型 delta_i 累加到各自历史梯度 H_i
2. 计算客户端对之间的余弦相似度矩阵
3. 执行 Pardoning 赦免操作, 避免误伤诚实客户端
4. 计算 alpha_i = 1 - max_j(cs_ij), 归一化后经 logit 非线性拉伸
"""

from typing import List, Dict, Any, Tuple, Sequence

import torch
from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("foolsgold")
class FoolsGoldScreener(BaseScreener):
    """
    FoolsGold 筛选器: 基于完整模型历史 delta 相似性为每个客户端分配学习率权重.

    核心逻辑嵌入在筛选阶段, 通过动态权重抑制 Sybil 恶意贡献, 而非直接丢弃.
    与 AvgAggregator 配合使用时, screen_scores 作为聚合权重.
    """

    def __init__(self, use_history: bool = True, **kwargs):
        """use_history: 是否使用历史梯度累加."""
        super().__init__(**kwargs)
        self.use_history = use_history
        self._history_features: Dict[str, torch.Tensor] = {}

    def _get_learnable_keys(
        self, global_model: torch.nn.Module, first_delta: Dict[str, torch.Tensor]
    ) -> Sequence[str]:
        if global_model is not None:
            return [name for name, param in global_model.named_parameters() if param.requires_grad]
        return list(first_delta.keys())

    def _flatten_delta(
        self, delta: Dict[str, torch.Tensor], keys: Sequence[str], device: torch.device
    ) -> torch.Tensor:
        flat_parts = [
            delta[name].detach().float().reshape(-1).to(device)
            for name in keys
            if name in delta
        ]
        if not flat_parts:
            return torch.zeros(0, device=device, dtype=torch.float32)
        return torch.cat(flat_parts, dim=0)

    def _pardoning(self, cs: torch.Tensor) -> torch.Tensor:
        """Pardoning: 若 v_j > v_i, 仅将 cs_ij 按 v_i/v_j 缩小 (有方向性)."""
        cs_masked = cs.clone().fill_diagonal_(0.0)
        v_initial = cs_masked.max(dim=1).values
        v_col, v_row = v_initial.unsqueeze(1), v_initial.unsqueeze(0)
        ratio = torch.clamp(torch.where(v_row > v_col, v_col / v_row.clamp(min=1e-8), torch.ones_like(v_row)), 0.0, 1.0)
        return (cs * ratio).fill_diagonal_(0.0)

    def screen(self, client_deltas: List[Dict[str, torch.Tensor]], num_samples: List[float], global_model: torch.nn.Module = None, context: Dict[str, Any] = None) -> Tuple[List[float], Dict[str, Any]]:
        """执行 FoolsGold 筛选, 返回每客户端的连续权重 alpha_i (0~1)."""
        context = context or {}
        client_ids = context["client_ids"]
        device = next(global_model.parameters()).device
        learnable_keys = self._get_learnable_keys(global_model, client_deltas[0])

        # 1. 更新历史梯度 H_i += delta_i, 提取完整模型特征
        feature_vectors = []
        for cid, delta in zip(client_ids, client_deltas):
            feat = self._flatten_delta(delta, learnable_keys, device)
            if self.use_history:
                self._history_features[cid] = self._history_features.get(cid, torch.zeros_like(feat)).to(device) + feat
                feature_vectors.append(self._history_features[cid])
            else:
                feature_vectors.append(feat)

        # 2. 计算余弦相似度矩阵
        stack = torch.stack(feature_vectors)
        norms = stack / torch.clamp(stack.norm(p=2, dim=1, keepdim=True), min=1e-8)
        cs = torch.clamp(torch.mm(norms, norms.t()), -1.0, 1.0)

        # 3. Pardoning 赦免
        cs_pardoned = self._pardoning(cs)

        # 4. 基于 Pardoning 后矩阵重算 v_i 并归一化
        alpha_raw = 1.0 - cs_pardoned.max(dim=1).values
        alpha_norm = (alpha_raw / alpha_raw.max().clamp(min=1e-8)).clamp(0.0, 1.0)
        alpha_norm[alpha_norm >= 1.0] = 0.99

        # 5. logit 非线性拉伸
        eps = 1e-7
        alpha_clamped = alpha_norm.clamp(eps, 1.0 - eps)
        alpha_stretched = (torch.log(alpha_clamped / (1.0 - alpha_clamped)) + 0.5).clamp(0.0, 1.0)

        return alpha_stretched.tolist(), context
