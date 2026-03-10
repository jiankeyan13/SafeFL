"""
FoolsGold 筛选器: 通过历史梯度相似性动态分配聚合权重, 抵御 Sybil 攻击.

算法流程:
1. 将 delta_i 累加到各自历史梯度 H_i
2. 提取输出层参数作为指示性特征
3. 计算客户端对之间的加权余弦相似度矩阵
4. 执行 Pardoning 赦免操作, 避免误伤诚实客户端
5. 计算 alpha_i = 1 - max_j(cs_ij), 归一化后经 logit 非线性拉伸
"""

from typing import List, Dict, Any, Tuple, Optional

import torch
from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("foolsgold")
class FoolsGoldScreener(BaseScreener):
    """
    FoolsGold 筛选器: 基于历史梯度相似性为每个客户端分配学习率权重.

    核心逻辑嵌入在筛选阶段, 通过动态权重抑制 Sybil 恶意贡献, 而非直接丢弃.
    与 AvgAggregator 配合使用时, screen_scores 作为聚合权重.
    """

    def __init__(self, use_history: bool = True, topk_prop: float = 0.1, **kwargs):
        """use_history: 是否使用历史梯度累加; topk_prop: 每类保留特征比例 (默认 0.1)."""
        super().__init__(**kwargs)
        self.use_history, self.topk_prop = use_history, topk_prop
        self._history_features: Dict[str, torch.Tensor] = {}
        self._output_layer_cache: Optional[Tuple[frozenset, Optional[str], Optional[str]]] = None

    def _importance_weights(self, global_model: torch.nn.Module, weight_name: str, bias_name: Optional[str], device: torch.device) -> torch.Tensor:
        """软加权 (论文 importanceFeatureMapLocal): 去均值 + 按类归一化 + top-k. bias 对应位置填 1."""
        state = global_model.state_dict()
        weight = state[weight_name].float().to(device)
        n_classes, class_d = weight.shape[0], weight.numel() // weight.shape[0]
        k = max(1, int(class_d * self.topk_prop))
        M = weight.reshape(n_classes, class_d)
        M_centered = (M - M.mean(dim=1, keepdim=True)).abs()
        M_norm = M_centered / M_centered.sum(dim=1, keepdim=True).clamp(min=1e-8)
        _, topk_idx = M_norm.topk(k, dim=1, largest=True)
        mask = torch.zeros_like(M_norm, device=device)
        mask.scatter_(1, topk_idx, 1.0)
        weight_imp = (M_norm * mask).flatten()
        if bias_name and bias_name in state:
            bias_ones = torch.ones(state[bias_name].numel(), device=device, dtype=weight_imp.dtype)
            return torch.cat([weight_imp, bias_ones])
        return weight_imp

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
        n = len(client_deltas)
        client_ids = context["client_ids"]

        # 输出层定位 (带缓存)
        state_keys = frozenset(global_model.state_dict().keys())
        if self._output_layer_cache is not None and self._output_layer_cache[0] == state_keys:
            weight_name, bias_name = self._output_layer_cache[1], self._output_layer_cache[2]
        else:
            keys = list(global_model.state_dict().keys())
            key_set = set(keys)
            candidates = [(k, k[:-6] + "bias") for k in keys if k.endswith(".weight") and k[:-6] + "bias" in key_set]
            weight_name, bias_name = candidates[-1] if candidates else (None, None)
            self._output_layer_cache = (state_keys, weight_name, bias_name)

        device = next(global_model.parameters()).device

        # 1. 更新历史梯度 H_i += delta_i, 提取输出层特征
        feature_vectors = []
        for cid, delta in zip(client_ids, client_deltas):
            parts = [delta[weight_name].flatten()]
            if bias_name and bias_name in delta:
                parts.append(delta[bias_name].flatten())
            feat = torch.cat(parts).float().to(device)
            if self.use_history:
                self._history_features[cid] = self._history_features.get(cid, torch.zeros_like(feat)).to(device) + feat
                feature_vectors.append(self._history_features[cid])
            else:
                feature_vectors.append(feat)

        # 1.5 特征重要性软加权 (论文 importanceFeatureMapLocal)
        imp = self._importance_weights(global_model, weight_name, bias_name, device)
        feature_vectors = [fv * imp for fv in feature_vectors]

        # 2. 计算加权余弦相似度矩阵
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
