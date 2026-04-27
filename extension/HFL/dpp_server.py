"""DPP 异构服务器: 用 BN gamma 幅值作质量项, 归一化卷积权重的 RBF 作相似度, greedy MAP 近似得到通道全序, 各剪枝率取前缀 top-k."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from extension.HFL.cap_manager import CapManager
from extension.HFL.hetero_server import HeteroServer


def _collect_conv_bn_pairs(model: nn.Module) -> List[Tuple[str, str]]:
    """与 MARSScreener 一致: 按模块遍历, 将每个 Conv2d 与紧随其后的同通道数 BatchNorm2d 配对."""
    pairs: List[Tuple[str, str]] = []
    pending_conv: Optional[Tuple[str, nn.Conv2d]] = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            pending_conv = (name, module)
            continue
        if isinstance(module, nn.BatchNorm2d):
            if pending_conv is not None:
                conv_name, conv_module = pending_conv
                if conv_module.out_channels == module.num_features:
                    pairs.append((conv_name, name))
            pending_conv = None
    return pairs


def _median_squared_distances_normalized(W: torch.Tensor, weight_eps: float) -> float:
    """W: [n, d] 行已 L2 归一化. 返回非对角两两 ||w_i - w_j||^2 的中位数作为 sigma^2."""
    n = int(W.shape[0])
    if n <= 1:
        return max(float(weight_eps), 1.0)
    G = W @ W.T
    d2 = 2.0 - 2.0 * G
    d2 = d2.clamp(min=0.0)
    triu = torch.triu_indices(n, n, offset=1)
    vals = d2[triu[0], triu[1]]
    med = float(vals.median().item())
    if not math.isfinite(med) or med <= 0.0:
        med = float(weight_eps)
    return max(med, float(weight_eps))


def _greedy_map_dpp_order(
    L_kernel: torch.Tensor, max_items: Optional[int] = None, eps: float = 1e-12
) -> List[int]:
    """Fast greedy MAP DPP, using incremental Schur complement marginals."""
    n = int(L_kernel.shape[0])
    target = n if max_items is None else max(0, min(int(max_items), n))
    if target == 0:
        return []
    if n == 1:
        return [0]

    device = L_kernel.device
    K = L_kernel
    order: List[int] = []
    selected = torch.zeros(n, device=device, dtype=torch.bool)
    diag = torch.nan_to_num(torch.diagonal(K).clone(), nan=float("-inf"))
    di2 = diag.clamp(min=0.0)
    cis = K.new_zeros((target, n))

    def append_remaining_by_quality() -> None:
        remaining = torch.nonzero(~selected, as_tuple=False).flatten().tolist()
        remaining.sort(key=lambda idx: float(diag[idx].item()), reverse=True)
        for idx in remaining:
            if len(order) >= target:
                break
            selected[idx] = True
            order.append(int(idx))

    for _ in range(target):
        scores = di2.masked_fill(selected, float("-inf"))
        best_j = int(torch.argmax(scores).item())
        best_m = float(scores[best_j].item())
        if not math.isfinite(best_m) or best_m <= eps:
            append_remaining_by_quality()
            break

        step = len(order)
        order.append(best_j)
        selected[best_j] = True
        if len(order) >= target:
            break

        if step == 0:
            residual = K[best_j, :]
        else:
            residual = K[best_j, :] - cis[:step, best_j] @ cis[:step, :]
        ei = residual / math.sqrt(max(best_m, eps))
        cis[step, :] = ei
        di2 = (di2 - ei.square()).clamp(min=0.0)
        di2[selected] = float("-inf")

    return order


class DPPServer(HeteroServer):
    """结构化剪枝时按 DPP (gamma 质量 + RBF 多样性) 生成通道全序, 再按 p 取前缀 top-k."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cap_manager: CapManager,
        gamma_eps: float = 1e-12,
        weight_eps: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__(model=model, device=device, cap_manager=cap_manager, **kwargs)
        self.gamma_eps = float(gamma_eps)
        self.weight_eps = float(weight_eps)
        self._dpp_full_order: Dict[str, List[int]] = {}
        self._conv_to_bn: Dict[str, str] = {}
        self._round_max_pruned_p: Optional[float] = None
        self._refresh_conv_bn_map()

    def _refresh_conv_bn_map(self) -> None:
        pairs = _collect_conv_bn_pairs(self.global_model)
        self._conv_to_bn = {c: b for c, b in pairs}

    def broadcast(self, selected_clients: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        self._dpp_full_order.clear()
        self._refresh_conv_bn_map()
        if not selected_clients:
            self._round_max_pruned_p = None
            return {}

        global_state = self.global_model.state_dict()
        clients_by_p: Dict[float, List[str]] = {}
        for cid in selected_clients:
            p = self.cap_manager.get_bucketed_capability(cid)
            clients_by_p.setdefault(p, []).append(cid)

        pruned_ps = [p for p in clients_by_p if p < 1.0]
        self._round_max_pruned_p = max(pruned_ps) if pruned_ps else None

        package: Dict[str, Dict[str, torch.Tensor]] = {}
        for p, client_ids in clients_by_p.items():
            template_cid = client_ids[0]
            template_state = self._extract_sub_model(global_state, p, template_cid)
            template_order = self._client_orders.get(template_cid, {})
            package[template_cid] = template_state

            for cid in client_ids[1:]:
                self._track_client_order(cid, template_order)
                package[cid] = {k: v.clone() for k, v in template_state.items()}

        return package

    @staticmethod
    def _layer_key_to_conv_prefix(layer_key: str) -> str:
        if layer_key.endswith(".weight"):
            return layer_key[: -len(".weight")]
        return layer_key

    def _compute_full_channel_order(
        self, layer_key: str, num_channels: int, max_items: Optional[int] = None
    ) -> List[int]:
        global_state = self.global_model.state_dict()
        target = num_channels if max_items is None else max(0, min(int(max_items), num_channels))
        conv_w = global_state.get(layer_key)
        if conv_w is None or conv_w.dim() != 4:
            return list(range(target))

        n = int(conv_w.shape[0])
        if n != num_channels:
            return list(range(target))

        conv_prefix = self._layer_key_to_conv_prefix(layer_key)
        bn_name = self._conv_to_bn.get(conv_prefix)
        if bn_name is not None:
            gkey = f"{bn_name}.weight"
            gamma = global_state.get(gkey)
        else:
            gamma = None

        if gamma is None or int(gamma.shape[0]) != n:
            q = torch.ones(n, dtype=torch.float32)
        else:
            q = gamma.detach().float().cpu().abs() + self.gamma_eps

        W = conv_w.detach().float().cpu().reshape(n, -1)
        norms = W.norm(dim=1, keepdim=True).clamp(min=self.weight_eps)
        W = W / norms

        sigma_sq = _median_squared_distances_normalized(W, self.weight_eps)
        G = W @ W.T
        d2 = (2.0 - 2.0 * G).clamp(min=0.0)
        S = torch.exp(-d2 / (2.0 * sigma_sq))

        q_col = q.unsqueeze(1)
        L_mat = S * (q_col @ q_col.T)
        L_mat = 0.5 * (L_mat + L_mat.T)

        order = _greedy_map_dpp_order(
            L_mat.to(dtype=torch.float32),
            max_items=target,
            eps=max(self.gamma_eps, self.weight_eps, 1e-12),
        )
        if len(order) != target or len(set(order)) != len(order):
            return list(range(target))
        return order

    def _get_layer_indices(
        self, num_channels: int, p: float, layer_key: str, client_id: str
    ) -> List[int]:
        del client_id
        n_keep = max(1, math.ceil(num_channels * p))
        if layer_key not in self._dpp_full_order:
            max_p = self._round_max_pruned_p if self._round_max_pruned_p is not None else p
            max_keep = max(n_keep, math.ceil(num_channels * max_p))
            self._dpp_full_order[layer_key] = self._compute_full_channel_order(
                layer_key, num_channels, max_items=max_keep
            )
        full_order = self._dpp_full_order[layer_key]
        take = min(n_keep, len(full_order))
        selected = full_order[:take]
        return sorted(selected)
