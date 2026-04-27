"""按 BN gamma 幅值 (|gamma|) 为各卷积层选通道, 与 DPPServer 同构但不做 DPP 只按重要性排序取 top-k."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch

from extension.HFL.cap_manager import CapManager
from extension.HFL.dpp_server import _collect_conv_bn_pairs
from extension.HFL.hetero_server import HeteroServer


class GammaMagnitudeServer(HeteroServer):
    """结构化剪枝时按全局 BN gamma 幅值降序得到通道全序, 再按 p 取前缀 top-k."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        cap_manager: CapManager,
        gamma_eps: float = 1e-12,
        **kwargs,
    ) -> None:
        super().__init__(model=model, device=device, cap_manager=cap_manager, **kwargs)
        self.gamma_eps = float(gamma_eps)
        self._gamma_full_order: Dict[str, List[int]] = {}
        self._conv_to_bn: Dict[str, str] = {}
        self._round_max_pruned_p: Optional[float] = None
        self._refresh_conv_bn_map()

    def _refresh_conv_bn_map(self) -> None:
        pairs = _collect_conv_bn_pairs(self.global_model)
        self._conv_to_bn = {c: b for c, b in pairs}

    def broadcast(self, selected_clients: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        self._gamma_full_order.clear()
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
            return list(range(target))

        q = gamma.detach().float().abs() + self.gamma_eps
        # 幅值从大到小, 与常见 BN 通道重要性一致
        order = torch.argsort(q, descending=True)[:target].cpu().tolist()
        if len(order) != target or len(set(order)) != len(order):
            return list(range(target))
        return order

    def _get_layer_indices(
        self, num_channels: int, p: float, layer_key: str, client_id: str
    ) -> List[int]:
        del client_id
        n_keep = max(1, math.ceil(num_channels * p))
        if layer_key not in self._gamma_full_order:
            max_p = self._round_max_pruned_p if self._round_max_pruned_p is not None else p
            max_keep = max(n_keep, math.ceil(num_channels * max_p))
            self._gamma_full_order[layer_key] = self._compute_full_channel_order(
                layer_key, num_channels, max_items=max_keep
            )
        full_order = self._gamma_full_order[layer_key]
        take = min(n_keep, len(full_order))
        selected = full_order[:take]
        return sorted(selected)
