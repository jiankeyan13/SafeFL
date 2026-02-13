import copy
import math
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch

from core.server.base_server import BaseServer
from .cap_manager import CapManager


class HeteroServer(BaseServer):
    """异构联邦学习服务器：通过结构化剪枝为不同容量的客户端生成个性化子模型，并将子模型更新对齐回全局参数空间"""

    def __init__(self, model: torch.nn.Module, device: torch.device, \
                 cap_manager: CapManager, **kwargs):
        super().__init__(model=model, device=device, **kwargs)
        self.cap_manager = cap_manager
        self._client_orders: Dict[str, Dict[str, Any]] = {}

    def broadcast(self, selected_clients: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """根据客户端容量动态生成并下发个性化子模型"""
        global_state = self.get_global_model()
        package: Dict[str, Dict[str, torch.Tensor]] = {}
        for cid in selected_clients:
            p = self.cap_manager.get_bucketed_capability(cid)
            package[cid] = self._extract_sub_model(global_state, p, cid)
        return package

    def _extract_sub_model(self, global_state: Dict[str, torch.Tensor], \
                          p: float, client_id: str) -> Dict[str, torch.Tensor]:
        """从全局模型提取子模型权重，自动处理层间拓扑依赖（输入/输出通道对齐、残差连接对齐）"""
        if p >= 1.0:
            self._track_client_order(client_id, {})
            return copy.deepcopy(global_state)

        state = copy.deepcopy(global_state)
        keys = list(state.keys())
        order_dict: Dict[str, Any] = {}
        prev_order: Optional[List[int]] = None
        prev_num_out: Optional[int] = None

        cur_block: Optional[str] = None
        blk_in_order: Optional[List[int]] = None
        blk_in_num: Optional[int] = None
        sc_buf: List[str] = []

        for k in keys:
            if 'num_batches_tracked' in k:
                continue
            tensor = state[k]

            if tensor.dim() == 0:
                if k.endswith('.scale'):
                    state[k] = torch.tensor(1.0 / p)
                continue

            bid = self._parse_block_id(k)
            if bid is not None and bid != cur_block:
                self._prune_shortcut(state, sc_buf, order_dict, \
                                    blk_in_order, blk_in_num, prev_order)
                sc_buf.clear()
                cur_block = bid
                blk_in_order = prev_order
                blk_in_num = prev_num_out

            if 'shortcut' in k:
                sc_buf.append(k)
                continue

            if self._is_conv_linear_weight(k):
                num_out = tensor.shape[0]
                order = (list(range(num_out)) if 'linear' in k \
                        else self._get_layer_indices(num_out, p, k, client_id))

                in_idx = list(range(tensor.shape[1]))
                if prev_order is not None:
                    in_dim = tensor.shape[1]
                    if prev_num_out != in_dim:
                        r = in_dim // prev_num_out
                        in_idx = [i * r + s for i in prev_order for s in range(r)]
                    else:
                        in_idx = list(prev_order)
                    state[k] = tensor[:, in_idx]

                state[k] = state[k][order]
                order_dict[k] = (order, in_idx)
                prev_order = order
                prev_num_out = num_out
            else:
                if prev_order is not None:
                    state[k] = tensor[prev_order]
                    order_dict[k] = list(prev_order)

        self._prune_shortcut(state, sc_buf, order_dict, \
                            blk_in_order, blk_in_num, prev_order)
        self._track_client_order(client_id, order_dict)
        return state

    def _get_layer_indices(self, num_channels: int, p: float, \
                          layer_key: str, client_id: str) -> List[int]:
        """随机采样保留通道索引，返回升序列表"""
        n_keep = max(1, math.ceil(num_channels * p))
        chosen = np.random.choice(num_channels, n_keep, replace=False)
        return sorted(chosen.tolist())

    def compute_delta(self, client_weights: Dict[str, torch.Tensor], \
                     client_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """将子模型更新映射回全局维度，并返回显式参与掩码"""
        order_book = self._client_orders.get(client_id, {})
        g_state = {k: v.float().cpu() for k, v in self.global_model.state_dict().items()}
        dlt: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in g_state.items() \
                                        if 'num_batches_tracked' not in k}
        msk: Dict[str, torch.Tensor] = {k: torch.zeros_like(v) for k, v in g_state.items() \
                                        if 'num_batches_tracked' not in k}

        if not order_book:
            for k in dlt:
                if k in client_weights:
                    c = client_weights[k].float().cpu()
                    dlt[k] = c - g_state[k]
                    msk[k] = torch.ones_like(dlt[k])
            return dlt, msk

        for k, mapping in order_book.items():
            if k not in client_weights:
                continue
            c = client_weights[k].float().cpu()
            if isinstance(mapping, tuple):
                idx_o, idx_i = mapping
                t_o = torch.tensor(idx_o, dtype=torch.long)
                t_i = torch.tensor(idx_i, dtype=torch.long)
                g = g_state[k][t_o][:, t_i]
                dlt[k][t_o[:, None], t_i] = c - g
                msk[k][t_o[:, None], t_i] = 1.0
            else:
                g = g_state[k][mapping]
                dlt[k][mapping] = c - g
                msk[k][mapping] = 1.0
        return dlt, msk

    def step(self, updates, proxy_loader=None):
        """防御流水线：通过client_id定位索引账本"""
        context: Dict[str, Any] = {}
        num_samples = [u['num_samples'] for u in updates]
        dlt_msk = [self.compute_delta(u['weights'], u['client_id']) for u in updates]
        dlt_list = [x[0] for x in dlt_msk]
        msk_list = [x[1] for x in dlt_msk]

        screen_scores, context = self.screener.screen(client_deltas=dlt_list, \
                                                      num_samples=num_samples, \
                                                      global_model=self.global_model, \
                                                      context=context)

        context['masks'] = msk_list
        agg_weights, context = self.aggregator.aggregate(updates=dlt_list, \
                                                         sample_weights=num_samples, \
                                                         screen_scores=screen_scores, \
                                                         global_model=self.global_model, \
                                                         context=context)

        self.updater.update(self.global_model, agg_weights, \
                           calibration_loader=proxy_loader, \
                           device=self.device, context=context)

    def _track_client_order(self, client_id: str, order_dict: Dict[str, Any]) -> None:
        """缓存本轮分发给客户端的索引布局（账本）"""
        self._client_orders[client_id] = order_dict

    def _prune_shortcut(self, state: Dict[str, torch.Tensor], buf: List[str], \
                       order_dict: Dict[str, Any], in_order: Optional[List[int]], \
                       in_num: Optional[int], out_order: Optional[List[int]]) -> None:
        """对缓冲的shortcut参数执行剪枝"""
        if not buf or out_order is None:
            return
        for k in buf:
            t = state[k]
            if t.dim() == 0:
                continue
            if self._is_conv_linear_weight(k):
                in_idx = list(in_order) if in_order else list(range(t.shape[1]))
                if in_num is not None and in_num != t.shape[1]:
                    r = t.shape[1] // in_num
                    in_idx = [i * r + s for i in in_order for s in range(r)]
                state[k] = t[:, in_idx][out_order]
                order_dict[k] = (list(out_order), in_idx)
            else:
                state[k] = t[out_order]
                order_dict[k] = list(out_order)

    @staticmethod
    def _parse_block_id(key: str) -> Optional[str]:
        """解析'layerX.Y' block标识"""
        if 'layer' not in key:
            return None
        parts = key.split('.')
        for i, part in enumerate(parts):
            if part.startswith('layer') and i + 1 < len(parts) and parts[i + 1].isdigit():
                return f"{part}.{parts[i + 1]}"
        return None

    @staticmethod
    def _is_conv_linear_weight(key: str) -> bool:
        """判断key是否为Conv/Linear权重（排除BN）"""
        return 'weight' in key and 'bn' not in key
