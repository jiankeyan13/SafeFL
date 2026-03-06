import math
from typing import Dict, List, Any, Optional, Set, Tuple

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
        """根据客户端容量动态生成并下发个性化子模型。先切片后按需拷贝, 避免整模型 clone/deepcopy。"""
        global_state = self.global_model.state_dict()
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
            return {k: v.clone() for k, v in global_state.items()}

        state: Dict[str, torch.Tensor] = {}
        keys = list(global_state.keys())
        order_dict: Dict[str, Any] = {}
        prev_order: Optional[List[int]] = None
        prev_num_out: Optional[int] = None

        cur_block: Optional[str] = None
        blk_in_order: Optional[List[int]] = None
        blk_in_num: Optional[int] = None
        sc_buf: List[str] = []

        block_has_shortcut = self._infer_block_shortcut(keys)

        for k in keys:
            if 'num_batches_tracked' in k:
                continue
            tensor = global_state[k]

            if tensor.dim() == 0:
                if k.endswith('.scale'):
                    state[k] = torch.tensor(1.0 / p, device=tensor.device, dtype=tensor.dtype)
                continue

            bid = self._parse_block_id(k)
            if bid is not None and bid != cur_block:
                self._prune_shortcut(state, sc_buf, order_dict, global_state, \
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
                is_identity_tail = (
                    bid is not None
                    and blk_in_order is not None
                    and not block_has_shortcut.get(bid, True)
                    and self._is_block_tail_conv(k)
                )
                if 'linear' in k:
                    order = list(range(num_out))
                elif is_identity_tail:
                    order = list(blk_in_order)
                else:
                    order = self._get_layer_indices(num_out, p, k, client_id)

                in_idx = list(range(tensor.shape[1]))
                if prev_order is not None:
                    in_dim = tensor.shape[1]
                    if prev_num_out != in_dim:
                        r = in_dim // prev_num_out
                        in_idx = [i * r + s for i in prev_order for s in range(r)]
                    else:
                        in_idx = list(prev_order)
                    sliced = tensor[:, in_idx]
                else:
                    sliced = tensor
                state[k] = sliced[order].clone()
                t_o = torch.tensor(order, dtype=torch.long)
                t_i = torch.tensor(in_idx, dtype=torch.long)
                order_dict[k] = (t_o, t_i)
                prev_order = order
                prev_num_out = num_out
            else:
                if prev_order is not None:
                    state[k] = tensor[prev_order].clone()
                    order_dict[k] = torch.tensor(prev_order, dtype=torch.long)

        self._prune_shortcut(state, sc_buf, order_dict, global_state, \
                            blk_in_order, blk_in_num, prev_order)
        self._track_client_order(client_id, order_dict)
        return state

    @staticmethod
    def _infer_block_shortcut(keys: List[str]) -> Dict[str, bool]:
        """推断每个 block 是否有显式 shortcut (无 shortcut 则为 identity residual)"""
        blocks: Set[str] = set()
        for k in keys:
            bid = HeteroServer._parse_block_id(k)
            if bid is not None:
                blocks.add(bid)
        result = {bid: False for bid in blocks}
        for k in keys:
            bid = HeteroServer._parse_block_id(k)
            if bid is not None and 'shortcut' in k:
                result[bid] = True
        return result

    @staticmethod
    def _is_block_tail_conv(key: str) -> bool:
        """判断是否为 block 尾卷积 (conv2 for PreActBlock, conv3 for PreActBottleneck)"""
        return 'conv2.weight' in key or 'conv3.weight' in key

    def _get_layer_indices(self, num_channels: int, p: float, \
                          layer_key: str, client_id: str) -> List[int]:
        """随机采样保留通道索引，返回升序列表"""
        n_keep = max(1, math.ceil(num_channels * p))
        chosen = np.random.choice(num_channels, n_keep, replace=False)
        return sorted(chosen.tolist())

    def align_delta(self, client_delta: Dict[str, torch.Tensor], \
                   client_id: str) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """将客户端子模型 delta 对齐回全局维度，并返回显式参与掩码"""
        order_book = self._client_orders.get(client_id, {})
        dlt, msk = self._init_aligned_delta_and_mask()

        if not order_book:
            for k, delta in client_delta.items():
                if k in dlt:
                    dlt[k] = delta.float().cpu()
                    msk[k] = torch.ones_like(dlt[k])
            return dlt, msk

        for k, mapping in order_book.items():
            if k not in client_delta:
                continue
            c = client_delta[k].float().cpu()
            if isinstance(mapping, tuple):
                t_o, t_i = mapping
                dlt[k][t_o[:, None], t_i] = c
                msk[k][t_o[:, None], t_i] = 1.0
            else:
                dlt[k][mapping] = c
                msk[k][mapping] = 1.0
        return dlt, msk

    def _init_aligned_delta_and_mask(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """按全局参数形状初始化全零 delta 与 mask 容器。排除 BN running stats, 由 proxy_loader 校准。"""
        global_state = self.global_model.state_dict()
        skip = ("num_batches_tracked", "running_mean", "running_var")

        def _ok(k: str) -> bool:
            return not any(k.endswith(s) for s in skip)

        dlt = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in global_state.items() if _ok(k)}
        msk = {k: torch.zeros_like(v, dtype=torch.float32, device="cpu") for k, v in global_state.items() if _ok(k)}
        return dlt, msk

    def step(self, updates, proxy_loader=None):
        """Hetero pipeline: client delta -> screen (expand if needed) -> streaming aggregate -> refiner."""
        if not updates:
            return

        context: Dict[str, Any] = {}
        num_samples = [u["num_samples"] for u in updates]
        sparse_deltas = [u["delta"] for u in updates]
        client_ids = [u["client_id"] for u in updates]
        order_books = [self._client_orders.get(cid, {}) for cid in client_ids]

        needs_dense = type(self.screener).__name__ != "BaseScreener"
        if needs_dense:
            delta_mask_pairs = [self.align_delta(d, cid) for d, cid in zip(sparse_deltas, client_ids)]
            delta_list = [x[0] for x in delta_mask_pairs]
            mask_list = [x[1] for x in delta_mask_pairs]
        else:
            delta_list = sparse_deltas
            mask_list = None

        screen_scores, context = self.screener.screen(
            client_deltas=delta_list,
            num_samples=num_samples,
            global_model=self.global_model,
            context=context,
        )

        context["order_books"] = order_books
        context["masks"] = mask_list
        aggregated_delta, context = self.aggregator.aggregate(
            updates=sparse_deltas,
            sample_weights=num_samples,
            screen_scores=screen_scores,
            global_model=self.global_model,
            context=context,
        )

        new_state = {}
        global_state = self.global_model.state_dict()
        for key, value in global_state.items():
            new_state[key] = value.clone()
            if key in aggregated_delta:
                new_state[key] += aggregated_delta[key].to(
                    device=value.device, dtype=value.dtype
                )

        self.refiner.process(
            self.global_model,
            new_state,
            calibration_loader=proxy_loader,
            device=self.device,
            context=context,
        )

    def _track_client_order(self, client_id: str, order_dict: Dict[str, Any]) -> None:
        """缓存本轮分发给客户端的索引布局（账本）"""
        self._client_orders[client_id] = order_dict

    def _prune_shortcut(self, state: Dict[str, torch.Tensor], buf: List[str], \
                       order_dict: Dict[str, Any], global_state: Dict[str, torch.Tensor], \
                       in_order: Optional[List[int]], in_num: Optional[int], \
                       out_order: Optional[List[int]]) -> None:
        """对缓冲的shortcut参数执行剪枝"""
        if not buf or out_order is None:
            return
        for k in buf:
            t = global_state[k]
            if t.dim() == 0:
                continue
            if self._is_conv_linear_weight(k):
                in_idx = list(in_order) if in_order else list(range(t.shape[1]))
                if in_num is not None and in_num != t.shape[1]:
                    r = t.shape[1] // in_num
                    in_idx = [i * r + s for i in in_order for s in range(r)]
                state[k] = t[:, in_idx][out_order].clone()
                order_dict[k] = (torch.tensor(out_order, dtype=torch.long), torch.tensor(in_idx, dtype=torch.long))
            else:
                state[k] = t[out_order].clone()
                order_dict[k] = torch.tensor(out_order, dtype=torch.long)

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
