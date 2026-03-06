import torch
from typing import List, Dict, Optional, Any, Tuple
from core.server.aggregator.base_aggregator import BaseAggregator
from core.utils.registry import AGGREGATOR_REGISTRY


@AGGREGATOR_REGISTRY.register("sub_avg")
class SubAvgAggregator(BaseAggregator):
    """
    子模型聚合器 (Sub-model Aggregator)，采用"掩码感知的动态归一化聚合"策略。
    
    适用场景：
        - 模型异构联邦学习：客户端通过结构化剪枝仅训练全局模型的子集
        - 稀疏更新:Delta 矩阵在未训练的参数位置被 Zero-Padding
        - 防御机制共存：系统包含后门/投毒防御，为客户端计算软信任分数
    
    核心区别于 AvgAggregator:
        - 采用 逐元素动态归一化 (Element-wise Dynamic Normalization)
        - 分母仅包含更新了该参数的客户端权重之和，消除稀释效应
    
    实现逻辑：
        1. 显式掩码 (Explicit Mask): 优先使用服务端提供的参与掩码
           （兼容回退：无掩码时使用零值推断）
        2. 软加权结合 (Soft-Weighted Integration): 分子加权求和，分母累加有效权重
        3. 按位还原 (Element-wise Restoration): 分子除以分母得到加权平均
    """

    def __init__(self, device='cpu', eps: float = 1e-10):
        """
        初始化子模型聚合器。
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda')
            eps: 防止除零的小量，默认 1e-10
        """
        super().__init__(device)
        self.eps = eps

    def aggregate(self,
                  updates: List[Dict[str, torch.Tensor]],
                  sample_weights: Optional[List[float]] = None,
                  screen_scores: Optional[List[float]] = None,
                  global_model: torch.nn.Module = None,
                  context: Dict[str, Any] = None,
                  **kwargs) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        聚合客户端 Deltas 并返回完整模型权重。
        
        采用掩码感知 (Mask-Aware) 的加权平均，通过动态调整分母解决
        异构联邦学习中参数更新频率不一致导致的梯度尺度失真问题。
        
        Args:
            updates: 客户端 Delta 列表，每个 Delta 是 {layer_name: tensor} 的字典
            sample_weights: 样本权重列表（如客户端样本数）
            screen_scores: 防御机制计算的软信任分数
            global_model: 全局模型，用于获取基础权重
            context: 上下文信息字典
            **kwargs: 其他参数
            
        Returns:
            (final_weights, context): 聚合后的完整模型权重和更新后的上下文
        """
        if not updates:
            raise ValueError("Updates list is empty")

        context = context or {}
        num_clients = len(updates)
        masks = context.get('masks')
        order_books = context.get('order_books')

        if sample_weights is None:
            sample_weights = [1.0] * num_clients
        if screen_scores is None:
            screen_scores = [1.0] * num_clients

        w = [s * sc for s, sc in zip(sample_weights, screen_scores)]
        self._check_inputs(updates, w)

        w_t = torch.tensor(w, dtype=torch.float32, device=self.device)

        if order_books is not None and global_model is not None:
            return self._aggregate_streaming(
                updates, order_books, w_t, global_model, context
            )

        agg_dlt = {}
        layer_names = updates[0].keys()

        for name in layer_names:
            dlt_s = torch.stack(
                [u[name].to(torch.float32) for u in updates]
            ).to(self.device)

            if masks is not None:
                msk_s = torch.stack(
                    [m[name].to(torch.float32) for m in masks]
                ).to(self.device)
            else:
                msk_s = (dlt_s != 0).float()

            w_shape = [num_clients] + [1] * (dlt_s.dim() - 1)
            w_view = w_t.view(*w_shape)

            den = torch.sum(msk_s * w_view, dim=0)
            num = torch.sum(dlt_s * w_view, dim=0)
            agg_dlt[name] = num / (den + self.eps)

        return agg_dlt, context

    def _aggregate_streaming(self,
                            updates: List[Dict[str, torch.Tensor]],
                            order_books: List[Dict],
                            w_t: torch.Tensor,
                            global_model: torch.nn.Module,
                            context: Dict[str, Any]) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """按索引流式累加, 避免 stack 全量 delta/mask 的峰值内存。"""
        global_state = global_model.state_dict()
        skip = ("num_batches_tracked", "running_mean", "running_var")

        def _ok(k: str) -> bool:
            return not any(k.endswith(s) for s in skip)

        layer_names = [k for k in global_state.keys() if _ok(k)]
        agg_dlt = {}
        num_clients = len(updates)

        for name in layer_names:
            ref = global_state[name]
            num = torch.zeros_like(ref, dtype=torch.float32, device=self.device)
            den = torch.zeros_like(ref, dtype=torch.float32, device=self.device)

            for i in range(num_clients):
                w_i = w_t[i].item()
                delta = updates[i]
                ob = order_books[i] if i < len(order_books) else {}

                if not ob:
                    if name in delta:
                        c = delta[name].float().to(self.device)
                        num.add_(c, alpha=w_i)
                        den.add_(torch.ones_like(c, device=self.device), alpha=w_i)
                elif name in ob and name in delta:
                    mapping = ob[name]
                    c = delta[name].float().to(self.device)
                    if isinstance(mapping, tuple):
                        t_o, t_i = mapping
                        t_o = t_o.to(self.device)
                        t_i = t_i.to(self.device)
                        num[t_o[:, None], t_i] = num[t_o[:, None], t_i] + c * w_i
                        den[t_o[:, None], t_i] = den[t_o[:, None], t_i] + w_i
                    else:
                        idx = mapping.to(self.device)
                        num[idx] = num[idx] + c.flatten() * w_i
                        den[idx] = den[idx] + w_i

            agg_dlt[name] = num / (den + self.eps)

        return agg_dlt, context
