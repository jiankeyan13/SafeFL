"""
FLTrust 筛选器: 基于 proxy 数据计算 server reference delta_0, 对客户端 delta 计算 cosine similarity 并过 ReLU 得到 trust score.
方向相反的 delta -> TS=0, 不参与后续聚合.
"""
from typing import List, Dict, Any, Tuple, Set

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


def _get_delta_keys(state_dict: Dict[str, torch.Tensor]) -> Set[str]:
    """与 BaseClient 一致的 delta 键过滤: 排除 BN running stats 和 num_batches_tracked."""
    return {
        k for k in state_dict.keys()
        if "num_batches_tracked" not in k
        and not k.endswith("running_mean")
        and not k.endswith("running_var")
    }


def _flatten_delta(delta: Dict[str, torch.Tensor], keys: Set[str], device: torch.device) -> torch.Tensor:
    """将 delta 按指定键展平为一维向量."""
    flat = []
    for k in sorted(keys):
        if k in delta:
            flat.append(delta[k].view(-1).to(device))
    return torch.cat(flat) if flat else torch.tensor([], device=device)


def _compute_server_delta(
    model: nn.Module,
    proxy_loader: DataLoader,
    device: torch.device,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    epochs: int = 1,
) -> Dict[str, torch.Tensor]:
    """
    在 proxy 数据上微调模型, 计算 server reference update delta_0.
    键集合与客户端 delta 一致. 微调后恢复原模型状态, 不改变全局模型.
    """
    keys = _get_delta_keys(model.state_dict())
    full_initial = {k: v.clone() for k, v in model.state_dict().items()}
    initial_state = {k: full_initial[k] for k in keys}

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for data, target in proxy_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    current_state = model.state_dict()
    delta = {
        k: current_state[k] - initial_state[k]
        for k in initial_state
        if k in current_state
    }

    model.load_state_dict(full_initial, strict=True)
    return delta


@SCREENER_REGISTRY.register("fltrust")
class FLTrustScreener(BaseScreener):
    """
    FLTrust 筛选器: 用 proxy 数据计算 delta_0, 对每个客户端 delta_i 计算 TS_i = ReLU(cos(delta_i, delta_0)).
    """

    def __init__(
        self,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        epochs: int = 1,
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.eps = eps

    def screen(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        num_samples: List[float],
        global_model: nn.Module = None,
        context: Dict[str, Any] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        context = context or {}
        n = len(client_deltas)
        if n == 0:
            return [], context

        proxy_loader = context.get("proxy_loader")
        if proxy_loader is None:
            raise ValueError("FLTrust requires proxy_loader in context; none provided.")

        device = next(global_model.parameters()).device
        delta_0 = _compute_server_delta(
            global_model,
            proxy_loader,
            device,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
        )

        keys = _get_delta_keys(delta_0)
        flat_0 = _flatten_delta(delta_0, keys, device)
        raw_ref_norm = torch.norm(flat_0).item()

        if raw_ref_norm < 1e-6:
            context["fltrust_fallback"] = "zero_reference_norm"
            return [1.0] * n, context

        ref_norm = raw_ref_norm + self.eps
        trust_scores = []
        client_norms = []
        for delta_i in client_deltas:
            flat_i = _flatten_delta(delta_i, keys, device)
            raw_norm_i = torch.norm(flat_i).item()
            norm_i = raw_norm_i + self.eps
            client_norms.append(raw_norm_i)
            cos_sim = (flat_0 @ flat_i) / (ref_norm * norm_i)
            cos_sim = cos_sim.clamp(-1.0, 1.0).item()
            ts = max(0.0, cos_sim)
            trust_scores.append(ts)

        context["reference_delta"] = delta_0
        context["reference_norm"] = raw_ref_norm
        context["delta_keys"] = keys
        context["client_norms"] = client_norms
        context["trust_scores"] = trust_scores
        return trust_scores, context
