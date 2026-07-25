from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.utils.extmath import randomized_svd

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("dnc")
class DnCScreener(BaseScreener):
    """
    DnC (Divide-and-Conquer) 筛选器.

    流程:
    1. 将可学习参数 delta 展平并中心化.
    2. 随机抽取 sub_dim 个原始坐标构成子空间 (默认 10000 维).
    3. 取第一右奇异向量, 用投影绝对值作为离群分数.
    4. 重复 num_iters 次独立坐标子采样, 每次剔除 num_outliers 个客户端.
    5. 对多次迭代的良性客户端集合取交集.
    """

    def __init__(
        self,
        sub_dim: int = 10000,
        num_iters: int = 3,
        num_outliers: int = 1,
        seed: int = 42,
        eps: float = 1e-12,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_dim = int(sub_dim)
        self.num_iters = int(num_iters)
        self.num_outliers = int(num_outliers)
        self.seed = int(seed)
        self.eps = float(eps)
        self._screen_round = 0

    def screen(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        num_samples: List[float],
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        context = context or {}
        num_clients = len(client_deltas)

        if num_clients < 2 or num_clients <= self.num_outliers:
            context["benign_indices"] = set(range(num_clients))
            context["dnc_iter_benign"] = [list(range(num_clients))] * max(self.num_iters, 1)
            context["dnc_iter_scores"] = []
            context["dnc_dropped"] = []
            return [1.0] * num_clients, context

        learnable_keys = self._get_learnable_keys(global_model, client_deltas[0])
        matrix = self._stack_vectors(client_deltas, learnable_keys).astype(np.float64, copy=False)
        centered = matrix - matrix.mean(axis=0, keepdims=True)

        round_offset = self._screen_round
        iter_benign: List[set[int]] = []
        iter_scores: List[List[float]] = []

        for iter_idx in range(self.num_iters):
            subsampled = self._random_subsample(centered, round_offset, iter_idx)
            scores = self._spectral_outlier_scores(subsampled)
            outlier_indices = np.argsort(scores)[-self.num_outliers:]
            benign = set(range(num_clients)) - set(int(i) for i in outlier_indices)
            iter_benign.append(benign)
            iter_scores.append(scores.tolist())

        benign_indices = set.intersection(*iter_benign) if iter_benign else set(range(num_clients))
        if not benign_indices:
            benign_indices = set(range(num_clients))

        dropped = sorted(set(range(num_clients)) - benign_indices)
        screen_scores = [1.0 if idx in benign_indices else 0.0 for idx in range(num_clients)]

        context.update({
            "benign_indices": benign_indices,
            "dnc_iter_benign": [sorted(indices) for indices in iter_benign],
            "dnc_iter_scores": iter_scores,
            "dnc_dropped": dropped,
        })
        self._screen_round += 1
        return screen_scores, context

    def _get_learnable_keys(
        self,
        global_model: torch.nn.Module,
        first_delta: Dict[str, torch.Tensor],
    ) -> Sequence[str]:
        if global_model is not None:
            return [name for name, param in global_model.named_parameters() if param.requires_grad]
        return list(first_delta.keys())

    def _stack_vectors(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        learnable_keys: Sequence[str],
    ) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for delta in client_deltas:
            flat_parts = [
                delta[name].detach().float().cpu().reshape(-1).numpy()
                for name in learnable_keys
                if name in delta
            ]
            vectors.append(np.concatenate(flat_parts, axis=0) if flat_parts else np.zeros(1, dtype=np.float32))
        return np.stack(vectors, axis=0)

    def _random_subsample(self, centered: np.ndarray, round_offset: int, iter_idx: int) -> np.ndarray:
        num_clients, dim = centered.shape
        if dim == 0:
            return np.zeros((num_clients, 0), dtype=np.float64)

        target_dim = min(self.sub_dim, dim)
        if dim <= self.sub_dim:
            return centered

        rng = np.random.RandomState(self.seed + round_offset * self.num_iters + iter_idx)
        coord_indices = rng.choice(dim, size=target_dim, replace=False)
        return centered[:, coord_indices]

    def _spectral_outlier_scores(self, subsampled: np.ndarray) -> np.ndarray:
        num_clients = subsampled.shape[0]
        if num_clients == 0:
            return np.zeros(0, dtype=np.float64)
        if subsampled.shape[1] == 0:
            return np.zeros(num_clients, dtype=np.float64)

        n_components = min(1, subsampled.shape[0], subsampled.shape[1])
        _, _, vh = randomized_svd(
            subsampled,
            n_components=n_components,
            random_state=self.seed,
        )
        direction = vh[0]
        return np.abs(subsampled @ direction)
