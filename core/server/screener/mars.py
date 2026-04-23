from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("mars")
class MARSScreener(BaseScreener):
    """
    MARS 筛选器:
    1. 用 global_model + delta 还原每个客户端本地模型状态
    2. 仅基于 Conv2d + BatchNorm2d 配对层提取 CBE
    3. 对客户端 CBE 的 Wasserstein 距离矩阵执行 K-Means 聚类
    4. 剔除异常簇对应客户端
    """

    def __init__(
        self,
        top_ratio: float = 0.05,
        cluster_distance_threshold: float = 0.03,
        n_clusters: int = 2,
        bn_eps: float = 1e-5,
        var_clamp_min: float = 0.0,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not (0.0 < float(top_ratio) <= 1.0):
            raise ValueError("top_ratio must be in (0, 1].")
        if int(n_clusters) < 2:
            raise ValueError("n_clusters must be >= 2 for MARS.")

        self.top_ratio = float(top_ratio)
        self.cluster_distance_threshold = float(cluster_distance_threshold)
        self.n_clusters = int(n_clusters)
        self.bn_eps = float(bn_eps)
        self.var_clamp_min = float(var_clamp_min)
        self.seed = int(seed)

    def screen(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        num_samples: List[float],
        global_model: nn.Module = None,
        context: Dict[str, Any] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        del num_samples
        context = context or {}

        num_clients = len(client_deltas)
        if num_clients < 2 or global_model is None:
            return [1.0] * num_clients, context

        conv_bn_pairs = self._collect_conv_bn_pairs(global_model)
        if not conv_bn_pairs:
            return [1.0] * num_clients, context

        global_state = global_model.state_dict()
        cbe_vectors: List[np.ndarray] = []
        for delta in client_deltas:
            local_state = self._reconstruct_local_state(global_state, delta, conv_bn_pairs)
            cbe = self._extract_cbe(local_state, conv_bn_pairs)
            if cbe.size == 0:
                return [1.0] * num_clients, context
            cbe_vectors.append(cbe)

        distance_matrix = self._compute_distance_matrix(cbe_vectors)
        if num_clients < self.n_clusters or np.allclose(distance_matrix, distance_matrix[0]):
            return [1.0] * num_clients, context

        labels = self._cluster_clients(distance_matrix)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2:
            return [1.0] * num_clients, context

        centers = self._compute_cluster_centers(cbe_vectors, labels, unique_labels)
        center_distance = wasserstein_distance(centers[unique_labels[0]], centers[unique_labels[1]])
        if center_distance <= self.cluster_distance_threshold:
            return [1.0] * num_clients, context

        anomaly_label = max(
            unique_labels,
            key=lambda label: float(np.linalg.norm(centers[label], ord=1)),
        )
        screen_scores = [0.0 if int(label) == int(anomaly_label) else 1.0 for label in labels]
        return screen_scores, context

    def _collect_conv_bn_pairs(self, model: nn.Module) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        pending_conv: Tuple[str, nn.Conv2d] | None = None

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

    def _reconstruct_local_state(
        self,
        global_state: Dict[str, torch.Tensor],
        delta: Dict[str, torch.Tensor],
        conv_bn_pairs: Sequence[Tuple[str, str]],
    ) -> Dict[str, torch.Tensor]:
        required_keys = set()
        for conv_name, bn_name in conv_bn_pairs:
            required_keys.add(f"{conv_name}.weight")
            required_keys.add(f"{bn_name}.weight")
            required_keys.add(f"{bn_name}.running_var")

        local_state: Dict[str, torch.Tensor] = {}
        for key in required_keys:
            if key not in global_state:
                continue
            base_value = global_state[key]
            local_value = base_value.detach().clone()
            if key in delta:
                delta_value = delta[key].to(device=base_value.device)
                if torch.is_floating_point(local_value):
                    local_value = local_value + delta_value.to(dtype=local_value.dtype)
                else:
                    local_value = local_value + torch.round(delta_value).to(dtype=local_value.dtype)
            local_state[key] = local_value
        return local_state

    def _bn_channel_scale(
        self,
        bn_weight: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        """
        简化版: 每通道缩放为 |gamma| / std, std = sqrt(running_var + eps),
        与 Conv 权重相乘后做谱范数, 对应常见 MARS 实现.
        """
        sanitized_var = torch.nan_to_num(running_var.detach().float(), nan=0.0).clamp(min=self.var_clamp_min)
        sanitized_weight = torch.nan_to_num(bn_weight.detach().float(), nan=0.0)
        std = torch.sqrt(sanitized_var + self.bn_eps)
        return torch.abs(sanitized_weight / std)

    def _extract_cbe(
        self,
        local_state: Dict[str, torch.Tensor],
        conv_bn_pairs: Sequence[Tuple[str, str]],
    ) -> np.ndarray:
        per_layer_features: List[torch.Tensor] = []

        for conv_name, bn_name in conv_bn_pairs:
            conv_weight = local_state.get(f"{conv_name}.weight")
            bn_weight = local_state.get(f"{bn_name}.weight")
            running_var = local_state.get(f"{bn_name}.running_var")
            if conv_weight is None or running_var is None:
                continue

            if bn_weight is None:
                bn_weight = torch.ones(
                    conv_weight.shape[0],
                    device=conv_weight.device,
                    dtype=conv_weight.dtype,
                )

            scale = self._bn_channel_scale(bn_weight, running_var)

            channel_lips = []
            conv_weight = conv_weight.detach().float()
            for channel_idx in range(conv_weight.shape[0]):
                channel_matrix = conv_weight[channel_idx].reshape(conv_weight.shape[1], -1)
                singular_values = torch.linalg.svdvals(channel_matrix * scale[channel_idx])
                channel_lips.append(singular_values.max())

            if not channel_lips:
                continue

            channel_lips_tensor = torch.stack(channel_lips)
            per_layer_features.append(self._select_top_ratio(channel_lips_tensor))

        if not per_layer_features:
            return np.zeros(0, dtype=np.float32)

        return torch.cat(per_layer_features).detach().cpu().numpy()

    def _select_top_ratio(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return values

        num_keep = max(1, int(math.ceil(values.numel() * self.top_ratio)))
        quantile = torch.quantile(values, 1.0 - self.top_ratio)
        selected = values[values >= quantile]

        if selected.numel() != num_keep:
            selected = torch.topk(values, k=num_keep, largest=True).values

        return selected

    def _compute_distance_matrix(self, cbe_vectors: Sequence[np.ndarray]) -> np.ndarray:
        num_clients = len(cbe_vectors)
        distance_matrix = np.zeros((num_clients, num_clients), dtype=np.float64)
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                distance = wasserstein_distance(cbe_vectors[i], cbe_vectors[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        return distance_matrix

    def _cluster_clients(self, distance_matrix: np.ndarray) -> np.ndarray:
        model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.seed,
            n_init=10,
        )
        return model.fit_predict(distance_matrix)

    def _compute_cluster_centers(
        self,
        cbe_vectors: Sequence[np.ndarray],
        labels: np.ndarray,
        unique_labels: np.ndarray,
    ) -> Dict[int, np.ndarray]:
        centers: Dict[int, np.ndarray] = {}
        for label in unique_labels:
            members = [cbe_vectors[idx] for idx, item in enumerate(labels) if int(item) == int(label)]
            centers[int(label)] = np.mean(np.stack(members, axis=0), axis=0)
        return centers


@SCREENER_REGISTRY.register("mars_gamma_bn")
class MARSGammaBnStatScreener(MARSScreener):
    """
    MARS 变体: 每通道缩放为 |gamma| * std, 其中 std = sqrt(running_var + eps).
    使用 BN 可学习 gamma 与运行统计量(标准差)的乘性组合; 与注册名 "mars" 的 |gamma|/std 除法形式不同.
    """

    def _bn_channel_scale(
        self,
        bn_weight: torch.Tensor,
        running_var: torch.Tensor,
    ) -> torch.Tensor:
        sanitized_var = torch.nan_to_num(running_var.detach().float(), nan=0.0).clamp(min=self.var_clamp_min)
        sanitized_weight = torch.nan_to_num(bn_weight.detach().float(), nan=0.0)
        std = torch.sqrt(sanitized_var + self.bn_eps)
        return torch.abs(sanitized_weight) * std
