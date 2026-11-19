from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import hdbscan
import numpy as np
import torch
from scipy.fft import dctn

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("freqfed")
class FreqFedScreener(BaseScreener):
    """
    FreqFed screener.

    Pipeline:
    1) Convert each client delta to a low-frequency feature vector via 2D-DCT.
    2) Build cosine distance matrix across clients.
    3) Run HDBSCAN and keep the largest cluster as benign clients.
    """

    def __init__(
        self,
        min_cluster_size: Optional[int] = None,
        min_samples: int = 1,
        allow_single_cluster: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_cluster_size = min_cluster_size
        self.min_samples = max(1, int(min_samples))
        self.allow_single_cluster = bool(allow_single_cluster)

    @staticmethod
    def _tensor_to_2d(array: np.ndarray) -> np.ndarray:
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim == 2:
            return array
        return array.reshape(array.shape[0], -1)

    @staticmethod
    def _extract_low_frequency(dct_matrix: np.ndarray) -> np.ndarray:
        h, w = dct_matrix.shape
        bound = min(h, w)
        threshold = bound // 2

        row_ids = np.arange(h).reshape(-1, 1)
        col_ids = np.arange(w).reshape(1, -1)
        mask = (row_ids + col_ids) <= threshold
        low_freq = dct_matrix[mask]

        if low_freq.size == 0:
            return dct_matrix.reshape(-1)[:1]
        return low_freq.reshape(-1)

    def _build_client_feature(
        self,
        delta_dict: Dict[str, torch.Tensor],
        learnable_params: Optional[Set[str]],
    ) -> np.ndarray:
        layer_features: List[np.ndarray] = []

        for name, tensor in delta_dict.items():
            if learnable_params is not None and name not in learnable_params:
                continue

            matrix_2d = self._tensor_to_2d(tensor.detach().float().cpu().numpy())
            dct_matrix = dctn(matrix_2d, type=2, norm="ortho")
            low_freq = self._extract_low_frequency(dct_matrix)
            layer_features.append(low_freq.astype(np.float32, copy=False))

        if not layer_features:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(layer_features, axis=0)

    @staticmethod
    def _pad_features(feature_list: List[np.ndarray]) -> np.ndarray:
        max_dim = max(feature.shape[0] for feature in feature_list)
        padded: List[np.ndarray] = []
        for feature in feature_list:
            if feature.shape[0] == max_dim:
                padded.append(feature)
                continue
            pad_width = max_dim - feature.shape[0]
            padded.append(np.pad(feature, (0, pad_width), mode="constant"))
        return np.stack(padded, axis=0)

    def screen(
        self,
        client_deltas: List[Dict[str, torch.Tensor]],
        num_samples: List[float],
        global_model: torch.nn.Module = None,
        context: Dict[str, Any] = None,
    ) -> Tuple[List[float], Dict[str, Any]]:
        context = context or {}
        num_clients = len(client_deltas)

        if num_clients < 2:
            scores = [1.0] * num_clients
            context["freqfed_labels"] = [0] * num_clients
            context["freqfed_benign_indices"] = list(range(num_clients))
            context["freqfed_feature_dim"] = 0
            return scores, context

        if global_model is not None:
            learnable_params = {
                name for name, param in global_model.named_parameters() if param.requires_grad
            }
        else:
            learnable_params = set(client_deltas[0].keys())

        feature_list = [
            self._build_client_feature(delta_dict, learnable_params)
            for delta_dict in client_deltas
        ]
        feature_matrix = self._pad_features(feature_list)

        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        normalized = feature_matrix / np.clip(norms, a_min=1e-12, a_max=None)
        cosine_sim = np.clip(normalized @ normalized.T, -1.0, 1.0)
        cosine_dist = np.clip(1.0 - cosine_sim, 0.0, 2.0).astype(np.float64)

        min_cluster_size = self.min_cluster_size
        if min_cluster_size is None:
            min_cluster_size = num_clients // 2 + 1
        min_cluster_size = max(2, min(int(min_cluster_size), num_clients))
        min_samples = min(self.min_samples, num_clients)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            allow_single_cluster=self.allow_single_cluster,
            metric="precomputed",
        ).fit(cosine_dist)

        labels = clusterer.labels_
        valid_mask = labels >= 0
        if np.any(valid_mask):
            cluster_ids, counts = np.unique(labels[valid_mask], return_counts=True)
            max_cluster = cluster_ids[np.argmax(counts)]
            benign_indices = set(np.where(labels == max_cluster)[0].tolist())
        else:
            benign_indices = set(range(num_clients))

        screen_scores = [1.0 if idx in benign_indices else 0.0 for idx in range(num_clients)]
        if sum(screen_scores) == 0:
            screen_scores = [1.0] * num_clients
            benign_indices = set(range(num_clients))

        context["freqfed_labels"] = labels.tolist()
        context["freqfed_benign_indices"] = sorted(benign_indices)
        context["freqfed_feature_dim"] = int(feature_matrix.shape[1])
        context["freqfed_min_cluster_size"] = int(min_cluster_size)
        context["freqfed_min_samples"] = int(min_samples)

        return screen_scores, context
