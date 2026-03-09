from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import randomized_svd

from core.utils.registry import SCREENER_REGISTRY
from .base_screener import BaseScreener


@SCREENER_REGISTRY.register("rflbat")
class RFLBATScreener(BaseScreener):
    """
    RFLBAT 筛选器。

    流程:
    1. 将高维 delta 展平并 PCA 到二维。
    2. 基于二维欧氏距离和做第一轮离群点剔除。
    3. 对候选集合做 KMeans 聚类。
    4. 用原始高维 delta 计算簇内平均余弦相似度, 选择得分最低的簇。
    5. 在该簇二维表示上做第二轮更严格的欧氏离群点剔除。
    """

    def __init__(self, epsilon1: float = 4.0, epsilon2: float = 2.0, num_clusters: int = 2,
                 max_kmeans_iters: int = 50, seed: int = 42, eps: float = 1e-12, **kwargs):
        super().__init__()
        self.epsilon1, self.epsilon2 = float(epsilon1), float(epsilon2)
        self.num_clusters, self.max_kmeans_iters = int(num_clusters), int(max_kmeans_iters)
        self.seed, self.eps = int(seed), float(eps)

    def screen(self, client_deltas: List[Dict[str, torch.Tensor]], num_samples: List[float],
               global_model: torch.nn.Module = None, context: Dict[str, Any] = None) -> Tuple[List[float], Dict[str, Any]]:
        context = context or {}
        num_clients = len(client_deltas)

        if num_clients < 2:
            context["benign_indices"] = set(range(num_clients))
            return [1.0] * num_clients, context

        learnable_keys = self._get_learnable_keys(global_model, client_deltas[0])
        high_dim = self._stack_vectors(client_deltas, learnable_keys)
        low_dim = self._project_to_2d(high_dim)

        candidate_indices, stage1_scores = self._euclidean_filter(low_dim, list(range(num_clients)), self.epsilon1)
        if not candidate_indices:
            candidate_indices = [int(np.argmin(stage1_scores))]

        chosen_cluster_indices: List[int]
        cluster_labels: Dict[int, int] = {}

        if len(candidate_indices) <= 1:
            chosen_cluster_indices = list(candidate_indices)
            if chosen_cluster_indices:
                cluster_labels[chosen_cluster_indices[0]] = 0
        else:
            candidate_low_dim = low_dim[candidate_indices]
            cluster_ids = self._kmeans(candidate_low_dim, min(self.num_clusters, len(candidate_indices)))
            cluster_labels = {original_idx: int(label) for original_idx, label in zip(candidate_indices, cluster_ids)}
            chosen_cluster_indices = self._select_low_similarity_cluster(candidate_indices, cluster_ids, high_dim)

        refined_indices, stage2_scores = self._euclidean_filter(low_dim, chosen_cluster_indices, self.epsilon2)
        if not refined_indices and chosen_cluster_indices:
            chosen_scores = [stage2_scores[idx] for idx in chosen_cluster_indices]
            refined_indices = [chosen_cluster_indices[int(np.argmin(chosen_scores))]]

        benign_indices = set(refined_indices)
        screen_scores = [1.0 if idx in benign_indices else 0.0 for idx in range(num_clients)]

        context.update({
            "benign_indices": benign_indices,
            "rflbat_candidate_indices": candidate_indices,
            "rflbat_cluster_labels": cluster_labels,
            "rflbat_selected_cluster": chosen_cluster_indices,
            "rflbat_stage1_scores": np.where(np.isfinite(stage1_scores), stage1_scores, -1.0).tolist(),
            "rflbat_stage2_scores": np.where(np.isfinite(stage2_scores), stage2_scores, -1.0).tolist()
        })
        return screen_scores, context

    def _get_learnable_keys(self, global_model: torch.nn.Module, first_delta: Dict[str, torch.Tensor]) -> Sequence[str]:
        if global_model is not None:
            return [name for name, param in global_model.named_parameters() if param.requires_grad]
        return list(first_delta.keys())

    def _stack_vectors(self, client_deltas: List[Dict[str, torch.Tensor]], learnable_keys: Sequence[str]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for delta in client_deltas:
            flat_parts = [delta[name].detach().float().cpu().reshape(-1).numpy() for name in learnable_keys if name in delta]
            vectors.append(np.concatenate(flat_parts, axis=0) if flat_parts else np.zeros(1, dtype=np.float32))
        return np.stack(vectors, axis=0)

    def _project_to_2d(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape[0] == 0: return np.zeros((0, 2), dtype=np.float64)
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        if centered.shape[1] == 0: return np.zeros((centered.shape[0], 2), dtype=np.float64)

        _, _, vh = randomized_svd(centered, n_components=2, random_state=self.seed)
        components = min(2, vh.shape[0])
        projected = centered @ vh[:components].T
        if components < 2: projected = np.pad(projected, ((0, 0), (0, 2 - components)))
        return projected.astype(np.float64, copy=False)

    def _euclidean_filter(self, points: np.ndarray, indices: Sequence[int], threshold: float) -> Tuple[List[int], np.ndarray]:
        scores = np.full(points.shape[0], np.inf, dtype=np.float64)
        index_list = list(indices)
        if not index_list: return [], scores
        if len(index_list) == 1:
            scores[index_list[0]] = 0.0
            return index_list, scores

        sub_points = points[index_list]
        dist_sum = cdist(sub_points, sub_points).sum(axis=1)
        normalized = dist_sum / max(float(np.median(dist_sum)), self.eps)

        kept = [idx for idx, score in zip(index_list, normalized) if score <= threshold]
        for idx, score in zip(index_list, normalized): scores[idx] = float(score)
        return kept, scores

    def _kmeans(self, points: np.ndarray, num_clusters: int) -> np.ndarray:
        if len(points) == 0: return np.zeros(0, dtype=np.int64)
        if num_clusters <= 1 or len(points) == 1: return np.zeros(len(points), dtype=np.int64)

        rng = np.random.default_rng(self.seed)
        centers = points[rng.choice(len(points), size=num_clusters, replace=False)].copy()
        labels = np.zeros(len(points), dtype=np.int64)

        for _ in range(self.max_kmeans_iters):
            new_labels = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
            if np.array_equal(new_labels, labels): break
            labels = new_labels
            for i in range(num_clusters):
                members = points[labels == i]
                centers[i] = members.mean(axis=0) if len(members) > 0 else points[rng.integers(0, len(points))]
        return labels

    def _select_low_similarity_cluster(self, candidate_indices: Sequence[int], cluster_ids: np.ndarray, high_dim: np.ndarray) -> List[int]:
        cluster_scores, cluster_members = {}, {}
        for original_idx, cluster_id in zip(candidate_indices, cluster_ids):
            cluster_members.setdefault(int(cluster_id), []).append(int(original_idx))

        for cluster_id, members in cluster_members.items():
            if len(members) <= 1:
                cluster_scores[cluster_id] = 1.0
                continue
            vectors = high_dim[members]
            normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), self.eps)
            cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
            per_client_means = (cosine.sum(axis=1) - 1.0) / max(len(members) - 1, 1)
            cluster_scores[cluster_id] = float(np.median(per_client_means))

        best_cluster = min(cluster_scores, key=lambda cid: (cluster_scores[cid], -len(cluster_members[cid])))
        return cluster_members[best_cluster]
