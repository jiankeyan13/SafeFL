"""RFLBAT PCA + KMeans two-stage Euclidean/cosine defense (AAAI 2022, single-file v2)."""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch.nn as nn
from scipy.spatial.distance import cdist
from sklearn.utils.extmath import randomized_svd

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("rflbat")
class RFLBATDefense(BaseDefense):
    """2D PCA projection + Euclidean outlier filtering + low-cosine KMeans cluster selection."""

    def __init__(
        self,
        epsilon1: float = 4.0,
        epsilon2: float = 2.0,
        num_clusters: int = 2,
        max_kmeans_iters: int = 50,
        seed: int = 42,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.epsilon1 = float(epsilon1)
        self.epsilon2 = float(epsilon2)
        self.num_clusters = int(num_clusters)
        self.max_kmeans_iters = int(max_kmeans_iters)
        self.seed = int(seed)
        self.eps = float(eps)

    def _project_2d(self, matrix: np.ndarray) -> np.ndarray:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        if centered.shape[1] == 0:
            return np.zeros((centered.shape[0], 2), dtype=np.float64)
        n_comp = min(2, centered.shape[0], centered.shape[1])
        _, _, vh = randomized_svd(centered, n_components=n_comp, random_state=self.seed)
        proj = centered @ vh[:n_comp].T
        if n_comp < 2:
            proj = np.pad(proj, ((0, 0), (0, 2 - n_comp)))
        return proj.astype(np.float64, copy=False)

    def _euclidean_filter(
        self,
        points: np.ndarray,
        indices: Sequence[int],
        threshold: float,
    ) -> Tuple[List[int], np.ndarray]:
        scores = np.full(points.shape[0], np.inf, dtype=np.float64)
        idx_list = list(indices)
        if len(idx_list) <= 1:
            if idx_list:
                scores[idx_list[0]] = 0.0
            return idx_list, scores
        dist_sum = cdist(points[idx_list], points[idx_list]).sum(axis=1)
        normed = dist_sum / max(float(np.median(dist_sum)), self.eps)
        for idx, s in zip(idx_list, normed):
            scores[idx] = float(s)
        kept = [idx for idx, s in zip(idx_list, normed) if s <= threshold]
        return kept, scores

    def _kmeans(self, points: np.ndarray, k: int) -> np.ndarray:
        if k <= 1 or len(points) <= 1:
            return np.zeros(len(points), dtype=np.int64)
        rng = np.random.default_rng(self.seed)
        centers = points[rng.choice(len(points), size=k, replace=False)].copy()
        labels = np.zeros(len(points), dtype=np.int64)
        for _ in range(self.max_kmeans_iters):
            new_labels = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2).argmin(axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for cid in range(k):
                members = points[labels == cid]
                centers[cid] = (
                    members.mean(axis=0)
                    if len(members) > 0
                    else points[rng.integers(0, len(points))]
                )
        return labels

    def _select_low_similarity_cluster(
        self,
        cand_indices: Sequence[int],
        cluster_ids: np.ndarray,
        high_dim: np.ndarray,
    ) -> List[int]:
        members_map: Dict[int, List[int]] = {}
        scores_map: Dict[int, float] = {}
        for orig_idx, cid in zip(cand_indices, cluster_ids):
            members_map.setdefault(int(cid), []).append(int(orig_idx))

        for cid, members in members_map.items():
            if len(members) <= 1:
                scores_map[cid] = 1.0
                continue
            vecs = high_dim[members]
            normed = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), self.eps)
            cos_m = np.clip(normed @ normed.T, -1.0, 1.0)
            means = (cos_m.sum(axis=1) - 1.0) / max(len(members) - 1, 1)
            scores_map[cid] = float(np.median(means))

        best_cid = min(scores_map, key=lambda c: (scores_map[c], -len(members_map[c])))
        return members_map[best_cid]

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        n = len(updates)
        if n < 2:
            param_delta, buffers = F.aggregate_updates(updates, device=ctx.device)
            return DefenseOutput(param_delta=param_delta, buffers=buffers)

        mat_t, _ = F.stack_flat_deltas(updates)
        high_dim = mat_t.cpu().numpy().astype(np.float64, copy=False)
        low_dim = self._project_2d(high_dim)

        cand_idx, s1 = self._euclidean_filter(low_dim, list(range(n)), self.epsilon1)
        if not cand_idx:
            cand_idx = [int(np.argmin(s1))]

        if len(cand_idx) <= 1:
            cluster_idx = list(cand_idx)
        else:
            c_ids = self._kmeans(low_dim[cand_idx], min(self.num_clusters, len(cand_idx)))
            cluster_idx = self._select_low_similarity_cluster(cand_idx, c_ids, high_dim)

        refined_idx, s2 = self._euclidean_filter(low_dim, cluster_idx, self.epsilon2)
        if not refined_idx and cluster_idx:
            refined_idx = [cluster_idx[int(np.argmin([s2[i] for i in cluster_idx]))]]

        kept = [updates[i] for i in sorted(set(refined_idx))]
        param_delta, buffers = F.aggregate_updates(kept, device=ctx.device)
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(kept))},
        )
