"""FreqFed frequency-domain DCT + HDBSCAN defense (NDSS 2024, single-file v2)."""
from __future__ import annotations

from typing import Any, List, Optional

import hdbscan
import numpy as np
import torch.nn as nn
from scipy.fft import dctn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, StateDict
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("freqfed")
class FreqFedDefense(BaseDefense):
    """2D-DCT low-frequency feature extraction + cosine HDBSCAN clustering."""

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
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
        return arr.reshape(arr.shape[0], -1)

    @staticmethod
    def _low_freq(dct_mat: np.ndarray) -> np.ndarray:
        h, w = dct_mat.shape
        threshold = min(h, w) // 2
        r_ids = np.arange(h).reshape(-1, 1)
        c_ids = np.arange(w).reshape(1, -1)
        lf = dct_mat[(r_ids + c_ids) <= threshold]
        return lf.reshape(-1) if lf.size > 0 else dct_mat.reshape(-1)[:1]

    def _extract_feature(self, param_delta: StateDict, keys: List[str]) -> np.ndarray:
        parts: List[np.ndarray] = []
        for k in keys:
            arr = param_delta[k].detach().float().cpu().numpy()
            dct_mat = dctn(self._to_2d(arr), type=2, norm="ortho")
            parts.append(self._low_freq(dct_mat).astype(np.float32, copy=False))
        return np.concatenate(parts, axis=0) if parts else np.zeros(1, dtype=np.float32)

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

        keys = F.ordered_keys(updates[0].param_delta)
        feats = [self._extract_feature(u.param_delta, keys) for u in updates]
        max_d = max(f.shape[0] for f in feats)
        feat_mat = np.stack(
            [f if f.shape[0] == max_d else np.pad(f, (0, max_d - f.shape[0])) for f in feats],
            axis=0,
        )

        norms = np.linalg.norm(feat_mat, axis=1, keepdims=True)
        normed = feat_mat / np.clip(norms, a_min=1e-12, a_max=None)
        cos_dist = np.clip(1.0 - np.clip(normed @ normed.T, -1.0, 1.0), 0.0, 2.0).astype(np.float64)

        mcs = n // 2 + 1 if self.min_cluster_size is None else int(self.min_cluster_size)
        mcs = max(2, min(mcs, n))

        labels = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=min(self.min_samples, n),
            allow_single_cluster=self.allow_single_cluster,
            metric="precomputed",
        ).fit_predict(cos_dist)

        valid = labels >= 0
        if np.any(valid):
            c_ids, counts = np.unique(labels[valid], return_counts=True)
            best_c = c_ids[np.argmax(counts)]
            benign_idx = np.where(labels == best_c)[0].tolist()
        else:
            benign_idx = list(range(n))

        kept = [updates[i] for i in benign_idx] or updates
        param_delta, buffers = F.aggregate_updates(kept, device=ctx.device)
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(kept))},
        )
