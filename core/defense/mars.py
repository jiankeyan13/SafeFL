"""MARS channel-wise Lipschitz CBE + Wasserstein K-Means defense (single-file v2)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wasserstein_distance
from sklearn.cluster import KMeans

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext, split_model_state
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("mars")
class MARSDefense(BaseDefense):
    """CBE spectral feature extraction over Conv2d+BN2d pairs + Wasserstein K-Means screening."""

    def __init__(
        self,
        top_ratio: float = 0.05,
        cluster_distance_threshold: float = 0.03,
        n_clusters: int = 2,
        bn_eps: float = 1e-5,
        var_clamp_min: float = 0.0,
        seed: int = 42,
        multiply_std: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not 0.0 < float(top_ratio) <= 1.0:
            raise ValueError("top_ratio must be in (0, 1]")
        if int(n_clusters) < 2:
            raise ValueError("n_clusters must be >= 2")
        self.top_ratio = float(top_ratio)
        self.cluster_distance_threshold = float(cluster_distance_threshold)
        self.n_clusters = int(n_clusters)
        self.bn_eps = float(bn_eps)
        self.var_clamp_min = float(var_clamp_min)
        self.seed = int(seed)
        self.multiply_std = bool(multiply_std)

    def _channel_scale(self, bn_weight: torch.Tensor, running_var: torch.Tensor) -> torch.Tensor:
        var = torch.nan_to_num(running_var.detach().float(), nan=0.0).clamp(min=self.var_clamp_min)
        gamma = torch.nan_to_num(bn_weight.detach().float(), nan=0.0).abs()
        std = torch.sqrt(var + self.bn_eps)
        return (gamma * std) if self.multiply_std else (gamma / std)

    def _select_top(self, values: torch.Tensor) -> torch.Tensor:
        k = max(1, int(math.ceil(values.numel() * self.top_ratio)))
        q = torch.quantile(values, 1.0 - self.top_ratio)
        sel = values[values >= q]
        if sel.numel() != k:
            sel = torch.topk(values, k=k, largest=True).values
        return sel

    def _extract_cbe(
        self,
        update: ClientUpdate,
        g_params: Dict[str, torch.Tensor],
        g_buffers: Dict[str, torch.Tensor],
        pairs: Sequence[Tuple[str, str]],
    ) -> np.ndarray:
        feats: List[torch.Tensor] = []
        for conv_name, bn_name in pairs:
            cw_key = f"{conv_name}.weight"
            bw_key = f"{bn_name}.weight"
            rv_key = f"{bn_name}.running_var"
            if cw_key not in g_params or rv_key not in g_buffers:
                continue

            base_cw = g_params[cw_key]
            conv_w = (base_cw + update.param_delta.get(cw_key, torch.zeros_like(base_cw)).to(base_cw.device)).detach().float()

            if bw_key in g_params:
                base_bw = g_params[bw_key]
                bn_w = (base_bw + update.param_delta.get(bw_key, torch.zeros_like(base_bw)).to(base_bw.device)).detach().float()
            else:
                bn_w = torch.ones(conv_w.shape[0], device=conv_w.device, dtype=torch.float32)

            run_var = update.buffers.get(rv_key, g_buffers[rv_key]).to(conv_w.device)
            scale = self._channel_scale(bn_w, run_var)

            # Vectorized batched SVD across all output channels at once!
            out_c, in_c = conv_w.shape[0], conv_w.shape[1]
            chan_mats = conv_w.reshape(out_c, in_c, -1) * scale.view(out_c, 1, 1)
            max_sv = torch.linalg.svdvals(chan_mats)[:, 0]
            feats.append(self._select_top(max_sv))

        if not feats:
            return np.zeros(0, dtype=np.float32)
        return torch.cat(feats).detach().cpu().numpy()

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        n = len(updates)
        pairs = F.find_conv_bn_pairs(global_model)
        if n < max(2, self.n_clusters) or not pairs:
            param_delta, buffers = F.aggregate_updates(updates, device=ctx.device)
            return DefenseOutput(param_delta=param_delta, buffers=buffers)

        g_params, g_buffers = split_model_state(global_model, trainable_only=False)
        cbe_vecs = [self._extract_cbe(u, g_params, g_buffers, pairs) for u in updates]
        if any(v.size == 0 for v in cbe_vecs):
            param_delta, buffers = F.aggregate_updates(updates, device=ctx.device)
            return DefenseOutput(param_delta=param_delta, buffers=buffers)

        dist_mat = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = wasserstein_distance(cbe_vecs[i], cbe_vecs[j])
                dist_mat[i, j] = dist_mat[j, i] = d

        if np.allclose(dist_mat, dist_mat[0]):
            kept = updates
        else:
            labels = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10).fit_predict(dist_mat)
            uniq = np.unique(labels)
            if uniq.size < 2:
                kept = updates
            else:
                centers = {
                    int(lab): np.mean(
                        np.stack([cbe_vecs[i] for i, l in enumerate(labels) if int(l) == int(lab)], axis=0),
                        axis=0,
                    )
                    for lab in uniq
                }
                c_dist = wasserstein_distance(centers[int(uniq[0])], centers[int(uniq[1])])
                if c_dist <= self.cluster_distance_threshold:
                    kept = updates
                else:
                    anom = max(uniq, key=lambda lab: float(np.linalg.norm(centers[int(lab)], ord=1)))
                    kept = [u for u, lab in zip(updates, labels) if int(lab) != int(anom)] or updates

        param_delta, buffers = F.aggregate_updates(kept, device=ctx.device)
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(kept))},
        )


@DEFENSE_REGISTRY.register("mars_normal")
class MARSNormalDefense(MARSDefense):
    """MARS variant scaling channels by ``|gamma| * std`` instead of ``|gamma| / std``."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("multiply_std", True)
        super().__init__(**kwargs)
