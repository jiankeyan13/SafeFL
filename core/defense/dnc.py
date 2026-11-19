"""DnC (Divide-and-Conquer) spectral defense (Shejwalkar & Houmansadr, NDSS 2021, single-file v2)."""
from __future__ import annotations

from typing import Any, List, Set

import numpy as np
import torch.nn as nn
from sklearn.utils.extmath import randomized_svd

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("dnc")
class DnCDefense(BaseDefense):
    """Random-subspace singular-vector outlier removal + sample-weighted averaging."""

    def __init__(
        self,
        sub_dim: int = 10000,
        num_iters: int = 3,
        num_outliers: int = 1,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sub_dim = int(sub_dim)
        self.num_iters = int(num_iters)
        self.num_outliers = int(num_outliers)
        self.seed = int(seed)

    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        del global_model
        n = len(updates)
        if n < 2 or n <= self.num_outliers:
            param_delta, buffers = F.aggregate_updates(updates, device=ctx.device)
            return DefenseOutput(param_delta=param_delta, buffers=buffers)

        mat_t, _ = F.stack_flat_deltas(updates)
        mat = mat_t.cpu().numpy().astype(np.float64, copy=False)
        centered = mat - mat.mean(axis=0, keepdims=True)
        dim = centered.shape[1]

        iter_sets: List[Set[int]] = []
        for it in range(self.num_iters):
            if dim > self.sub_dim:
                rng = np.random.RandomState(self.seed + ctx.round_idx * self.num_iters + it)
                cols = rng.choice(dim, size=self.sub_dim, replace=False)
                sub = centered[:, cols]
            else:
                sub = centered

            if sub.shape[1] == 0:
                scores = np.zeros(n, dtype=np.float64)
            else:
                _, _, vh = randomized_svd(sub, n_components=1, random_state=self.seed)
                scores = np.abs(sub @ vh[0])

            outliers = set(int(i) for i in np.argsort(scores)[-self.num_outliers :])
            iter_sets.append(set(range(n)) - outliers)

        benign_set = set.intersection(*iter_sets) if iter_sets else set(range(n))
        if not benign_set:
            benign_set = set(range(n))

        kept = [updates[i] for i in sorted(benign_set)]
        param_delta, buffers = F.aggregate_updates(kept, device=ctx.device)
        return DefenseOutput(
            param_delta=param_delta,
            buffers=buffers,
            metrics={"defense/kept_clients": float(len(kept))},
        )
