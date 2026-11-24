"""Pure index-array data partitioners for v2."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from data.registry import PARTITIONER_REGISTRY
from data.spec import PartitionMap

if TYPE_CHECKING:
    from core.utils.config import PartitionerConfig


class Partitioner(ABC):
    """Base class partitioning 1D label arrays into per-client int32 index arrays."""

    @abstractmethod
    def partition(self, targets: np.ndarray, num_clients: int) -> Dict[int, np.ndarray]:
        """Return ``{client_id: int32_indices}`` for ``client_id in 0..num_clients-1``."""


@PARTITIONER_REGISTRY.register("iid")
class IIDPartitioner(Partitioner):
    """Uniform random IID split across clients."""

    def __init__(self, seed: int = 42, **_: Any) -> None:
        self.seed = int(seed)

    def partition(self, targets: np.ndarray, num_clients: int) -> Dict[int, np.ndarray]:
        n_samples = int(len(targets))
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if n_samples < num_clients:
            raise ValueError(f"n_samples ({n_samples}) < num_clients ({num_clients})")

        indices = np.arange(n_samples, dtype=np.int32)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)

        splits = np.array_split(indices, num_clients)
        return {i: np.ascontiguousarray(split, dtype=np.int32) for i, split in enumerate(splits)}


@PARTITIONER_REGISTRY.register("dirichlet")
class DirichletPartitioner(Partitioner):
    """Balanced-constrained Dirichlet(alpha) Non-IID split across clients."""

    def __init__(self, alpha: float = 1.0, seed: int = 42, max_retries: int = 100, **_: Any) -> None:
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.max_retries = int(max_retries)

    def partition(self, targets: np.ndarray, num_clients: int) -> Dict[int, np.ndarray]:
        labels = np.asarray(targets)
        n_samples = int(labels.size)
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)

        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if n_samples < num_clients:
            raise ValueError(f"n_samples ({n_samples}) < num_clients ({num_clients})")

        rng = np.random.default_rng(self.seed)
        min_threshold = min(num_classes, 1)
        client_lists: List[List[int]] = []
        min_size = 0

        for _ in range(self.max_retries):
            client_lists = [[] for _ in range(num_clients)]
            for label_val in unique_labels:
                idk = np.where(labels == label_val)[0]
                rng.shuffle(idk)
                proportion = rng.dirichlet(np.repeat(self.alpha, num_clients))
                proportion = np.array(
                    [
                        p * (len(idx_j) < n_samples / num_clients)
                        for idx_j, p in zip(client_lists, proportion)
                    ],
                    dtype=np.float64,
                )
                proportion = proportion / proportion.sum()
                split_points = (np.cumsum(proportion) * len(idk)).astype(int)[:-1]
                for cid, batch in enumerate(np.split(idk, split_points)):
                    client_lists[cid].extend(batch.tolist())

            min_size = min(len(c_idx) for c_idx in client_lists)
            if min_size >= min_threshold:
                break

        if min_size < min_threshold:
            raise RuntimeError(
                f"DirichletPartitioner failed to converge after {self.max_retries} retries "
                f"(min_samples_per_client={min_size}, threshold={min_threshold}). "
                "Try increasing alpha or reducing num_clients."
            )

        out: Dict[int, np.ndarray] = {}
        for cid, c_list in enumerate(client_lists):
            arr = np.asarray(c_list, dtype=np.int32)
            rng.shuffle(arr)
            out[cid] = arr
        return out


@PARTITIONER_REGISTRY.register("balanced_dirichlet")
class BalancedDirichletPartitioner(Partitioner):
    """Strictly sample-balanced Dirichlet(alpha) Non-IID split across clients.

    Every client receives exactly ``n_samples // num_clients`` samples (or ``+1`` for
    the first ``n_samples % num_clients`` clients when not evenly divisible), while
    every sample is assigned to exactly one client without replacement.
    """

    def __init__(self, alpha: float = 0.5, seed: int = 42, ipf_iters: int = 200, **_: Any) -> None:
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.ipf_iters = int(ipf_iters)

    def partition(self, targets: np.ndarray, num_clients: int) -> Dict[int, np.ndarray]:
        labels = np.asarray(targets)
        n_samples = int(labels.size)
        unique_labels = np.sort(np.unique(labels))
        num_classes = len(unique_labels)

        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if n_samples < num_clients:
            raise ValueError(f"n_samples ({n_samples}) < num_clients ({num_clients})")

        rng = np.random.default_rng(self.seed)

        # Build shuffled per-class index pools and exact target quotas.
        class_pools: List[np.ndarray] = []
        class_counts = np.empty(num_classes, dtype=np.int64)
        for c, label_val in enumerate(unique_labels):
            idx = np.where(labels == label_val)[0].astype(np.int32)
            rng.shuffle(idx)
            class_pools.append(idx)
            class_counts[c] = len(idx)

        base_quota, rem_quota = divmod(n_samples, num_clients)
        quotas = np.full(num_clients, base_quota, dtype=np.int64)
        if rem_quota > 0:
            extra_clients = rng.choice(num_clients, size=rem_quota, replace=False)
            quotas[extra_clients] += 1

        if num_classes == 1:
            counts = quotas[:, None]
        else:
            # Draw per-client Dirichlet class preferences and balance row/column marginals
            # via Iterative Proportional Fitting (Sinkhorn scaling) without client-order bias.
            mat = rng.dirichlet(np.repeat(self.alpha, num_classes), size=num_clients).astype(np.float64)
            mat = np.maximum(mat, 1e-12)
            mat *= quotas[:, None]

            target_rows = quotas.astype(np.float64)
            target_cols = class_counts.astype(np.float64)
            for _ in range(self.ipf_iters):
                col_sums = mat.sum(axis=0)
                mat *= target_cols[None, :] / np.maximum(col_sums[None, :], 1e-12)
                row_sums = mat.sum(axis=1)
                mat *= target_rows[:, None] / np.maximum(row_sums[:, None], 1e-12)
                if (
                    np.max(np.abs(mat.sum(axis=0) - target_cols)) < 1e-6
                    and np.max(np.abs(mat.sum(axis=1) - target_rows)) < 1e-6
                ):
                    break

            counts = np.floor(mat).astype(np.int64)
            row_def = quotas - counts.sum(axis=1)
            col_def = class_counts - counts.sum(axis=0)

            # Fix any rare overshoot caused by floating-point rounding.
            while np.any(row_def < 0):
                i = int(np.argmin(row_def))
                valid_cols = np.where(counts[i] > 0)[0]
                c = int(valid_cols[np.argmin(mat[i, valid_cols] - counts[i, valid_cols])])
                counts[i, c] -= 1
                row_def[i] += 1
                col_def[c] += 1
            while np.any(col_def < 0):
                c = int(np.argmin(col_def))
                valid_rows = np.where(counts[:, c] > 0)[0]
                i = int(valid_rows[np.argmin(mat[valid_rows, c] - counts[valid_rows, c])])
                counts[i, c] -= 1
                row_def[i] += 1
                col_def[c] += 1

            # Distribute remaining units by largest fractional part, then fallback to largest weight.
            frac = mat - counts
            order = np.argsort(frac.ravel())[::-1]
            for flat_idx in order:
                if row_def.sum() == 0:
                    break
                i, c = divmod(int(flat_idx), num_classes)
                if row_def[i] > 0 and col_def[c] > 0:
                    counts[i, c] += 1
                    row_def[i] -= 1
                    col_def[c] -= 1

            if row_def.sum() > 0:
                rem_rows = np.where(row_def > 0)[0]
                for i in rem_rows:
                    while row_def[i] > 0:
                        avail_cols = np.where(col_def > 0)[0]
                        c = int(avail_cols[np.argmax(mat[i, avail_cols])])
                        take = int(min(row_def[i], col_def[c]))
                        counts[i, c] += take
                        row_def[i] -= take
                        col_def[c] -= take

        # Slice per-class pools according to the exact integer count matrix.
        cursors = np.zeros(num_classes, dtype=np.int64)
        out: Dict[int, np.ndarray] = {}
        for cid in range(num_clients):
            parts: List[np.ndarray] = []
            for c in range(num_classes):
                cnt = int(counts[cid, c])
                if cnt > 0:
                    start = int(cursors[c])
                    end = start + cnt
                    parts.append(class_pools[c][start:end])
                    cursors[c] = end
            arr = np.concatenate(parts) if parts else np.empty(0, dtype=np.int32)
            rng.shuffle(arr)
            out[cid] = np.ascontiguousarray(arr, dtype=np.int32)
        return out


@PARTITIONER_REGISTRY.register("q_non_iid")
class QNonIIDPartitioner(Partitioner):
    """q-based Non-IID split (Fang et al., 2020; Cao et al., 2021)."""

    def __init__(self, q: float = 0.5, seed: int = 42, **_: Any) -> None:
        self.q = float(q)
        self.seed = int(seed)

    def _assign_class_to_groups(
        self,
        sample_indices: np.ndarray,
        label_group: int,
        num_classes: int,
        rng: np.random.Generator,
    ) -> List[np.ndarray]:
        num_idx_k = int(len(sample_indices))
        shuffled = sample_indices.copy()
        rng.shuffle(shuffled)

        main_count = int(num_idx_k * self.q)
        other_count = int(num_idx_k * (1.0 - self.q) / (num_classes - 1))
        group_counts = np.full(num_classes, other_count, dtype=int)
        group_counts[label_group] = main_count

        remainder = num_idx_k - int(group_counts.sum())
        if remainder > 0:
            extra_groups = rng.permutation(num_classes)
            for i in range(remainder):
                group_counts[extra_groups[i % num_classes]] += 1

        splits: List[np.ndarray] = []
        cursor = 0
        for count in group_counts:
            end = cursor + int(count)
            splits.append(shuffled[cursor:end])
            cursor = end
        return splits

    def partition(self, targets: np.ndarray, num_clients: int) -> Dict[int, np.ndarray]:
        labels = np.asarray(targets)
        unique_labels = np.sort(np.unique(labels))
        num_classes = len(unique_labels)

        if num_clients <= 0:
            raise ValueError("num_clients must be positive")
        if not 0.0 < self.q <= 1.0:
            raise ValueError("q must be in (0, 1]")
        if num_classes < 2:
            raise ValueError("q_non_iid requires at least 2 classes")
        if num_clients % num_classes != 0:
            raise ValueError(
                f"num_clients ({num_clients}) must be divisible by num_classes ({num_classes})"
            )

        clients_per_group = num_clients // num_classes
        rng = np.random.default_rng(self.seed)
        group_chunks: List[List[np.ndarray]] = [[] for _ in range(num_classes)]

        for label_group, label_val in enumerate(unique_labels):
            sample_indices = np.where(labels == label_val)[0].astype(np.int32)
            class_splits = self._assign_class_to_groups(
                sample_indices, label_group, num_classes, rng
            )
            for group_id, chunk in enumerate(class_splits):
                group_chunks[group_id].append(chunk)

        out: Dict[int, np.ndarray] = {}
        for group_id, chunks in enumerate(group_chunks):
            merged = np.concatenate(chunks)
            rng.shuffle(merged)
            start_client = group_id * clients_per_group
            for offset, split_arr in enumerate(np.array_split(merged, clients_per_group)):
                arr = np.ascontiguousarray(split_arr, dtype=np.int32)
                rng.shuffle(arr)
                out[start_client + offset] = arr

        return out


def build_partitioner(config: "PartitionerConfig", seed: int) -> Partitioner:
    """Build a partitioner from ``PartitionerConfig`` and global ``seed``."""
    params = dict(config.params)
    params.setdefault("seed", int(seed))
    return PARTITIONER_REGISTRY.build(config.name, **params)


def sample_proxy_indices(
    targets: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample class-balanced server proxy indices and return ``(proxy, remaining)``."""
    labels = np.asarray(targets)
    n_total = int(labels.size)
    if per_class <= 0 or n_total == 0:
        return np.empty(0, dtype=np.int32), np.arange(n_total, dtype=np.int32)

    proxy_parts: List[np.ndarray] = []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        chosen = rng.choice(cls_idx, size=min(int(per_class), len(cls_idx)), replace=False)
        proxy_parts.append(chosen.astype(np.int32))

    proxy_indices = np.concatenate(proxy_parts)
    keep_mask = np.ones(n_total, dtype=bool)
    keep_mask[proxy_indices] = False
    remaining_indices = np.nonzero(keep_mask)[0].astype(np.int32)
    return proxy_indices, remaining_indices


def create_partition_map(
    targets: np.ndarray,
    partitioner: Partitioner,
    num_clients: int,
    *,
    seed: int = 42,
    enable_proxy: bool = False,
    proxy_per_class: int = 10,
    exclude_indices: Optional[Sequence[int] | np.ndarray] = None,
) -> PartitionMap:
    """Carve out optional exclusions/proxy set and partition remaining samples across clients."""
    rng = np.random.default_rng(seed)
    n_total = int(len(targets))
    available_mask = np.ones(n_total, dtype=bool)
    if exclude_indices is not None:
        excl = np.asarray(list(exclude_indices), dtype=np.int64).reshape(-1)
        if excl.size > 0:
            if int(excl.min()) < 0 or int(excl.max()) >= n_total:
                raise ValueError(
                    f"exclude_indices out of range [0, {n_total}): "
                    f"min={int(excl.min())}, max={int(excl.max())}"
                )
            available_mask[excl] = False

    avail_idx = np.nonzero(available_mask)[0].astype(np.int32)
    if enable_proxy:
        rel_proxy, rel_rem = sample_proxy_indices(targets[avail_idx], proxy_per_class, rng)
        proxy_idx = avail_idx[rel_proxy].astype(np.int32, copy=False)
        remaining_idx = avail_idx[rel_rem].astype(np.int32, copy=False)
    else:
        proxy_idx = np.empty(0, dtype=np.int32)
        remaining_idx = avail_idx

    rel_map = partitioner.partition(targets[remaining_idx], num_clients)
    client_indices: Dict[int, np.ndarray] = {}
    for cid in range(num_clients):
        rel = rel_map.get(cid)
        if rel is None or len(rel) == 0:
            raise ValueError(f"Client {cid} received 0 samples after partitioning.")
        abs_idx = remaining_idx[rel].astype(np.int32, copy=True)
        rng.shuffle(abs_idx)
        client_indices[cid] = abs_idx

    return PartitionMap(
        num_clients=num_clients,
        client_indices=client_indices,
        proxy_indices=proxy_idx,
    )
