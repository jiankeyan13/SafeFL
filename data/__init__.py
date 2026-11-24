"""SafeFL v2 zero-copy vectorized data layer."""
from __future__ import annotations

from data.hub import DataHub
from data.loader import BatchPoisoner, DataView, FastBatchLoader, apply_batch_augment
from data.partition import (
    PARTITIONER_REGISTRY,
    BalancedDirichletPartitioner,
    DirichletPartitioner,
    IIDPartitioner,
    Partitioner,
    QNonIIDPartitioner,
    build_partitioner,
    create_partition_map,
)
from data.registry import DATASET_REGISTRY
from data.source import SharedDataSource, ensure_chw_mmap_cache
from data.spec import (
    AugmentSpec,
    DatasetPairSpec,
    FederatedDataSpec,
    PartitionMap,
    SplitSpec,
)

__all__ = [
    "AugmentSpec",
    "BalancedDirichletPartitioner",
    "BatchPoisoner",
    "DATASET_REGISTRY",
    "DataHub",
    "DataView",
    "DatasetPairSpec",
    "DirichletPartitioner",
    "FastBatchLoader",
    "FederatedDataSpec",
    "IIDPartitioner",
    "PARTITIONER_REGISTRY",
    "PartitionMap",
    "Partitioner",
    "QNonIIDPartitioner",
    "SharedDataSource",
    "SplitSpec",
    "apply_batch_augment",
    "build_partitioner",
    "create_partition_map",
    "ensure_chw_mmap_cache",
]
