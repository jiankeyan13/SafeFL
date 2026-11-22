"""Top-level DataHub assembling sources, partitions, and zero-copy client/server views."""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np
import torch

from data.loader import DataView
from data.partition import build_partitioner, create_partition_map
from data.registry import DATASET_REGISTRY
from data.source import SharedDataSource
from data.spec import FederatedDataSpec

if TYPE_CHECKING:
    from core.utils.config import DataConfig


class DataHub:
    """Unified federated data hub used identically on Driver and Ray Workers.

    - Driver builds once via ``DataHub.build(data_config, num_clients, seed)``
      and passes ``hub.spec`` (a lightweight ``FederatedDataSpec``, <200KB) to workers.
    - Each worker reconstructs ``DataHub.from_spec(spec, device=...)`` with zero
      image serialization overhead (sharing OS mmap page cache or preloading VRAM).
    """

    def __init__(
        self,
        spec: FederatedDataSpec,
        *,
        train_source: Optional[SharedDataSource] = None,
        test_source: Optional[SharedDataSource] = None,
    ) -> None:
        self.spec = spec
        self.train_source = train_source or SharedDataSource(spec.dataset.train)
        self.test_source = test_source or SharedDataSource(spec.dataset.test)

    @classmethod
    def build(
        cls,
        data_config: "DataConfig",
        num_clients: int,
        seed: int = 42,
        *,
        exclude_train_indices: Optional[Sequence[int] | np.ndarray] = None,
    ) -> "DataHub":
        """Build dataset caches (if needed), partition labels, and return ``DataHub``."""
        import data.datasets  # noqa: F401 - ensure built-in datasets are registered

        pair_spec = DATASET_REGISTRY.build(data_config.dataset, data_config.root)
        train_source = SharedDataSource(pair_spec.train)
        test_source = SharedDataSource(pair_spec.test)

        partitioner = build_partitioner(data_config.partitioner, seed=seed)
        partition_map = create_partition_map(
            targets=train_source.targets,
            partitioner=partitioner,
            num_clients=num_clients,
            seed=seed,
            enable_proxy=data_config.enable_proxy,
            proxy_per_class=data_config.proxy_per_class,
            exclude_indices=exclude_train_indices,
        )

        fed_spec = FederatedDataSpec(
            dataset=pair_spec,
            partition=partition_map,
            use_augmentation=data_config.use_augmentation,
            preload_gpu=data_config.preload_gpu,
        )
        return cls(fed_spec, train_source=train_source, test_source=test_source)

    @classmethod
    def from_spec(
        cls,
        spec: FederatedDataSpec,
        device: Optional[Union[str, torch.device]] = None,
    ) -> "DataHub":
        """Reconstruct ``DataHub`` in a worker process from ``FederatedDataSpec``."""
        hub = cls(spec)
        if spec.preload_gpu and device is not None:
            dev = torch.device(device)
            if dev.type == "cuda":
                hub.preload(dev)
        return hub

    @property
    def dataset_name(self) -> str:
        return self.spec.dataset.name

    @property
    def num_classes(self) -> int:
        return self.spec.dataset.num_classes

    @property
    def num_clients(self) -> int:
        return self.spec.partition.num_clients

    @property
    def client_ids(self) -> List[int]:
        return list(range(self.num_clients))

    def preload(self, device: Union[str, torch.device], *, include_test: bool = False) -> None:
        """Preload raw uint8 training (and optionally test) tensors onto ``device``."""
        self.train_source.preload(device)
        if include_test:
            self.test_source.preload(device)

    def evict_device(self) -> None:
        self.train_source.evict_device()
        self.test_source.evict_device()

    def client(self, client_id: int) -> DataView:
        """Return zero-copy clean local ``DataView`` for ``client_id``."""
        cid = int(client_id)
        if cid not in self.spec.partition.client_indices:
            raise KeyError(f"Unknown client_id={cid} (num_clients={self.num_clients})")
        return DataView(
            source=self.train_source,
            indices=self.spec.partition.client_indices[cid],
            client_id=cid,
            default_use_augmentation=self.spec.use_augmentation,
        )

    def test(self) -> DataView:
        """Return global clean test ``DataView``."""
        return DataView(
            source=self.test_source,
            indices=None,
            client_id=None,
            default_use_augmentation=False,
        )

    def proxy(self) -> Optional[DataView]:
        """Return server proxy ``DataView``, or ``None`` when proxy is disabled."""
        proxy_idx = self.spec.partition.proxy_indices
        if proxy_idx.size == 0:
            return None
        return DataView(
            source=self.train_source,
            indices=proxy_idx,
            client_id=None,
            default_use_augmentation=False,
        )
