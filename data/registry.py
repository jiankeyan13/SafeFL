"""Type-safe dataset registry for v2."""
from __future__ import annotations

from typing import Callable, Dict, List

from data.spec import DatasetPairSpec

DatasetBuilder = Callable[[str], DatasetPairSpec]


class DatasetRegistry:
    """Registry mapping canonical dataset names (e.g. 'cifar10') to builders."""

    def __init__(self) -> None:
        self._builders: Dict[str, DatasetBuilder] = {}

    def register(self, name: str) -> Callable[[DatasetBuilder], DatasetBuilder]:
        key = name.strip().lower()

        def decorator(func: DatasetBuilder) -> DatasetBuilder:
            if key in self._builders:
                raise ValueError(f"Dataset '{key}' is already registered.")
            self._builders[key] = func
            return func

        return decorator

    def build(self, name: str, root: str) -> DatasetPairSpec:
        key = name.strip().lower()
        if key not in self._builders:
            raise KeyError(
                f"Dataset '{name}' is not registered. "
                f"Available: {self.available()}"
            )
        return self._builders[key](root)

    def available(self) -> List[str]:
        return sorted(self._builders.keys())

    def __contains__(self, name: str) -> bool:
        return name.strip().lower() in self._builders


DATASET_REGISTRY = DatasetRegistry()
