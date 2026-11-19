"""Unified registry for datasets and partitioners in the v2 data layer."""
from __future__ import annotations

from typing import Any, Callable, Dict, Generic, List, TypeVar

T = TypeVar("T")


class DataRegistry(Generic[T]):
    """Case-insensitive registry mapping canonical names to builders or classes."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: Dict[str, Callable[..., T]] = {}

    @staticmethod
    def _norm(name: str) -> str:
        return name.strip().lower()

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        key = self._norm(name)

        def decorator(fn_or_cls: Callable[..., T]) -> Callable[..., T]:
            if key in self._entries:
                raise ValueError(f"{self._kind} '{key}' is already registered.")
            self._entries[key] = fn_or_cls
            return fn_or_cls

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        key = self._norm(name)
        if key not in self._entries:
            raise KeyError(
                f"Unknown {self._kind.lower()} '{name}'. "
                f"Available: {self.available()}"
            )
        return self._entries[key]

    def build(self, name: str, *args: Any, **kwargs: Any) -> T:
        return self.get(name)(*args, **kwargs)

    def available(self) -> List[str]:
        return sorted(self._entries.keys())

    def __contains__(self, name: str) -> bool:
        return self._norm(name) in self._entries


DATASET_REGISTRY: DataRegistry[Any] = DataRegistry("Dataset")
PARTITIONER_REGISTRY: DataRegistry[Any] = DataRegistry("Partitioner")
