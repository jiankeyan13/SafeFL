"""Generic component registry for SafeFL v2."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Type


class Registry:
    """Decorator-based class/callable registry."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}

    def register(self, name: Optional[str] = None):
        def _register(cls: Type[Any]) -> Type[Any]:
            key = name if name is not None else cls.__name__
            if key in self._module_dict:
                raise ValueError(f"'{key}' is already registered in {self._name}.")
            self._module_dict[key] = cls
            return cls

        return _register

    def get(self, key: str) -> Type[Any]:
        if key not in self._module_dict:
            raise KeyError(
                f"'{key}' not found in {self._name} registry. "
                f"Available: {self.available()}"
            )
        return self._module_dict[key]

    def build(self, key: str, **kwargs: Any) -> Any:
        return self.get(key)(**kwargs)

    def available(self) -> List[str]:
        return sorted(self._module_dict.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._module_dict


MODEL_REGISTRY = Registry("Models")
ATTACK_REGISTRY = Registry("Attacks")
DEFENSE_REGISTRY = Registry("Defenses")
