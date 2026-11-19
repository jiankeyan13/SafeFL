"""Generic component registry and built-in bootstrapper for SafeFL v2."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

if TYPE_CHECKING:
    import torch.nn as nn

    from core.utils.config import ComponentConfig
    from data.spec import FederatedDataSpec


class Registry:
    """Case-insensitive decorator-based class/callable registry."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module_dict: Dict[str, Type[Any]] = {}

    @staticmethod
    def _norm(key: str) -> str:
        return key.strip().lower()

    def register(self, name: Optional[str] = None):
        def _register(cls: Type[Any]) -> Type[Any]:
            raw_key = name if name is not None else cls.__name__
            key = self._norm(raw_key)
            if key in self._module_dict:
                raise ValueError(f"'{key}' is already registered in {self._name}.")
            self._module_dict[key] = cls
            return cls

        return _register

    def get(self, key: str) -> Type[Any]:
        norm_key = self._norm(key)
        if norm_key not in self._module_dict:
            raise KeyError(
                f"'{key}' not found in {self._name} registry. "
                f"Available: {self.available()}"
            )
        return self._module_dict[norm_key]

    def build(self, key: str, **kwargs: Any) -> Any:
        return self.get(key)(**kwargs)

    def available(self) -> List[str]:
        return sorted(self._module_dict.keys())

    def __contains__(self, key: str) -> bool:
        return self._norm(key) in self._module_dict


MODEL_REGISTRY = Registry("Models")
ATTACK_REGISTRY = Registry("Attacks")
DEFENSE_REGISTRY = Registry("Defenses")

_BOOTSTRAPPED = False


def bootstrap_registries() -> None:
    """Idempotently import built-in dataset, model, attack, and defense modules."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    import core.attack  # noqa: F401
    import core.defense  # noqa: F401
    import data.datasets  # noqa: F401
    import models  # noqa: F401

    _BOOTSTRAPPED = True


def build_model(
    model_cfg: "ComponentConfig",
    data_spec: Optional["FederatedDataSpec"] = None,
) -> "nn.Module":
    """Instantiate a registered model, auto-injecting ``num_classes`` & ``input_channels`` from ``data_spec``."""
    params: Dict[str, Any] = {}
    if data_spec is not None:
        params["num_classes"] = int(data_spec.dataset.num_classes)
        params["input_channels"] = int(data_spec.dataset.train.channels)
    params.update(model_cfg.params)
    return MODEL_REGISTRY.build(model_cfg.name, **params)
