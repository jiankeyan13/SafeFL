"""Bootstrap built-in registries for SafeFL v2."""
from __future__ import annotations


def bootstrap_registries() -> None:
    """Import built-in dataset, model, attack, and defense modules."""
    import data.datasets  # noqa: F401
    import models.fashion_cnn  # noqa: F401
    import models.resnet  # noqa: F401
