def bootstrap_registries() -> None:
    """Import modules that register datasets, models, attacks, and algorithms."""
    import algorithms  # noqa: F401
    import core.attack  # noqa: F401
    import data.datasets  # noqa: F401
    import models.fashion_cnn  # noqa: F401
    import models.resnet  # noqa: F401
