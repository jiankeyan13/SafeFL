def bootstrap_registries() -> None:
    """Import modules that register datasets, models, attacks, and algorithms."""
    import algorithms  # noqa: F401
    import core.attack  # noqa: F401
    import data.datasets.cifar10  # noqa: F401
    import data.datasets.cifar100  # noqa: F401
    import data.datasets.tiny_imagenet  # noqa: F401
    import models.resnet  # noqa: F401
