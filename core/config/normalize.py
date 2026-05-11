from typing import Any

from omegaconf import DictConfig, OmegaConf

from core.config.schema import GlobalConfig


def build_global_config(cfg: DictConfig | dict[str, Any]) -> GlobalConfig:
    """Convert a Hydra/OmegaConf config into SafeFL's runtime config object."""
    if isinstance(cfg, DictConfig):
        raw = OmegaConf.to_container(cfg, resolve=True)
    else:
        raw = dict(cfg)

    if not isinstance(raw, dict):
        raise TypeError("SafeFL config must resolve to a mapping.")

    raw.pop("hydra", None)
    return GlobalConfig.from_dict(raw)
