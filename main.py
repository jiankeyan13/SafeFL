from __future__ import annotations

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from core.simulation.runner import Runner
from core.utils.config import build_global_config
from core.utils.registry import bootstrap_registries


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    bootstrap_registries()
    config = build_global_config(cfg)
    run_dir = HydraConfig.get().runtime.output_dir
    Runner(config, run_dir=run_dir).run()


if __name__ == "__main__":
    main()
