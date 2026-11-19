import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from core.bootstrap import bootstrap_registries
from core.config import build_global_config
from core.simulation.runner import Runner


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    bootstrap_registries()
    config = build_global_config(cfg)
    config.logger_config.log_root = HydraConfig.get().runtime.output_dir
    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
