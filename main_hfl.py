"""最小运行 HFL (异构联邦学习) 的主程序."""

from extension.HFL.hetero_runner import HeteroRunner
from extension.HFL.config import get_default_hfl_config

# 导入注册表模块 (触发自动注册)
import models.hetero_resnet
import data.datasets.cifar10
import extension.HFL.sub_aggregator  # noqa: F401


def main():
    config = get_default_hfl_config(overrides={
        "experiment_name": "rolex_02",
        "logging": {
            "project": "FL_Base",
            "use_wandb": True,
        },
        "dataset": "cifar10",
    })

    runner = HeteroRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
