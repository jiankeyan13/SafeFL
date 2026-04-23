import argparse

from core.simulation.runner import Runner
from core.utils.configs import GlobalConfig, load_config_from_yaml

# 导入注册表模块 (触发自动注册)
import models.resnet
import data.datasets.cifar10
import core.attack
import algorithms


def main():
    parser = argparse.ArgumentParser(description="SafeFL Vanilla Runner")
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the config file (e.g., ./configs/vanilla/fedavg.yaml)",
    )
    args, unknown = parser.parse_known_args()

    # 兼容 python main_vanilla.py --./configs/xxx.yaml 的写法
    config_file = args.config_path
    if not config_file:
        for arg in unknown:
            if arg.startswith("--./") or arg.startswith("--configs/"):
                config_file = arg[2:]
                break

    if config_file:
        config = load_config_from_yaml(config_file)
    else:
        config = GlobalConfig().to_dict()

    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
