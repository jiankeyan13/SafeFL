import argparse
from extension.HFL.hetero_runner import HeteroRunner
from extension.HFL.config import get_default_hfl_config
from core.utils.configs import load_config_from_yaml

# 导入注册表模块 (触发自动注册)
import models.hetero_resnet
import data.datasets.cifar10
import core.attack


def main():
    parser = argparse.ArgumentParser(description="SafeFL Runner")
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the config file (e.g., ./configs/xxx.yaml)",
    )
    args, unknown = parser.parse_known_args()

    # 兼容 python main.py --./configs/xxx.yaml 的写法
    config_file = args.config_path
    if not config_file:
        for arg in unknown:
            if arg.startswith("--./") or arg.startswith("--configs/"):
                config_file = arg[2:]  # 去掉开头的 --
                break

    if config_file:
        # 如果指定了配置文件，从 YAML 加载
        config = load_config_from_yaml(config_file)
    else:
        # 否则使用默认配置
        config = get_default_hfl_config()

    runner = HeteroRunner(config)
    runner.run()


if __name__ == "__main__":
    main()