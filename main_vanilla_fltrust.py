from core.simulation.runner import Runner

# 导入注册表模块 (触发自动注册)
import algorithms.fedavg
import algorithms.fltrust
import models.resnet
import data.datasets.cifar10
import core.attack.data.badnets  # 导入攻击实现


def main():
    config = {
        "experiment_name": "vanilla_fltrust_cifar10_resnet18",
        "project": "FL_Base",
        "use_wandb": False,
        "model": {"name": "resnet18", "params": {"num_classes": 10, "input_channels": 3}},
        "dataset": "cifar10",
        "algorithm": {"name": "fltrust"},
        "attack": {"enabled": True}
    }
    runner = Runner(config)
    runner.run()


if __name__ == "__main__":
    main()
