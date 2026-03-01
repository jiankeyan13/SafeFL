from core.simulation.base_runner import BaseRunner

# 导入注册表模块 (触发自动注册)
import algorithms.fedavg
import models.resnet
import data.datasets.cifar10

def main():
    config = {
        "lr": 0.001,
        "experiment_name": "minimal_fedavg_test"
    }
    # 创建并运行仿真器
    runner = BaseRunner(config)
    runner.run()

if __name__ == "__main__":
    main()
