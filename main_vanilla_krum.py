from core.simulation.runner import Runner

# 导入注册表模块 (触发自动注册)
import algorithms.fedavg
import algorithms.multi_krum
import models.resnet
import data.datasets.cifar10
import core.attack.data.badnets  # 导入攻击实现

def main():
    # 默认配置下运行 BadNets 攻击 + Multi-Krum 防御
    config = {
        "experiment_name": "vanilla_multikrum_cifar10_resnet18",
        "project": "FL_Base",
        "use_wandb": True,  # 如果需要上传到 wandb, 请改为 True
        "model": {
            "name": "resnet18",
            "params": {"num_classes": 10, "input_channels": 3}
        },
        "dataset": "cifar10",
        "algorithm": {
            "name": "multi_krum",
            "params": {
                "screener": {
                    "params": {
                        "f": 4,  # 预期的最大恶意客户端数量 (20 * 0.2 = 4)
                        "m": 14  # 最终选取的客户端数量 (通常设为 n - f - 2 = 20 - 4 - 2 = 14)
                    }
                }
            }
        },
        "attack": {
            "enabled": True,
        }
    }
    
    # 创建并运行仿真器
    runner = Runner(config)
    runner.run()

if __name__ == "__main__":
    main()
