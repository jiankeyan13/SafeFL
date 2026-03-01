import yaml
import argparse
import os
import sys

# 将当前目录加入 Python 路径，确保能 import core/models 等
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.runner import FederatedRunner

def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description='Federated Learning Framework')
    parser.add_argument('--config', type=str, default='configs/cifar10_fedavg.yaml', 
                        help='Path to the YAML config file')
    args = parser.parse_args()

    # 2. 读取 YAML 配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found at: {args.config}")
        
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Loading config from {args.config}...")

    # 3. 实例化并运行 Runner
    # 此时 Runner 内部会自动触发 Registry 的 import，完成组件组装
    runner = FederatedRunner(config)
    runner.run()

if __name__ == '__main__':
    main()