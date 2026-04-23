# SafeFL 配置说明

本目录包含从各 `main_*.py` 及 `core.utils.configs` 汇总的 YAML 配置文件.

## 文件说明

按攻击方式分类, vanilla 为 BadNets 攻击场景:

| 文件 | 说明 |
|------|------|
| `fedavg.yaml` | 完整默认配置 schema, 含所有可调参数 |
| `vanilla/fedavg.yaml` | FedAvg + BadNets 攻击 |
| `vanilla/fltrust.yaml` | FLTrust + BadNets 攻击 |
| `vanilla/multikrum.yaml` | Multi-Krum 防御 + BadNets 攻击 |
| `HFL/FedRolex.yaml` | 异构联邦学习 FedRoLex |

## 使用方式

```python
from core.utils.configs import load_config_from_yaml
from core.simulation.runner import Runner

# 加载配置 (若存在 default.yaml 则自动合并)
config = load_config_from_yaml("vanilla/fedavg.yaml")
# 可选: 运行时覆盖
config = load_config_from_yaml("vanilla/fedavg.yaml", overrides={"use_wandb": False})

runner = Runner(config)
runner.run()
```

或直接使用非异构入口:

```bash
python main_vanilla.py configs/vanilla/fedavg.yaml
```

HFL 使用 `HeteroRunner` 和 `get_default_hfl_config`, 可先加载 YAML 再合并:

```python
from extension.HFL.config import get_default_hfl_config
from core.utils.configs import load_config_from_yaml

config = load_config_from_yaml("HFL/FedRolex.yaml")
# 或与 get_default_hfl_config 合并
base = get_default_hfl_config()
base.update(config)
```
