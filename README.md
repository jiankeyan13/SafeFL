# SafeFL v2

轻量、低心智负担、面向科研快速上手与多卡并行复现的**联邦学习后门攻防研究框架**。

## 目录结构

```text
SafeFL/
├── main.py                      # Hydra 实验入口
├── configs/                     # Hydra 分组配置 (defense / attack / dataset / model / ...)
├── data/                        # 零拷贝 CHW uint8 向量化数据层 (DataHub / DataView / FastBatchLoader)
├── models/                      # 模型注册 (ResNet18 / ResNet34 / FashionMNISTCNN)
└── core/
    ├── types.py                 # 跨层通信三数据盒 (ServerBroadcast / ClientContext / ClientUpdate)
    ├── attack/                  # 1 算法 = 1 文件 (base.py + poisoner.py + 11 种单文件攻击，无子文件夹)
    ├── defense/                 # 1 算法 = 1 文件 (base.py + 14 种单文件防御，无子文件夹)
    ├── simulation/              # 联邦系统角色与调度执行 (client / server / coordinator / executor / runner)
    └── utils/                   # 通用底座 (functional / trainer / evaluator / config / registry / logger / lr_schedule)
```

## 支持算法

### 攻击算法（Attack）

| 名称 | Official Repo |
| --- | --- |
| 17arXiv_BadNets | [https://github.com/Kooscii/BadNets](https://github.com/Kooscii/BadNets) |
| 17arXiv_Blend | - |
| 20ICLR_DBA | [https://github.com/AI-secure/DBA](https://github.com/AI-secure/DBA) |
| 20NeurIPS_PGD | [https://github.com/ksreenivasan/OOD_Federated_Learning](https://github.com/ksreenivasan/OOD_Federated_Learning) |
| 22ICML_Neurotoxin | [https://github.com/jhcknzzm/Federated-Learning-Backdoor](https://github.com/jhcknzzm/Federated-Learning-Backdoor) |
| 23AAAI_F3BA | [https://github.com/jinghuichen/Focused-Flip-Federated-Backdoor-Attack](https://github.com/jinghuichen/Focused-Flip-Federated-Backdoor-Attack) |
| 23ICML_Chameleon | [https://github.com/ybdai7/Chameleon-durable-backdoor](https://github.com/ybdai7/Chameleon-durable-backdoor) |
| 23NeurIPS_A3FL | [https://github.com/hfzhang31/A3FL](https://github.com/hfzhang31/A3FL) |
| 24AAAI_FCBA | [https://github.com/PhD-TaoLiu/FCBA](https://github.com/PhD-TaoLiu/FCBA) |
| 24ICLR_LP-Attack | [https://github.com/zhmzm/Poisoning_Backdoor-critical_Layers_Attack](https://github.com/zhmzm/Poisoning_Backdoor-critical_Layers_Attack) |
| 25CVPR_Mirage | [https://github.com/NUAA-SmartSensing/Mirage](https://github.com/NUAA-SmartSensing/Mirage) |
| 25ICCV_LGA | [https://github.com/yqqhyqq/LGA](https://github.com/yqqhyqq/LGA) |
| 26CVPR_Batman | [https://github.com/WenddHe0119/Batman](https://github.com/WenddHe0119/Batman) |

### 防御算法（Defense）

| 名称 | Official Repo |
| --- | --- |
| 17AISTATS_FedAvg | - |
| 17NeurIPS_Multi-Krum | [https://github.com/LPD-EPFL/AggregaThor](https://github.com/LPD-EPFL/AggregaThor) |
| 18ICML_TrimmedMean | - |
| 20RAID_FoolsGold | [https://github.com/DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold) |
| 21AAAI_RLR | [https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate](https://github.com/TinfoilHat0/Defending-Against-Backdoors-with-Robust-Learning-Rate) |
| 21NDSS_DnC | [https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning](https://github.com/vrt1shjwlkr/NDSS21-Model-Poisoning) |
| 21NDSS_FLTrust | - |
| 22AAAI_RFLBAT | - |
| 22NDSS_DeepSight | - |
| 22TSP_RFA | [https://github.com/krishnap25/rfa](https://github.com/krishnap25/rfa) |
| 22USENIX_FLAME | - |
| 24AAAI_Snowball | [https://github.com/zhenqincn/Snowball](https://github.com/zhenqincn/Snowball) |
| 24NDSS_FreqFed | - |
| 24NeurIPS_FDCR | [https://github.com/WenkeHuang/FDCR](https://github.com/WenkeHuang/FDCR) |
| 24USENIX_BackdoorIndicator | [https://github.com/ybdai7/Backdoor-indicator-defense](https://github.com/ybdai7/Backdoor-indicator-defense) |
| 25CVPR_AlignIns | [https://github.com/JiiahaoXU/AlignIns](https://github.com/JiiahaoXU/AlignIns) |
| 25ICML_SPMC | [https://github.com/WenddHe0119/SPMC](https://github.com/WenddHe0119/SPMC) |
| 25NeurIPS_MARS | [https://github.com/yunming181920/MARS](https://github.com/yunming181920/MARS) |

### 3. 支持触发器类型（Trigger）

<table>
<thead>
<tr><th>名称</th><th>可选项</th></tr>
</thead>
<tbody>
<tr><th colspan="2" align="center" style="text-align:center;background-color:#f6f8fa;">General</th></tr>
<tr><td>静态局部</td><td><code>patch</code></td></tr>
<tr><td>分布式</td><td><code>dba</code>、<code>fcba</code></td></tr>
<tr><td>混合</td><td><code>blend</code></td></tr>
<tr><td>语义触发器</td><td><code>semantic</code></td></tr>
<tr><td>边缘触发器</td><td><code>edge_case</code></td></tr>
<tr><th colspan="2" align="center" style="text-align:center;background-color:#f6f8fa;">More</th></tr>
<tr><td>动态优化</td><td><code>a3fl</code>、<code>f3ba</code>、<code>mirage</code></td></tr>
<tr><td>自然语言文本</td><td>暂未实现</td></tr>
</tbody>
</table>

## 快速运行

```bash
# 默认运行 (Ray 多卡执行器，无 GPU/Ray 时自动回退到 LocalExecutor)
python main.py

# 本地单进程模式（支持 IDE 直接打断点单步调试）
python main.py experiment=quick_debug defense=fltrust attack=badnets

# 正交组合任意攻击 × 防御（FLTrust 会自动开启 server proxy）
python main.py attack=batman defense=fltrust

# 运行论文场景 Preset 或 Hydra 多实验网格扫描
python main.py -m experiment=paper_batman attack=badnets,batman,neurotoxin defense=fedavg,fltrust,alignins
python main.py -m defense=fedavg,flame,alignins attack=badnets,neurotoxin,lga seed=42
```

## 5 分钟新增算法指南（1 算法 = 1 文件）

### 1. 新增后门攻击 (`core/attack/<my_attack>.py`)
攻击者是与 `BenignClient` 平级的参与者（绝不继承 `BenignClient`），直接操作 `ClientContext`：

```python
from core.attack.base import BaseAttacker
from core.types import ClientContext, ClientUpdate
from core.utils.registry import ATTACK_REGISTRY
from core.utils.trainer import train_model


@ATTACK_REGISTRY.register("my_attack")
class MyAttack(BaseAttacker):
    def __init__(self, scale: float = 1.5, **kwargs):
        super().__init__(**kwargs)  # 自动支持 target_label / poison_ratio / poisoner 等配置
        self.scale = float(scale)

    def step(self, ctx: ClientContext) -> ClientUpdate:
        stats = train_model(ctx.model, ctx.loader(poisoner=self.poisoner), ctx.train_cfg)
        delta = {k: (p.detach() - ctx.initial_params[k]) * self.scale
                 for k, p in ctx.model.named_parameters() if k in ctx.initial_params}
        return ctx.pack_update(param_delta=delta, metrics={"loss": stats.loss})
```

### 2. 新增服务端防御 (`core/defense/<my_defense>.py`)
所有客户端上传更新均已物理分离可训练 `param_delta` 与 BN `buffers`，可直接组合 `core.utils.functional` 中的向量化算子：

```python
from typing import List
import torch.nn as nn

from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
from core.utils import functional as F
from core.utils.registry import DEFENSE_REGISTRY


@DEFENSE_REGISTRY.register("my_defense")
class MyDefense(BaseDefense):
    def defend(
        self,
        global_model: nn.Module,
        updates: List[ClientUpdate],
        ctx: ServerContext,
    ) -> DefenseOutput:
        delta_mat, keys = F.stack_flat_deltas(updates, device=ctx.device)
        # 在 [N, D] 矩阵上筛选良性下标 benign_idx ...
        benign_idx = list(range(len(updates)))
        param_delta, buffers = F.aggregate_updates([updates[i] for i in benign_idx], device=ctx.device)
        return DefenseOutput(param_delta=param_delta, buffers=buffers)
```
