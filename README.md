# SafeFL v2

轻量、低心智负担、面向科研快速上手与多卡并行复现的**联邦学习后门攻防研究框架**。

## 目录结构

```text
SafeFL/
├── main.py                      # Hydra 实验入口
├── configs/                     # Hydra 分组配置 (algorithm / attack / dataset / model / ...)
├── data/                        # 零拷贝 CHW uint8 向量化数据层 (DataHub / DataView / FastBatchLoader)
├── models/                      # 模型注册 (ResNet18 / ResNet34 / FashionMNISTCNN)
└── core/
    ├── types.py                 # 跨层通信三数据盒 (ServerBroadcast / ClientContext / ClientUpdate)
    ├── attack/                  # 1 算法 = 1 文件 (BaseAttacker + 11 种攻击 + AttackCoordinator + Poisoner)
    ├── defense/                 # 1 算法 = 1 文件 (BaseDefense + 14 种防御 + functional.py 数学算子库)
    ├── simulation/              # 联邦系统角色与同构执行层 (BenignClient / Server / ClientWorker / Runner)
    └── utils/                   # 基础设施 (config / registry / trainer / evaluator / logger / lr_schedule)
```

## 快速运行

```bash
# 默认运行 (Ray 多卡执行器，无 GPU/Ray 时自动回退到 LocalExecutor)
python main.py

# 本地单进程模式（支持 IDE 直接打断点单步调试）
python main.py parallel.backend=local training.rounds=5 algorithm=fltrust data.enable_proxy=true attack=badnets logging=local

# 运行预设实验或 Hydra 多实验扫描
python main.py experiment=batman_fltrust
python main.py -m algorithm=fedavg,flame,alignins attack=badnets,neurotoxin,lga seed=42
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
所有客户端上传更新均已物理分离可训练 `param_delta` 与 BN `buffers`，可直接组合 `core.defense.functional` 中的向量化算子：

```python
from typing import List
import torch.nn as nn

from core.defense import functional as F
from core.defense.base import BaseDefense
from core.types import ClientUpdate, DefenseOutput, ServerContext
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
