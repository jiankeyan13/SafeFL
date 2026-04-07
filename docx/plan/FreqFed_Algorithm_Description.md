# FreqFed 算法描述（筛选-聚合-后处理框架）

> 原始论文：*FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks in Federated Learning*（NDSS 2024）

---

## 总体流程概述

FreqFed 是一种联邦学习聚合防御机制，核心思路是：将客户端模型更新（delta）变换到频域，利用低频分量携带足够权重信息且对投毒操作敏感这一性质，通过无监督聚类自动识别并排除恶意更新，最终只对良性客户端的 delta 做加权平均聚合。

整体流程与框架的三阶段映射如下：

```
客户端 delta (W_i^t) 
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  阶段 1：筛选（Screener）                            │
│   DCT 变换 → 低频提取 → HDBSCAN 聚类 → 加权分数     │
└─────────────────────────────────────────────────────┘
    │  screen_scores（每客户端 ∈ [0,1]，0 表示排除）
    ▼
┌─────────────────────────────────────────────────────┐
│  阶段 2：数学聚合（Aggregator）                      │
│   FedAvg（仅对 score > 0 的客户端 delta 加权平均）   │
└─────────────────────────────────────────────────────┘
    │  aggregated_delta
    ▼
┌─────────────────────────────────────────────────────┐
│  阶段 3：后处理（Refiner）                           │
│   global_weights + aggregated_delta → 载入模型       │
│   （可选：BN 校准 / 差分隐私噪声等）                 │
└─────────────────────────────────────────────────────┘
    │
    ▼
新全局模型 G_{t+1}
```

---

## 阶段 1：筛选（Screener）

### 输入
- `client_deltas`：所有 K 个客户端上传的模型 delta，即 $W_i^t - G^t$，每个 delta 与全局模型同构（layer name → tensor）
- `num_samples`：各客户端本地样本数（用于后续加权，筛选阶段本身不依赖此值）
- `global_model`：当前轮次全局模型（筛选阶段不修改，仅作参考）
- `context`：跨阶段共享的上下文字典（可传递中间状态）

### 输出
- `screen_scores`：长度为 K 的字典或列表，每个客户端对应一个标量分数
  - 分数 = **1**：该客户端 delta 被判定为良性，参与聚合
  - 分数 = **0**：该客户端 delta 被判定为恶意，完全排除

### 内部步骤

#### 步骤 1：对每个客户端 delta 计算 DCT 系数矩阵

对客户端 $i$ 的 delta $W_i$，遍历其所有参数层（weight tensor）：

1. 将每一层的参数张量 reshape 为二维矩阵（若已是二维则直接使用，否则展平为 $N \times M$ 形状）。
2. 对该二维矩阵施加**二维离散余弦变换（2D-DCT）**，得到系数矩阵 $V_i$：

$$
V_i(k, l) = \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} c_1 c_2 \cdot W_i(m,n) \cdot \cos\!\left(\frac{k\pi}{2M}(2m+1)\right) \cos\!\left(\frac{l\pi}{2N}(2n+1)\right)
$$

3. 对模型所有层的 DCT 系数矩阵分别执行后续低频提取，再将结果拼接为一个完整向量 $F_i$。

> **设计依据**：神经网络权重的能量绝大部分集中在低频 DCT 分量；投毒攻击会在模型权重中引入额外结构，导致低频能量分布发生偏移，从而在频域中可被区分。

#### 步骤 2：提取低频分量（Filtering）

对系数矩阵 $V_i$（大小为 $|V| \times |V|$），仅保留满足以下条件的元素：

$$
\text{保留} \quad V_i[k][l] \quad \text{当且仅当} \quad k + l \leq \left\lfloor \frac{|V|}{2} \right\rfloor
$$

将所有满足条件的元素展平，拼接为低频向量 $F_i \in \mathbb{R}^d$，其中 $d \ll |V|^2$。

对所有 K 个客户端执行同样操作，得到 $\{F_1, F_2, \ldots, F_K\}$。

> **设计依据**：低频分量对应权重的整体分布模式，包含充足的辨别信息；高频分量对应细粒度扰动，噪声较多，对聚类稳定性干扰大。

#### 步骤 3：基于余弦距离的 HDBSCAN 聚类（Clustering）

1. 构建 $K \times K$ 的成对距离矩阵，使用**余弦距离**（而非欧氏距离）：

$$
\text{dist}(i, j) = 1 - \text{CosineSimilarity}(F_i, F_j)
$$

2. 以该距离矩阵作为输入，运行 **HDBSCAN**（层次密度聚类）：
   - 无需预先指定簇的数量。
   - 能够自适应地处理不同形状和大小的簇。
   - 将离群点标记为噪声（不属于任何簇），不强制归类。

3. 获取聚类结果后，找到**样本数最多的簇**（即最大簇）：

$$
C^* = \arg\max_c \left| \{ i \mid \text{cluster\_id}(i) = c \} \right|
$$

> **设计假设**：良性客户端数量超过半数（$k_A < K/2$），因此良性模型的低频向量会自然聚集在最大簇中；恶意模型的低频分布存在偏移，会落入较小的簇或被标记为噪声。

#### 步骤 4：生成加权分数

$$
\text{screen\_score}[i] = \begin{cases} 1 & \text{if } i \in C^* \\ 0 & \text{otherwise} \end{cases}
$$

返回 `screen_scores` 字典（client index → score）。

---

## 阶段 2：数学聚合（Aggregator）

### 输入
- `updates`（即 `client_deltas`）：所有 K 个客户端的 delta（与筛选阶段相同）
- `sample_weights`（即 `num_samples`）：各客户端本地样本数 $n_i$
- `screen_scores`：来自筛选阶段，每个客户端对应 0 或 1 的分数
- `global_model`：当前全局模型（聚合阶段不修改，不使用）
- `context`：跨阶段上下文

### 输出
- `aggregated_delta`：聚合后的全局 delta（layer name → tensor），与全局模型同构

### 内部步骤

1. **过滤**：仅保留 `screen_score[i] == 1` 的客户端集合，设其索引集合为 $\mathcal{B} = \{b_1, b_2, \ldots, b_L\}$，$L = |\mathcal{B}|$。

2. **加权平均（FedAvg）**：对筛选通过的客户端 delta 按样本数加权平均：

$$
\Delta_{\text{agg}} = \frac{\sum_{l=1}^{L} n_{b_l} \cdot W_{b_l}^{\Delta}}{\sum_{l=1}^{L} n_{b_l}}
$$

其中 $W_{b_l}^{\Delta}$ 为客户端 $b_l$ 的 delta（逐层 tensor）。

3. 返回 `aggregated_delta`，逐层存储，结构与全局模型 `state_dict` 一致。

> **说明**：若所有客户端样本数相同，则等价于简单平均 $\Delta_{\text{agg}} = \frac{1}{L} \sum_{l=1}^{L} W_{b_l}^{\Delta}$，与论文中 Algorithm 1 第 7 行一致。

---

## 阶段 3：后处理（Refiner）

### 输入
- `global_model`：当前全局模型对象（将被原地更新）
- `new_state`：由框架在阶段 2 与阶段 3 之间完成的合成结果，即：

$$
\text{new\_state}[\text{key}] = G^t[\text{key}] + \Delta_{\text{agg}}[\text{key}]
$$

（此步骤在框架主循环中完成，不属于 Refiner 内部逻辑）

- `calibration_loader`（即 `proxy_loader`）：可选的代理数据集 DataLoader，用于 BN 校准等操作
- `device`：目标设备
- `context`：跨阶段上下文

### 输出
- 原地更新 `global_model` 的权重为 `new_state`（无返回值）

### 内部步骤

#### 基础操作（必须执行）
1. 将 `new_state` 载入 `global_model`：

```python
global_model.load_state_dict(new_state)
```

#### 可选操作（根据配置执行）

| 可选操作 | 触发条件 | 说明 |
|---|---|---|
| **Batch Normalization 校准** | 提供 `calibration_loader` 且模型含 BN 层 | 在代理数据上前向传播若干批次，重新统计 BN 层的 running_mean / running_var，消除因聚合引入的统计偏差 |
| **差分隐私噪声注入** | 配置启用 DP | 在模型权重上叠加校准后的高斯噪声，增强隐私保护（FreqFed 原文未采用，但框架支持扩展） |
| **权重裁剪/归一化** | 配置启用 clipping | 对聚合后权重的 L2 范数进行裁剪，防止极端更新 |

> **注意**：FreqFed 原文的后处理仅为"载入新权重"，不包含额外精炼操作。BN 校准等属于框架扩展能力，在代理数据可用时可选启用。

---

## 关键设计约束与边界条件

| 约束 | 说明 |
|---|---|
| **攻击者比例上限** | 要求恶意客户端数量严格小于总数一半，即 $k_A < K/2$，否则最大簇可能被恶意模型主导 |
| **数据分布无假设** | FreqFed 不假设客户端数据为 iid 或 non-iid，在完全 non-iid（iid=0.0）场景下仍有效 |
| **攻击类型无假设** | 同时防御有目标攻击（后门）和无目标攻击（Label Flipping、Random Updates、PGD），无需针对特定攻击调整参数 |
| **不直接操作权重** | 所有防御逻辑在频域中完成，不依赖权重的 L2 范数、余弦距离等直接度量，从而规避对抗性权重操纵 |

---

## 与框架接口的对应关系总结

```
screen_scores ∈ {0, 1}^K
    └── 由 HDBSCAN 最大簇成员资格决定
    └── 0：完全排除，不参与任何加权平均
    └── 1：参与 FedAvg 加权（权重由 num_samples 决定）

aggregated_delta
    └── 仅包含通过筛选的客户端 delta 的加权平均
    └── 结构与 global_model.state_dict() 完全一致

Refiner
    └── 执行 new_state = global_state + aggregated_delta 后的载入
    └── 可选附加 BN 校准 / 噪声注入等精炼操作
```
