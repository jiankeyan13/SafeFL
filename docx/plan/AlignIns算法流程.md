# AlignIns 算法流程详解

> 论文：*Detecting Backdoor Attacks in Federated Learning via Direction Alignment Inspection*

---

## 背景设定

联邦学习（FL）系统中有 $n$ 个客户端，其中 $m$ 个是恶意的。每一轮训练结束后，服务器收到所有客户端提交的模型更新 $\{\Delta_i\}_{i=1}^n$，然后执行以下流程。

---

## 整体流程

```
收到所有客户端更新
        ↓
Step 1：计算 TDA 值（粗粒度方向检测）
        ↓
Step 2：计算 MPSA 值（细粒度方向检测）
        ↓
Step 3：Z Score 异常检测 → 过滤恶意更新，得到良性集合 S
        ↓
Step 4：幅度裁剪
        ↓
Step 5：聚合 → 更新全局模型
```

---

## Step 1：TDA——粗粒度方向检测

对每个客户端的更新 $\Delta_i$，计算它与当前全局模型 $\theta^t$ 的 Cosine 相似度：

$$\omega_i = \frac{\langle \Delta_i,\ \theta^t \rangle}{\|\Delta_i\|\ \|\theta^t\|}$$

**直觉：** 良性客户端的更新方向应与全局模型参数方向大体一致；恶意客户端由于同时优化后门任务，方向会产生偏差。

---

## Step 2：MPSA——细粒度方向检测

这是本文最核心的创新，分三个子步骤：

### ① 计算主符号 $p$

对所有客户端的更新，在每个参数维度上做**符号投票**：

$$p = \text{sgn}\!\left(\sum_{i=1}^n \text{sgn}(\Delta_i)\right)$$

正号多的维度记 $+1$，负号多的维度记 $-1$，得到代表"多数方向"的主符号向量 $p$。

### ② 提取重要参数（Top-k 掩码）

对每个客户端的更新，用 **Top-30%** 绝对值最大的参数作为掩码，忽略不重要的参数。

> **关键设计：** 重要参数最能代表更新的真实意图；不重要的参数引入噪声，在 non-IID 场景下尤其干扰检测。

### ③ 计算符号对齐率 $\rho_i$

$$\rho_i = 1 - \frac{\left\|(\text{sgn}(\Delta_i) - p)\ \odot\ \text{Top}_k(\Delta_i)\right\|_0}{k}$$

- $\rho_i \in [0,\ 1]$，越接近 1 说明该更新的重要参数符号与主方向越一致，越像良性更新。
- $\odot$ 为逐元素乘积，$\|\cdot\|_0$ 统计不对齐的元素个数。

---

## Step 3：Z Score 异常检测

对每个客户端，分别基于 TDA 值 $\omega_i$ 和 MPSA 值 $\rho_i$ 计算 **Z Score**（标准分数）：

$$\lambda_i = \frac{|x_i - \text{med}(X)|}{\sigma}$$

其中 $\text{med}(X)$ 是中位数，$\sigma$ 是标准差。

**同时满足以下两个条件**的客户端才保留进良性集合 $S$：

$$\lambda_{i,c} \leq \lambda_c \quad \text{且} \quad \lambda_{i,s} \leq \lambda_s$$

- 默认阈值均为 $\lambda_c = \lambda_s = 1.0$，超参数极少。
- 任意一个指标异常的客户端都会被过滤掉。

---

## Step 4：幅度裁剪

对所有客户端更新，使用良性集合 $S$ 中更新的 L2 范数**中位数**作为裁剪阈值 $c$：

$$c = \text{med}\!\left(\{\|\Delta_i\|\}_{i \in S}\right)$$

$$\Delta_i \leftarrow \Delta_i \cdot \min\!\left\{1,\ \frac{c}{\|\Delta_i\|}\right\}$$

> 裁剪操作应用于所有客户端更新，但阈值由检测出的良性更新决定。

---

## Step 5：聚合

对裁剪后的良性集合 $S$ 中所有更新做简单平均，得到本轮全局模型更新：

$$\Delta_e = \frac{1}{|S|} \sum_{i \in S} \Delta_i$$

---

## 两个指标为什么要同时用？

消融实验结果（CIFAR-10，Badnet 攻击）：

| 配置 | IID RA | non-IID RA |
|------|:------:|:----------:|
| 只用 MPSA | 85.02% | 5.79% |
| 只用 TDA | 83.88% | 21.31% |
| TDA + MPSA | 85.82% | 45.30% |
| **AlignIns（含裁剪）** | **85.27%** | **81.32%** |

在 IID 场景下单个指标已够用；在 non-IID 场景下良性更新本身高度分散，单一指标容易误判，两者结合才能互补，大幅提升鲁棒性。

---

## 核心 Insight 一句话总结

> 恶意更新在**重要参数的符号分布**上与良性更新存在系统性差异，这一细粒度方向信息是现有基于幅度或粗粒度方向方法所忽略的，也是 AlignIns 有效性的根本来源。
