# FoolsGold 复现代码审查报告

对比对象：`model_aggregator.py`（官方参考）与 `foolsgold.py`（复现代码）

---

## 总体流程对照

官方算法（Algorithm 1）流程：

1. 累加历史梯度 `H_i += Δ_i`
2. 提取指示性特征并加权（输出层参数重要性）
3. 计算加权余弦相似度矩阵 `cs_ij`
4. **Pardoning**：若 `v_j > v_i`，则 `cs_ij *= v_i / v_j`
5. 重算每行最大值 `v_i = max_j(cs_ij)`（基于 Pardoning 后的矩阵）
6. `α_i = 1 - v_i`，归一化，logit 拉伸

---

## 问题一：Pardoning 逻辑与官方不一致（严重）

### 官方代码逻辑（`model_aggregator.py`）

```python
# 先计算 maxcs（即 v_i），作为 Pardoning 的依据
maxcs = np.max(cs, axis=1) + epsilon

for i in range(n):
    for j in range(n):
        if i == j:
            continue
        if maxcs[i] < maxcs[j]:          # 若 v_i < v_j
            cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

# Pardoning 后，直接从修改后的 cs 矩阵提取最大值
wv = 1 - (np.max(cs, axis=1))
```

关键点：**`maxcs`（即 `v_i`）在 Pardoning 之前一次性固定，Pardoning 过程中不会更新**。Pardoning 后，直接对修改过的 `cs` 矩阵取每行最大值，作为最终的 `v_i`。

### 复现代码逻辑（`foolsgold.py`）

```python
# 先屏蔽对角线，计算初始 v_i
cs_masked = cs.clone()
cs_masked.fill_diagonal_(0.0)
v_initial = cs_masked.max(dim=1).values   # 用于 Pardoning 的 v_i

# Pardoning（向量化实现）
v_col, v_row = v_initial.unsqueeze(1), v_initial.unsqueeze(0)
ratio = torch.where(v_row > v_col, v_col / v_row.clamp(min=1e-8), ...)
cs_pardoned = cs * ratio

# 重新屏蔽对角线，再次取最大值
cs_pardoned.fill_diagonal_(0.0)
max_sim = cs_pardoned.max(dim=1).values   # 用于计算 alpha 的 v_i
alpha_raw = 1.0 - max_sim
```

复现代码实际上也是先固定 `v_initial`，再 Pardoning，再取最大值，**大方向正确**，但存在以下细节差异：

**差异点：Pardoning 方向的理解**

官方代码：`if maxcs[i] < maxcs[j]: cs[i][j] *= maxcs[i]/maxcs[j]`
即：当 `v_j > v_i` 时，缩小 `cs[i][j]`（第 `i` 行第 `j` 列），即对"高相似度客户端 j 对客户端 i 的影响"进行赦免。

复现代码中 `ratio` 的计算：
```python
ratio = torch.where(v_row > v_col, v_col / v_row.clamp(min=1e-8), ones)
```
这里 `v_row = v_initial.unsqueeze(0)`（形状 `[1, n]`，代表列方向的 `v_j`），`v_col = v_initial.unsqueeze(1)`（形状 `[n, 1]`，代表行方向的 `v_i`）。当 `v_row > v_col`（即 `v_j > v_i`）时，`ratio[i,j] = v_col[i] / v_row[j] = v_i / v_j`。

**这与官方逻辑一致**，但有一个隐患：`cs_pardoned = cs * ratio` 中包含了对角线元素（对角线上 `v_i == v_j`，ratio 为 1，不影响结果）。复现代码随后调用 `cs_pardoned.fill_diagonal_(0.0)` 处理，**逻辑最终正确**，但不够清晰。

**实质问题：Pardoning 基于的是含对角线的原始 `cs` 矩阵**

官方代码执行的是 `cs = smp.cosine_similarity(...) - np.eye(n)`，即在计算余弦相似度后**立即减去单位矩阵**，使对角线为 0。之后 `maxcs = np.max(cs, axis=1)` 不会受对角线影响。

复现代码计算的 `cs` 对角线为 1（余弦相似度自身为 1），在取 `v_initial` 时调用了 `cs_masked.fill_diagonal_(0.0)` 屏蔽对角线。但 Pardoning 时用的是原始 `cs`（含对角线 1）乘以 ratio，虽然随后再次置零，**中间步骤引入了不必要的对角线干扰**。建议在计算 `v_initial` 前直接 `cs.fill_diagonal_(0.0)`，而非单独克隆一份。

---

## 问题二：特征提取范围与官方不同（中等）

### 官方实现

官方在 `foolsgold()` 函数中通过 `sig_features_idx` 传入显式的特征索引：

```python
sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)
```

`sig_features_idx` 由外部决定（通常为输出层参数的索引），作用于**累加的历史梯度 `summed_deltas`**（全参数空间），再截取重要特征子集。

论文原文（Section 6.2）对指示性特征的描述：通过输出层参数的**幅度**来确定，并支持"硬过滤"（`importanceFeatureHard`）和"软加权"（`importanceFeatureMapLocal`）两种模式。

### 复现实现

```python
weight_name, bias_name = _resolve_output_layer(global_model)
feat = _extract_output_layer_features(delta, weight_name, bias_name)
```

复现代码直接提取输出层 weight + bias 的 delta，**不对特征做重要性加权**，相当于官方代码中 `importance=False, importanceHard=False` 且 `sig_features_idx` 恰好等于输出层参数的情况。

这在论文默认实验设置下是合理的，但**丢失了软加权和硬过滤机制**，对抗智能扰动攻击（Section 7.3）时防御能力会下降。

---

## 问题三：历史梯度累加的对象不同（中等）

### 官方实现

```python
# summed_deltas 是所有参数的全量历史梯度
sd = summed_deltas.copy()
sig_filtered_deltas = np.take(sd, sig_features_idx, axis=1)
```

历史梯度 `H_i` 存储的是**全参数**的累加梯度，特征提取是在计算相似度时才做截取。

### 复现实现

```python
feat = _extract_output_layer_features(delta, ...)  # 只取输出层
self._history_features[cid] += feat  # 只累加输出层特征
```

复现代码**只对输出层特征做历史累加**，不保存全参数历史。这意味着若后续希望切换特征集（如从输出层改为全参数），现有历史记录无法复用。在当前固定使用输出层的前提下，功能等价，但与论文原始设计的扩展性不符。

---

## 问题四：logit 拉伸的中心点处理方式（轻微）

### 官方实现（Algorithm 1 第 20 行）

```
α = κ * (ln(α / (1 - α)) + 0.5)
```

即：`logit(α) + 0.5`，再乘以置信度参数 `κ`（默认 `κ=1`）。官方代码中：

```python
wv = (np.log((wv / (1 - wv)) + epsilon) + 0.5)
wv[(np.isinf(wv) + wv > 1)] = 1
wv[(wv < 0)] = 0
```

注意官方对 logit 结果做**线性平移（+0.5）**，然后直接截断到 `[0, 1]`，**没有经过 sigmoid**。

### 复现实现

```python
logit_alpha = torch.log(alpha_clamped / (1.0 - alpha_clamped))
logit_center_val = math.log(center / (1.0 - center))  # center=0.5 → logit_center_val=0
alpha_stretched = torch.sigmoid(logit_alpha - logit_center_val)
```

当 `center=0.5` 时，`logit_center_val = 0`，`alpha_stretched = sigmoid(logit(alpha)) = alpha`。这实际上**没有做任何拉伸**，完全等价于跳过 logit 步骤，与官方的 `logit(α) + 0.5` 截断逻辑产生显著差异。

**官方逻辑的实际效果：**
- `α` 接近 1（高相似）→ `logit(α)` 很大正数 + 0.5 → 截断为 1 → 权重为 1（即被完全压制）
- `α` 接近 0（低相似）→ `logit(α)` 很大负数 + 0.5 → 截断为 0 → 权重为 0（不受惩罚）
- `α = 0.5` → `logit(0.5) + 0.5 = 0 + 0.5 = 0.5` → 权重 0.5

**复现代码的实际效果：**
- `sigmoid(logit(α)) = α`，即恒等变换，无拉伸效果

**正确复现应为：**

```python
logit_alpha = torch.log(alpha_clamped / (1.0 - alpha_clamped)) + 0.5
alpha_stretched = logit_alpha.clamp(0.0, 1.0)
```

当需要支持 `κ` 参数时：`alpha_stretched = (κ * (logit_alpha + 0.5)).clamp(0.0, 1.0)`

---

## 问题五：epsilon 的使用位置（轻微）

### 官方实现

```python
epsilon = 1e-5
maxcs = np.max(cs, axis=1) + epsilon  # 加在 v_i 上，防止除零
```

epsilon 加在分母 `maxcs[j]` 上，防止 `v_j = 0` 时除零。

### 复现实现

```python
row_norms = torch.clamp(row_norms, min=1e-8)  # 归一化时防零
...
v_col / v_row.clamp(min=1e-8)  # Pardoning 除法防零
alpha_max_val = alpha_raw.max().clamp(min=1e-8)  # 归一化防零
```

复现代码的防零处理更分散，但覆盖了所有除法场景，**逻辑上无误**，只是处理方式与官方不同。

---

## 汇总

| # | 问题 | 严重程度 | 是否影响结果 |
|---|------|----------|------------|
| 1 | Pardoning 基于含对角线的 cs 矩阵（中间步骤有冗余），最终结果等价 | 低 | 否（结果等价） |
| 2 | 特征提取只用输出层，未实现软加权/硬过滤 | 中 | 是（对抗扰动攻击时防御变弱） |
| 3 | 历史梯度只累加输出层特征，不保留全参数历史 | 中 | 当前设置下等价，扩展性差 |
| **4** | **logit 拉伸用 sigmoid 实现，实际等于恒等变换，与官方线性平移截断不符** | **严重** | **是（logit 拉伸失效，无法抵抗大规模 sybil）** |
| 5 | epsilon 位置不同，但覆盖完整 | 低 | 否 |

---

## 最高优先级修复建议

**针对问题四**，将 logit 拉伸部分替换为：

```python
eps = 1e-7
alpha_clamped = alpha_norm.clamp(eps, 1.0 - eps)
# 官方：logit(α) + 0.5，再截断到 [0, 1]
alpha_stretched = (torch.log(alpha_clamped / (1.0 - alpha_clamped)) + 0.5).clamp(0.0, 1.0)
```

若要支持 `κ`（置信度参数）：

```python
alpha_stretched = (self.kappa * (torch.log(alpha_clamped / (1.0 - alpha_clamped)) + 0.5)).clamp(0.0, 1.0)
```
