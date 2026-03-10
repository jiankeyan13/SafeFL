# FoolsGold 问题2修改计划

## 背景

当前 `_compute_importance_weights` 与论文 `importanceFeatureMapLocal` / `importanceFeatureHard` 存在两处语义偏差：

1. **软加权**：当前用 `|w|` 衡量特征重要性；论文用 `|w - mean(w)|`（偏离类内均值的程度）
2. **归一化粒度**：当前对整个输出层做全局 `max=1` 归一化；论文对每个输出类别单独做 `sum=1` 归一化

---

## 前提：输出层形状语义

官方代码将输出层参数 reshape 为 `[n_classes, class_d]`，其中：

- `n_classes`：分类数，从 `weight_tensor.shape[0]` 推断
- `class_d`：每个类别对应的特征维度，等于 `total_dim / n_classes`

---

## Step 1：重写 `_compute_importance_weights`

### 修改签名

```python
# 修改前
def _compute_importance_weights(global_model, weight_name, bias_name, device):

# 修改后
def _compute_importance_weights(global_model, weight_name, bias_name, device,
                                 mode="soft", topk_prop=0.1):
```

### 软加权路径（对应论文 `importanceFeatureMapLocal`）

```
1. 取 global_model.state_dict()[weight_name]，reshape 为 [n_classes, class_d]
2. 对每个类别 i：
     a. M[i] = |M[i] - mean(M[i])|       # 去均值后取绝对值，衡量偏离类内均值的程度
     b. M[i] = M[i] / sum(M[i])          # 按类归一化，使每类权重和为 1
     c. k = int(class_d * topk_prop)
        将 M[i] 中最小的 (class_d - k) 个位置置 0（只保留 top-k）
3. flatten 回 [n_classes * class_d]
4. 若有 bias：在末尾拼接全 1 向量（bias 不做重要性过滤，直接保留）
```

### 硬过滤路径（对应论文 `importanceFeatureHard`）

```
1. 取 global_model.state_dict()[weight_name]，reshape 为 [n_classes, class_d]
2. 对每个类别 i：
     a. k = int(class_d * topk_prop)
     b. 取 M[i] 中最大的 k 个位置，对应位置置 1，其余置 0（二值 mask）
3. flatten 回 [n_classes * class_d]
4. 若有 bias：在末尾拼接全 1 向量
```

### n_classes 推断

```python
weight_tensor = global_model.state_dict()[weight_name]
n_classes = weight_tensor.shape[0]   # nn.Linear 的 weight shape 为 [out_features, in_features]
```

> 对卷积输出层（如 SqueezeNet 的 1×1 conv），先 flatten 再按 `n_classes` 等分。

---

## Step 2：修改 `__init__` 参数

```python
# 修改前
def __init__(self, ..., importance_hard_percentile: float = 0.1, ...):
    self.importance_hard_percentile = importance_hard_percentile

# 修改后
def __init__(self, ..., topk_prop: float = 0.1, ...):
    self.topk_prop = topk_prop
```

`topk_prop` 含义：每个类别中保留的特征比例。论文 Figure 10 显示，该值低于 0.1（10%）时智能扰动攻击失效，防御生效。

---

## Step 3：bias 的处理方式

官方代码的重要性计算**只针对 weight**，bias 不参与过滤。当前复现代码将 weight 和 bias 拼接后统一处理，为保持接口不变，采用以下方案：

**选项 B（推荐）**：importance 向量中 bias 对应位置填全 1，使 bias 不受过滤影响。

```
importance_weights = [weight部分的重要性向量] ++ [全1向量，长度等于bias维度]
```

此方案改动最小，不破坏现有 `_extract_output_layer_features` 接口。

---

## Step 4：修改 `screen` 中的调用点

```python
# 修改前
importance = _compute_importance_weights(global_model, weight_name, bias_name, device)
mask = torch.ones_like(importance, ...)
if self.importance_hard:
    threshold = torch.quantile(importance.float(), self.importance_hard_percentile)
    mask = (importance >= threshold).float()
w = (importance + 1e-8) if self.importance_soft else torch.ones_like(importance, ...)
weight_and_mask = w * mask
feature_vectors = [fv * weight_and_mask for fv in feature_vectors]

# 修改后
if self.importance_soft or self.importance_hard:
    importance_weights = _compute_importance_weights(
        global_model, weight_name, bias_name, device,
        mode="soft" if self.importance_soft else "hard",
        topk_prop=self.topk_prop,
    )
    feature_vectors = [fv * importance_weights for fv in feature_vectors]
```

soft 与 hard 的结果在函数内部统一为一个权重向量，外部调用无需区分。

---

## 修改影响范围汇总

| 修改点 | 涉及位置 | 改动量 |
|--------|----------|--------|
| 重写软加权逻辑（去均值 + 按类归一化 + top-k） | `_compute_importance_weights` | 大（核心改动） |
| 重写硬过滤逻辑（按类二值 mask） | `_compute_importance_weights` | 大（核心改动） |
| bias 对应位置补全 1 | `_compute_importance_weights` | 小 |
| `topk_prop` 替换 `importance_hard_percentile` | `__init__` | 小 |
| 统一调用接口，移除外部 mask 逻辑 | `screen` Step 1.5 | 小 |

---

## 注意事项

- 软加权和硬过滤**不应同时启用**，二者互斥（与官方代码一致）；若同时传入 `importance_soft=True, importance_hard=True`，建议以 soft 优先或抛出异常。
- `topk_prop` 过小（< 0.01）时，论文 Figure 10 显示诚实客户端也可能被误判为相似，导致误伤率上升，需在实验中根据数据集调整。
- 当 `n_classes=1` 或输出层维度无法被 `n_classes` 整除时，需做异常处理后回退到全局归一化。
