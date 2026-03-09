# RFLBAT_debug_01_0309 — RFLBAT 实现问题分析

日期: 2026-03-09
文件: `core/server/screener/rflbat.py`, `algorithms/rflbat.py`

---

## 一、正确性问题 (Correctness Bugs)

### [BUG-1] 关键：簇选择逻辑反向 (Critical)

**位置**: `_select_low_similarity_cluster` L225

```python
best_cluster = min(cluster_scores, key=lambda cid: (cluster_scores[cid], -len(cluster_members[cid])))
```

**问题**: 代码选择了**簇内平均余弦相似度最低**的簇，视其为"良性簇"并保留。

**正确逻辑**: RFLBAT 的核心假设是良性客户端的更新方向一致，因此良性簇的**簇内余弦相似度应更高**。应选择相似度**最高**的簇，即把 `min` 改为 `max`：

```python
best_cluster = max(cluster_scores, key=lambda cid: (cluster_scores[cid], len(cluster_members[cid])))
```

当前实现将防御逻辑完全反转——它选出的是最不一致（最可疑）的簇并认定为良性，等价于没有防御甚至帮助攻击者通过筛选。

### [BUG-2] 单成员簇得分硬编码为 1.0

**位置**: `_select_low_similarity_cluster` L208-L209

```python
if len(members) <= 1:
    cluster_scores[cluster_id] = 1.0
```

**问题**: 单成员簇得分固定为 1.0（最高），这意味着即使某个孤立的攻击者单独成一簇，当前逻辑（`min`选最低）不会选它；但修正为 `max` 后，单成员攻击者会以 1.0 分竞争被选中，仍不合理。

**建议**: 单成员簇应赋予一个特殊标记（如 `NaN` 或排除在选举之外），优先选成员数量更多且相似度更高的簇；或对单成员簇赋予惩罚分（如 `0.0`），鼓励选多成员高相似簇。

### [BUG-3] `stage2_scores` 含大量 `inf` 值写入 context

**位置**: `screen` L99

```python
context["rflbat_stage2_scores"] = stage2_scores.tolist()
```

`stage2_scores` 初始化为全 `inf`，只有 `chosen_cluster_indices` 中的索引会被赋值。其余客户端的分数为 `inf`，如果下游代码做 JSON 序列化或数值处理会出错（JSON 不支持 `Infinity`）。

---

## 二、显著性能问题 (Performance Bottlenecks)

### [PERF-1] `_euclidean_filter` — O(N²·2) 广播张量

**位置**: L159

```python
pairwise = np.linalg.norm(sub_points[:, None, :] - sub_points[None, :, :], axis=2)
```

对 N 个 2D 点构造 N×N×2 中间张量，内存是 `O(N²)`。虽然已投影到 2D（dim=2），当 N 较大时仍有不必要开销。可用 `scipy.spatial.distance.cdist` 或以下等价但更高效的写法：

```python
from scipy.spatial.distance import cdist
pairwise = cdist(sub_points, sub_points)
```

### [PERF-2] `_project_to_2d` — 全量 SVD

**位置**: L137

```python
_, _, vh = np.linalg.svd(centered, full_matrices=False)
```

对 N×D 矩阵做全量截断 SVD（`full_matrices=False` 仍计算 min(N,D) 个奇异向量），但只需要前 2 个主成分。当模型参数量大（D >> 2）时，这是最主要的性能瓶颈。

**建议**: 改用随机化 SVD（只算 k=2 个分量），速度可提升数量级：

```python
from sklearn.utils.extmath import randomized_svd
_, _, vh = randomized_svd(centered, n_components=2, random_state=self.seed)
```

### [PERF-3] `_select_low_similarity_cluster` — 余弦均值计算有冗余循环

**位置**: L217-L221

```python
per_client_means = []
for row_idx in range(len(members)):
    mask = np.ones(len(members), dtype=bool)
    mask[row_idx] = False
    per_client_means.append(float(cosine[row_idx, mask].mean()))
```

每次迭代都创建一个 mask 数组，实际上排除对角线均值可向量化：

```python
N = len(members)
row_sums = cosine.sum(axis=1) - 1.0          # 减去对角线的 1.0
per_client_means = row_sums / max(N - 1, 1)
cluster_score = float(np.median(per_client_means))
```

这将 O(N²) 的 Python 循环替换为纯 numpy 向量化操作。

### [PERF-4] `_stack_vectors` — 全量参数展平后驻留内存

**位置**: L116-L127

所有客户端的完整参数向量同时存在内存中（`high_dim` 矩阵）。对参数量大的模型（如 ResNet、ViT），这会显著消耗内存。在 `_project_to_2d` 之后 `high_dim` 仍需保留用于余弦相似度计算，无法提前释放，但可考虑用 float32 代替 float64 减半内存：

当前 L127 做了 `astype(np.float64)`，若改为保持 float32 精度，内存减半，且对 PCA/余弦相似度精度影响可忽略。

---

## 三、轻微问题 (Minor)

- `_kmeans` 使用随机初始化而非 k-means++，在极端分布下收敛慢或分簇差，但因点数少（投影后 2D）影响有限。
- `algorithms/rflbat.py` 中 `screener_conf.get("params", screener_conf)` 当 `screener_conf` 本身就是参数 dict 时降级正确，但逻辑稍混乱，可能在配置格式变化时引入 bug。

---

## 修复优先级

| 优先级 | 项目 | 影响 |
|--------|------|------|
| P0 | BUG-1 簇选择方向反向 | 防御完全失效，甚至反效果 |
| P1 | PERF-2 全量 SVD | 大模型时训练变慢 |
| P1 | PERF-3 余弦均值循环 | 客户端多时显著慢 |
| P2 | BUG-2 单成员簇分数 | 边缘场景误判 |
| P2 | PERF-4 float64 内存 | 大模型内存压力 |
| P3 | BUG-3 inf 序列化 | 日志/监控问题 |
