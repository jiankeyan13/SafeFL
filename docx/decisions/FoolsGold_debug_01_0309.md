# FoolsGold Debug Review 01 03/09

文件: `core/server/screener/foolsgold.py`, `core/server/base_server.py`, `core/server/aggregator/avg_aggregator.py`

## 结论

当前实现已经接入了框架的 `筛选 -> 数学聚合 -> 精炼` 主流程:

- `BaseServer.step()` 先调用 screener 产出 `screen_scores`
- 再由 `AvgAggregator` 将 `sample_weights` 与 `screen_scores` 相乘后聚合
- 最后由 refiner 将聚合后的 `global_delta` 加回全局模型

所以从“框架接线”角度看, FoolsGold 已经被放在了正确的阶段里。

但从“算法是否按 FoolsGold 本意实现”来看, 当前实现有几处明显偏差。其中至少 3 处会直接影响防御效果, 不只是实现风格差异。

## Findings

### 1. P0: Pardoning 被实现成了对称缩放, 会错误赦免更可疑客户端

- 位置: `core/server/screener/foolsgold.py:127-132`

当前代码:

```python
v_col = v.unsqueeze(1)
v_row = v.unsqueeze(0)
ratio = torch.where(v_row > 1e-8, v_col / v_row.clamp(min=1e-8), torch.ones_like(v_row))
ratio = torch.clamp(ratio, 0.0, 1.0)
cs_pardoned = cs * torch.minimum(ratio, ratio.t())
```

标准 FoolsGold 的 pardoning 是有方向性的:

- 若 `v_j > v_i`, 仅缩小 `cs_ij`
- `cs_ji` 不应该被同样缩小

而这里用了 `torch.minimum(ratio, ratio.t())`, 结果变成双向同时缩小。这样会把本来“更可疑”的那个客户端也一起赦免掉。

影响:

- Sybil 之间的高相似度会被整体压低
- `max_sim` 被低估
- 最终 `alpha_i` 会偏高
- 防御强度明显下降

这是当前最严重的问题, 建议优先修复。

### 2. P0: 当所有客户端的 `alpha` 一样时, 实现会直接返回全 1 权重, 使 FoolsGold 失效

- 位置: `core/server/screener/foolsgold.py:140-145`, `core/server/screener/foolsgold.py:161-166`

当前逻辑有两次“区间长度为 0 就返回全 1”:

```python
if alpha_max - alpha_min > 1e-8:
    alpha_norm = (alpha_raw - alpha_min) / (alpha_max - alpha_min)
else:
    alpha_norm = torch.ones_like(alpha_raw)
```

以及:

```python
if a_max - a_min > 1e-8:
    screen_scores = ((alpha_stretched - a_min) / (a_max - a_min)).tolist()
else:
    screen_scores = [1.0] * n
```

这会在一个非常关键的场景下出错:

- 多个 Sybil 更新几乎完全一致
- 所有人的 `max_sim` 都一样高
- `alpha_raw` 全相等

此时实现不会把这些客户端压到低权重, 反而直接给 `1.0`。也就是说, 越是“成批高度相似”的攻击者, 越可能触发这个退化分支并绕过筛选。

这与 FoolsGold 的设计目标正好相反。

### 3. P1: 聚合阶段仍然乘了 `num_samples`, 与你描述的 FoolsGold 聚合公式不一致

- 位置: `core/server/base_server.py:102-107`
- 位置: `core/server/aggregator/avg_aggregator.py:27-37`

当前实际聚合公式是:

```python
combined_weights = [num_samples_i * alpha_i]
global_delta = sum_i(normalized(combined_weights)_i * delta_i)
```

而你描述的目标流程是:

```python
global_delta = sum_i(alpha_i * delta_i)
```

这两者差别不小:

- 当前实现里, 大样本量客户端即使被 FoolsGold 降权, 仍可能靠 `num_samples` 把影响力拉回来
- 算法语义从“FoolsGold learning-rate style weighting”变成了“FedAvg 权重再乘一个筛选系数”

如果你的目标是忠实复现 FoolsGold, 这里属于算法偏差, 不是小细节。

如果你是有意保留 FedAvg 的样本量加权, 那至少需要在文档和配置层明确说明: 这是 “FedAvg x FoolsGold score” 的混合变体, 不是标准 FoolsGold。

### 4. P1: 当前的归一化和 logit 拉伸不符合常见 FoolsGold 实现, 会放大排序误差

- 位置: `core/server/screener/foolsgold.py:139-158`

当前实现是:

1. 先对 `alpha_raw` 做 min-max 归一化
2. 再做一次以 `logit_center` 为中心的 sigmoid 变换
3. 最后再做一次 min-max

这个流程和常见 FoolsGold 实现差别较大。典型实现更接近:

- `alpha = 1 - max_sim`
- `alpha = alpha / max(alpha)`
- `alpha[alpha == 1] = 0.99`
- 再做 logit 风格拉伸和裁剪

当前这版的风险是:

- 最小值会被强制映射到 0
- 最大值会被强制映射到 1
- 中间客户端的相对距离会被两次 min-max 改写

结果是输出权重更像“排序拉满后的分段器”, 而不是 FoolsGold 原本那种对相似度的连续抑制。

这不一定每轮都错, 但很容易让算法行为偏离论文和公开实现。

### 5. P2: 输出层定位规则过于脆弱, 最终可能静默退化成“全部客户端权重为 1”

- 位置: `core/server/screener/foolsgold.py:22-35`
- 位置: `core/server/screener/foolsgold.py:93-95`

当前只接受:

- `fc.weight` + `fc.bias`
- `linear.weight` + `linear.bias`
- 或最后一个同时带 `weight` / `bias` 的模块

问题在于:

- 一些模型最后一层可能没有 bias
- 有些头部命名不是 `fc` / `linear`
- 有些任务模型最后可学习输出头不满足这套命名假设

一旦没命中, screener 直接 `return [1.0] * n`, 整个 FoolsGold 静默失效。

这不是性能问题, 但会显著影响可用性和调试难度。

## 性能观察

### A. O(n^2) 相似度矩阵是算法本身成本, 不是主要实现缺陷

- 位置: `core/server/screener/foolsgold.py:114-125`

`torch.mm(norms, norms.t())` 会构造完整的 `n x n` 相似度矩阵。对于 FoolsGold 这是基本成本, 不算实现失误。

只要单轮参与客户端数不是特别大, 这部分可以接受。

### B. 更值得注意的性能点是历史特征长期累积, 内存随“出现过的 client_id 数量”增长

- 位置: `core/server/screener/foolsgold.py:68-69`
- 位置: `core/server/screener/foolsgold.py:105-110`

`self._history_features` 会为每个见过的 `client_id` 永久保存一份输出层历史向量。

影响:

- 长时间训练或大规模客户端池下, 内存占用会持续增长
- 即使本轮未被采样的客户端, 历史也不会清理

如果客户端池很大, 这是需要关注的。尤其是在输出层维度较大时, 会成为稳定的常驻内存成本。

### C. 每轮都对历史向量执行 `.to(device)` 有额外复制风险

- 位置: `core/server/screener/foolsgold.py:105-106`

```python
self._history_features[cid] = self._history_features[cid].to(device) + feat
```

如果历史本来就在同一个 device 上, 这里通常只是多一次检查; 如果不在, 则会发生真实拷贝。它不是最大的热点, 但写法上不够干净。

更稳妥的做法是:

- 初始化时就统一 device / dtype
- 后续只做原位累计或同设备累计

### D. `state_dict()` 扫描输出层名称是轻微开销, 不属于显著性能问题

- 位置: `core/server/screener/foolsgold.py:24-35`

每轮扫一遍 `state_dict().keys()` 能优化, 但相比 pairwise cosine 和聚合 tensor stack, 不是主要瓶颈。

## 建议优先级

1. 先修 P0: pardoning 的对称缩放问题
2. 再修 P0: “零方差时返回全 1” 的退化逻辑
3. 明确聚合公式是否要保留 `num_samples`; 若要忠实复现 FoolsGold, 应移除这部分
4. 将权重映射过程改回更接近标准 FoolsGold 的实现
5. 再补可用性和性能改进: 输出层定位缓存、历史特征清理策略、device/dtype 一致化

## 总评

这份代码在框架接线层面是通的, `FoolsGoldScreener -> AvgAggregator -> Refiner` 的落点也合理。

但当前实现还不能算“可靠复现 FoolsGold”。最大的问题不是工程结构, 而是权重计算链路里有几处关键数学逻辑被改写了:

- pardoning 方向性丢失
- 等值退化时直接放行
- 聚合公式仍混入 `num_samples`

如果不修这些点, 实验上即使“能跑”, 也很可能测到的是一个行为偏移较大的 FoolsGold 变体。
