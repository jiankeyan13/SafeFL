# FLTrust 实现审查与改进计划 01_0309

## 结论

当前实现的主路径和 FLTrust 原始思路大体一致:

- `FLTrustScreener` 使用 `proxy_loader` 在服务端计算 reference update `delta_0`
- 对每个客户端更新计算 `TS_i = ReLU(cos(delta_i, delta_0))`
- `FLTrustAggregator` 将每个正信任分数的客户端更新缩放到与 `delta_0` 同范数
- 最终按 trust score 做加权平均

也就是说, 在以下前提同时满足时, 这套实现基本符合 FLTrust 的核心公式:

- 运行时一定提供 `proxy_loader`
- `delta_0` 的范数不是 0
- 所有客户端 `delta` 的键集合与形状都和服务端 reference update 一致

但它还不能算“严格正确”, 目前至少有 2 个会改变算法语义或引入错误结果的问题, 以及 2 个明显的性能问题。

参考的 FLTrust 核心描述可见:

- NDSS 页面: https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/
- 相关公式转述: trust score 为 ReLU-clipped cosine similarity, 聚合为 trust-score-weighted average of normalized updates

## 关键问题

### 1. P0: 缺少 `proxy_loader` 或 reference update 退化时, 实现会静默降级成 Avg/FedAvg 路径

涉及位置:

- `core/server/screener/fltrust_screener.py:109-112`
- `core/server/screener/fltrust_screener.py:127-131`
- `core/server/aggregator/fltrust_aggregator.py:36-46`

现状:

- 当 `proxy_loader is None` 时, screener 直接返回 `[1.0] * n`
- 当 `reference_norm <= eps` 时, screener 也直接返回 `[1.0] * n`
- 这两个分支都没有把 `trust_scores` / `reference_norm` 放进 `context`
- 聚合器随后进入 fallback 分支, 实际执行的是按 `sample_weights * screen_scores` 的普通加权平均

影响:

- 这时系统对外仍然叫 “fltrust”, 但实际运行的已经不是 FLTrust
- 实验配置一旦漏传 `proxy_loader`, 结果会被静默污染, 很难从日志里第一时间发现
- 这会直接影响论文复现、算法对比和安全性结论

判断:

- 这是正确性问题, 优先级最高
- 对 FLTrust 来说, root/proxy data 不是“可选优化”, 而是算法成立前提

### 2. P1: fallback 聚合分支把 `screen_scores=0` 错当成 `1.0`

涉及位置:

- `core/server/aggregator/fltrust_aggregator.py:39-40`

问题代码:

```python
combined = [s * (sc or 1.0) for s, sc in zip(sample_weights or [1.0] * len(updates), screen_scores or [1.0] * len(updates))]
```

问题本质:

- Python 里 `0` 是 falsy
- 所以当某个 `screen_score == 0` 时, `(sc or 1.0)` 会变成 `1.0`
- 结果就是“本应被筛掉的客户端”重新获得完整权重

影响:

- 只要 fallback 分支被触发, 这个 bug 就可能让被屏蔽的更新重新参与聚合
- 目前 FLTrust 自己的 fallback 返回的是全 1 分数, 所以这个问题暂时不一定暴露
- 但从实现本身看, 这是明确 bug, 不是风格问题

正确写法应是:

```python
sc if sc is not None else 1.0
```

### 3. P1: 同一批客户端更新在 screener 和 aggregator 中被重复 flatten / 拷贝 / 求范数

涉及位置:

- `core/server/screener/fltrust_screener.py:25-31`
- `core/server/screener/fltrust_screener.py:133-137`
- `core/server/aggregator/fltrust_aggregator.py:51-61`

现状:

- screener 为了计算 cosine similarity, 会对每个 `delta_i` 做一次 flatten + `torch.cat`
- aggregator 为了做 norm scaling, 又对同一份 `delta_i` 再做一次 flatten + 求范数
- `_flatten_delta()` 内部还会对每个 tensor 做 `.to(device)`, 带来额外设备搬运或至少额外分配

影响:

- 大模型、多客户端场景下, 这会产生明显的 round-time 开销
- `torch.cat` 会构造大的临时向量, 额外占显存/内存
- 这类开销不改变数学结果, 但会显著拖慢仿真

判断:

- 这是显著性能问题
- 可以通过“流式累计 dot/norm”或者“在 screener 里缓存 norm 和正样本索引”来消除重复计算

### 4. P2: 服务端 reference update 通过“原地训练 + 全量恢复”实现, 每轮都有一次完整 state clone/load

涉及位置:

- `core/server/screener/fltrust_screener.py:47-49`
- `core/server/screener/fltrust_screener.py:51-71`

现状:

- 每轮计算 `delta_0` 时, 都会先 clone 一份完整 `state_dict`
- 然后在原模型上做一次服务端训练
- 训练完再 `load_state_dict(full_initial, strict=True)` 把模型恢复回去

影响:

- 对小模型问题不大
- 对大模型来说, 每轮一次完整 clone + reload 是稳定的额外成本
- 如果后续引入更大的 backbone, 这里会成为 round latency 的固定项

判断:

- 这是性能问题, 但优先级低于前 3 项
- 它不是数学错误, 但应该被纳入后续优化计划

## 不算错误, 但需要明确的点

### 1. 当前主路径里没有使用 `sample_weights`, 这本身不一定是 bug

涉及位置:

- `core/server/aggregator/fltrust_aggregator.py:68-75`

说明:

- FLTrust 的经典公式就是对“归一化后的客户端更新”按 trust score 加权平均
- 它不等同于 FedAvg, 也不要求再乘 `num_samples`
- 所以主路径没有把 `sample_weights` 合进来, 从算法上可以成立

### 2. `algorithms/fltrust.py` 本身没有错误, 但缺少“FLTrust 必须依赖 proxy/root data”的显式约束

涉及位置:

- `algorithms/fltrust.py:15-41`

说明:

- builder 成功注册了 `fltrust` 组合
- 但从接口上看, 没有任何地方强制调用方在 round 执行时提供 `proxy_loader`
- 这正是上面 P0 问题容易静默发生的原因之一

## 改进计划

### P0: 先修正确性, 防止“名义 FLTrust, 实际 Avg”

1. 把 `proxy_loader` 视为 FLTrust 的硬前置条件
2. 当 `proxy_loader is None` 时直接抛错, 不要静默 fallback
3. 当 `reference_norm` 过小导致无法建立 trust anchor 时, 也不要直接退成 Avg
4. 至少增加一个显式配置项, 例如 `strict=True`
5. 如果确实要保留退化模式, 必须把运行模式明确记录到日志和结果文件中, 例如 `mode=fltrust_degraded`

### P1: 修 fallback 权重 bug

1. 把 `(sc or 1.0)` 改成 `sc if sc is not None else 1.0`
2. 为 fallback 分支补一个单元测试
3. 测试至少覆盖 `screen_scores=[1, 0, 0.5]` 的情况
4. 测试断言被筛掉的客户端不会重新进入聚合

### P1: 去掉重复 flatten / cat / norm 计算

1. 在 screener 内一次性计算并缓存每个客户端的:
   - `client_norm`
   - `trust_score`
   - `positive_trust_mask`
2. 将这些结果放入 `context`
3. aggregator 直接复用 `client_norm`, 不再二次 flatten
4. 将 cosine 计算改成按层流式累计:
   - `dot += sum(delta_i[k] * delta_0[k])`
   - `norm_i_sq += sum(delta_i[k] ** 2)`
   - `norm_0_sq += sum(delta_0[k] ** 2)`
5. 避免为每个客户端构造完整拼接向量

### P2: 优化 reference update 的实现路径

1. 明确 `delta_0` 的 optimizer 配置来源
2. 尽量与实验配置保持一致并写入日志
3. 评估是否可以复用服务端专用 shadow model / functional update 路径, 减少整模型 clone/load
4. 如果模型较大, 优先度会明显上升

## 建议落地顺序

1. 先修 P0: 禁止无 `proxy_loader` 时静默退化
2. 再修 P1: 修掉 `screen_scores=0` 被覆盖的 fallback bug
3. 然后做 P1 性能优化: 缓存 norm, 去掉重复 flatten / cat
4. 最后再考虑 P2: 优化 `delta_0` 计算路径

## 最终判断

一句话总结:

- 当前 FLTrust 实现的“主公式路径”基本是对的
- 但它还不够严格, 尤其是“缺少 proxy/root data 时静默退化”为普通加权平均, 这是最需要先修的地方
- 如果你要拿这版代码做正式实验, 我不建议在修掉 P0 和 fallback bug 之前直接把结果记为 FLTrust
