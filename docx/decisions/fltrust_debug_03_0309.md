# FLTrust 实现审查与改进计划 03_0309 (综合版)

## 1. 核心结论与环境前提

当前 FLTrust 实现的主路径与原始算法思路基本一致，但在正确性、数值稳定性和性能方面存在多处待修复问题。

**运行环境前提：**
- 运行时**一定提供** `proxy_loader`。
- 运行时**一定支持** `cuda`。
- 对于 `Bug A` (BN buffer 缩放问题)，选择在 `Screener` 中进行处理，并删除相关的安全性检查（如 P0 静默降级）。

---

## 2. 关键问题汇总 (P0/P1 优先级)

### 2.1 P0: 正确性与数值错误 (必须修复)

#### 问题 1: `ref_norm` 重复加 `eps` (Bug A)
- **涉及位置**: `fltrust_screener.py:127` 和 `fltrust_aggregator.py:49`
- **现象**: Screener 计算 `ref_norm` 时已加 `self.eps` 并存入 `context`；Aggregator 取出后又加了一次。
- **影响**: 导致最终缩放因子 `scale = (norm_0 + 2*eps) / (norm_i + eps)`，偏离算法语义。
- **修复方案**: 统一在一处加 `eps`。建议 Screener 存原始 norm，Aggregator 统一处理。

#### 问题 2: BN Buffer 被错误执行梯度幅度缩放 (Bug C)
- **涉及位置**: `fltrust_aggregator.py:51-62`
- **现象**: `scale` 是基于可学习参数计算的，但缩放时对 `delta.items()` 所有键（包括 `running_mean`/`running_var`）都乘了 `scale`。
- **影响**: BN 统计量失真，随训练轮数累积偏差，影响模型收敛稳定性。
- **修复方案**: 缩放时仅针对可学习参数（或排除 BN buffer 键）应用 `scale`。

#### 问题 3: `ref_norm <= eps` 检查失效 (Bug B)
- **涉及位置**: `fltrust_screener.py:127-129`
- **现象**: 先执行 `ref_norm = raw_norm + eps`，再检查 `ref_norm <= eps`。这等价于检查 `raw_norm <= 0`。
- **影响**: 对于极小的 `raw_norm`（如 `1e-15`），检查无法触发，导致后续计算产生巨大噪声。
- **修复方案**: 在加 `eps` 前检查 `raw_norm` 是否小于阈值（如 `1e-6`）。

### 2.2 P1: 逻辑漏洞与性能问题

#### 问题 4: Fallback 分支将 `screen_scores=0` 误判为 `1.0`
- **涉及位置**: `fltrust_aggregator.py:39-40`
- **现象**: 代码使用 `(sc or 1.0)`，当 `sc=0` 时 Python 判定为 falsy，导致被筛掉的客户端重新获得权重。
- **修复方案**: 改为 `sc if sc is not None else 1.0`。

#### 问题 5: 重复 Flatten/Norm 计算 (性能损耗)
- **涉及位置**: Screener 和 Aggregator 均对同一批 `delta_i` 执行 flatten 和 norm。
- **影响**: 大模型场景下产生明显的显存/内存开销和计算延迟。
- **修复方案**: 在 Screener 中计算并缓存 `client_norm` 到 `context`，Aggregator 直接复用。

#### 问题 6: 键过滤逻辑不一致 (Bug E)
- **涉及位置**: Screener（按名称过滤）与 Aggregator（按 `requires_grad` 过滤）。
- **影响**: 在模型包含冻结参数时，两处计算的向量维度不一致，导致 trust score 语义错位。
- **修复方案**: 统一使用同一种过滤逻辑（推荐按名称模式排除 BN buffer）。

---

## 3. 改进实施计划

### 第一阶段：修复数值与逻辑错误 (正确性优先)
1. **统一 `eps` 处理**: 修正 `ref_norm` 重复加 `eps` 的问题。
2. **修正 BN 缩放**: 在 Aggregator 聚合循环中增加判断，确保只缩放参数，不缩放 BN buffer。
3. **修正 Fallback 权重**: 修复 `(sc or 1.0)` 的 falsy 逻辑 bug。
4. **增强 `ref_norm` 检查**: 在加 `eps` 前进行阈值判定。

### 第二阶段：性能优化与逻辑统一
1. **缓存 Norm**: 在 Screener 计算 trust score 时顺便存下 `client_norm`，避免 Aggregator 重复计算。
2. **统一键过滤**: 确保 Screener 和 Aggregator 看到的“向量”维度完全一致。
3. **优化 `delta_0` 计算**: 减少每轮 `state_dict` 的全量 clone/load（可选）。

### 第三阶段：清理与简化
1. **移除静默降级**: 既然环境保证有 `proxy_loader`，移除 Screener 中 `proxy_loader is None` 时返回全 1 的逻辑，改为直接报错或严格执行。
2. **移除 CUDA Fallback**: 既然环境保证有 `cuda`，移除 Aggregator 中针对 `global_model is None` 的 CPU fallback 逻辑。

---

## 4. 最终状态检查清单
- [ ] `ref_norm` 是否只加了一次 `eps`？
- [ ] BN buffer (`running_mean`/`var`) 是否被排除了缩放？
- [ ] `screen_score=0` 的客户端是否真的被剔除了？
- [ ] Screener 和 Aggregator 的 `delta` 键集合是否一致？
- [ ] 是否移除了不必要的 `is None` 容错和静默降级逻辑？
