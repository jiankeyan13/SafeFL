# FLTrust 实现审查与改进计划 02_0309

## 前置说明

本文是 `fltrust_debug_01_0309.md` 的补充审查。

第一份文档的结论基本正确, 所有 P0/P1/P2 问题经代码验证均属实, 关键判断无误。本文聚焦于原文档未覆盖的 bug, 且排除"缺少 proxy_loader 时静默降级"类安全性/容错性问题, 只讨论在 proxy_loader 正常提供时仍然存在的数学错误和逻辑错误。

---

## 对原文档的验证结论

| 原文档问题 | 验证结果 | 备注 |
|---|---|---|
| P0: 无 proxy_loader 时静默 fallback | ✓ 正确 | 本文不再展开 |
| P1: `(sc or 1.0)` 当 sc=0 时错误 | ✓ 正确 | Python falsy 逻辑 bug |
| P1: screener 和 aggregator 重复 flatten/norm | ✓ 正确 | 性能问题 |
| P2: 每轮完整 clone/load state\_dict | ✓ 正确 | 性能问题 |
| 说明项: 主路径不用 sample\_weights 符合算法 | ✓ 正确 | 非 bug |

---

## 新发现的 Bug

### Bug A (P0): aggregator 对 `ref_norm` 重复加 `eps`

涉及位置:

- `core/server/screener/fltrust_screener.py:127`
- `core/server/aggregator/fltrust_aggregator.py:49`

问题代码:

```python
# screener.py:127
ref_norm = torch.norm(flat_0).item() + self.eps   # 已经加过一次 eps
context["reference_norm"] = ref_norm              # 存入 context

# aggregator.py:49
ref_norm = float(ref_norm) + self.eps             # 从 context 取出后又加一次 eps
```

问题本质:

- screener 计算 `ref_norm` 时已经加了 `eps`, 并原封不动地存入 `context["reference_norm"]`
- aggregator 从 `context` 取出该值后, 又再次加 `eps`
- 最终用于缩放的分子是 `norm_0 + 2*eps`, 而正确值应为 `norm_0 + eps`

实际影响:

- 对于大范数的模型 (e.g., ResNet-18), `2*eps ≈ 2e-12`, 数值影响极小
- 对于近零范数场景, 分子被额外抬高, 导致 scale 偏大, 客户端更新被过度放大
- 这是一个明确的实现错误, 与算法语义不一致

修正方案:

```python
# aggregator.py:49 — 去掉二次加 eps, 直接用 context 里的值
ref_norm = float(ref_norm)   # 已经含 eps, 不再重复加
```

或统一改为: screener 存原始 norm, aggregator 加 eps。二选一, 不能两处都加。

---

### Bug B (P1): `ref_norm <= self.eps` 检查是近似死代码

涉及位置:

- `core/server/screener/fltrust_screener.py:127-129`

问题代码:

```python
ref_norm = torch.norm(flat_0).item() + self.eps   # line 127: 已加 eps
if ref_norm <= self.eps:                           # line 129: 等价于 norm <= 0
    context["fltrust_fallback"] = "zero_reference_norm"
    return [1.0] * n, context
```

问题本质:

- 第 127 行: `ref_norm = raw_norm + eps`
- 第 129 行: `ref_norm <= eps` ⟺ `raw_norm + eps <= eps` ⟺ `raw_norm <= 0`
- 由于范数非负, 这个条件只在 `raw_norm` 精确为 `0.0` 时触发

实际影响:

- 对于"近似零"的 reference norm (例如 `raw_norm = 1e-15`):
  - `ref_norm = 1e-15 + 1e-12 ≈ 1.001e-12 > 1e-12 = eps`
  - 检查不触发, 系统用接近零的方向作为信任锚点, 产生噪声极大的 trust score
- 守护条件形同虚设, 只有恰好全零梯度更新才能触发 fallback

修正方案:

```python
raw_norm = torch.norm(flat_0).item()
ref_norm = raw_norm + self.eps
if raw_norm < some_threshold:   # 在加 eps 之前检查原始 norm
    context["fltrust_fallback"] = "zero_reference_norm"
    return [1.0] * n, context
```

`some_threshold` 建议与 `eps` 解耦, 使用独立的 `min_ref_norm` 参数 (如 `1e-6`)。

---

### Bug C (P1): BN buffer 的 delta 被错误地乘以梯度幅度缩放因子

涉及位置:

- `core/server/aggregator/fltrust_aggregator.py:51-62`

问题代码:

```python
learnable_keys = self._get_learnable_keys(global_model)   # requires_grad=True 的参数
...
flat = self._flatten_update(delta, learnable_keys)         # 仅含可学习参数
norm_i = torch.norm(flat).item() + self.eps               # 范数基于可学习参数
scale = ref_norm / norm_i                                  # 缩放因子

normalized_updates.append({
    k: (v.to(device, dtype=torch.float32) * scale)        # 对 delta 所有 key 都乘 scale
    for k, v in delta.items()                             # 包括 running_mean/running_var
})
```

问题本质:

- `scale` 由可学习参数的 L2 范数算出, 语义是"把梯度方向幅度归一化到与 delta_0 相同"
- 但缩放时对 `delta.items()` 所有键逐一乘以 `scale`, 包括 BN 的 `running_mean` 和 `running_var`
- `running_mean` / `running_var` 是统计量 buffer, 不是梯度, 不应参与梯度幅度归一化
- `_get_delta_keys()` 在 screener 里之所以存在, 正是为了把这些 buffer 排除在向量之外; 但 aggregator 在缩放时并未做同样的排除

实际影响:

- BN running statistics 被错误放大或缩小, 聚合后全局模型的 BN 统计量失真
- 对 ResNet 等大量使用 BN 的模型, 每轮都会累积 BN 统计量的偏差
- 模型的 Batch Normalization 行为在多轮后会出现漂移, 影响收敛稳定性

修正方案:

缩放时只对可学习参数键应用 `scale`, BN buffer 直接保留原始 delta:

```python
normalized_updates.append({
    k: (v.to(device, dtype=torch.float32) * scale)
        if k in learnable_keys
        else v.to(device, dtype=torch.float32)
    for k, v in delta.items()
})
```

---

### Bug D (P1): `global_model` 为 None 时设备硬编码为 `"cuda"`

涉及位置:

- `core/server/aggregator/fltrust_aggregator.py:48`

问题代码:

```python
device = next(global_model.parameters()).device if global_model else torch.device("cuda")
```

问题本质:

- 当 `global_model` 为 `None` 时, fallback 到 `torch.device("cuda")`
- 在 CPU-only 环境下, 后续 `.to(device)` 调用会触发 `RuntimeError: No CUDA GPUs are available`
- 这一 fallback 分支在 FLTrust 正常路径不会触发 (trust_scores/ref_norm 均有效时 global_model 不应为 None), 但防御性编程角度仍是错误

实际影响:

- 在 CPU 实验环境 (如本地调试、CI 环境) 下, 若 global_model 为 None, 系统会崩溃
- 错误信息不直观, 难以定位

修正方案:

```python
device = next(global_model.parameters()).device if global_model else torch.device("cpu")
```

或更稳健的写法:

```python
if global_model is None:
    raise ValueError("FLTrustAggregator requires global_model when trust_scores are available")
device = next(global_model.parameters()).device
```

---

### Bug E (P2): screener 与 aggregator 使用不同的键过滤方法计算范数

涉及位置:

- `core/server/screener/fltrust_screener.py:15-22` (`_get_delta_keys`)
- `core/server/aggregator/fltrust_aggregator.py:51` (`_get_learnable_keys`)

两种过滤方式:

```python
# screener: 基于 key 名称模式排除 BN buffer
def _get_delta_keys(state_dict):
    return {k for k in state_dict if "num_batches_tracked" not in k
            and not k.endswith("running_mean")
            and not k.endswith("running_var")}

# aggregator: 基于 requires_grad 过滤
def _get_learnable_keys(global_model):
    return set(name for name, param in global_model.named_parameters()
               if param.requires_grad)
```

问题本质:

- 二者在标准 ResNet 上结果等价: BN running stats 既匹配名称模式, 又是 buffer (不在 `named_parameters()` 里)
- 但对于有冻结参数的模型 (e.g., `requires_grad=False` 的 backbone 层), 两者结果不同:
  - `_get_delta_keys` 会包含冻结层的 delta
  - `_get_learnable_keys` 不包含冻结层的参数
- 结果: screener 在计算 cosine similarity 时包含了冻结层的 delta, 但 aggregator 在计算 norm_i 时没有
- 两个 norm_i 不一致, 导致 trust score 和 scale 的数学语义错位

实际影响:

- 在当前实验 (标准 ResNet-18, 全参数可训练) 下无影响
- 引入迁移学习或部分冻结场景时, 会静默产生错误的范数缩放

建议:

统一使用同一个键集合, 推荐在两处都用 `_get_delta_keys`, 因为它不依赖 global_model 的 `requires_grad` 状态, 更直接反映 delta 的实际内容。

---

## 汇总与优先级

| 编号 | 优先级 | 文件 | 行号 | 问题类型 | 描述 |
|---|---|---|---|---|---|
| Bug A | P0 | `fltrust_aggregator.py` | 49 | 数学错误 | `ref_norm` 重复加 `eps`, 导致 scale 分子偏高 |
| Bug B | P1 | `fltrust_screener.py` | 127-129 | 逻辑错误 | `ref_norm <= eps` 检查等价于 `norm == 0`, 近似零无法保护 |
| Bug C | P1 | `fltrust_aggregator.py` | 59-62 | 语义错误 | BN buffer delta 被错误乘以梯度缩放因子 |
| Bug D | P1 | `fltrust_aggregator.py` | 48 | 运行时崩溃 | `global_model=None` 时硬编码 `cuda`, CPU 环境会 crash |
| Bug E | P2 | screener + aggregator | — | 潜在不一致 | 键过滤逻辑不同, 冻结参数场景下范数计算错位 |

---

## 建议修复顺序

1. **Bug A**: 去掉 aggregator 里的二次 `+ self.eps`, 改为直接用 context 的值
2. **Bug B**: 将 ref_norm 检查改为先检查 raw norm, 再加 eps
3. **Bug C**: 缩放时跳过 BN buffer (运行 mean/var) 的 key
4. **Bug D**: 将 CUDA fallback 改为 CPU 或直接报错
5. **Bug E**: 统一两处键过滤逻辑

---

## 关于原文档准确性

原文档对所有 P0/P1/P2 问题的描述经逐行验证均属实, 主路径正确性判断也准确。本文发现的新 bug 均在原文档审查范围之外, 不构成对原文档结论的修正, 而是补充。

其中 Bug C (BN buffer 被缩放) 在正式实验中的影响最值得关注: 它不是边缘情况, 每轮聚合都会发生, 且会在多轮后累积 BN 统计量偏差。如果当前实验结果看起来收敛但比预期差, 这是一个需要排查的候选原因。
