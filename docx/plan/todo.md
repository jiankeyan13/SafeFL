# Todo

## HeteroRunner proxy 开关适配

- 要做什么: 后续评估是否让 `extension/HFL/hetero_runner.py` 中的 `_setup_global_proxy_loader()` 遵守 `data.enable_proxy` 开关。若启用, 才重建全局 proxy loader；若关闭, 不覆盖 core 的 `proxy_loader` 状态。
- 在什么改动后提出的: 在 core 中规划新增默认关闭的 `data.enable_proxy` 开关后提出。该开关用于控制是否从训练集划出每类 10 条 proxy 样本, 以及是否通过 proxy loader 触发 server 端 BN calibration。
- 为什么要做: `HeteroRunner` 当前会无条件调用 `_setup_global_proxy_loader()`, 可能绕过 core 层默认关闭 proxy/BN calibration 的语义。暂时不修改 HFL 路径, 先记录为后续兼容事项, 避免本次改动扩大范围。
