# SafeFL Hydra 配置说明

`configs/` 使用 Hydra config groups 管理实验配置。入口是项目根目录的 `main.py`。

## 目录

```text
configs/
  config.yaml
  dataset/
  partitioner/
  model/
  defense/
  attack/
  training/
  logging/
  experiment/
```

## 运行

```bash
python main.py
python main.py attack=batman defense=fltrust
python main.py experiment=quick_debug attack=badnets defense=fedavg
python main.py -m experiment=paper_batman attack=badnets,batman defense=fedavg,fltrust,alignins
python main.py training.rounds=10 training.eval_interval=5 client.lr=0.001
```

> 说明：
> - `attack`、`defense`、`dataset`、`model`、`partitioner` 均正交解耦，直接在命令行组合或使用 `-m` 扫网格即可。
> - `FLTrust` 等需要服务端根数据的防御声明了 `requires_proxy = True`，选中 `defense=fltrust` 时框架会自动开启 `data.enable_proxy=true`。
> - `configs/experiment/` 仅存放**训练协议 / 调试场景 Preset**（如 `quick_debug`、`paper_batman`），不再维护 `attack × defense` 笛卡尔积文件。

## 多实验

```bash
python main.py -m defense=fedavg,fltrust,flame seed=1,2,3
```

Hydra 负责 YAML 加载、配置组合、命令行覆盖和 sweep。SafeFL 内部配置结构位于 `core/utils/config.py`。
