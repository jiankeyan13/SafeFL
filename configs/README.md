# SafeFL Hydra 配置说明

`configs/` 使用 Hydra config groups 管理实验配置。入口是项目根目录的 `main.py`。

## 目录

```text
configs/
  config.yaml
  dataset/
  partitioner/
  model/
  algorithm/
  attack/
  training/
  logging/
  experiment/
```

## 运行

```bash
python main.py
python main.py algorithm=fltrust
python main.py experiment=vanilla_fltrust_badnets
python main.py training.rounds=10 client.lr=0.001
```

## 多实验

```bash
python main.py -m algorithm=fedavg,fltrust,flame seed=1,2,3
```

Hydra 负责 YAML 加载、配置组合、命令行覆盖和 sweep。SafeFL 内部配置结构位于 `core/config/`。
