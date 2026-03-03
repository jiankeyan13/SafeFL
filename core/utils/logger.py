"""
logger.py — SafeFL 统一日志管理器

设计原则：
  1. 构造与启动分离：Logger(...) 仅设置参数，start() 才真正触发 IO / 网络
  2. 支持 Context Manager，保证 close() 在异常时仍被执行
  3. 后端抽象：MetricBackend protocol，WandB / TensorBoard / CSV 可插拔
  4. logging 命名唯一化，避免多实验共进程时 handler 重复追加
  5. log_level 可配置，无 WandB 时指标回退打印到控制台
  6. 各子组件可通过 logging.getLogger(__name__) 对接，无需持有 Logger 实例
"""

from __future__ import annotations

import csv
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.utils.configs import LoggerConfig
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from io import TextIOWrapper
from typing import IO, Any, Dict, List, Optional

# ──────────────────────────────────────────────────────────────────
# 可选依赖
# ──────────────────────────────────────────────────────────────────
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    SummaryWriter = None  # type: ignore[assignment,misc]


# ══════════════════════════════════════════════════════════════════
# MetricBackend — 可插拔指标后端抽象
# ══════════════════════════════════════════════════════════════════

class MetricBackend(ABC):
    """所有指标后端须实现的最小接口。"""

    @abstractmethod
    def start(self) -> None:
        """触发真正的 IO / 网络初始化，由 Logger.start() 统一调用。"""
        ...

    @abstractmethod
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class WandBBackend(MetricBackend):
    """
    WandB 后端。
    wandb.init() 在 start() 阶段才被调用，构造时不触发任何网络请求。
    """

    def __init__(
        self,
        project: str,
        name: str,
        config: Dict[str, Any],
        run_dir: str,
        reinit: bool = True,
    ) -> None:
        self._project = project
        self._name = name
        self._config = config
        self._run_dir = run_dir
        self._reinit = reinit
        self._run: Any = None

    def start(self) -> None:
        if not WANDB_AVAILABLE:
            raise RuntimeError(
                "WandBBackend: wandb 未安装，请执行 `pip install wandb`。"
            )
        self._run = wandb.init(
            project=self._project,
            name=self._name,
            config=self._config,
            dir=self._run_dir,
            reinit=self._reinit,
        )

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._run is not None:
            wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._run is not None:
            wandb.finish()
            self._run = None

    @property
    def run(self) -> Any:
        """返回 wandb run 对象，供需要直接调用 wandb API 的场景使用。"""
        return self._run


class TensorBoardBackend(MetricBackend):
    """TensorBoard 后端，使用 torch.utils.tensorboard.SummaryWriter。"""

    def __init__(self, log_dir: str) -> None:
        self._log_dir = log_dir
        self._writer: Optional[Any] = None  # SummaryWriter | None

    def start(self) -> None:
        if not TB_AVAILABLE or SummaryWriter is None:
            raise RuntimeError(
                "TensorBoardBackend: tensorboard 未安装，"
                "请执行 `pip install tensorboard`。"
            )
        self._writer = SummaryWriter(log_dir=self._log_dir)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._writer is None:
            return
        for k, v in metrics.items():
            self._writer.add_scalar(k, v, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class CSVBackend(MetricBackend):
    """将指标追加写入 CSV 文件，适合离线分析。"""

    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path
        self._file: Optional[IO[str]] = None
        self._writer: Optional[csv.DictWriter] = None  # type: ignore[type-arg]
        self._fieldnames: List[str] = ["step"]
        self._header_written: bool = False

    def start(self) -> None:
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._file is None:
            return
        row: Dict[str, Any] = {"step": step, **metrics}
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys:
            self._fieldnames.extend(new_keys)
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        if not self._header_written and self._writer is not None:
            self._writer.writeheader()
            self._header_written = True
        if self._writer is not None:
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class ConsoleBackend(MetricBackend):
    """
    纯控制台后端（兜底）。
    当没有任何可视化工具时，将指标以 INFO 级别打印到 stdout。
    """

    def __init__(self, logger_name: str) -> None:
        self._log = logging.getLogger(logger_name)

    def start(self) -> None:
        pass  # 无需初始化

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        step_str = f"[Step {step}] " if step is not None else ""
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._log.info(f"[Metrics] {step_str}{metric_str}")

    def close(self) -> None:
        pass


# ══════════════════════════════════════════════════════════════════
# Logger — 框架统一日志管理器
# ══════════════════════════════════════════════════════════════════

class Logger:
    """
    SafeFL 统一日志管理器。

    职责：
      1. 文本日志 (Console + File)，通过标准 logging 模块透传给子组件
      2. 指标可视化，通过可插拔 MetricBackend（WandB / TensorBoard / CSV / Console）
      3. 实验配置持久化 (config.json)

    用法（推荐 Context Manager）：
    ```python
    with Logger.from_config(config) as logger:
        runner.run()    # 任何异常都能保证 close() 被调用
    ```

    用法（手动管理）：
    ```python
    logger = Logger.from_config(config)
    logger.start()
    ...
    logger.close()
    ```
    """

    # ------------------------------------------------------------------
    # 工厂方法 — 从框架 config 字典或 LoggerConfig 快速构建
    # ------------------------------------------------------------------

    @classmethod
    def from_logger_config(
        cls,
        logger_config: "LoggerConfig",
        full_config: Dict[str, Any],
    ) -> "Logger":
        """
        从 LoggerConfig 构建 Logger, 不触发任何 IO.

        Args:
            logger_config: 来自 GlobalConfig.logger_config
            full_config: 完整实验配置 (用于 config.json 持久化)
        """
        d = logger_config.to_dict()
        d.update(full_config)  # 合并以便 from_config 读取
        return cls.from_config(d)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Logger":
        """
        从框架 config 字典构建 Logger，不触发任何 IO。

        config 支持的相关字段：
          - project (str):          项目名，默认 "FL_Project"
          - name (str):             实验名，默认 "experiment"
          - log_root (str):         日志根目录，默认 "./logs"
          - log_level (str):        日志等级，默认 "INFO"
          - use_wandb (bool):       是否启用 WandB，默认 False
          - use_tensorboard (bool): 是否启用 TensorBoard，默认 False
          - use_csv (bool):         是否启用 CSV 记录，默认 False
          - console_metrics (bool): 无可视化后端时是否在控制台打印指标，默认 True
        """
        return cls(
            project_name=config.get("project", "FL_Project"),
            experiment_name=config.get("name", "experiment"),
            config=config,\
            log_root=config.get("log_root", "./logs"),
            log_level=config.get("log_level", "INFO"),
            use_wandb=config.get("use_wandb", False),
            use_tensorboard=config.get("use_tensorboard", False),
            use_csv=config.get("use_csv", False),
            console_metrics=config.get("console_metrics", True),
        )

    # ------------------------------------------------------------------
    # 构造函数 — 纯参数设置，无任何 IO / 网络副作用
    # ------------------------------------------------------------------

    def __init__(
        self,
        project_name: str,
        experiment_name: str,
        config: Dict[str, Any],
        log_root: str = "./logs",
        log_level: str = "INFO",
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        use_csv: bool = False,
        console_metrics: bool = True,
    ) -> None:
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self._log_level: int = getattr(logging, log_level.upper(), logging.INFO)
        self._console_metrics = console_metrics

        # 实验目录（不在构造函数中创建，等 start() 再创建）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            log_root, project_name, f"{experiment_name}_{timestamp}"
        )

        # 唯一 logger name，避免 getLogger 全局复用导致 handler 叠加
        self._logger_name = f"safefl.{project_name}.{experiment_name}.{timestamp}"

        # 已激活的后端（start() 后填充）
        self._backends: List[MetricBackend] = []
        # 待启动后端（构造时登记，start() 时依次调用 .start()）
        self._pending_backends: List[MetricBackend] = []
        # 降级警告标志（start() 后打印）
        self._wandb_warn: bool = False
        self._tb_warn: bool = False

        # 登记各后端
        if use_wandb:
            if WANDB_AVAILABLE:
                self._pending_backends.append(
                    WandBBackend(
                        project=project_name,
                        name=experiment_name,
                        config=config,
                        run_dir=self.run_dir,
                    )
                )
            else:
                self._wandb_warn = True

        if use_tensorboard:
            if TB_AVAILABLE:
                self._pending_backends.append(
                    TensorBoardBackend(log_dir=os.path.join(self.run_dir, "tb"))
                )
            else:
                self._tb_warn = True

        if use_csv:
            self._pending_backends.append(
                CSVBackend(csv_path=os.path.join(self.run_dir, "metrics.csv"))
            )

        # 控制台兜底后端（只在无其他后端时激活）
        self._console_backend: Optional[ConsoleBackend] = (
            ConsoleBackend(self._logger_name) if console_metrics else None
        )

        # Python logging.Logger（start() 后赋值）
        self._logger: Optional[logging.Logger] = None
        self._started: bool = False

    # ------------------------------------------------------------------
    # 启动 — 真正触发 IO / 网络的时机
    # ------------------------------------------------------------------

    def start(self) -> "Logger":
        """
        显式启动：创建目录、初始化 handlers、启动各 MetricBackend。
        返回 self，支持链式调用：logger = Logger(...).start()
        """
        if self._started:
            return self

        # 1. 创建实验目录
        os.makedirs(self.run_dir, exist_ok=True)

        # 2. 初始化 Python logging
        self._setup_text_logging()

        # 3. 启动各可视化后端（wandb.init 等网络请求在此发生）
        for backend in self._pending_backends:
            backend.start()
            self._backends.append(backend)

        # 4. 若无可视化后端，激活控制台兜底
        if not self._backends and self._console_backend is not None:
            self._console_backend.start()
            self._backends.append(self._console_backend)

        # 5. 降级警告（logging 就绪后才打印）
        if self._wandb_warn:
            self.warning(
                "use_wandb=True 但 wandb 未安装，指标将回退到控制台。"
                "安装命令: pip install wandb"
            )
        if self._tb_warn:
            self.warning(
                "use_tensorboard=True 但 tensorboard 未安装。"
                "安装命令: pip install tensorboard"
            )

        # 6. 持久化实验配置
        self._save_config()

        self._started = True
        self.info(f"Logger started. Run directory: {self.run_dir}")
        return self

    def _setup_text_logging(self) -> None:
        """初始化 Python logging，使用时间戳保证 logger name 唯一。"""
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(self._log_level)
        # 不向 root logger 传播，防止重复打印
        logger.propagate = False

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Handler 1: 文件
        file_handler = logging.FileHandler(
            os.path.join(self.run_dir, "run.log"), mode="w", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Handler 2: 控制台 (stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._logger = logger

    def _save_config(self) -> None:
        """将实验配置序列化为 config.json 存入实验目录。"""
        path = os.path.join(self.run_dir, "config.json")
        serializable: Dict[str, Any] = {}
        for k, v in self.config.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable[k] = v
            else:
                serializable[k] = str(v)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 公共文本日志接口
    # ------------------------------------------------------------------

    def _emit(self, level: int, msg: str) -> None:
        """内部统一分发，_logger 未就绪时打印到 stderr 以防信息丢失。"""
        if self._logger is not None:
            self._logger.log(level, msg)
        else:
            print(f"[PRE-START][{logging.getLevelName(level)}] {msg}", file=sys.stderr)

    def debug(self, msg: str) -> None:
        self._emit(logging.DEBUG, msg)

    def info(self, msg: str) -> None:
        self._emit(logging.INFO, msg)

    def warning(self, msg: str) -> None:
        self._emit(logging.WARNING, msg)

    def error(self, msg: str) -> None:
        self._emit(logging.ERROR, msg)

    def critical(self, msg: str) -> None:
        self._emit(logging.CRITICAL, msg)

    # ------------------------------------------------------------------
    # 公共指标接口
    # ------------------------------------------------------------------

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        *,
        always_print: bool = False,
    ) -> None:
        """
        记录指标 (Loss, Acc, ASR 等) 到所有已激活的后端。

        Args:
            metrics:       字典，如 {'train_loss': 0.5, 'test_acc': 0.8}
            step:          当前轮次 (Round Index)
            always_print:  强制在控制台打印指标（即使已有可视化后端）
        """
        if not metrics:
            return

        for backend in self._backends:
            backend.log(metrics, step=step)

        # 有可视化后端但仍希望控制台回显时
        if always_print and self._console_backend is not None:
            if self._console_backend not in self._backends:
                self._console_backend.log(metrics, step=step)

    # ------------------------------------------------------------------
    # WandB 专属接口
    # ------------------------------------------------------------------

    @property
    def wandb_run(self) -> Any:
        """
        返回当前活跃的 wandb run 对象，若未启用则返回 None。
        用于需要直接调用 wandb API（如 wandb.watch、wandb.save）的场景。
        """
        for b in self._backends:
            if isinstance(b, WandBBackend):
                return b.run
        return None

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭所有后端与 logging handlers，释放资源。"""
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                pass
        self._backends.clear()

        if self._logger is not None:
            for handler in list(self._logger.handlers):
                handler.flush()
                handler.close()
                self._logger.removeHandler(handler)
            self._logger = None

        self._started = False

    def __enter__(self) -> "Logger":
        return self.start()

    def __exit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> bool:
        self.close()
        return False  # 不吞掉异常