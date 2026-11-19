"""Lightweight, pluggable experiment logger for SafeFL v2."""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from core.utils.config import LoggerConfig

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore[import-untyped]
except ImportError:
    SummaryWriter = None  # type: ignore[assignment,misc]


class MetricBackend(ABC):
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class WandBBackend(MetricBackend):
    def __init__(self, project: str, name: str, config: Dict[str, Any], run_dir: str) -> None:
        self._project = project
        self._name = name
        self._config = config
        self._run_dir = run_dir
        self._run: Any = None

    def start(self) -> None:
        if wandb is None:
            raise RuntimeError("wandb is not installed (`pip install wandb`).")
        self._run = wandb.init(
            project=self._project,
            name=self._name,
            config=self._config,
            dir=self._run_dir,
            reinit=True,
        )

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._run is not None:
            wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._run is not None:
            wandb.finish()
            self._run = None


class TensorBoardBackend(MetricBackend):
    def __init__(self, log_dir: str) -> None:
        self._log_dir = log_dir
        self._writer: Optional[Any] = None

    def start(self) -> None:
        if SummaryWriter is None:
            raise RuntimeError("tensorboard is not installed (`pip install tensorboard`).")
        self._writer = SummaryWriter(log_dir=self._log_dir)

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, global_step=step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


class CSVBackend(MetricBackend):
    def __init__(self, csv_path: str) -> None:
        self._csv_path = csv_path
        self._file: Optional[IO[str]] = None
        self._writer: Optional[csv.DictWriter[str]] = None
        self._fieldnames: List[str] = ["step"]
        self._rows: List[Dict[str, Any]] = []

    def start(self) -> None:
        self._file = open(self._csv_path, "w", newline="", encoding="utf-8")

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if self._file is None:
            return
        row: Dict[str, Any] = {"step": step, **metrics}
        self._rows.append(row)
        new_keys = [k for k in row if k not in self._fieldnames]
        if new_keys or self._writer is None:
            self._fieldnames.extend(new_keys)
            # Rewrite header + buffered rows when new metric columns appear mid-run (e.g. start_round > 0).
            self._file.seek(0)
            self._file.truncate()
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._writer.writerows(self._rows)
        else:
            self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class ConsoleBackend(MetricBackend):
    def __init__(self, logger_name: str) -> None:
        self._log = logging.getLogger(logger_name)

    def start(self) -> None:
        pass

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        step_str = f"[Step {step}] " if step is not None else ""
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        self._log.info(f"[Metrics] {step_str}{metric_str}")

    def close(self) -> None:
        pass


class Logger:
    """Unified text + metric logger for SafeFL v2.

    When ``run_dir`` is provided explicitly (e.g. Hydra's ``runtime.output_dir``),
    logs and ``config.json`` are written directly into ``run_dir`` without creating
    redundant nested timestamp folders.
    """

    def __init__(
        self,
        cfg: "LoggerConfig",
        full_config: Dict[str, Any],
        *,
        run_dir: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.full_config = full_config
        self._log_level: int = getattr(logging, cfg.log_level.upper(), logging.INFO)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        if run_dir is not None:
            self.run_dir = os.path.abspath(run_dir)
        else:
            self.run_dir = os.path.abspath(
                os.path.join(cfg.log_root, cfg.project, f"{cfg.name}_{ts[:15]}")
            )

        self._logger_name = f"safefl.{cfg.project}.{cfg.name}.{ts}"
        self._logger: Optional[logging.Logger] = None
        self._backends: List[MetricBackend] = []
        self._console_backend: Optional[ConsoleBackend] = (
            ConsoleBackend(self._logger_name) if cfg.console_metrics else None
        )
        self._started = False

    def start(self) -> "Logger":
        if self._started:
            return self

        os.makedirs(self.run_dir, exist_ok=True)
        self._setup_text_logging()

        if self.cfg.use_wandb:
            if wandb is not None:
                b_wb = WandBBackend(
                    self.cfg.project, self.cfg.name, self.full_config, self.run_dir
                )
                b_wb.start()
                self._backends.append(b_wb)
            else:
                self.warning("use_wandb=True but wandb is not installed; skipping WandB.")

        if self.cfg.use_tensorboard:
            if SummaryWriter is not None:
                b_tb = TensorBoardBackend(os.path.join(self.run_dir, "tb"))
                b_tb.start()
                self._backends.append(b_tb)
            else:
                self.warning("use_tensorboard=True but tensorboard is not installed; skipping TB.")

        if self.cfg.use_csv:
            b_csv = CSVBackend(os.path.join(self.run_dir, "metrics.csv"))
            b_csv.start()
            self._backends.append(b_csv)

        if self._console_backend is not None:
            self._console_backend.start()
            self._backends.append(self._console_backend)

        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.full_config, f, indent=2, ensure_ascii=False, default=str)

        self._started = True
        self.info(f"Logger started | run_dir={self.run_dir}")
        return self

    def _setup_text_logging(self) -> None:
        logger = logging.getLogger(self._logger_name)
        logger.setLevel(self._log_level)
        logger.propagate = False
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(os.path.join(self.run_dir, "run.log"), mode="w", encoding="utf-8")
        fh.setFormatter(formatter)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)
        self._logger = logger

    def _emit(self, level: int, msg: str) -> None:
        if self._logger is not None:
            self._logger.log(level, msg)
        else:
            print(f"[{logging.getLevelName(level)}] {msg}", file=sys.stderr)

    def debug(self, msg: str) -> None:
        self._emit(logging.DEBUG, msg)

    def info(self, msg: str) -> None:
        self._emit(logging.INFO, msg)

    def warning(self, msg: str) -> None:
        self._emit(logging.WARNING, msg)

    def error(self, msg: str) -> None:
        self._emit(logging.ERROR, msg)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if not metrics:
            return
        for backend in self._backends:
            backend.log(metrics, step=step)

    def close(self) -> None:
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

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.close()
        return False
