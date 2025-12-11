import os
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# 尝试导入 wandb，如果没有安装则降级处理
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

class Logger:
    """
    统一日志管理器。
    职责：
    1. 文本日志 (Console + File)
    2. 指标可视化 (WandB / TensorBoard)
    3. 配置保存 (Config Dump)
    """
    def __init__(self, 
                 project_name: str, 
                 experiment_name: str, 
                 config: Dict[str, Any], 
                 log_root: str = "./logs",
                 use_wandb: bool = True):
        
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        
        # 1. 创建实验目录
        # 格式: logs/project/exp_name_timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(log_root, project_name, f"{experiment_name}_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        
        # 2. 初始化文本日志 (Python logging)
        self._setup_text_logging()
        
        # 3. 初始化 WandB
        if self.use_wandb:
            self._setup_wandb()
        elif use_wandb and not WANDB_AVAILABLE:
            self.info("Warning: WandB is enabled but not installed. Skipping.")

        # 4. 保存 Config
        self._save_config()
        
        self.info(f"Logger initialized. Run Dir: {self.run_dir}")

    def _setup_text_logging(self):
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # 格式
        formatter = logging.Formatter(
            fmt='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler 1: 文件
        file_handler = logging.FileHandler(os.path.join(self.run_dir, "run.log"), mode='w')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Handler 2: 控制台 (Stdout)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_wandb(self):
        # 初始化 WandB
        wandb.init(
            project=self.project_name,
            name=self.experiment_name,
            config=self.config,
            dir=self.run_dir,
            reinit=True
        )

    def _save_config(self):
        path = os.path.join(self.run_dir, "config.json")
        # 处理 config 中可能包含的不可序列化对象 (如 class, function)
        # 这里做一个简单的过滤，只保存基础类型
        serializable_config = {}
        for k, v in self.config.items():
            if isinstance(v, (int, float, str, bool, list, dict, type(None))):
                serializable_config[k] = v
            else:
                serializable_config[k] = str(v) # 强转字符串
                
        with open(path, 'w') as f:
            json.dump(serializable_config, f, indent=4)

    # --- 公共接口 ---

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        记录指标 (Loss, Acc, etc.)
        Args:
            metrics: 字典，如 {'train_loss': 0.5, 'test_acc': 0.8}
            step: 当前轮次 (Round Index)
        """
        # 1. 文本日志 (可选，避免刷屏只打印特定 key)
        # log_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        # self.logger.info(f"[Metrics Step {step}] {log_str}")
        
        # 2. WandB
        if self.use_wandb:
            wandb.log(metrics, step=step)

    def close(self):
        if self.use_wandb:
            wandb.finish()
        # 关闭 logging handlers
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)