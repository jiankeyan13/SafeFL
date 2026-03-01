import math
from typing import Any, Dict, List, Optional

import numpy as np


class CapManager:
    """
    管理客户端能力值 p 的分配与查询。
    由 runner 传入异构配置和 seed,支持 uniform 与 beta 区间采样。
    bucket 规则固定为 k/8 向上取整，用于 hetero_resnet 的宽度缩放。
    """

    BUCKET_STEP = 1.0 / 8.0

    def __init__(self, config: Dict[str, Any], seed: int, sort_client_ids: bool = True):
        self.config = config
        self.seed = int(seed)
        self.sort_client_ids = bool(sort_client_ids)
        self.rng = np.random.default_rng(self.seed)
        self.capabilities_map: Dict[str, float] = {}

        self.sample_mode = config.get("sample")
        self.p_list = config.get("p_list")

        if not isinstance(self.p_list, (list, tuple)) or len(self.p_list) == 0:
            raise ValueError("CapManager requires a non-empty config['p_list'].")

        if self.sample_mode not in ("uniform", "beta"):
            raise ValueError("CapManager config['sample'] must be 'uniform' or 'beta'.")

        self.alpha: Optional[float] = None
        self.beta: Optional[float] = None
        if self.sample_mode == "beta":
            if len(self.p_list) < 2:
                raise ValueError("Beta sampling requires config['p_list'] like [low, high].")
            self.alpha = float(config.get("alpha", np.nan))
            self.beta = float(config.get("beta", np.nan))
            if not np.isfinite(self.alpha) or not np.isfinite(self.beta) or self.alpha <= 0 or self.beta <= 0:
                raise ValueError("Beta sampling requires positive config['alpha'] and config['beta'].")

    # ------------------------------------------------------------------ #
    # 注册与采样
    # ------------------------------------------------------------------ #
    def register_clients(self, all_clients: List[str]) -> None:
        """
        为传入的客户端分配能力 p 并建立映射表。
        uniform: 严格均匀配额（balanced_quota）后再打乱。
        beta: 在 [low, high] 内按 Beta(alpha, beta) 采样。
        """
        if not all_clients:
            self.capabilities_map = {}
            return

        clients = sorted(all_clients) if self.sort_client_ids else list(all_clients)
        n = len(clients)

        if self.sample_mode == "uniform":
            m = len(self.p_list)
            base = n // m
            remainder = n % m

            allocation: List[float] = []
            for p in self.p_list:
                allocation.extend([float(p)] * base)

            if remainder > 0:
                extra_idx = self.rng.choice(len(self.p_list), size=remainder, replace=False)
                for idx in extra_idx:
                    allocation.append(float(self.p_list[idx]))

            # 防御性：长度不足时补齐（理论上不会发生）
            while len(allocation) < n:
                allocation.append(float(self.rng.choice(self.p_list)))
            allocation = allocation[:n]

            allocation = list(self.rng.permutation(allocation))
        else:  # beta
            low, high = sorted(self.p_list[:2])
            allocation = []
            for _ in range(n):
                x = self.rng.beta(self.alpha, self.beta)  # type: ignore[arg-type]
                p = float(low + x * (high - low))
                allocation.append(p)

        self.capabilities_map = {cid: float(p) for cid, p in zip(clients, allocation)}

    # 查询接口
    def get_capability(self, client_id: str) -> float:
        return self.capabilities_map[client_id]

    def get_capabilities(self, client_ids: List[str]) -> List[float]:
        return [self.get_capability(cid) for cid in client_ids]

    def bucket_p(self, p: float) -> float:
        bucket = math.ceil(p / self.BUCKET_STEP) * self.BUCKET_STEP
        return round(bucket, 10)

    def p_to_w(self, p: float) -> Dict[str, float]:
        return {"width": self.bucket_p(p)}

    def get_bucketed_capability(self, client_id: str) -> float:
        return self.bucket_p(self.get_capability(client_id))

    def get_width(self, client_id: str) -> float:
        return self.get_bucketed_capability(client_id)

    def get_widths(self, client_ids: List[str]) -> List[float]:
        return [self.get_width(cid) for cid in client_ids]

    # ------------------------------------------------------------------ #
    # 辅助：总结与状态
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        if not self.capabilities_map:
            return {"num_clients": 0, "sample": self.sample_mode}

        vals = np.array(list(self.capabilities_map.values()), dtype=float)
        bucket_counts: Dict[float, int] = {}
        for p in vals:
            b = self.bucket_p(float(p))
            bucket_counts[b] = bucket_counts.get(b, 0) + 1

        return {
            "num_clients": len(vals),
            "sample": self.sample_mode,
            "p_list": list(self.p_list),
            "alpha": self.alpha,
            "beta": self.beta,
            "seed": self.seed,
            "min_p": float(vals.min()),
            "max_p": float(vals.max()),
            "mean_p": float(vals.mean()),
            "bucket_counts": bucket_counts,
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "capabilities_map": self.capabilities_map,
            "seed": self.seed,
            "sort_client_ids": self.sort_client_ids,
            "sample_mode": self.sample_mode,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.config = state.get("config", self.config)
        self.seed = int(state.get("seed", self.seed))
        self.sort_client_ids = bool(state.get("sort_client_ids", self.sort_client_ids))
        self.sample_mode = state.get("sample_mode", self.sample_mode)
        self.p_list = self.config.get("p_list", self.p_list)
        self.alpha = self.config.get("alpha", self.alpha)
        self.beta = self.config.get("beta", self.beta)
        self.rng = np.random.default_rng(self.seed)
        saved_map = state.get("capabilities_map", {})
        self.capabilities_map = {cid: float(p) for cid, p in saved_map.items()}

if __name__ == "__main__":
    clients = [f"client_{i}" for i in range(10)]

    # uniform + 严格均匀配额
    conf_uniform = {"sample": "uniform", "p_list": [0.5, 1.0]}
    cm_u = CapManager(conf_uniform, seed=42)
    cm_u.register_clients(clients)
    print("[uniform] summary:", cm_u.summary())
    print("[uniform] first widths:", cm_u.get_widths(clients[:5]))

    # beta 区间采样
    conf_beta = {"sample": "beta", "p_list": [0.25, 1.0], "alpha": 3.0, "beta": 3.0}
    cm_b = CapManager(conf_beta, seed=42)
    cm_b.register_clients(clients)
    print("[beta] summary:", cm_b.summary())
    print("[beta] first widths:", cm_b.get_widths(clients[:5]))
