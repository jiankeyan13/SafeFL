from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from core.utils.evaluator import Evaluator

from .screener.base_screener import BaseScreener
from .aggregator.base_aggregator import BaseAggregator
from .aggregator.avg_aggregator import AvgAggregator
from .refiner.base_refiner import BaseRefiner
from torch.utils.data import DataLoader

class BaseServer:
    """
    联邦学习服务器基类。
    职责：
    1. 调度训练流程 (Select -> Broadcast -> Step)
    2. 执行防御流水线 (Screen -> Aggregate -> Base+Delta -> Refine)
    """

    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device,
                 screener: Optional[BaseScreener] = None,
                 aggregator: Optional[BaseAggregator] = None,
                 refiner: Optional[BaseRefiner] = None,
                 seed: int = 42,
                 ):
        """
        Args:
            model: 全局模型实例
            device: 运行设备
            screener: 筛选器 (e.g., Krum, HDBSCAN) - 负责剔除恶意更新
            aggregator: 聚合器 (e.g., FedAvg, Median) - 负责数学聚合 delta
            refiner: 精炼器 (e.g., Standard, Noise) - 负责将聚合结果应用到模型
        """
        self.global_model = model.to(device)
        self.device = device
        self.rng = random.Random(seed)

        self.screener = screener or BaseScreener()
        self.aggregator = aggregator or AvgAggregator()
        self.refiner = refiner or BaseRefiner()

    def select_clients(self, client_ids: List[str], num_select: int) -> List[str]:
        """
        从给定的客户端 ID 列表中随机选择指定数量的客户端。
        Args:
            client_ids: 候选客户端 ID 列表。
            num_select: 要选择的数量。
        Returns:
            被选中的客户端 ID 列表。
        """
        # 确保选择数量不超过候选总数
        num_to_sample = min(num_select, len(client_ids))
        
        selected = self.rng.sample(client_ids, num_to_sample)
        
        return selected

    def broadcast(self, selected_clients: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        向选中的客户端分发模型。
        S1 优化: 直接分发 GPU 上的 state_dict, 避免 CPU 搬运; load_state_dict 会执行数值拷贝.
        """
        global_weights = self.global_model.state_dict()

        package = {client_id: global_weights for client_id in selected_clients}
        
        return package

    def step(self, updates: List[Dict[str, Any]], proxy_loader: Optional[DataLoader] = None):
        """
        [核心防御流水线] 筛选 -> 聚合 -> Base+Delta 合成 -> 后处理。
        context 用于在各阶段之间传递信息。

        Args:
            updates: 客户端上传的 payload 列表，每项须包含 'delta' 与 'num_samples'。
            proxy_loader: 可选的代理数据加载器，用于 BN 校准等后处理。
        """
        context = {
            "proxy_loader": proxy_loader,
            "client_ids": [up.get("client_id") for up in updates],
        }
        num_samples = [up['num_samples'] for up in updates]
        client_deltas = [up['delta'] for up in updates]
        
        # 阶段1: 筛选
        screen_scores, context = self.screener.screen(
            client_deltas=client_deltas,
            num_samples=num_samples,
            global_model=self.global_model,
            context=context
        )

        # 阶段2: 聚合 → 返回纯 delta
        aggregated_delta, context = self.aggregator.aggregate(
            updates=client_deltas,
            sample_weights=num_samples,
            screen_scores=screen_scores,
            global_model=self.global_model,
            context=context
        )

        # 阶段3: Base + Delta 合成
        new_state = {}
        global_state = self.global_model.state_dict()
        for key, value in global_state.items():
            new_state[key] = value.clone()
            if key in aggregated_delta:
                new_state[key] += aggregated_delta[key].to(device=value.device, dtype=value.dtype)

        # 阶段4: 精炼（加载模型 + 可选 BN 校准 / 加噪声等）
        self.refiner.process(
            self.global_model,
            new_state,
            calibration_loader=proxy_loader,
            device=self.device,
            context=context,
        )

    def eval(
        self,
        evaluator: "Evaluator",
        dataloader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> Dict[str, float]:
        """
        将全局模型评估委托给 Evaluator，Server 不自持推理循环。

        Args:
            evaluator:  已初始化的 Evaluator 实例（含 metrics 字典）。
            dataloader: 评估数据加载器；传入毒化 loader 时 accuracy 即为 ASR。
            criterion:  损失函数（用于 AverageLoss 等需要 loss 值的指标）。
            device:     推理所在设备。

        Returns:
            指标字典，如 {"accuracy": 0.92, "loss": 0.21}。
        """
        return evaluator.evaluate(self.global_model, dataloader, criterion, device)
