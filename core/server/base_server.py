import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import random

from .screener.base_screener import BaseScreener
from .aggregator.base_aggregator import BaseAggregator
from .aggregator.avg_aggregator import AvgAggregator
from .updater.base_updater import BaseUpdater
from torch.utils.data import DataLoader

class BaseServer:
    """
    联邦学习服务器基类。
    职责：
    1. 管理全局模型状态
    2. 调度训练流程 (Select -> Broadcast -> Step)
    3. 执行防御流水线 (Screen -> Aggregate -> Update)
    4. 全局评估
    """

    def __init__(self, 
                 model: nn.Module, 
                 device: torch.device,
                 screener: Optional[BaseScreener] = None,
                 aggregator: Optional[BaseAggregator] = None,
                 updater: Optional[BaseUpdater] = None,
                 test_loader: Optional[DataLoader] = None,
                 seed: int = 42,
                 ):
        """
        Args:
            model: 全局模型实例
            test_loader: 服务器持有的全局测试集 (用于 evaluate_global)
            device: 运行设备
            screener: 筛选器 (e.g., Krum, MARS) - 负责剔除恶意更新
            aggregator: 聚合器 (e.g., FedAvg, Median) - 负责数学平均
            updater: 更新器 (e.g., Standard, RLR) - 负责将聚合结果应用到模型
        """
        self.global_model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.rng = random.Random(seed)

        self.screener = screener
        self.aggregator = aggregator or AvgAggregator() #默认使用 FedAvg 聚合
        self.updater = updater or BaseUpdater()


    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数副本 (CPU)"""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def save_checkpoint(self, path: str, round_idx: int):
        """保存检查点"""
        torch.save({
            'round': round_idx,
            'model_state_dict': self.global_model.state_dict(),
        }, path)

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
        """
        # deepcopy 防止 Client 训练时意外修改了 Server 的原版模型
        global_weights = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

        package = {client_id: global_weights for client_id in selected_clients}
        
        return package

    def step(self, updates: List[Dict[str, Any]], proxy_loader: Optional[DataLoader] = None):
        """
        [核心防御流水线] 处理客户端上传的更新。
        proxy_loader: 如果提供，将用于BN校准
        """

        # 输入：原始更新列表 -> 输出：清洗后的更新列表 (可能变短)
        if self.screener:
            # 传入全局模型作为参考 (某些防御如 FLTrust 需要)
            updates = self.screener.screen(updates, self.global_model)

        client_weights = [up['weights'] for up in updates]
        num_samples = [up['num_samples'] for up in updates]
        # 输入：更新列表 -> 输出：聚合后的权重 (weights dict)
        aggregated_weights = self.aggregator.aggregate(updates=client_weights, weights=num_samples)

        # 输入：聚合结果 + 当前模型 -> 输出：原地修改模型
        self.updater.update(self.global_model, aggregated_weights, calibration_loader=proxy_loader, device=self.device)

    def eval(self, metrics: List[Callable], dataloader=None) -> Dict[str, float]:
        """
        在服务器持有的测试集上评估全局模型。
        """
        self.global_model.eval()
        device = self.device
        
        loader_to_use = dataloader if dataloader is not None else self.test_loader

        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in loader_to_use:
                data, target = data.to(device), target.to(device)
                output = self.global_model(data)
                
                all_preds.append(output.cpu())
                all_targets.append(target.cpu())
                
        if len(all_preds) == 0:
            return {}
            
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        
        results = {}
        for metric in metrics:
            results[metric.name] = metric(all_preds, all_targets)
            
        return results
    
 ###################################
    def aggregate_metrics(self, client_payloads):
        """
        聚合客户端返回的 metrics (如 train_loss)。
        通常是按样本数加权平均。
        """
        total_samples = sum(p['num_samples'] for p in client_payloads)
        if total_samples == 0:
            return {}

        aggregated = {}
        # 假设所有 client 返回的 metrics 键值都一样
        first_metrics = client_payloads[0]['metrics']
        
        for key in first_metrics.keys():
            weighted_sum = sum(p['metrics'][key] * p['num_samples'] for p in client_payloads)
            aggregated[f"avg_{key}"] = weighted_sum / total_samples
            
        return aggregated