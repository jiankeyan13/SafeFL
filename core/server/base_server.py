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
                 test_loader: Any, 
                 device: torch.device,
                 client_ids: List[str],
                 attacker_ids: List[str] = [],
                 seed: int = 42,
                 screener: Optional[BaseScreener] = None,
                 aggregator: Optional[BaseAggregator] = None,
                 updater: Optional[BaseUpdater] = None):
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

        self.all_clients = client_ids
        self.attacker_ids = set(attacker_ids) # 用 set 查找更快
        self.benign_ids = list(set(client_ids) - self.attacker_ids)
        self.attacker_list = list(self.attacker_ids) # 用于 sample

        self.screener = screener
        self.aggregator = aggregator or AvgAggregator() #默认使用 FedAvg 聚合
        self.updater = updater

    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数副本 (CPU)"""
        return {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

    def save_checkpoint(self, path: str, round_idx: int):
        """保存检查点"""
        torch.save({
            'round': round_idx,
            'model_state_dict': self.global_model.state_dict(),
        }, path)

    def select_clients(self, ratio, attacker_ratio=0.0):
        num_total_select = int(len(self.all_clients) * ratio)
        
        if attacker_ratio <= 0:
            return self.rng.sample(self.all_clients, num_total_select)
            
        target_attackers = int(num_total_select * attacker_ratio)
        
        # 修正坏人数,不能超过实际存在的坏人总数
        attackers = min(target_attackers, len(self.attacker_ids))
        
        benigns = num_total_select - attackers
        
        selected_attackers = self.rng.sample(self.attacker_list, attackers)
        selected_benign = self.rng.sample(self.benign_ids, benigns)
        
        return selected_attackers + selected_benign

    def broadcast(self, selected_clients: List[str]) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        向选中的客户端分发模型。
        """
        # deepcopy 防止 Client 训练时意外修改了 Server 的原版模型
        global_weights = {k: v.cpu().clone() for k, v in self.global_model.state_dict().items()}

        package = {client_id: global_weights for client_id in selected_clients}
        
        return package

    def step(self, updates: List[Dict[str, Any]]):
        """
        [核心防御流水线] 处理客户端上传的更新。
        
        Args:
            updates: 列表，每个元素是 {'client_id': str, 'weights': dict, 'num_samples': int, ...}
        """

        # 输入：原始更新列表 -> 输出：清洗后的更新列表 (可能变短)
        if self.screener:
            # 传入全局模型作为参考 (某些防御如 FLTrust 需要)
            updates = self.screener.screen(updates, self.global_model)
        
        # 输入：更新列表 -> 输出：聚合后的权重 (weights dict)
        aggregated_weights = self.aggregator.aggregate(updates)

        # 输入：聚合结果 + 当前模型 -> 输出：原地修改模型
        if self.updater:
            self.updater.update(self.global_model, aggregated_weights)
        else:
            # 默认行为：直接覆盖
            self.global_model.load_state_dict(aggregated_weights)

    def eval(self, metrics: List[Callable]) -> Dict[str, float]:
        """
        在服务器持有的测试集上评估全局模型。
        """
        self.global_model.eval()
        device = self.device
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
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