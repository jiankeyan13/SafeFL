import torch
import torch.nn as nn
from typing import Dict, Optional

class BaseMetric:
    """评估指标基类，支持增量计算。"""

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None):
        """累积批次数据。"""
        raise NotImplementedError

    def compute(self) -> float:
        """计算最终指标值。"""
        raise NotImplementedError

    def reset(self):
        """重置内部状态以进行新一轮评估。"""
        raise NotImplementedError


class Accuracy(BaseMetric):
    """计算分类准确率。可用于CDA（干净数据准确率）或ASR（攻击成功率）。"""
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None):
        if preds.ndim > 1:
            preds = preds.argmax(dim=1)
        self.correct += preds.eq(targets).sum().item()
        self.total += targets.size(0)

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0


class AverageLoss(BaseMetric):
    """计算加权平均损失。"""
    def __init__(self):
        self.total_loss = 0.0
        self.total_samples = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None):
        if loss is not None:
            self.total_loss += loss.item() * targets.size(0)
            self.total_samples += targets.size(0)

    def compute(self) -> float:
        return self.total_loss / self.total_samples if self.total_samples > 0 else 0.0

    def reset(self):
        self.total_loss = 0.0
        self.total_samples = 0


class Evaluator:
    """模型评估器，支持多指标计算。适用于服务端和客户端模型评估。"""

    def __init__(self, metrics: Optional[Dict[str, BaseMetric]] = None):
        """
        Args:
            metrics: 指标字典，如{"acc": Accuracy(), "loss": AverageLoss()}。
                     默认使用accuracy和loss。
        """
        if metrics is None:
            self.metrics = {
                "accuracy": Accuracy(),
                "loss": AverageLoss()
            }
        else:
            self.metrics = metrics

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> Dict[str, float]:
        """
        在指定数据集上评估模型。

        注意：传入毒化数据(poisoned_loader)时, accuracy即代表ASR。
        """
        was_training = model.training
        model.eval()
        model.to(device)

        for metric in self.metrics.values():
            metric.reset()

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets) if criterion else None

            for metric in self.metrics.values():
                metric.update(outputs, targets, loss)

        if was_training:
            model.train()

        return {name: metric.compute() for name, metric in self.metrics.items()}