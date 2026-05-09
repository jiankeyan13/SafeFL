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
    """计算分类准确率。可用于 CDA 或基于目标标签的 ASR。"""
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

        注意：若 poisoned_loader 已将标签替换为攻击目标标签,
        则 accuracy 即代表 targeted ASR。
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

    @torch.no_grad()
    def evaluate_backdoor(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: Optional[nn.Module],
        device: torch.device,
    ) -> Dict[str, float]:
        """
        在带触发器的测试集上评估, 一次前向同时计算 targeted ASR 与 Backdoor Acc.

        要求 dataloader 每条样本为 (inputs, target_label, original_label).
        - ASR: 预测等于攻击目标标签的比例.
        - backdoor_acc: 预测等于原始真实标签的比例.
        """
        was_training = model.training
        model.eval()
        model.to(device)

        asr_correct = 0
        backdoor_correct = 0
        total = 0
        total_loss = 0.0
        loss_count = 0

        for inputs, target_labels, original_labels in dataloader:
            inputs = inputs.to(device)
            target_labels = target_labels.to(device)
            original_labels = original_labels.to(device)

            outputs = model(inputs)
            if criterion is not None:
                loss = criterion(outputs, original_labels)
                total_loss += loss.item() * original_labels.size(0)
                loss_count += original_labels.size(0)

            preds = outputs.argmax(dim=1)
            asr_correct += preds.eq(target_labels).sum().item()
            backdoor_correct += preds.eq(original_labels).sum().item()
            total += target_labels.size(0)

        if was_training:
            model.train()

        if total == 0:
            out = {"asr": 0.0, "backdoor_acc": 0.0}
        else:
            out = {"asr": asr_correct / total, "backdoor_acc": backdoor_correct / total}
        if criterion is not None and loss_count > 0:
            out["backdoor_loss"] = total_loss / loss_count
        return out
