import torch
from core.utils.registry import METRIC_REGISTRY
class Metric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        raise NotImplementedError
@METRIC_REGISTRY.register("acc")
class Acc(Metric):
    def __init__(self):
        super().__init__("acc")

    def __call__(self, preds, targets):
        # preds: [N, C] or [N] (indices)
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
        correct = preds.eq(targets).sum().item()
        return correct / len(targets)
@METRIC_REGISTRY.register("asr")
class ASR(Metric):
    """
    计算攻击成功率 (ASR)
    """
    def __init__(self, target_label: int):
        super().__init__("asr")
        self.target_label = target_label

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Args:
            preds: 模型的输出 logits [N, C]
            targets: 样本的原始、干净标签 [N]
        """
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
        
        # 只关注那些原始标签不是目标标签的样本
        non_target_mask = (targets != self.target_label)
        preds_on_non_targets = preds[non_target_mask]
        
        # 如果没有非目标样本（例如一个 batch 全是目标类），则 ASR 为 0
        if len(preds_on_non_targets) == 0:
            return 0.0
            
        # 在筛选后的样本上，计算有多少被错误地预测成了目标标签
        success = preds_on_non_targets.eq(self.target_label).sum().item()
        
        return success / len(preds_on_non_targets)
@METRIC_REGISTRY.register("loss")
class Loss(Metric):
    def __init__(self):
        super().__init__("loss")
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def __call__(self, preds, targets):
        # 这里 preds 必须是 Logits
        return self.criterion(preds, targets).item()