import torch

class Metric:
    def __init__(self, name: str):
        self.name = name

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        raise NotImplementedError

class Acc(Metric):
    def __init__(self):
        super().__init__("acc")

    def __call__(self, preds, targets):
        # preds: [N, C] or [N] (indices)
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
        correct = preds.eq(targets).sum().item()
        return correct / len(targets)

class ASR(Metric):
    """
    计算攻击成功率 (ASR)。
    ASR = (预测为 Target 的样本数) / (总样本数)
    通常只在全是 Trigger 的数据集上计算。
    """
    def __init__(self, target_label: int):
        super().__init__("asr")
        self.target_label = target_label

    def __call__(self, preds, targets):
        # 注意：targets 参数在这里其实没用，因为我们只关心是否变成了 target_label
        # 但为了接口一致性，还是保留
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)
        
        success = preds.eq(self.target_label).sum().item()
        return success / len(preds)

class Loss(Metric):
    def __init__(self, criterion):
        super().__init__("loss")
        self.criterion = criterion
    
    def __call__(self, preds, targets):
        # 这里 preds 必须是 Logits
        return self.criterion(preds, targets).item()