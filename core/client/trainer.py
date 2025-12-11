import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

class StandardTrainer:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, attack_hooks=None):

        self.model.train()
        
        # 1. 配置优化器和调度器
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer, len(dataloader))
        
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            
            output = self.model(data)
            
            # 自定义 Loss (如 CerP)
            if attack_hooks and 'compute_loss' in attack_hooks:
                loss = attack_hooks['compute_loss'](output, target, self.model, self.criterion)
            else:
                loss = self.criterion(output, target)
            
            loss.backward()
            
            # 梯度处理 (如 Neurotoxin)
            if attack_hooks and 'on_after_backward' in attack_hooks:
                attack_hooks['on_after_backward'](self.model)
            
            # 可以在这里加 Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                
            # [Hook] 参数处理 (如 PGD)
            if attack_hooks and 'on_after_step' in attack_hooks:
                attack_hooks['on_after_step'](self.model)

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def inference(self, dataloader):
        """
        执行纯推理，返回所有预测和标签。
        用于 Eval 阶段。
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            
            # 为了节省显存，立即转回 CPU
            all_preds.append(output.cpu())
            all_targets.append(target.cpu())

        return torch.cat(all_preds), torch.cat(all_targets)

    def _get_optimizer(self):
        # ... 封装优化器创建逻辑 ...
        return torch.optim.SGD(self.model.parameters(), 
                               lr=self.config.get('lr', 0.01), 
                               momentum=0.9, 
                               weight_decay=5e-4)
