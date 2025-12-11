import copy
import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Any, Optional, Callable

from data.task import Task
from data.dataset_store import DatasetStore
from core.client.trainer import StandardTrainer

class BaseClient:
    """
    联邦学习客户端基类。
    职责：
    1. 协调数据流 (Dataset -> DataLoader)
    2. 协调控制流 (Trainer -> Training Loop)
    3. 协调攻击流 (AttackProfile -> Hooks)
    4. 协调模型流 (Receive -> Update)
    """

    def __init__(self, client_id: str, device: torch.device, model_fn: Callable[[], torch.nn.Module]):
        """
        Args:
            client_id: 客户端唯一标识
            device: 运行设备 (cpu/cuda)
            model_fn: 模型构建函数 (无参工厂函数，使用 functools.partial 封装参数)
        """
        self.client_id = client_id
        self.device = device
        # 延迟实例化：调用工厂函数创建本地模型
        self.model = model_fn().to(self.device)
        self.num_train_samples = 0

    def execute(self, 
                global_state_dict: Dict[str, torch.Tensor],
                task: Task,
                dataset_store: DatasetStore,
                config: Dict[str, Any],
                attack_profile: Optional[Any] = None) -> Dict[str, Any]:
        """
        执行一轮本地训练流程。
        
        Args:
            global_state_dict: 服务器下发的模型参数
            task: 本轮分配的数据任务
            dataset_store: 全局数据源
            config: 训练配置
            attack_profile: 攻击插件
        """
        self.receive_model(global_state_dict)
        
        # 保存初始权重，供某些攻击（如 Scaling Attack）计算 Delta 使用
        # 注意：这里存的是 CPU 上的副本，不占显存
        initial_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # 数据准备 (可能触发 BadNets/DBA)
        dataloader = self.data_load(task, dataset_store, config, attack_profile, mode='train')

        # 本地训练 (可能触发 Neurotoxin/PGD)
        train_metrics = self.train(dataloader, config, attack_profile)

        payload = self.update(initial_weights, attack_profile)

        payload['metrics'] = train_metrics
        
        self.model.cpu()
        torch.cuda.empty_cache()
        
        return payload

    def receive_model(self, state_dict: Dict[str, torch.Tensor]):
        """
        接收并加载模型权重。
        """
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

    def data_load(self, 
                  task: Task, 
                  dataset_store: DatasetStore, 
                  config: Dict[str, Any], 
                  attack_profile: Optional[Any] = None,
                  mode: str = 'train') -> DataLoader:
        """
        数据加载阶段。支持 Trigger 注入和 Label Flipping。
        """
        # 获取 Subset
        local_dataset = Subset(dataset_store, task.indices)
        
        if mode == 'train':
            self.num_train_samples = len(local_dataset)

        # [Hook] 攻击者修改数据集
        if attack_profile is not None and hasattr(attack_profile, 'modify_dataset'):
            local_dataset = attack_profile.modify_dataset(local_dataset, mode=mode)

        shuffle = (mode == 'train')
        loader = DataLoader(
            local_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=shuffle,
            num_workers=config.get('num_workers', 0),
            pin_memory=True
        )
        return loader

    def train(self, 
              dataloader: DataLoader, 
              config: Dict[str, Any], 
              attack_profile: Optional[Any] = None) -> Dict[str, float]:
        """
        训练阶段。委托给 Trainer 处理具体的 SGD/Scheduler 逻辑。
        """
        trainer = StandardTrainer(self.model, self.device, config)
        
        # 提取并组装攻击钩子 (Dependency Injection)
        hooks = {}
        if attack_profile:
            if hasattr(attack_profile, 'compute_loss'):
                hooks['compute_loss'] = attack_profile.compute_loss
            if hasattr(attack_profile, 'on_after_backward'):
                hooks['on_after_backward'] = attack_profile.on_after_backward
            if hasattr(attack_profile, 'on_after_step'):
                hooks['on_after_step'] = attack_profile.on_after_step

        epochs = config.get('epochs', 1)
        total_loss = 0.0
        
        for epoch in range(epochs):
            epoch_loss = trainer.train_epoch(dataloader, attack_hooks=hooks)
            total_loss += epoch_loss
            
        return {"train_loss": total_loss / epochs}

    def update(self, 
               initial_weights: Dict[str, torch.Tensor], 
               attack_profile: Optional[Any] = None) -> Dict[str, Any]:
        """
        更新打包阶段。支持 Scaling Attack / Noise Masking。
        """
        # 必须 clone 到 CPU，否则返回的是引用，后续会被修改
        final_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        # [Hook] 攻击者修改最终权重
        if attack_profile and hasattr(attack_profile, 'modify_weights'):
            final_state_dict = attack_profile.modify_weights(
                final_state_dict, 
                initial_weights
            )

        return {
            "client_id": self.client_id,
            "weights": final_state_dict,
            "num_samples": self.num_train_samples, # 真实样本数
        }

    def eval(self, 
                 global_state_dict: Dict[str, torch.Tensor],
                 task: Task,
                 dataset_store: DatasetStore,
                 config: Dict[str, Any],
                 metrics: List[Callable],
                 attack_profile: Optional[Any] = None) -> Dict[str, float]:
        """
        通用评估函数。支持 Metric 注入。
        """

        self.model.eval()
        
        loader = self.data_load(task, dataset_store, config, attack_profile, mode='test')
        
        trainer = StandardTrainer(self.model, self.device, config)
        all_preds, all_targets = trainer.inference(loader)
        
        self.model.cpu()
        torch.cuda.empty_cache()
        
        if len(all_preds) == 0:
            return {}

        # 5. 计算指标
        results = {}
        for metric in metrics:
            val = metric(all_preds, all_targets)
            results[metric.name] = val
            
        return results