from typing import Dict, Any, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from core.utils.evaluator import Evaluator
from core.utils.configs import ClientConfig, TrainerConfig
from data.dataset_store import DatasetStore
from data.task import TaskSet
from data.constants import SPLIT_TRAIN, SPLIT_TEST, client_owner


class BaseClient:
    """
    联邦学习客户端基类。
    """

    def __init__(
        self,
        client_id: Union[int, str],
        task_set: TaskSet,
        stores: Dict[str, DatasetStore],
        model: nn.Module,
        device: torch.device,
        config: ClientConfig,
        evaluator: Optional[Evaluator] = None
    ):
        """
        初始化客户端状态。

        Args:
            client_id: 客户端标识，支持 int 或 str (如 "client_0") 。
            task_set: task_generator 的划分结果集合，包含各客户端的数据集索引。
            stores: 所有真实数据的映射。
            model: 网络模型，必传。
            device: 模型训练/推理所在设备。
            config: 客户端配置 (ClientConfig)，包含 TrainerConfig。
            evaluator: 联邦指标评估器，若无则默认使用内置 Evaluator。
        """
        if isinstance(client_id, int):
            self.client_id = client_id
            self.owner_id = client_owner(client_id)
        else:
            self.owner_id = client_id
            self.client_id = int(client_id.split("_")[-1])

        self.task_set = task_set
        self.stores = stores
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.evaluator = evaluator or Evaluator()
        self.loss = config.trainer_config.build_criterion()

        self.train_loader = self._create_train_dataloader()
        self.test_loader = self._create_test_dataloader()

    def _build_dataset(self, split: str) -> Optional[DatasetStore]:
        """
        [内部辅助方法]：根据 client_id 和 split (如 'train') 从 task_set 与 stores 中解析构建出对应的数据子集。

        Args:
            split: 获取的数据划分标识（如 constants.SPLIT_TRAIN）。

        Returns:
            重新用 Subset 包装的数据，包含 get_label 等方法；如果此客户端没有对应数据则返回 None。
        """
        task = self.task_set.try_get_task(self.owner_id, split)
        if task is None:
            return None
        store = self.stores[task.dataset_tag]
        subset = Subset(store.dataset, task.indices)
        return DatasetStore(
            name=f"{self.owner_id}_{split}",
            split=split,
            dataset=subset,
        )

    def _create_train_dataloader(self) -> DataLoader:
        """
        构建训练用 DataLoader。
        默认会调用自身 _build_dataset 抽出训练数据。任何需要自定义采样规则（Sampler）的算法都可通过重写本方法实现。

        Returns:
            用于向后传递进行模型更新的 PyTorch 数据加载器。
        """
        ds = self._build_dataset(SPLIT_TRAIN)
        if ds is None:
            raise RuntimeError(f"Client {self.owner_id} has no training data.")
        return DataLoader(
            ds, batch_size=self.config.batch_size, shuffle=True, drop_last=False,
            num_workers=self.config.num_workers, pin_memory=True,
            persistent_workers=self.config.num_workers > 0
        )

    def _create_test_dataloader(self) -> Optional[DataLoader]:
        """
        构建验证用 DataLoader。

        Returns:
            若该客户端不具备测试划分，则返回 None。
        """
        ds = self._build_dataset(SPLIT_TEST)
        if ds is None:
            return None
        return DataLoader(
            ds, batch_size=self.config.batch_size, shuffle=False, drop_last=False,
            num_workers=self.config.num_workers, pin_memory=True,
            persistent_workers=self.config.num_workers > 0
        )

    def receive(self, server_payload: Dict[str, Any]) -> None:
        """
        接收服务端下发的内容并解析、执行同步（例如同步 global_model 状态到本地 base_model）。

        Args:
            server_payload: 来自 server.broadcast() 的 state_dict。
        """
        self.model.load_state_dict(server_payload, strict=False)

    def train(self) -> Dict[str, Any]:
        """
        执行客户端本地模型训练流程。
        调用 _create_train_dataloader() 推进数个 Local Epoch 的前向和反向传播。

        Returns:
            本地训练期的指标流（含 train_loss 和 delta）。
        """
        self.model.train()

        # 保存训练前模型状态，用于计算 delta（含 BN running_mean / running_var / num_batches_tracked）
        # S2 优化: 保留在 GPU 上 clone, 避免 CPU 搬运
        initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        optimizer = self.config.trainer_config.build_optimizer(self.model)
        local_epochs = self.config.trainer_config.epochs

        total_loss = 0.0
        total_samples = 0

        for _ in range(local_epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                output = self.model(data)
                loss_val = self.loss(output, target)
                loss_val.backward()
                optimizer.step()

                total_loss += loss_val.item() * target.size(0)
                total_samples += target.size(0)

        # 计算前后模型变化 delta (S2 优化: GPU 内直接计算)
        current_state = self.model.state_dict()
        delta = {
            k: current_state[k] - initial_state[k]
            for k in initial_state
            if k in current_state
        }
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        return {"train_loss": avg_loss, "delta": delta}

    def evaluate(self) -> Dict[str, float]:
        """
        在本地测试集执行测试模型评估。
        依赖内置 Evaluator。若输入被恶意污染(如触发后门),该方法的准确率返回值即对应 ASR 攻击成功率。

        Returns:
            各项评估指标得分(例如 {"accuracy": 0.98, "loss": 0.05})。
        """
        if self.test_loader is None:
            return {}
        return self.evaluator.evaluate(
            self.model, self.test_loader, self.loss, self.device
        )

    def package(self, train_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        打包客户端想要向上传递的数据集合，用于被服务端的聚合器收集。

        Returns:
            最终 payload 字典(包含 "client_id", "delta", "num_samples")。
            delta 含可学习参数与 BN 等 buffer 的差分。
        """
        return {
            "client_id": self.owner_id,
            "delta": train_metrics["delta"],
            "metrics": train_metrics["train_loss"],
            "num_samples": len(self.train_loader.dataset),
        }

    def step(self, server_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        客户端单轮工作主令控制流(Template Method)。
        依次发生：接收 -> 训练 -> 评估 -> 组装并发送。

        Args:
            server_payload: Server 对 Client 在这一 Round 投喂的完整通信包裹(state_dict)。

        Returns:
            Client 投递给 Server 的所有必要信息(结合 package 的结果加上评价指标)。
        """
        self.receive(server_payload)
        train_metrics = self.train()
        payload = self.package(train_metrics)
        return payload
