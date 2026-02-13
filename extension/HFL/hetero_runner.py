import copy
import random
import torch
import numpy as np
from functools import partial
from typing import Dict, Any, List, Callable

from core.runner import (
    FederatedRunner,
    DEFAULT_LOCAL_EVAL_RATIO,
)
from core.utils.registry import MODEL_REGISTRY, AGGREGATOR_REGISTRY, SCREENER_REGISTRY
from core.utils.scheduler import build_scheduler
from core.client.base_client import BaseClient
from core.server.updater.base_updater import BaseUpdater
from extension.HFL.fedrolex_server import FedrolexServer
from extension.HFL.cap_manager import CapManager
import extension.HFL.sub_aggregator  # noqa: F401 - 注册 sub_avg 聚合器
from data.constants import (
    OWNER_SERVER,
    SPLIT_TRAIN,
    SPLIT_TEST,
    SPLIT_TEST_GLOBAL,
)


class HeteroRunner(FederatedRunner):
    """
    异构联邦学习主控循环 (Heterogeneous Federated Learning Runner)。
    
    继承自 FederatedRunner，针对模型异构场景（结构化剪枝）进行扩展：
    1. 初始化 CapManager 并注册客户端能力
    2. 构建 FedrolexServer（支持动态子模型提取与滚动选取）
    3. 为每个客户端动态构建匹配其 p 值的模型架构
    4. 在训练和评估时确保模型架构与剪枝结构一致
    """

    def _setup(self):
        """
        初始化异构联邦学习组件。
        
        流程：
        1. 调用父类方法完成数据、全局模型等基础初始化
        2. 初始化 CapManager 并注册所有客户端
        3. 构建 FedrolexServer（传入 cap_manager）
        4. 设置学习率调度器和攻击管理器
        """
        self.logger.info(">>> Initializing Heterogeneous FL components...")

        # 1. 准备数据（复用父类逻辑）
        global_seed = self.seed
        self._setup_data_pipeline()

        # 2. 准备全局模型（全量模型，p=1.0）
        model_conf = self.config['model']
        model_cls = MODEL_REGISTRY.get(model_conf['name'])
        # 全局模型使用 p=1.0（全量）
        self.model_fn = partial(model_cls, **model_conf.get('params', {}))
        self.global_model = self.model_fn().to(self.device)
        self.model_cls = model_cls  # 保存模型类，用于后续动态构建

        # 3. 提取 Server 任务
        self.server_test_task = self.task_set.get_task(OWNER_SERVER, SPLIT_TEST_GLOBAL)
        self.server_dataset_store = self.dataset_stores[self.server_test_task.dataset_tag]

        # 5. 初始化 CapManager 并注册客户端
        self.client_ids = self.task_set.list_client_ids(exclude_server=True)
        hetero_config = self.config.get('hetero', {})
        if not hetero_config:
            raise ValueError("HeteroRunner requires 'hetero' config section with 'sample' and 'p_list'.")
        
        self.cap_manager = CapManager(hetero_config, seed=global_seed)
        self.cap_manager.register_clients(self.client_ids)
        self.logger.info(f"CapManager initialized. Summary: {self.cap_manager.summary()}")

        # 6. 构建 FedrolexServer（直接构建，不通过算法注册表）
        server_conf = self.config.get('server', {})
        
        # 构建聚合器
        aggregator_conf = server_conf.get('aggregator', {})
        aggregator_name = aggregator_conf.get('name', 'sub_avg')  # 默认使用 sub_avg
        aggregator = AGGREGATOR_REGISTRY.build(aggregator_name, **aggregator_conf.get('params', {}))
        
        # 构建筛选器
        screener_conf = server_conf.get('screener', {})
        screener = None
        if screener_conf.get('name'):
            screener = SCREENER_REGISTRY.build(
                screener_conf['name'], 
                **screener_conf.get('params', {})
            )
        
        # 构建更新器
        updater_conf = server_conf.get('updater', {})
        updater = BaseUpdater(config=updater_conf)
        
        # 使用标准 BaseClient 构建 dataloader
        self.client_class = BaseClient

        # 构建测试集 DataLoader
        test_loader = None
        if self.server_test_task:
            test_loader = self._build_client_dataloader(
                client_id="server_test_loader",
                task=self.server_test_task,
                dataset_store=self.server_dataset_store,
                mode='test',
            )

        # 创建 FedrolexServer（关键：传入 cap_manager，使用滚动选取替代随机选取）
        self.server = FedrolexServer(
            model=self.global_model,
            device=self.device,
            cap_manager=self.cap_manager,
            screener=screener,
            aggregator=aggregator,
            updater=updater,
            test_loader=test_loader,
            seed=global_seed
        )
        # 构造 Server Proxy Loader（用于 BN 校准）
        self._setup_server_proxy_loader()

        self.logger.info(f"Server Type: {type(self.server).__name__}")
        self.logger.info(f"Client Type: {self.client_class.__name__}")

        # 7. 学习率调度器
        self.lr_scheduler = build_scheduler(self.config['training'])

        # 8. 设置攻击者（复用父类逻辑）
        self._setup_attack_manager()

    def _get_client_model_fn(self, client_id: str) -> Callable[[], torch.nn.Module]:
        """
        根据客户端 ID 生成专属的模型构建函数。
        
        功能简介：
            查询 CapManager 获取该客户端的 p 值，返回一个固定了 p 参数的模型构建函数。
        
        参数说明：
            client_id (str): 客户端唯一标识符
        
        返回值：
            Callable[[], nn.Module]: 无参工厂函数，调用后返回匹配该客户端 p 值的模型实例
        
        逻辑要点：
            - 使用 partial 固定 p 参数
            - p 值通过 cap_manager.get_bucketed_capability() 获取（已做 bucket 处理）
        """
        p = self.cap_manager.get_bucketed_capability(client_id)
        model_conf = self.config['model']
        model_params = model_conf.get('params', {}).copy()
        model_params['p'] = p
        return partial(self.model_cls, **model_params)

    def _run_local_training(self, client_ids, client_models, config):
        """
        执行本地训练循环（异构版本）。
        
        功能简介：
            遍历选中客户端执行训练，为每个客户端动态构建与其 p 值匹配的模型架构。
        
        参数说明：
            client_ids (List[str]): 本轮选中的客户端 ID 列表
            client_models (Dict[str, Dict[str, torch.Tensor]]): Server 下发的模型参数（已剪枝）
            config (Dict): 训练配置（学习率、epochs 等）
        
        返回值：
            updates (List[Dict]): 包含各客户端训练后参数差值及元数据的列表
        
        逻辑要点：
            - 不能直接使用 self.model_fn（它是全量模型）
            - 必须为每个客户端调用 _get_client_model_fn 获取定制的模型构建函数
            - 确保模型架构与 Server 下发的剪枝权重维度一致
        """
        updates = []
        
        for cid in client_ids:
            attack_strategy = None
            if self.attack_manager:
                attack_strategy = self.attack_manager.get_strategy(cid)
            
            # 关键：为每个客户端动态构建模型函数
            client_model_fn = self._get_client_model_fn(cid)
            client = self.client_class(cid, self.device, client_model_fn)
            
            task = self.task_set.get_task(cid, SPLIT_TRAIN)
            store = self.dataset_stores[task.dataset_tag]
            
            payload = client.execute(
                global_state_dict=client_models[cid],
                task=task,
                dataset_store=store,
                config=config,
                attack_profile=attack_strategy
            )
            
            updates.append(payload)
            del client
            
        return updates

    def _run_local_evaluation(self, round_idx, config, metrics):
        """
        抽样评估客户端本地性能（异构版本）。
        
        功能简介：
            在本地测试集上评估模型性能，确保测试时实例化的模型架构与该客户端的剪枝结构一致。
        
        参数说明：
            round_idx (int): 当前轮次索引
            config (Dict): 评估配置
            metrics (List[Metric]): 评估指标对象列表
        
        返回值：
            None (结果直接记录到 Logger)
        
        逻辑要点：
            - 与 _run_local_training 类似，实例化 Client 前必须获取该客户端的 p 值
            - 构建对应的 HeteroResNet(p=...) 进行评估
            - 否则加载 Server 下发的子模型权重时会报 Shape Mismatch 错误
        """
        client_candidates = self.task_set.list_client_ids(exclude_server=True)
        
        eval_ids = random.sample(
            client_candidates, 
            k=max(1, int(len(client_candidates) * DEFAULT_LOCAL_EVAL_RATIO))
        )

        results_collector = {m.name: [] for m in metrics}
        
        for cid in eval_ids:
            # 关键：为每个客户端动态构建模型函数
            client_model_fn = self._get_client_model_fn(cid)
            client = self.client_class(cid, self.device, client_model_fn)
            
            task = self.task_set.get_task(cid, SPLIT_TEST)
            store = self.dataset_stores[task.dataset_tag]
            
            # 获取模型（Server 会下发剪枝后的子模型）
            model_dict = self.server.broadcast([cid])[cid]
            
            # 执行评估
            res = client.evaluate(
                global_state_dict=model_dict,
                task=task,
                dataset_store=store,
                config=config,
                metrics=metrics
            )
            
            # 收集结果
            for key, val in res.items():
                if key in results_collector:
                    results_collector[key].append(val)
            
            del client
        
        # 记录日志
        log_dict = {}
        for metric_name, values in results_collector.items():
            if len(values) > 0:
                log_dict[f"local/avg_{metric_name}"] = np.mean(values)
                log_dict[f"local/std_{metric_name}"] = np.std(values)
                
        self.logger.log_metrics(log_dict, step=round_idx)
