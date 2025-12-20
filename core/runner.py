import os
import torch
import numpy as np
import random
import copy
from functools import partial
from typing import Dict, Any, List, Optional

from core.utils.logger import Logger
from core.utils.scheduler import build_scheduler
from core.client.attack_manager import AttackManager
import core.utils.metrics
import models
import core.server.aggregator
import core.server.screener
import core.server.updater
import core.attack
import algorithms
from core.utils.registry import (
    MODEL_REGISTRY, 
    AGGREGATOR_REGISTRY, 
    SCREENER_REGISTRY, 
    UPDATER_REGISTRY,
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    ATTACK_REGISTRY
)

from data.task_generator import TaskGenerator
from core.client.base_client import BaseClient
from core.server.base_server import BaseServer

import data.datasets

class FederatedRunner:
    """
    联邦学习主控循环 (Simulator)。
    职责：
    1. 初始化 (数据、模型、服务器、日志)。
    2. 执行 Round 循环 (选人 -> 训练 -> 聚合 -> 评估)。
    3. 管理全局状态 (轮次、学习率、Checkpoints)。
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.logger = Logger(
            project_name=config.get('project', 'FL_Project'),
            experiment_name=config.get('name', 'experiment'),
            config=config,
            use_wandb=config.get('use_wandb', False)
        )
        self.logger.info(f"Runner is configured to use device: {self.device}")
        self._set_seed(config.get('seed', 42))
        self.client_ids = []
        self._setup()

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)

    def _setup(self):
        self.logger.info(">>> Initializing components...")

        # 准备数据
        # 假设 config['data'] 包含了 root, dataset_name, partitioner 等参数
        from data.partitioner import DirichletPartitioner, IIDPartitioner # 临时引入
        global_seed = self.config.get('seed', 42)
        part_conf = self.config['data']['partitioner']
        if part_conf['name'] == 'dirichlet':
            partitioner = DirichletPartitioner(alpha=part_conf.get('alpha', 0.5), seed = global_seed)
        else:
            partitioner = IIDPartitioner(seed = global_seed)

        self.task_generator = TaskGenerator(
            dataset_name=self.config['data']['dataset'],
            root=self.config['data']['root'],
            partitioner=partitioner,
            num_clients=self.config['data']['num_clients'],
            val_ratio=self.config['data'].get('val_ratio', 0.1),
            seed = global_seed
        )
        # 生成任务和数据源
        self.task_set, self.dataset_stores = self.task_generator.generate()
        self.logger.info(f"Data setup complete. Clients: {self.config['data']['num_clients']}")

        # 准备模型构建函数 (Model Fn)
        model_conf = self.config['model']
        model_cls = MODEL_REGISTRY.get(model_conf['name'])
        self.model_fn = partial(model_cls, **model_conf.get('params', {}))
        self.global_model = self.model_fn().to(self.device)

        self.server_test_task = self.task_set.get_task("server", "test_global")
        self.server_dataset_store = self.dataset_stores[self.server_test_task.dataset_tag]

        algo_conf = self.config['algorithm'] # e.g. {'name': 'fedavg', 'params': {...}}
        self.server, self.client_class = ALGORITHM_REGISTRY.build(
            algo_conf['name'],
            # --- 传递给 build_xxx_algorithm 的参数 ---
            model=self.global_model, device=self.device, dataset_store=self.dataset_stores, config=self.config,\
                server_test_task=self.server_test_task, **algo_conf.get('params', {}))

        self.logger.info(f"Algorithm Loaded: {algo_conf['name']}")
        self.logger.info(f"Server Type: {type(self.server).__name__}")
        self.logger.info(f"Client Type: {self.client_class.__name__}")

        # 学习率调度器
        self.lr_scheduler = build_scheduler(self.config['training'])

        # --- 设置攻击者 ---
        self.logger.info(">>> Setting up attackers...")
        self.client_ids = [cid for cid in self.task_set._tasks.keys() if cid != 'server']

        # AttackManager 内部会处理 config 为 None 的情况
        self.attack_manager = AttackManager(config=self.config.get('attack'), \
                                            all_client_ids=self.client_ids, seed=self.config.get('seed', 42))
        
        # 攻击者名单
        self.attacker_ids = self.attack_manager.get_attacker_ids()

    def run(self):
        total_rounds = self.config['training']['rounds']
        eval_interval = self.config['training'].get('eval_interval', 5)
        
        self.logger.info(">>> Start Training...")
        
        best_acc = 0.0

        for round_idx in range(total_rounds):
            self.logger.info(f"--- Round {round_idx} ---")

            # 计算全局 LR
            current_lr = self.lr_scheduler.get_lr(round_idx)
            round_config = copy.deepcopy(self.config['client'])
            round_config['lr'] = current_lr
            round_config['current_round'] = round_idx

            selected_ids = self._select_clients(round_idx)

            self.logger.info(f"Selected clients ({len(selected_ids)}): {selected_ids}")
            
            client_models = self.server.broadcast(selected_ids)

            updates = self._run_local_training(selected_ids, client_models, round_config)

            # Server Step
            self.server.step(updates)
            train_metrics = self.server.aggregate_metrics(updates)
            self.logger.log_metrics(train_metrics, step=round_idx)

            # 全局评估
            test_metrics = self._run_global_evaluation(round_idx)

            # 本地抽样评估 (Local Eval)
            if round_idx % eval_interval == 0:
                local_metric_confs = self.config.get('evaluation', {}).get('local', [{'name': 'acc'}])
                local_metrics = self._build_metrics(local_metric_confs)
                self._run_local_evaluation(round_idx, round_config, local_metrics)

            # 保存 Checkpoint
            if test_metrics.get('acc', 0) > best_acc:
                best_acc = test_metrics['acc']
                self._save_checkpoint(round_idx, is_best=True)

        self.logger.info(f"Training Finished. Best Acc: {best_acc:.4f}")
        self.logger.close()
    def _run_local_training(self, client_ids, client_models, config):
        """
        执行本地训练循环。
        """
        updates = []
        
        for cid in client_ids:
            attack_strategy = None
            if self.attack_manager:
                attack_strategy = self.attack_manager.get_strategy(cid)
            
            client = self.client_class(cid, self.device, self.model_fn)
            task = self.task_set.get_task(cid, "train")
            store = self.dataset_stores[task.dataset_tag]
            
            payload = client.execute(
                global_state_dict=client_models[cid],
                task=task,
                dataset_store=store,
                config=config,
                attack_profile=attack_strategy
            )
            
            updates.append(payload)
            
            # 5. 显式销毁 (Python GC 会处理，但显式删除更保险)
            del client
            
        return updates

    def _run_global_evaluation(self, round_idx):
        """
        执行全局评估（主任务 + 各攻击组任务）。
        
        流程：
        1. 在干净的全局测试集上评估主任务指标 (ACC, Loss)。
        2. 遍历配置中的每个攻击组，在被该组策略污染的全局测试集上评估攻击指标 (ASR)。
        """
        
        # 从配置读取全局指标 (通常是 acc, loss)
        global_metric_confs = self.config.get('evaluation', {}).get('global', [{'name': 'acc'}])
        clean_metric_objs = self._build_metrics(global_metric_confs)
        # 默认使用 Server 内部的干净 test_loader
        clean_metrics = self.server.eval(metrics=clean_metric_objs)
        
        # 记录日志
        self.logger.log_metrics(clean_metrics, step=round_idx)
        
        # 打印主任务结果 (用于控制台监控)
        first_key = list(clean_metrics.keys())[0] if clean_metrics else "N/A"
        first_val = clean_metrics.get(first_key, 0)
        self.logger.info(f"Global Eval (Clean): {first_key} = {first_val:.4f}")
        
        # 获取攻击策略配置
        attackers_conf = self.config.get('attack', {}).get('strategies', {})
        # 如果没有配置攻击，或者没有 AttackManager，直接返回
        if not attackers_conf or not self.attack_manager:
            return clean_metrics 

        # 实例化一个工具人 Client，用于复用 data_load 逻辑生成带毒数据
        # 注意：这里我们使用 self.client_class 确保兼容不同算法定制的 Client
        tool_client = self.client_class(client_id="server_eval_tool", device=self.device, model_fn=self.model_fn)

        # 遍历每个攻击组 (例如 'badnets_group', 'scaling_group')
        for group_name, group_config in attackers_conf.items():
        
            # 检查该组是否定义了评估指标
            eval_conf = group_config.get('evaluation')
            if not eval_conf:
                continue 
                
            # 该组专属的 Metrics (通常是 ASR)
            group_metrics = self._build_metrics(eval_conf)
            
            # 构建攻击策略实例
            strategy_conf = group_config['strategy']
            params = strategy_conf.get('params', {}).copy()
            params['seed'] = self.config.get('seed', 42)
            eval_strategy = ATTACK_REGISTRY.build(
                strategy_conf['name'], 
                **strategy_conf.get('params', {})
            )
            
            # 生成带毒测试集 (通过工具人 Client)
            poisoned_loader = tool_client.data_load(
                task=self.server_test_task,
                dataset_store=self.server_dataset_store,
                config=self.config['client'], # 复用 Client 的 batch_size 等配置
                attack_profile=eval_strategy, # 注入攻击策略
                mode='test' 
            )
            attack_metrics = self.server.eval(
                metrics=group_metrics,
                dataloader=poisoned_loader
            )
            
            # 记录日志 (加前缀区分，例如 "badnets_group/asr")
            prefixed_metrics = {f"{group_name}/{k}": v for k, v in attack_metrics.items()}
            self.logger.log_metrics(prefixed_metrics, step=round_idx)
            res_str = ", ".join([f"{k}={v:.4f}" for k, v in attack_metrics.items()])
            self.logger.info(f"Global Eval ({group_name}): {res_str}")
            
            # 及时清理大对象
            del poisoned_loader

        # 清理工具人
        del tool_client

        # 返回干净指标，供外部使用 (如保存 Checkpoint)
        return clean_metrics
    def _run_local_evaluation(self, round_idx, config, metrics):
        """
        抽样评估客户端本地性能。
        Args:
            metrics: 由 _build_metrics 构建好的 Metric 对象列表
        """
        
        all_clients = list(self.task_set._tasks.keys())
        
        client_candidates = [c for c in all_clients if c != 'server']
        
        eval_ids = random.sample(client_candidates, k=max(1, int(len(client_candidates) * 0.2)))

        results_collector = {m.name: [] for m in metrics}
        
        for cid in eval_ids:
            # 使用动态 Client 类
            client = self.client_class(cid, self.device, self.model_fn)
            task = self.task_set.get_task(cid, "test") 
            store = self.dataset_stores[task.dataset_tag]
            
            # 获取模型
            model_dict = self.server.broadcast([cid])[cid]
            
            # 执行评估
            # 这里的 metrics 是外面传进来的通用列表
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
            
        log_dict = {}
        for metric_name, values in results_collector.items():
            if len(values) > 0:
                log_dict[f"local/avg_{metric_name}"] = np.mean(values)
                log_dict[f"local/std_{metric_name}"] = np.std(values)
                
        self.logger.log_metrics(log_dict, step=round_idx)

    def _save_checkpoint(self, round_idx, is_best=False):

        state = {
            'round': round_idx,
            'model_state_dict': self.global_model.state_dict(),
            'config': self.config
        }
        filename = "checkpoint_best.pth" if is_best else f"checkpoint_{round_idx}.pth"
        path = os.path.join(self.logger.run_dir, filename)
        torch.save(state, path)

    def _build_metrics(self, metric_configs: List[Dict]) -> List[Any]:
        """
        根据配置列表构建 Metric 对象
        """
        metrics = []
        for conf in metric_configs:
            name = conf['name']
            params = conf.get('params', {})
            # 从注册表构建
            metrics.append(METRIC_REGISTRY.build(name, **params))
        return metrics
    
    def _select_clients(self, round_idx: int) -> List[str]:
        """
        选人
        区别于单纯调用server.select_clients,本函数目的为服务后续攻击者采样的拓展
        """
        
        clients_frac = self.config['training'].get('clients_fraction', 0.2)
        num_select = int(len(self.client_ids) * clients_frac)
        selected_ids = []
        attacker_frac = self.config['training'].get('attackers_frac', None)
        #鲁棒性测试(每轮不设定攻击者的数量)
        if attacker_frac is None:
            selected_ids = self.server.select_clients(self.client_ids, \
                                    num_select)
            return selected_ids
        else: #稳定性测试(一定包含攻击者)
            num_attackers = int(num_select * attacker_frac)
            selected_attackers = self.attack_manager.sample_attackers(num_attackers)

            # 好人量
            num_benigns = num_select - len(selected_attackers)
            
            benign_ids = [cid for cid in self.client_ids if cid not in self.attacker_ids]
            
            selected_benign = []
            if num_benigns > 0 and benign_ids:
                selected_benign = self.server.select_clients(client_ids=benign_ids, num_select=num_benigns)

            selected_ids = selected_attackers + selected_benign
            random.shuffle(selected_ids)
            
            return selected_ids