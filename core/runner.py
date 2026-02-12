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
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    ATTACK_REGISTRY,
)
from data.constants import (
    OWNER_SERVER,
    SPLIT_TRAIN,
    SPLIT_TEST,
    SPLIT_PROXY,
    SPLIT_TEST_GLOBAL,
)
from data.task_generator import TaskGenerator
from core.client.base_client import BaseClient
from core.server.base_server import BaseServer

import data.datasets

# Runner 默认常量
DEFAULT_SEED = 42
DEFAULT_EVAL_INTERVAL = 5
DEFAULT_LOCAL_EVAL_RATIO = 0.2
DEFAULT_CLIENTS_FRACTION = 0.2

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
        self.seed = config.get('seed', DEFAULT_SEED)
        self._set_seed(self.seed)
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

    def _setup_data_pipeline(self):
        """准备数据管道：划分器、TaskGenerator、任务与数据源。"""
        from data.partitioner import DirichletPartitioner, IIDPartitioner

        part_conf = self.config['data']['partitioner']
        if part_conf['name'] == 'dirichlet':
            partitioner = DirichletPartitioner(
                alpha=part_conf.get('alpha', 0.9), seed=self.seed
            )
        else:
            partitioner = IIDPartitioner(seed=self.seed)

        self.task_generator = TaskGenerator(
            dataset_name=self.config['data']['dataset'],
            root=self.config['data']['root'],
            partitioner=partitioner,
            num_clients=self.config['data']['num_clients'],
            val_ratio=self.config['data'].get('val_ratio', 0.1),
            seed=self.seed,
        )
        self.task_set, self.dataset_stores = self.task_generator.generate()
        self.logger.info(f"Data setup complete. Clients: {self.config['data']['num_clients']}")

    def _setup_model(self):
        """准备模型构建函数与全局模型。"""
        model_conf = self.config['model']
        model_cls = MODEL_REGISTRY.get(model_conf['name'])
        self.model_fn = partial(model_cls, **model_conf.get('params', {}))
        self.global_model = self.model_fn().to(self.device)

    def _setup_server_and_algorithm(self):
        """提取 Server 任务、构建 Server 与 Client 类。"""
        self.server_test_task = self.task_set.get_task(OWNER_SERVER, SPLIT_TEST_GLOBAL)
        self.server_dataset_store = self.dataset_stores[self.server_test_task.dataset_tag]

        algo_conf = self.config['algorithm']
        self.server, self.client_class = ALGORITHM_REGISTRY.build(
            algo_conf['name'],
            model=self.global_model,
            device=self.device,
            dataset_store=self.dataset_stores,
            config=self.config,
            seed=self.seed,
            server_test_task=self.server_test_task,
            **algo_conf.get('params', {}),
        )
        self.logger.info(f"Algorithm Loaded: {algo_conf['name']}")
        self.logger.info(f"Server Type: {type(self.server).__name__}")
        self.logger.info(f"Client Type: {self.client_class.__name__}")

        self.lr_scheduler = build_scheduler(self.config['training'])

    def _setup_server_proxy_loader(self):
        """构造 Server Proxy Loader（用于 BN 校准）。"""
        proxy_task = self.task_set.try_get_task(OWNER_SERVER, SPLIT_PROXY)
        if proxy_task is None:
            self.server_proxy_loader = None
            return
        proxy_store = self.dataset_stores[proxy_task.dataset_tag]
        self.server_proxy_loader = self._build_client_dataloader(
            client_id="server_proxy_loader",
            task=proxy_task,
            dataset_store=proxy_store,
            mode='train',
        )

    def _setup_attack_manager(self):
        """设置攻击者与攻击者名单。"""
        self.logger.info(">>> Setting up attackers...")
        self.client_ids = self.task_set.list_client_ids(exclude_server=True)
        self.attack_manager = AttackManager(
            config=self.config.get('attack'),
            all_client_ids=self.client_ids,
            seed=self.seed,
        )
        self.attacker_ids = self.attack_manager.get_attacker_ids()

    def _setup(self):
        self.logger.info(">>> Initializing components...")
        self._setup_data_pipeline()
        self._setup_model()
        self._setup_server_and_algorithm()
        self._setup_server_proxy_loader()
        self._setup_attack_manager()

    def _build_client_dataloader(
        self,
        client_id: str,
        task,
        dataset_store,
        mode: str = 'test',
        attack_profile=None,
    ):
        """使用临时 Client 构建 DataLoader，复用 data_load 逻辑，构建后立即销毁 Client。"""
        temp_client = self.client_class(client_id, self.device, self.model_fn)
        loader = temp_client.data_load(
            task=task,
            dataset_store=dataset_store,
            config=self.config['client'],
            mode=mode,
            attack_profile=attack_profile,
        )
        del temp_client
        return loader

    def run(self):
        total_rounds = self.config['training']['rounds']
        eval_interval = self.config['training'].get('eval_interval', DEFAULT_EVAL_INTERVAL)
        
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

            # Server Step (传入Proxy Loader进行可能的BN校准)
            self.server.step(updates, proxy_loader=self.server_proxy_loader)
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
            
            # 5. 显式销毁 (Python GC 会处理，但显式删除更保险)
            del client
            
        return updates

    def _run_global_evaluation(self, round_idx):
        """
        执行全局评估（主任务 + 各攻击组任务）。
        流程:1. 干净全局测试; 2. 各攻击组评估。
        """
        clean_metrics = self._eval_clean_global(round_idx)
        self._eval_attack_groups(round_idx)
        return clean_metrics

    def _eval_clean_global(self, round_idx):
        """在干净全局测试集上评估主任务指标 (ACC, Loss)。"""
        global_metric_confs = self.config.get('evaluation', {}).get('global', [{'name': 'acc'}])
        clean_metric_objs = self._build_metrics(global_metric_confs)
        clean_metrics = self.server.eval(metrics=clean_metric_objs)

        self.logger.log_metrics(clean_metrics, step=round_idx)
        first_key = list(clean_metrics.keys())[0] if clean_metrics else "N/A"
        first_val = clean_metrics.get(first_key, 0)
        self.logger.info(f"Global Eval (Clean): {first_key} = {first_val:.4f}")
        return clean_metrics

    def _eval_attack_groups(self, round_idx):
        """遍历各攻击组，在被污染测试集上评估攻击指标 (ASR)。"""
        attackers_conf = self.config.get('attack', {}).get('strategies', {})
        if not attackers_conf or not self.attack_manager:
            return

        for group_name, group_config in attackers_conf.items():
            eval_conf = group_config.get('evaluation')
            if not eval_conf:
                continue

            group_metrics = self._build_metrics(eval_conf)
            strategy_conf = group_config['strategy']
            eval_strategy = ATTACK_REGISTRY.build(
                strategy_conf['name'],
                **strategy_conf.get('params', {}),
            )

            poisoned_loader = self._build_client_dataloader(
                client_id="server_eval_tool",
                task=self.server_test_task,
                dataset_store=self.server_dataset_store,
                mode='test',
                attack_profile=eval_strategy,
            )
            attack_metrics = self.server.eval(
                metrics=group_metrics,
                dataloader=poisoned_loader,
            )
            prefixed_metrics = {f"{group_name}/{k}": v for k, v in attack_metrics.items()}
            self.logger.log_metrics(prefixed_metrics, step=round_idx)
            res_str = ", ".join([f"{k}={v:.4f}" for k, v in attack_metrics.items()])
            self.logger.info(f"Global Eval ({group_name}): {res_str}")
            del poisoned_loader
        
    def _run_local_evaluation(self, round_idx, config, metrics):
        """
        抽样评估客户端本地性能。
        Args:
            metrics: 由 _build_metrics 构建好的 Metric 对象列表
        """
        
        client_candidates = self.task_set.list_client_ids(exclude_server=True)
        eval_ids = random.sample(
            client_candidates,
            k=max(1, int(len(client_candidates) * DEFAULT_LOCAL_EVAL_RATIO)),
        )

        results_collector = {m.name: [] for m in metrics}
        
        for cid in eval_ids:
            # 使用动态 Client 类
            client = self.client_class(cid, self.device, self.model_fn)
            task = self.task_set.get_task(cid, SPLIT_TEST) 
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
        
        clients_frac = self.config['training'].get('clients_fraction', DEFAULT_CLIENTS_FRACTION)
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