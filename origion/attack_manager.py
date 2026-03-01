import random
import numpy as np
from typing import Dict, Any, List, Optional

from core.utils.registry import ATTACK_REGISTRY
from core.attack.base_strategy import AttackStrategy

class AttackManager:
    """
    攻击管理器 (Attack Scenario Generator & Dispatcher)。
    
    职责:
    1. 在实验开始时，根据配置（比例）随机但可复现地分配攻击者角色及其策略。
    2. 在训练过程中，为 Runner 提供查询接口，获取攻击策略和参与本轮的攻击者。
    """

    def __init__(self, 
                 config: Optional[Dict[str, Any]], 
                 all_client_ids: List[str], 
                 seed: int):
        """
        Args:
            config: 'attack' 部分的配置，如果为 None 则无攻击。
            all_client_ids: 所有可用客户端的 ID 列表。
            seed: 全局随机种子，用于保证角色分配的可复现性。
        """
        self.config = config
        self.all_client_ids = all_client_ids
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        #  { "client_id": AttackStrategy 实例 }
        self._attacker_map: Dict[str, AttackStrategy] = {}
        self.attacker_ids: List[str] = []
        
        self._assign_roles()

    def _assign_roles(self):
        """
        [核心逻辑] 解析配置，分配攻击者角色和策略。
        这是一个两阶段的采样过程。
        """
        if not self.config or self.config.get('fraction', 0.0) <= 0:
            print("INFO: No attackers configured for this run.")
            return

        # --- 阶段一：宏观采样 (确定谁是攻击者) ---
        total_attack_fraction = self.config['fraction']
        num_attackers = int(len(self.all_client_ids) * total_attack_fraction)
        
        if num_attackers == 0:
            return

        # rng = np.random.default_rng(self.seed)
        self.attacker_ids = list(self.rng.choice(self.all_client_ids, size=num_attackers, replace=False))
        
        print(f">>> Assigning {num_attackers} attackers from {len(self.all_client_ids)} total clients...")

        # --- 阶段二：微观分配 (在攻击者内部分工) ---
        unassigned_attackers = list(self.attacker_ids)
        self.rng.shuffle(unassigned_attackers)

        strategies_config = self.config.get('strategies', {})
        if not strategies_config:
            print("WARNING: Attack fraction > 0 but no strategies defined. All attackers will be passive.")
            return

        for group_name, group_config in strategies_config.items():
            # a. 计算本组需要分配多少人
            group_fraction = group_config.get('fraction', 0.0)
            num_to_sample = int(num_attackers * group_fraction)
            
            if num_to_sample <= 0:
                continue

            # b. 从待分配池中安全地采样
            num_to_sample = min(num_to_sample, len(unassigned_attackers))
            if num_to_sample == 0:
                break # 没有更多攻击者可供分配

            sampled_clients = unassigned_attackers[:num_to_sample]
            unassigned_attackers = unassigned_attackers[num_to_sample:]
            
            # c. 为采样出的客户端构建并分配策略
            strategy_conf = group_config['strategy']
            strategy_name = strategy_conf['name']
            strategy_params = strategy_conf.get('params', {})
            
            for client_id in sampled_clients:
                # 注入种子，保证每个客户端的攻击行为可复现
                client_seed = self.seed + int(client_id.split('_')[-1])
                strategy_params['seed'] = client_seed
                
                strategy_instance = ATTACK_REGISTRY.build(strategy_name, **strategy_params)
                self._attacker_map[client_id] = strategy_instance
                print(f"    - Client {client_id} assigned to group '{group_name}' (Strategy: {strategy_name})")

    # =========================================================================
    # 公共接口 (Public API for Runner)
    # =========================================================================

    def get_strategy(self, client_id: str) -> Optional[AttackStrategy]:
        """
        查询指定客户端的攻击策略。
        """
        return self._attacker_map.get(client_id)

    def get_attacker_ids(self) -> List[str]:
        """
        返回所有被分配为攻击者的客户端ID列表。
        """
        return self.attacker_ids

    def sample_attackers(self, num_to_sample: int) -> List[str]:
        """
        从所有攻击者中随机抽取一部分参与本轮训练。
        """
        if not self.attacker_ids:
            return []
            
        num_to_sample = min(num_to_sample, len(self.attacker_ids))
        selected = self.rng.choice(self.attacker_ids, size=num_to_sample, replace=False)
        return selected.tolist()