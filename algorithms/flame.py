from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.screener.hdbscan import HdbscanScreener
from core.server.aggregator.flame_aggregator import FlameAggregator
from core.server.refiner.noise_refiner import NoiseRefiner
from core.utils.registry import ALGORITHM_REGISTRY

# 客户端不进行梯度裁剪效果更好 
# https://wandb.ai/jiankeyan13-lab/FL-Test?nw=nwuserjiankeyan13xyz

@ALGORITHM_REGISTRY.register("flame")
def build_flame_algorithm(model, device, dataset_store, config, **kwargs) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    构建 FLAME 算法：HDBSCAN 筛选 + 范数裁剪聚合 + 噪声添加。
    
    Returns:
        (server_instance, client_class)
    """
    server_conf = config.get('server', {})
    seed = kwargs.get('seed', config.get('seed', 42))
    
    # 初始化 FLAME 三组件
    screener_conf = server_conf.get('screener', {})
    screener = HdbscanScreener(**screener_conf.get('params', {}))
    
    aggregator_conf = server_conf.get('aggregator', {})
    aggregator = FlameAggregator(**aggregator_conf.get('params', {}))
    
    refiner_conf = server_conf.get('refiner', {})
    refiner = NoiseRefiner(**refiner_conf.get('params', {}))

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        screener=screener,
        refiner=refiner,
        device=device,
        seed=seed,
    )

    return server, BaseClient
