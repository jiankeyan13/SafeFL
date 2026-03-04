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
def build_flame_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    构建 FLAME 算法: HDBSCAN 筛选 + 范数裁剪聚合 + 噪声添加.

    Returns:
        (server_instance, client_class)
    """
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = HdbscanScreener(**screener_params)

    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = FlameAggregator(**aggregator_params)

    refiner_conf = params.get("refiner", {})
    refiner_params = refiner_conf.get("params", refiner_conf)
    refiner = NoiseRefiner(**refiner_params)

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        screener=screener,
        refiner=refiner,
        device=device,
        seed=seed,
    )
    return server, BaseClient
