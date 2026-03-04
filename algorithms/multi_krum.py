from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.krum import KrumScreener
from core.utils.registry import ALGORITHM_REGISTRY

@ALGORITHM_REGISTRY.register("multi_krum")
def build_multi_krum_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    Multi-Krum 聚合算法: 先用 Krum 筛选, 再用 FedAvg 聚合.

    Returns:
        (server_instance, client_class)
        返回 server 实例和 client 类, 不能在此创建 client 对象.
    """
    screener_params = params.get("screener", {}).get("params", params.get("screener", {}))
    f = screener_params.get("f", 0)
    m = screener_params.get("m", 1)
    screener = KrumScreener(f=f, m=m)
    aggregator = AvgAggregator()
    refiner = BaseRefiner(config=params.get("refiner", {}))

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        screener=screener,
        refiner=refiner,
        device=device,
        seed=seed,
    )

    return server, BaseClient
