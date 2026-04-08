from typing import Tuple, Type

from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.base_server import BaseServer
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.mars import MARSScreener
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("mars")
def build_mars_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    MARS: MARS 筛选 + 平均聚合 + 标准模型更新。
    """
    del config
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = MARSScreener(seed=seed, **screener_params)

    aggregator = AvgAggregator(device=device)
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
