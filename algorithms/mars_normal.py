from typing import Tuple, Type

from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.base_server import BaseServer
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.mars import MARSGammaBnStatScreener
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("mars_normal")
def build_mars_normal_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    MarsNormal: 基于 MARSGammaBnStatScreener 的 MARS, 每通道缩放置为 |gamma| * std.
    其余流程 (CBE, Wasserstein, K-Means) 与 mars 相同.
    """
    del config
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = MARSGammaBnStatScreener(seed=seed, **screener_params)

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
