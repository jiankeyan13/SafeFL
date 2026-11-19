from typing import Tuple, Type

from core.client.base_client import BaseClient
from core.server.aggregator.alignins_aggregator import AlignInsAggregator
from core.server.base_server import BaseServer
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.alignins import AlignInsScreener
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("alignins")
def build_alignins_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    AlignIns:
    - Screener: TDA + MPSA + Z-score filtering
    - Aggregator: benign-median norm clipping + benign average
    - Refiner: standard model update
    """
    del config, seed

    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = AlignInsScreener(**screener_params)

    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = AlignInsAggregator(device=device, **aggregator_params)

    refiner_conf = params.get("refiner", {})
    refiner_params = refiner_conf.get("params", refiner_conf)
    refiner = BaseRefiner(config=refiner_params)

    server = BaseServer(
        model=model,
        device=device,
        screener=screener,
        aggregator=aggregator,
        refiner=refiner,
    )
    return server, BaseClient
