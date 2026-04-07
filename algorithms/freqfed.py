from typing import Tuple, Type

from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.base_server import BaseServer
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.freqfed import FreqFedScreener
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("freqfed")
def build_freqfed_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    Build FreqFed with:
    - FreqFedScreener (frequency-based filtering)
    - AvgAggregator (FedAvg over screened clients)
    - BaseRefiner (load new_state and optional BN calibration)
    """
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = FreqFedScreener(**screener_params)

    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = AvgAggregator(device=device, **aggregator_params)

    refiner_conf = params.get("refiner", {})
    refiner_params = refiner_conf.get("params", refiner_conf)
    refiner = BaseRefiner(config=refiner_params)

    server = BaseServer(
        model=model,
        device=device,
        screener=screener,
        aggregator=aggregator,
        refiner=refiner,
        seed=seed,
    )
    return server, BaseClient
