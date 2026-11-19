from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.rlr_aggregator import RLRAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("rlr")
def build_rlr_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    RLR 防御: 逐维符号投票 + Robust Learning Rate 掩码.

    默认 robustLR_threshold=4 (Ozdayi et al., AAAI 2021).
    """
    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    robustLR_threshold = aggregator_params.get("robustLR_threshold", 4)
    aggregator = RLRAggregator(
        robustLR_threshold=robustLR_threshold,
        device=device,
    )
    refiner = BaseRefiner(config=params.get("refiner", {}))

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        refiner=refiner,
        device=device,
        seed=seed,
    )
    return server, BaseClient
