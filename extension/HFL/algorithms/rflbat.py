"""HFL RFLBAT: FedrolexServer + RFLBAT 筛选 + sub_avg 聚合."""

from typing import Tuple, Type

import extension.HFL.sub_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.rflbat import RFLBATScreener
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer


@ALGORITHM_REGISTRY.register("rflbat_fedrolex")
def build_rflbat_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = RFLBATScreener(seed=seed, **screener_params)

    aggregator = AGGREGATOR_REGISTRY.build("sub_avg")
    refiner = BaseRefiner(config=params.get("refiner", {}))

    server = FedrolexServer(
        model=model,
        device=device,
        cap_manager=cap_manager,
        screener=screener,
        aggregator=aggregator,
        refiner=refiner,
        seed=seed,
        rolex_to_next=params.get("server", {}).get("rolex_to_next", True),
    )
    return server, BaseClient
