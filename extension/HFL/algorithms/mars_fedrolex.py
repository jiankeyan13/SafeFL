"""HFL MARS: FedrolexServer + MARSScreener + sub_avg 聚合."""

from typing import Tuple, Type

import extension.HFL.sub_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.server.screener.mars import MARSScreener
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer
from extension.HFL.hfl_refiner import HFLRefiner


@ALGORITHM_REGISTRY.register("mars_fedrolex")
def build_mars_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    """
    异构联邦学习 MARS: FedrolexServer + CBE 聚类筛选 + sub_avg 聚合.
    """
    del config
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = MARSScreener(seed=seed, **screener_params)

    aggregator = AGGREGATOR_REGISTRY.build("sub_avg")
    refiner = HFLRefiner(config=params.get("refiner", {}))

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
