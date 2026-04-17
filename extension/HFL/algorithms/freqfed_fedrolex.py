"""HFL FreqFed: FedrolexServer + FreqFedScreener + sub_avg 聚合."""

from typing import Tuple, Type

import extension.HFL.sub_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.server.screener.freqfed import FreqFedScreener
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer
from extension.HFL.hfl_refiner import HFLRefiner


@ALGORITHM_REGISTRY.register("freqfed_fedrolex")
def build_freqfed_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    """
    异构联邦学习 FreqFed: FedrolexServer + 频域特征 HDBSCAN 筛选 + sub_avg 聚合.
    """
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = FreqFedScreener(**screener_params)

    aggregator = AGGREGATOR_REGISTRY.build("sub_avg")

    refiner_conf = params.get("refiner", {})
    refiner_params = refiner_conf.get("params", refiner_conf)
    refiner = HFLRefiner(config=refiner_params)

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
