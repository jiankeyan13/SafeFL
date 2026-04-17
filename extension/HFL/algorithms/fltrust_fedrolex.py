"""HFL FLTrust: FedrolexServer + FLTrustScreener + FLTrustAggregator."""

from typing import Tuple, Type

import core.server.aggregator.fltrust_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.server.screener.fltrust_screener import FLTrustScreener
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer
from extension.HFL.hfl_refiner import HFLRefiner


@ALGORITHM_REGISTRY.register("fltrust_fedrolex")
def build_fltrust_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    """
    异构联邦学习 FLTrust: FedrolexServer + proxy 锚点余弦信任分数 + FLTrust 聚合.
    需 Runner 传入 proxy_loader (见 HeteroRunner).
    """
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = FLTrustScreener(**screener_params)

    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = AGGREGATOR_REGISTRY.build("fltrust", device=device, **aggregator_params)

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
