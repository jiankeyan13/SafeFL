"""HFL DeepSight: 异构联邦学习专用的 DeepSight 算法 (FedrolexServer + DeepSightScreener + SubFlameAggregator)."""

from typing import Tuple, Type

import extension.HFL.sub_flame_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from extension.HFL.hfl_refiner import HFLRefiner
from core.server.screener.deepsight_screener import DeepSightScreener
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer


@ALGORITHM_REGISTRY.register("deepsight_fedrolex")
def build_deepsight_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    """
    异构联邦学习专用 DeepSight: FedrolexServer + DeepSightScreener + sub_flame 聚合.

    流程: DeepSight 筛选 -> sub_flame 聚合 (掩码感知 + 范数裁剪).

    Returns:
        (server_instance, client_class)
    """
    screener_conf = params.get("screener", {})
    screener = DeepSightScreener(**screener_conf) if screener_conf else DeepSightScreener()

    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = AGGREGATOR_REGISTRY.build("sub_flame", device=device, **aggregator_params)

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
