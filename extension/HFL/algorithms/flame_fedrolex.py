"""HFL FLAME: 异构联邦学习专用的 FLAME 算法 (FedrolexServer + HdbscanScreener + SubFlameAggregator + NoiseRefiner)."""

from typing import Tuple, Type

import extension.HFL.sub_flame_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from extension.HFL.hfl_refiner import HFLNoiseRefiner
from core.server.screener.hdbscan import HdbscanScreener
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer


@ALGORITHM_REGISTRY.register("flame_fedrolex")
def build_flame_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    """
    异构联邦学习专用 FLAME: FedrolexServer + HDBSCAN 筛选 + sub_flame 聚合 + 噪声精炼.

    流程: HDBSCAN 筛选 -> sub_flame 聚合 (掩码感知 + 范数裁剪) -> 噪声添加.

    Returns:
        (server_instance, client_class)
    """
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = HdbscanScreener(**screener_params)

    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = AGGREGATOR_REGISTRY.build("sub_flame", device=device, **aggregator_params)

    refiner_conf = params.get("refiner", {})
    refiner_params = refiner_conf.get("params", refiner_conf)
    refiner = HFLNoiseRefiner(**refiner_params)

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
