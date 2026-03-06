"""HFL Fedrolex: 异构联邦学习专用, 仅使用 sub_avg 聚合 (无 screener/refiner 等无关组件)."""

from typing import Tuple, Type

import extension.HFL.sub_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.server.refiner.base_refiner import BaseRefiner
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer


@ALGORITHM_REGISTRY.register("fedrolex")
def build_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    """
    异构联邦学习 Fedrolex: FedrolexServer + sub_avg 聚合.

    仅 sub_avg 聚合, 无 screener/refiner 等可替换组件.
    """
    aggregator = AGGREGATOR_REGISTRY.build("sub_avg")
    refiner = BaseRefiner(config={})

    server = FedrolexServer(
        model=model,
        device=device,
        cap_manager=cap_manager,
        screener=None,
        aggregator=aggregator,
        refiner=refiner,
        seed=seed,
        rolex_to_next=params.get("server", {}).get("rolex_to_next", True),
    )
    return server, BaseClient
