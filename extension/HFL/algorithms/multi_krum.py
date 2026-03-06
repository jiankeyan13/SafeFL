"""HFL Multi-Krum: 异构联邦学习专用的严格 Multi-Krum 算法 (FedrolexServer + KrumScreener)."""

from typing import Tuple, Type

import core.server.screener  # noqa: F401
import extension.HFL.sub_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.server.refiner.base_refiner import BaseRefiner
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY, SCREENER_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.fedrolex_server import FedrolexServer


@ALGORITHM_REGISTRY.register("multi_krum_fedrolex")
def build_multi_krum_fedrolex_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[FedrolexServer, Type[BaseClient]]:
    """
    异构联邦学习专用 Multi-Krum: FedrolexServer + Krum 筛选 + sub_avg 聚合.

    严格 Multi-Krum 流程: Krum 筛选 -> sub_avg 聚合 (无其他可替换组件).

    Args:
        params.screener.f: 假设攻击者数量, 默认 0
        params.screener.m: 保留客户端数量, 默认 1 (m=1 即标准 Krum)

    Returns:
        (server_instance, client_class)
    """
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    f = screener_params.get("f", 0)
    m = screener_params.get("m", 1)

    screener = SCREENER_REGISTRY.build("krum", f=f, m=m)
    aggregator = AGGREGATOR_REGISTRY.build("sub_avg")
    refiner = BaseRefiner(config={})

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
