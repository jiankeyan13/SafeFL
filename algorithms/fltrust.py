"""
FLTrust 算法: 基于 proxy 数据的 server reference update 作为信任锚点,
对客户端 delta 计算 cosine similarity 过 ReLU 得 TS, 同范数归一化后 TS 加权聚合.
"""
from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.fltrust_aggregator import FLTrustAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.fltrust_screener import FLTrustScreener
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("fltrust")
def build_fltrust_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    FLTrust 算法: BaseServer + FLTrustScreener + FLTrustAggregator + BaseRefiner.
    delta_0 在 screener 中基于 proxy 数据计算, 不参与加权求和.
    """
    screener_conf = params.get("screener", {})
    screener_params = screener_conf.get("params", screener_conf)
    screener = FLTrustScreener(**screener_params)

    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = FLTrustAggregator(device=device, **aggregator_params)

    refiner = BaseRefiner(config=params.get("refiner", {}))

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        screener=screener,
        refiner=refiner,
        device=device,
        seed=seed,
    )
    return server, BaseClient
