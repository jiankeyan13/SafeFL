"""
FLTrust 算法: 基于 proxy 数据的 server reference update 作为信任锚点,
对客户端 delta 计算 cosine similarity 过 ReLU 得 TS, 同范数归一化后 TS 加权聚合.
"""
from typing import Any, Dict, Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.fltrust_aggregator import FLTrustAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.fltrust_screener import FLTrustScreener
from core.utils.registry import ALGORITHM_REGISTRY


def _client_aligned_screener_params(config: dict, screener_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    论文要求 server / client 共用同一套 ModelUpdate(β, R_l).
    未在 algorithm.params.screener 中显式覆盖的项, 从 client 配置继承.
    """
    client_conf = (config or {}).get("client", {})
    aligned = {
        "lr": client_conf.get("lr", 0.1),
        "momentum": client_conf.get("momentum", 0.9),
        "weight_decay": client_conf.get("weight_decay", 5e-4),
        "epochs": client_conf.get("epochs", 1),
    }
    aligned.update(screener_params or {})
    return aligned


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
    screener = FLTrustScreener(**_client_aligned_screener_params(config, screener_params))

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
