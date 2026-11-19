"""FoolsGold 算法: 基于历史梯度相似性的 Sybil 防御."""

from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.foolsgold import FoolsGoldScreener
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("foolsgold")
def build_foolsgold_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    FoolsGold 聚合算法: 筛选阶段使用 FoolsGold 动态权重, 聚合阶段使用加权 FedAvg.

    筛选阶段为每个客户端分配 alpha_i (0~1), 与 AvgAggregator 的 screen_scores 结合,
    实现 global_delta = sum_i(alpha_i * delta_i) 的加权聚合.

    Args:
        params.screener.use_history: 是否使用历史梯度累加, 默认 True
        params.screener.logit_center: logit 拉伸中心, 默认 0.5

    Returns:
        (server_instance, client_class)
    """
    screener_params = params.get("screener", {}).get("params", params.get("screener", {}))
    use_history = screener_params.get("use_history", True)
    logit_center = screener_params.get("logit_center", 0.5)

    screener = FoolsGoldScreener(use_history=use_history, logit_center=logit_center)
    aggregator = AvgAggregator()
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
