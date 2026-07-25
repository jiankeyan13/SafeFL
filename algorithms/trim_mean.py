from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.trim_mean_aggregator import TrimmedMeanAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("trim_mean")
def build_trim_mean_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    Trimmed Mean 防御: 按坐标维截断两端后取平均.

    默认两端各去除 30% (trim_ratio=0.3).
    """
    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    trim_ratio = aggregator_params.get("trim_ratio", 0.3)
    aggregator = TrimmedMeanAggregator(trim_ratio=trim_ratio, device=device)
    refiner = BaseRefiner(config=params.get("refiner", {}))

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        refiner=refiner,
        device=device,
        seed=seed,
    )
    return server, BaseClient
