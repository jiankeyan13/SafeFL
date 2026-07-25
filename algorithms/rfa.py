from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.rfa_aggregator import RFAAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("rfa")
def build_rfa_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    RFA 防御: 对客户端更新做加权几何中位数聚合 (平滑 Weiszfeld).

    默认 num_iters=4, nu=1e-6.
    """
    aggregator_conf = params.get("aggregator", {})
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    num_iters = aggregator_params.get("num_iters", 4)
    nu = aggregator_params.get("nu", 1e-6)
    aggregator = RFAAggregator(num_iters=num_iters, nu=nu, device=device)
    refiner = BaseRefiner(config=params.get("refiner", {}))

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        refiner=refiner,
        device=device,
        seed=seed,
    )
    return server, BaseClient
