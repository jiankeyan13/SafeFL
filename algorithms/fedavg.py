from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("fedavg")
def build_fedavg_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Type[BaseClient]]:
    """返回 server 实例和 client 类."""
    
    server = BaseServer(
        model=model,
        aggregator=AvgAggregator(),
        refiner=BaseRefiner(config=params.get("refiner", {})),
        device=device,
        seed=seed,
    )
    return server, BaseClient
