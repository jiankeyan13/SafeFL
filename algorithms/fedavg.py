from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.utils.registry import ALGORITHM_REGISTRY

@ALGORITHM_REGISTRY.register("fedavg")
def build_fedavg_algorithm(model, device, dataset_store, config, **kwargs) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    Returns:
        (server_instance, client_class)
    返回server 实例和 client 类-> 不能在此创建client对象
    """
    server_conf = config.get('server', {})
    seed = kwargs.get('seed', config.get('seed', 42))
    
    screener = None
    aggregator = AvgAggregator()
    refiner = BaseRefiner(config=server_conf.get('refiner', {}))

    server = BaseServer(model=model, aggregator=aggregator, screener=screener, refiner=refiner, device=device, seed=seed)

    return server, BaseClient