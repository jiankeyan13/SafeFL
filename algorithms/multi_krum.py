from typing import Tuple, Type

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.refiner.base_refiner import BaseRefiner
from core.server.screener.krum import KrumScreener
from core.utils.registry import ALGORITHM_REGISTRY

@ALGORITHM_REGISTRY.register("multi_krum")
def build_multi_krum_algorithm(model, device, dataset_store, config, **kwargs) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    Multi-Krum 聚合算法：先用 Krum 筛选，再用 FedAvg 聚合。
    
    Returns:
        (server_instance, client_class)
        返回 server 实例 and client 类 -> 不能在此创建 client 对象
    """
    server_conf = config.get('server', {})
    seed = kwargs.get('seed', config.get('seed', 42))
    screener_conf = server_conf.get('screener', {})
    
    # 从配置中读取 Krum 参数
    screener_params = screener_conf.get('params', {})
    f = screener_params.get('f', 0)  # 假设的攻击者数量
    m = screener_params.get('m', 1)  # Multi-Krum 保留的客户端数量
    
    # 构建组件
    screener = KrumScreener(f=f, m=m)
    aggregator = AvgAggregator()
    refiner = BaseRefiner()

    server = BaseServer(
        model=model,
        aggregator=aggregator,
        screener=screener,
        refiner=refiner,
        device=device,
        seed=seed,
    )

    return server, BaseClient
