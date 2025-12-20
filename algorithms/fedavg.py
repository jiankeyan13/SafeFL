from typing import Tuple, Type
from torch.utils.data import Subset,DataLoader

from core.server.base_server import BaseServer
from core.client.base_client import BaseClient
from core.server.aggregator.avg_aggregator import AvgAggregator
from core.server.updater.base_updater import BaseUpdater
from core.utils.registry import ALGORITHM_REGISTRY

@ALGORITHM_REGISTRY.register("fedavg")
def build_fedavg_algorithm(model, device, dataset_store, config, **kwargs) -> Tuple[BaseServer, Type[BaseClient]]:
    """
    Returns:
        (server_instance, client_class)
    返回server 实例和 client 类->不能在此创建client对象
    """
    server_conf = config.get('server', {})
    
    screener = None
    aggregator = AvgAggregator()
    updater = BaseUpdater(config=server_conf.get('updater', {}))
    
    # 从 dataset_store 中提取 server 测试集，
    server_test_task = kwargs.get('server_test_task') 
    
    test_loader = None
    if server_test_task:
        test_ds_store = dataset_store[server_test_task.dataset_tag]
        server_dataset = Subset(test_ds_store, server_test_task.indices)
        batch_size = config.get('client', {}).get('batch_size', 64)
        test_loader = DataLoader(server_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    server = BaseServer(model=model, aggregator=aggregator, screener=screener, updater=updater, \
                        device=device, test_loader=test_loader)

    return server, BaseClient