from functools import partial
from typing import Callable, Tuple

from core.client.base_client import BaseClient
from core.client.lockdown_client import LockdownClient
from core.server.aggregator.lockdown_aggregator import LockdownAggregator
from core.server.base_server import BaseServer
from core.server.refiner.lockdown_refiner import LockdownRefiner
from core.server.screener.lockdown_screener import LockdownScreener
from core.utils.registry import ALGORITHM_REGISTRY


@ALGORITHM_REGISTRY.register("lockdown")
def build_lockdown_algorithm(
    model, device, config: dict, seed: int, **params
) -> Tuple[BaseServer, Callable[..., BaseClient]]:
    """
    Build Lockdown: isolated subspace client training + sparse FedAvg +
    consensus-fusion refinement.
    """
    training_conf = config.get("training", {}) if config else {}

    client_conf = params.get("client", {}) or {}
    client_params = client_conf.get("params", client_conf)
    client_params = dict(client_params)
    client_params.setdefault("total_rounds", training_conf.get("rounds", 200))
    client_params.setdefault("mask_seed", seed)

    screener_conf = params.get("screener", {}) or {}
    screener_params = screener_conf.get("params", screener_conf)
    screener = LockdownScreener(**screener_params)

    aggregator_conf = params.get("aggregator", {}) or {}
    aggregator_params = aggregator_conf.get("params", aggregator_conf)
    aggregator = LockdownAggregator(device=device, **aggregator_params)

    refiner_conf = params.get("refiner", {}) or {}
    refiner_params = refiner_conf.get("params", refiner_conf)
    refiner = LockdownRefiner(**refiner_params)

    server = BaseServer(
        model=model,
        device=device,
        screener=screener,
        aggregator=aggregator,
        refiner=refiner,
        seed=seed,
    )
    return server, partial(LockdownClient, **client_params)
