"""HFL DPP: DPPServer + sub_avg 聚合."""

from functools import partial
from typing import Callable, Tuple

import extension.HFL.sub_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.client.gamma_l1_client import GammaL1Client
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.dpp_server import DPPServer
from extension.HFL.hfl_refiner import HFLRefiner


@ALGORITHM_REGISTRY.register("dpp")
@ALGORITHM_REGISTRY.register("hetero_dpp")
def build_dpp_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[DPPServer, Callable[..., BaseClient]]:
    """异构联邦学习 DPP 通道采样: DPPServer + sub_avg."""
    del config
    server_params = params.get("server", {}) or {}
    client_params = params.get("client", {}) or {}
    aggregator = AGGREGATOR_REGISTRY.build("sub_avg")
    refiner = HFLRefiner(config=params.get("refiner", {}))

    server = DPPServer(
        model=model,
        device=device,
        cap_manager=cap_manager,
        screener=None,
        aggregator=aggregator,
        refiner=refiner,
        seed=seed,
        gamma_eps=float(server_params.get("gamma_eps", 1e-12)),
        weight_eps=float(server_params.get("weight_eps", 1e-12)),
    )
    return server, partial(GammaL1Client, **client_params)
