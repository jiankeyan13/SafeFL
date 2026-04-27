"""HFL: 仅按 gamma 幅值选通道的异构 server + sub_avg."""

from functools import partial
from typing import Callable, Tuple

import extension.HFL.sub_aggregator  # noqa: F401
from core.client.base_client import BaseClient
from core.client.gamma_l1_client import GammaL1Client
from core.utils.registry import AGGREGATOR_REGISTRY, ALGORITHM_REGISTRY
from extension.HFL.cap_manager import CapManager
from extension.HFL.gamma_magnitude_server import GammaMagnitudeServer
from extension.HFL.hfl_refiner import HFLRefiner


@ALGORITHM_REGISTRY.register("gamma_magnitude")
@ALGORITHM_REGISTRY.register("hetero_gemma")
def build_gamma_magnitude_algorithm(
    model, device, config: dict, seed: int, cap_manager: CapManager, **params
) -> Tuple[GammaMagnitudeServer, Callable[..., BaseClient]]:
    """异构 FL: BN |gamma| 降序选通道, 子模型聚合为 sub_avg."""
    del config
    server_params = params.get("server", {}) or {}
    client_params = params.get("client", {}) or {}
    aggregator = AGGREGATOR_REGISTRY.build("sub_avg")
    refiner = HFLRefiner(config=params.get("refiner", {}))

    server = GammaMagnitudeServer(
        model=model,
        device=device,
        cap_manager=cap_manager,
        screener=None,
        aggregator=aggregator,
        refiner=refiner,
        seed=seed,
        gamma_eps=float(server_params.get("gamma_eps", 1e-12)),
    )
    return server, partial(GammaL1Client, **client_params)
