"""Simulation & system-roles layer for SafeFL v2."""
from __future__ import annotations

from core.simulation.client import BenignClient
from core.simulation.coordinator import AttackCoordinator, ParticipantTask
from core.simulation.executor import (
    BaseExecutor,
    ClientWorker,
    LocalExecutor,
    RayExecutor,
    create_executor,
)
from core.simulation.runner import Runner
from core.simulation.server import Server

__all__ = [
    "AttackCoordinator",
    "BaseExecutor",
    "BenignClient",
    "ClientWorker",
    "LocalExecutor",
    "ParticipantTask",
    "RayExecutor",
    "Runner",
    "Server",
    "create_executor",
]
