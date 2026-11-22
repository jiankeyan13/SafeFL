"""SafeFL v2 server-side backdoor defense package (1 defense = 1 file)."""
from __future__ import annotations

from core.defense import (
    alignins,
    backdoor_indicator,
    deepsight,
    dnc,
    fdcr,
    flame,
    fltrust,
    foolsgold,
    freqfed,
    mars,
    multi_krum,
    rfa,
    rlr,
    snowball,
    spmc,
    trim_mean,
)
from core.defense.base import BaseDefense, FedAvgDefense, build_defense

__all__ = [
    "BaseDefense",
    "FedAvgDefense",
    "alignins",
    "backdoor_indicator",
    "build_defense",
    "deepsight",
    "dnc",
    "fdcr",
    "flame",
    "fltrust",
    "foolsgold",
    "freqfed",
    "mars",
    "multi_krum",
    "rfa",
    "rlr",
    "snowball",
    "spmc",
    "trim_mean",
]
