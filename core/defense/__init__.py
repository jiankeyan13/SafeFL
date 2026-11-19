"""SafeFL v2 server-side backdoor defense package (1 defense = 1 file)."""
from __future__ import annotations

from core.defense import (
    alignins,
    deepsight,
    dnc,
    flame,
    fltrust,
    foolsgold,
    freqfed,
     functional,
    mars,
    multi_krum,
    rfa,
    rlr,
    trim_mean,
)
from core.defense.base import BaseDefense, FedAvgDefense, build_defense

__all__ = [
    "BaseDefense",
    "FedAvgDefense",
    "alignins",
    "build_defense",
    "deepsight",
    "dnc",
    "flame",
    "fltrust",
    "foolsgold",
    "freqfed",
    "functional",
    "mars",
    "multi_krum",
    "rfa",
    "rlr",
    "trim_mean",
]
