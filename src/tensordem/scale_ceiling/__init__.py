"""IN383-DEM01 — scale ceiling of the granular DEM contact search.

Public surface
--------------
``FloeBed`` / ``FloeBedConfig``  : scale-clean Hertz–Mindlin DEM kernel
``cell_list_pairs``             : O(N) counting-sort contact search
``brute_force_pairs``           : O(N^2) reference search (verification only)
``fit_loglog`` / ``strong_scaling`` / ``weak_scaling`` / ``memory_linearity``
``negative_controls``           : false-negative audit of the acceptance gates
"""

from __future__ import annotations

from .kernel import BedState, FloeBed, FloeBedConfig, NeighbourList
from .search import brute_force_pairs, cell_list_pairs
from .scaling import (
    amdahl_serial_fraction,
    fit_loglog,
    memory_linearity,
    strong_scaling,
    weak_scaling,
)

__all__ = [
    "BedState",
    "FloeBed",
    "FloeBedConfig",
    "NeighbourList",
    "amdahl_serial_fraction",
    "brute_force_pairs",
    "cell_list_pairs",
    "fit_loglog",
    "memory_linearity",
    "strong_scaling",
    "weak_scaling",
]

__version__ = "0.1.0"