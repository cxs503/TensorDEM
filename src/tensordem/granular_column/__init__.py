"""Free-surface granular flows: rectangular column collapse and mu(I) rheology.

This package adds the macroscopic free-surface-flow verification surface to
TensorDEM (as opposed to the quasi-static sample verification in
``tensordem.granular2d``): a three-dimensional rectangular column-collapse
kernel with a closed energy ledger (:mod:`.column3d`), the runout/Midi scaling
analysis (:mod:`.scaling`) and the steady-shear mu(I) calibration
(:mod:`.rheology`).
"""
from __future__ import annotations

from .column3d import (
    Column3DCase,
    Column3DError,
    LUBE_EXPONENT,
    PACKING_STATES,
    build_column,
    lube_axisymmetric_runout,
    simulate_column,
)
from .kernel import PairSpringState, WallSpringState, pair_contact_forces, wall_contact_forces

__all__ = [
    "Column3DCase",
    "Column3DError",
    "LUBE_EXPONENT",
    "PACKING_STATES",
    "PairSpringState",
    "WallSpringState",
    "build_column",
    "lube_axisymmetric_runout",
    "pair_contact_forces",
    "simulate_column",
    "wall_contact_forces",
]
