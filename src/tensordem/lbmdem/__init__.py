"""Two-way LBM--DEM coupling: immersed-boundary drag closure and its calibration.

Sub-modules
-----------
``lattice``      D3Q19 BGK + Smagorinsky fluid solver with exact momentum bookkeeping
``ibm``          partially saturated bounce-back (MEM) and direct-forcing immersed
                 boundaries, plus sub-cell solid fractions
``drag``         analytic anchors (Stokes, Schiller--Naumann, periodic-array
                 correction, tandem pair reflection series)
``calibration``  case drivers (fixed/pair sphere, free settling) and gate evaluation
"""

from .lattice import D3Q19_C, D3Q19_W, Lattice, LatticeShape
from .ibm import DirectForcingIBM, MomentumExchangeIBM, Sphere, solid_fraction
from .drag import (
    array_correction,
    drag_from_cd,
    reynolds,
    schiller_naumann_cd,
    stokes_drag,
    tandem_pair_ratio,
    terminal_velocity_schiller_naumann,
    terminal_velocity_stokes,
)
from .calibration import (
    FixedSphereSpec,
    SettlingSpec,
    TowedSphereSpec,
    evaluate_gates,
    run_fixed_sphere,
    run_settling,
    run_towed_sphere,
)

__all__ = [
    "D3Q19_C",
    "D3Q19_W",
    "Lattice",
    "LatticeShape",
    "Sphere",
    "solid_fraction",
    "MomentumExchangeIBM",
    "DirectForcingIBM",
    "stokes_drag",
    "schiller_naumann_cd",
    "drag_from_cd",
    "array_correction",
    "tandem_pair_ratio",
    "terminal_velocity_stokes",
    "terminal_velocity_schiller_naumann",
    "reynolds",
    "FixedSphereSpec",
    "TowedSphereSpec",
    "SettlingSpec",
    "run_fixed_sphere",
    "run_towed_sphere",
    "run_settling",
    "evaluate_gates",
]