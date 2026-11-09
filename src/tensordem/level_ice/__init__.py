"""Level-ice resistance: bonded-particle DEM section plus engineering anchors.

Public surface
--------------
``formula``        Lindqvist (1989) three-component anchor, Riska-type speed-linear
                   anchor, MT Uikku experimental reference table, provenance hashes.
``multiyear_ice``  Multi-year (MY) ice parameterization, mixed ice field FSD sampling,
                   and Lindqvist MY-extended model with k_MY correction factor.
``sheet``          two-dimensional parallel-bond level-ice sheet with a buoyancy
                   foundation and a rigid hull face (:class:`LevelIceSheet`).
``campaign``       the verification campaign: sweeps, scale mapping, mesh
                   convergence (GCI), mass-scaling and step-size sensitivity,
                   dimensional check and negative controls.

Evidence grades are stated in each module docstring; the campaign artifact
carries per-criterion verdicts with the numbers behind them.
"""
from .formula import (ICEBREAKER, TRANSPORT, HullForm, IceState,
                      lindqvist_components, riska_components)
from .multiyear_ice import (
    FY_DEFAULT,
    MY_DEFAULT,
    FY_RANGES,
    MY_RANGES,
    IceAgeProperties,
    MixedIceField,
    lindqvist_my_components,
    lindqvist_my_sweep,
    my_correction_factor,
    my_ledger_sha256,
)
from .sheet import LevelIceSheet, RunResult, SheetConfig

__all__ = [
    "ICEBREAKER", "TRANSPORT", "HullForm", "IceState", "lindqvist_components",
    "riska_components",
    "FY_DEFAULT", "MY_DEFAULT", "FY_RANGES", "MY_RANGES",
    "IceAgeProperties", "MixedIceField",
    "lindqvist_my_components", "lindqvist_my_sweep",
    "my_correction_factor", "my_ledger_sha256",
    "LevelIceSheet", "RunResult", "SheetConfig",
]
