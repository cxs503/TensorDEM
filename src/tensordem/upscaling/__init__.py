"""DEM coarse-graining (upscaling) scaling laws + fine/coarse macro-anchor tests.

Public API
----------
law      : :class:`BaseMaterial`, :class:`ScaledMaterial`, :func:`scale_material`,
           rule table, applicability domain, stability gates.
engine   : vectorised 3-D DEM integrator (same contact laws as
           :mod:`tensordem.granular3d_core`), analytic walls.
cases    : direct-shear cell, rotating drum, uniaxial compression.
runner   : f-sweep, error budget, negative-control adjudication, JSON artefact.
"""
from .law import (  # noqa: F401
    BaseMaterial,
    ScaledMaterial,
    scale_material,
    scaling_rule_table,
    applicability_domain,
    stability_gate,
    damping_ratio_from_restitution,
    restitution_from_damping_ratio,
    pair_damping,
    APPROACHES,
    NEGATIVE_CONTROLS,
)
from .engine import Sim, PlaneWall, CylinderWall, pair_forces, build_pairs  # noqa: F401
from .cases import run_case, CaseResult  # noqa: F401
