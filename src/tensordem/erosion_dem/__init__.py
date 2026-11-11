"""DEM-side solid-particle erosion verification face (node p176w2-n11).

The face answers one question end to end: *given a sand jet, how much wall
material disappears, and does the number survive cross-adjudication?*

    contact.py          published contact mechanics: plastic normal restitution
                        (Johnson quarter power) and Coulomb oblique restitution;
                        closed-form ballistics for the flight to the wall
    crater.py           single impact: erosion ratio -> removed volume ->
                        spherical-cap crater (depth, radius) -> volumetric deposit
                        on the wall patch
    stream.py           multi-particle stream: trajectory-measured impacts,
                        bounces, cumulative E [kg/kg], mass ledger, sweeps
    breakage_bridge.py  optional comminution coupling + the static check that the
                        erosion core has no breakage dependency (criterion C7)

Everything is re-used rather than re-implemented: material data, the Oka/Finnie
angle functions, the DNV-RP-O501 ductile coefficients and the velocity-exponent
fitter all come from :mod:`tensordem.erosion`.  The only calibrated number in
this package is Oka's leading anchor, whose provenance travels with it
(:data:`tensordem.erosion_dem.stream.CASE_ANCHOR_PROVENANCE`).

Entry points: ``python -m tensordem.erosion_dem {impact,stream,pipe,verify}``
(console script ``tensordem-erosion``); the same recorder is driven by the
slurry pipe solver, see ``benchmarks/erosion_pipe_coupled.py``.

Reproduce: ``python benchmarks/erosion_dem_verification.py``
Tests: ``python -m pytest tests/test_erosion_dem.py -q``
"""
from __future__ import annotations

from .breakage_bridge import (
    FragmentationPolicy,
    erosion_core_imports_comminution,
    fragmentation_hook,
    mass_scale,
)
from .contact import (
    HERTZ_PLASTIC_PREFACTOR,
    JointImpactKinematics,
    oblique_restitution,
    plastic_normal_restitution,
    yield_velocity_m_s,
)
from .crater import (
    SPHERE_FACTOR,
    CraterGeometry,
    OkaAnchoredLaw,
    WallSurface,
    indentation_depth_m,
    single_impact_crater,
    sphere_cap_depth_from_volume,
    sphere_cap_radius,
    sphere_cap_volume,
)
from .recorder import ImpactErosionRecorder
from .stream import (
    CASE_ANCHOR_PROVENANCE,
    CASE_LAW,
    CASE_MATERIAL,
    StreamConfig,
    aggregate_exponent,
    angle_sweep,
    monotonicity,
    peak_angle_from_samples,
    run_impact_stream,
    velocity_sweep,
)

__all__ = [
    "ImpactErosionRecorder",
    "HERTZ_PLASTIC_PREFACTOR",
    "JointImpactKinematics",
    "oblique_restitution",
    "plastic_normal_restitution",
    "yield_velocity_m_s",
    "SPHERE_FACTOR",
    "CraterGeometry",
    "OkaAnchoredLaw",
    "WallSurface",
    "indentation_depth_m",
    "single_impact_crater",
    "sphere_cap_depth_from_volume",
    "sphere_cap_radius",
    "sphere_cap_volume",
    "CASE_ANCHOR_PROVENANCE",
    "CASE_LAW",
    "CASE_MATERIAL",
    "StreamConfig",
    "aggregate_exponent",
    "angle_sweep",
    "monotonicity",
    "peak_angle_from_samples",
    "run_impact_stream",
    "velocity_sweep",
    "FragmentationPolicy",
    "erosion_core_imports_comminution",
    "fragmentation_hook",
    "mass_scale",
]