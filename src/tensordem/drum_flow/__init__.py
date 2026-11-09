"""Rotary-drum granular flow: DEM kernel, flow-regime map, power ledger.

Layout
------
``kernel``  2-D monodisperse disk DEM with a *rotating* circular shell, contact
            torque, and a mechanical power ledger (torque x omega in, contact
            dissipation out).
``regimes`` Froude-number classification, analytic release-angle/cataract
            criteria, free-surface parabola fit vs the centrifugal analytic
            curvature, and the phase-map cell driver.

Units SI throughout (m, kg, s, N, rad, W).
"""

from .kernel import (
    DrumConfig,
    DrumState,
    cross_z,
    damping_ratio,
    make_drum_state,
    run_drum,
)
from .regimes import (
    FR_BANDS,
    REGIMES,
    analytic_surface_curvature,
    cataract_onset_froude,
    classify_froude,
    classify_kinematic,
    detachment_angle_deg,
    fill_to_chord_angle,
    fit_free_surface,
    free_surface_fit_quality,
    froude,
    omega_for_froude,
    simulate_point,
)

__all__ = [
    "DrumConfig",
    "DrumState",
    "cross_z",
    "damping_ratio",
    "make_drum_state",
    "run_drum",
    "FR_BANDS",
    "REGIMES",
    "analytic_surface_curvature",
    "cataract_onset_froude",
    "classify_froude",
    "classify_kinematic",
    "detachment_angle_deg",
    "fill_to_chord_angle",
    "fit_free_surface",
    "free_surface_fit_quality",
    "froude",
    "omega_for_froude",
    "simulate_point",
]