"""Tests for the R11 phi-state-point sensitivity sweep (CASE-BIAXIAL-SHEAR-BAND).

R11 closes the R9/R10 ``not_run`` debt (the state-matching arm ``cj4770`` that
rebuilds ``phi`` over the late window) *arithmetically*: it sweeps ``phi`` over
the physically-plausible state-point range ``[30, 40]`` and asks whether the
R9/R10 verdict (Arthur supported at every declared ``phi_cv``) is robust to
which state point ``phi`` is taken at.

The honest finding (asserted here): the robustness window is **narrow** --
``[34.0, 35.5]`` (1.5 deg wide), containing the measured peak ``phi = 35.45``
but **not** covering the pre-registered ``[33, 38]`` margin.  So the
phi-state-point sensitivity is a *real, quantified* concern that the in-flight
``cj4770`` arm must resolve -- the R9/R10 verdict holds at the peak but is not
robust to a wide ``phi`` shift.

These tests run the pure-arithmetic sweep (no DEM cell) and assert the
invariants of the finding.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensordem.band_angle_laws import (  # noqa: E402
    adjudicate,
    stress_dilatancy_psi_deg,
)

THETA_MEAS_DEG = 55.00
PHI_PEAK_DEG = 35.45
PHI_CV_RANGE = (26.0, 28.0, 30.0, 32.0, 34.0)
TOLERANCE = 0.15
PRECISION = 0.03


def _arthur_supported_at_every_phi_cv(phi_deg: float) -> bool:
    """The R9 'supported at every declared phi_cv' bar at one phi state-point."""
    return all(
        "arthur" in adjudicate(THETA_MEAS_DEG, phi_deg,
                               stress_dilatancy_psi_deg(phi_deg, phi_cv),
                               tolerance=TOLERANCE, precision=PRECISION)
        ["within_precision"]
        for phi_cv in PHI_CV_RANGE)


def test_peak_phi_is_inside_robustness_window():
    """The measured peak phi=35.45 must be inside the robustness window."""
    # window is [34.0, 35.5] (from the R11 sweep); the peak must be inside
    assert _arthur_supported_at_every_phi_cv(35.0)
    assert _arthur_supported_at_every_phi_cv(35.5)
    assert _arthur_supported_at_every_phi_cv(PHI_PEAK_DEG)


def test_window_is_narrow_not_wide():
    """The robustness window must be narrow: [34, 35.5], not [33, 38].

    This is the honest finding -- the pre-registered [33, 38] margin is NOT
    met, so the phi-state-point sensitivity is a real concern.
    """
    # inside the window
    assert _arthur_supported_at_every_phi_cv(34.0)
    assert _arthur_supported_at_every_phi_cv(35.5)
    # outside the window (below 34 and above 35.5)
    assert not _arthur_supported_at_every_phi_cv(33.5)
    assert not _arthur_supported_at_every_phi_cv(36.0)
    # the pre-registered wide margin [33, 38] is NOT met
    assert not _arthur_supported_at_every_phi_cv(33.0)
    assert not _arthur_supported_at_every_phi_cv(38.0)


def test_mohr_coulomb_never_within_precision_across_sweep():
    """Mohr-Coulomb must never be within 3 % precision at any phi in [30, 40].

    The discrimination is driven by the phi/4 vs phi/2 coefficient; even at
    the window edges Mohr-Coulomb stays above 3 % (8-15 %).
    """
    for phi in np.arange(30.0, 40.0 + 1e-9, 0.5):
        for phi_cv in PHI_CV_RANGE:
            adj = adjudicate(THETA_MEAS_DEG, float(phi),
                             stress_dilatancy_psi_deg(float(phi), phi_cv),
                             tolerance=TOLERANCE, precision=PRECISION)
            assert "mohr_coulomb" not in adj["within_precision"], (
                f"MC within precision at phi={phi}, phi_cv={phi_cv}")


def test_arthur_is_closest_law_at_peak():
    """At the measured peak, Arthur must be the closest law (R9/R10 verdict)."""
    adj = adjudicate(THETA_MEAS_DEG, PHI_PEAK_DEG,
                     stress_dilatancy_psi_deg(PHI_PEAK_DEG, 32.0),
                     tolerance=TOLERANCE, precision=PRECISION)
    assert adj["closest_law"] == "arthur"
    assert "arthur" in adj["within_precision"]
    # discrimination vs Mohr-Coulomb is the phi/4 vs phi/2 gap (~7 deg)
    assert adj["discrimination_deg"] >= 6.0


def test_arthur_error_at_peak_is_below_3pct():
    """At the peak, the Arthur prediction must be within 3 % (R9/R10)."""
    adj = adjudicate(THETA_MEAS_DEG, PHI_PEAK_DEG,
                     stress_dilatancy_psi_deg(PHI_PEAK_DEG, 32.0),
                     tolerance=TOLERANCE, precision=PRECISION)
    err = adj["errors"]["arthur"]["abs_relative_error"]
    assert err <= 0.03
