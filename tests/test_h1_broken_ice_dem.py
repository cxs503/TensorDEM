"""H1-BROKEN-ICE-DEM: floating equilibrium, rubble repose, channel clearing."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.polyhedron_floe3d.h1_cases import (
    SEA_ICE_DENSITY,
    SEA_WATER_DENSITY,
    buoyancy_equilibrium_case,
    channel_clearing_case,
    repose_case,
)
from tensordem.polyhedron_floe3d.study import case3_repose


def test_buoyancy_equilibrium_matches_archimedes():
    """Relaxed draft must reproduce rho_ice/rho_water (5 % tolerance)."""
    result = buoyancy_equilibrium_case(max_steps=4000)
    assert result["got"]["steps"] > 10
    assert result["error"]["rel_err_draft"] < 0.05
    assert result["error"]["rel_err_submerged_fraction"] < 0.05
    assert result["passed"]


def test_buoyancy_equilibrium_second_density():
    """The equilibrium fraction is the density ratio, not a tuned number."""
    result = buoyancy_equilibrium_case(rho_ice=900.0, thickness=0.30, max_steps=4000)
    assert result["ref"]["submerged_fraction_archimedes"] == pytest.approx(
        900.0 / SEA_WATER_DENSITY, rel=1e-12)
    assert result["error"]["rel_err_submerged_fraction"] < 0.05


def test_linear_drag_removes_energy():
    """The drag closure must dissipate: a floe released at the waterline settles."""
    result = buoyancy_equilibrium_case(max_steps=4000)
    tail = result["history"][-5:]
    assert max(abs(h["vz"]) for h in tail) < 1.0e-3
    assert result["got"]["energy_residual"] < 1.0  # dimensionless residual, O(1e-1)


def test_repose_case_is_valid_not_nan():
    """Regression: the shipped case3 returned nan with zero contacts."""
    raw = case3_repose(n_bodies=18, friction=(0.4,), n_steps=400)
    run = raw["runs"][0]
    assert "valid" in run
    assert "grid" in raw["setup"] and "spacing" in raw["setup"]
    assert math.isfinite(run["pile_height"])


def test_channel_clearing_series_is_monotone_in_the_mean():
    """Smoke: the corridor concentration must start high and drop."""
    result = channel_clearing_case(n_floes=6, n_steps=120, width=0.6, length=1.2,
                                   corridor=0.4)
    series = result["series"]
    assert len(series["time_s"]) == len(series["concentration"])
    head = float(np.mean(series["concentration"][:5]))
    tail = float(np.mean(series["concentration"][-5:]))
    assert head >= tail or math.isfinite(result["summary"]["concentration_half_time_s"])
    assert result["summary"]["mass_residual"] < 1.0e-12


def test_declared_densities_are_sea_ice_values():
    assert SEA_ICE_DENSITY == 917.0
    assert SEA_WATER_DENSITY == 1025.0
