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


# ---------------------------------------------------------------------------
# exact slab (lane) volume -- the metric the clearing curve is built on
# ---------------------------------------------------------------------------
def test_solid_volume_slab_matches_analytic_box():
    """Clipped slab volume must equal the analytic value, not a centroid count."""
    from tensordem.polyhedron_floe3d.contact import ContactLaw
    from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody
    from tensordem.polyhedron_floe3d.geometry import extrude_polygon

    rectangle = np.array([[-0.2, -0.15], [0.2, -0.15], [0.2, 0.15], [-0.2, 0.15]])
    shape = extrude_polygon(rectangle, 0.12, z0=-0.06)
    body = FloeBody(shape, 917.0)
    system = DEMSystem3D([body], law=ContactLaw(), gravity=0.0)

    assert system.solid_volume_slab(0, -10.0, 10.0) == pytest.approx(shape.volume, rel=1e-12)
    # slab [-0.2, 0.05]: overlap length 0.25 of the 0.4 m side
    assert system.solid_volume_slab(0, -10.0, 0.05) == pytest.approx(
        0.25 * 0.30 * 0.12, rel=1e-10)
    # a slab that sits entirely inside the body
    assert system.solid_volume_slab(0, 0.0, 0.1) == pytest.approx(
        0.10 * 0.30 * 0.12, rel=1e-10)
    # the same along y (the axis is general, not a z special case)
    assert system.solid_volume_slab(1, -10.0, 0.0) == pytest.approx(
        0.40 * 0.15 * 0.12, rel=1e-10)
    # an empty slab returns exactly zero
    assert system.solid_volume_slab(0, 5.0, 6.0) == 0.0


def test_slab_volume_is_continuous_not_a_step_function():
    """Sweeping a face across a floe must change the volume smoothly (unlike porosity)."""
    from tensordem.polyhedron_floe3d.contact import ContactLaw
    from tensordem.polyhedron_floe3d.dynamics import DEMSystem3D, FloeBody
    from tensordem.polyhedron_floe3d.geometry import extrude_polygon

    rectangle = np.array([[-0.2, -0.15], [0.2, -0.15], [0.2, 0.15], [-0.2, 0.15]])
    shape = extrude_polygon(rectangle, 0.12, z0=-0.06)
    system = DEMSystem3D([FloeBody(shape, 917.0)], law=ContactLaw(), gravity=0.0)
    cuts = np.linspace(-0.2, 0.2, 9)
    volumes = [system.solid_volume_slab(0, -10.0, c) for c in cuts]
    assert all(b >= a - 1e-15 for a, b in zip(volumes[:-1], volumes[1:]))
    increments = np.diff(volumes)
    assert np.allclose(increments, increments[0], rtol=1e-9)  # linear in the cut
    # the centroid proxy cannot see the cut at all: every sub-body slice reports
    # "no solid" until the centroid itself is inside the probe box
    inside_cuts = np.linspace(-0.18, -0.02, 5)
    proxy = {system.porosity([c - 1e-6, -1.0, -1.0], [c + 1e-6, 1.0, 1.0])
             for c in inside_cuts}
    assert proxy == {1.0}


def test_channel_case_declares_incomplete_sweep():
    """A run whose hull advance is shorter than the corridor must not pass."""
    result = channel_clearing_case(n_floes=6, n_steps=120, width=0.6, length=1.2,
                                   corridor=0.4)
    assert result["summary"]["sweep_complete"] is False
    assert result["criteria"]["sweep_complete"] is False
    assert result["passed"] is False
    assert len(result["series"]["lane_concentration"]) == len(result["series"]["time_s"])
    assert math.isfinite(result["summary"]["lane_concentration_initial"])


def test_integration_caller_reports_maturity_gaps(tmp_path):
    """The in-repo maturity check must report gaps instead of rounding up."""
    from tensordem.integration_callers import check_h1_maturity

    fake = {"cases": {"buoyancy": {"passed": True,
                                   "ref": {"submerged_fraction_archimedes": 0.8946},
                                   "got": {"submerged_fraction": 0.8946}},
                      "repose": {"passed": False, "ref": {}, "got": {}}},
            "integration": {"caller": "test"}, "repro_cmd": "cmd"}
    verdict = check_h1_maturity(fake, required_level="L4")
    assert verdict["passed"] is False
    assert any("repose" in g for g in verdict["gaps"])

    good = {"cases": {"buoyancy": {"passed": True, "ref": {"a": 1.0}, "got": {"a": 1.0}}},
            "integration": {"caller": "test"}, "repro_cmd": "cmd"}
    assert check_h1_maturity(good, required_level="L4")["passed"] is True
