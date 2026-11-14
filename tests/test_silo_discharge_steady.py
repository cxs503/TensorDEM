"""Regression tests for the m20055 silo steady-discharge protocol.

Pins the three additions to ``tensordem.silo_discharge``:

* ``steady_rate`` -- the steady-state estimator (start-up / tail rejection);
* ``SiloCase.packing="settled"`` -- a non-overlapping loose lattice that is
  gravity-settled into a disordered bed;
* ``SiloCase.plug_during_settle`` -- the orifice is closed during settling, so
  the measured rate cannot be fed by mass that left before the window opened.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.silo_discharge import (
    SiloCase, SiloError, _initial_packing, _plate_forces, simulate_silo,
    steady_rate,
)


def _tiny_case(**kw) -> SiloCase:
    params = dict(radius=0.010, density=2500.0, mu=0.30, mu_rolling=0.02,
                  restitution=0.30, kn=1.0e5, silo_radius=0.050,
                  fill_height=0.060, orifice_diameter=0.065,
                  plate_thickness=0.010, wall_mu=0.0, packing="settled",
                  plug_during_settle=True, settle_loose_gap=0.30, jitter=0.05,
                  seed=7, settle_time=0.10, discharge_time=0.10, window=0.02,
                  n_windows=5)
    params.update(kw)
    return SiloCase(**params)


# ---------------------------------------------------------------------------
# 1. the estimator itself
# ---------------------------------------------------------------------------
def test_steady_rate_recovers_a_constant_rate():
    """A constant-rate stream must be recovered exactly, with zero scatter."""
    mass = 2.5e-3
    n = np.arange(1, 401)
    t = n * 0.01                     # 100 particles/s -> 0.25 kg/s
    out = steady_rate(t, n, mass, 0.5, 5, (0.2, 0.8))
    assert out["steady_flow_rate_kg_s"] == pytest.approx(100.0 * mass, rel=1e-12)
    assert out["steady_rate_cv"] < 1e-12
    assert out["n_discharged_total"] == 400


def test_steady_rate_rejects_a_slow_start_and_a_slow_tail():
    """A stream with 20 % slow start and 20 % slow tail must report the plateau.

    Counts: 40 particles over 2 s (slow), then 120 over 1 s (plateau), then
    40 over 2 s (slow).  Quantiles 0.2-0.8 span exactly the plateau, so the
    reported rate has to be the plateau rate, not the stream average
    (200/5 s = 40 part./s).
    """
    mass = 1.0
    t = np.concatenate([np.linspace(0.0, 2.0, 41)[1:],
                        np.linspace(2.0, 3.0, 121)[1:],
                        np.linspace(3.0, 5.0, 41)[1:]])
    n = np.arange(1, 201)
    out = steady_rate(t, n, mass, 0.5, 5, (0.2, 0.8))
    assert out["steady_flow_rate_kg_s"] == pytest.approx(120.0, rel=0.02)
    assert out["steady_interval_s"][0] == pytest.approx(2.0, abs=0.01)
    assert out["steady_interval_s"][1] == pytest.approx(3.0, abs=0.01)


def test_steady_rate_is_degenerate_safe():
    out = steady_rate([], [], 1.0, 0.1, 5)
    assert out["steady_flow_rate_kg_s"] == 0.0
    assert math.isnan(out["steady_rate_cv"])


# ---------------------------------------------------------------------------
# 2. plugged / open orifice
# ---------------------------------------------------------------------------
def _plate_force_on_one(z_frac: float, radial: float, closed: bool) -> float:
    case = _tiny_case()
    mat = case.material
    r = case.radius
    pos = np.array([[radial, 0.0, z_frac * r]])
    vel = np.zeros((1, 3))
    f = np.zeros((1, 3))
    _plate_forces(pos, vel, mat, case, f, closed=closed)
    return float(f[0, 2])


def test_plugged_plate_supports_a_particle_over_the_orifice_centre():
    """Over the hole centre the plate is solid when plugged and open otherwise."""
    assert _plate_force_on_one(0.999, 0.0, closed=True) > 0.5
    assert _plate_force_on_one(0.999, 0.0, closed=False) == 0.0


def test_plug_does_not_change_the_solid_part_of_the_plate():
    """Away from the aperture both plate models must agree (same top face)."""
    z = _plate_force_on_one(0.999, 0.045, closed=True)
    o = _plate_force_on_one(0.999, 0.045, closed=False)
    assert z == pytest.approx(o, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. settled bed
# ---------------------------------------------------------------------------
def test_settled_packing_is_born_without_overlaps():
    case = _tiny_case()
    pos = _initial_packing(case)
    assert len(pos) > 10
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    # settle_loose_gap = 0.3 r  ->  centre spacing 1.3 * 2r minus the jitter
    assert d.min() > 2.0 * case.radius


def test_legacy_hcp_packing_is_unchanged():
    pos = _initial_packing(_tiny_case(packing="hcp", packing_gap=0.02, jitter=0.0))
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    assert d.min() == pytest.approx(2.0 * 0.010 * 1.02, rel=2e-3)


def test_invalid_packing_mode_is_rejected():
    with pytest.raises(SiloError):
        _tiny_case(packing="crystal")


# ---------------------------------------------------------------------------
# 4. end-to-end: nothing leaves the silo while the plug is in
# ---------------------------------------------------------------------------
def test_no_discharge_before_the_orifice_opens():
    case = _tiny_case()
    res = simulate_silo(case)
    times = res["discharge_time_s"]
    assert times, "the settled bed must discharge once the plug is removed"
    assert min(times) >= case.settle_time - case.timestep
    assert res["n_discharged_total"] == len(times) and res["n_discharged_total"] > 0
    assert res["bed_height_m"] > 0.0
    assert 0.0 < res["packing_fraction_bed"] < 0.75


def test_open_orifice_from_the_first_step_keeps_the_legacy_protocol():
    """plug_during_settle=False must behave exactly like the legacy run."""
    case = _tiny_case(plug_during_settle=False)
    res = simulate_silo(case)
    assert res["bed_height_m"] == 0.0          # no snapshot without a plug
    assert np.isfinite(res["flow_rate_kg_s"])