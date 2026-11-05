"""Tests for the bounded-domain boundary verification of the D3Q19 race solver.

Every assertion is anchored on a closed form or on an exact discrete identity:

* a uniform stream must be a fixed point of the *complete* bounded scheme
  (reference: zero drift),
* a specular (free-slip) wall must be impermeable and must transfer exactly zero
  axial momentum (reference: zero; the negative control corrupts the reflection
  table and must be caught),
* ``plane_mass_flux`` must reproduce ``rho u_x dA`` of a known equilibrium state,
* the open-boundary ledger must report a small box-mass drift and a positive thrust
  for the forced case, and the fill-time helper must report immediate convergence
  for a case that starts in the steady state.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.threefield import boundary_verification as bv
from tensordem.threefield.lbm_transport import shear_wave_decay, shear_wave_rate
from tensordem.threefield.wake_lbm import _D3Q19_VELOCITIES, LBMConfig, WakeLBM

INFLOW = 0.04


def forced_config(*, nx=24, ny=12, nz=12, tau0=0.5005, cs=0.14, radius=3.0):
    return LBMConfig(
        nx=nx,
        ny=ny,
        nz=nz,
        tau0=tau0,
        smagorinsky_cs=cs,
        inflow_velocity=INFLOW,
        jet_velocity=0.12,
        disk_x=8,
        disk_radius=radius,
        disk_thickness=2,
        disk_profile="gaussian",
        disk_perturbation=0.15,
        inlet_perturbation=0.02,
        quarter_domain=True,
    )


# -- A. uniform stream is an exact fixed point of the bounded scheme ----------
def test_uniform_stream_is_an_exact_fixed_point():
    result = bv.uniform_stream_check(nx=24, ny=12, nz=12, tau0=0.6, smagorinsky_cs=0.0, steps=150)
    assert result.cells == 24 * 12 * 12
    assert result.velocity_drift < 1e-12
    assert result.density_drift < 1e-12
    assert result.mass_drift < 1e-12
    # a stream along x may not generate any transverse velocity at all
    assert result.lateral_drift == 0.0
    assert result.exact


def test_uniform_stream_with_smagorinsky_is_also_exact():
    """The closure must not break the fixed point: zero strain gives tau = tau0."""
    result = bv.uniform_stream_check(tau0=0.5005, smagorinsky_cs=0.14, steps=150)
    assert result.exact


# -- B. specular walls: impermeable and frictionless -------------------------
def test_specular_walls_have_zero_normal_velocity_and_no_drag():
    inst = WakeLBM(forced_config())
    inst.run(240, average_from=120)
    audit = bv.wall_slip_audit(inst)
    assert audit.normal_velocity == 0.0
    assert audit.mass_crossing_mismatch < 1e-14
    assert audit.axial_momentum_exchange < 1e-14
    assert audit.reflection_tangential_mismatch == 0.0
    assert audit.interior_speed > 0.0
    assert audit.impermeable_and_frictionless


def test_corrupted_reflection_table_is_caught():
    """Negative control: a table that also flips the axial component is a no-slip defect."""
    inst = WakeLBM(forced_config())
    inst.run(60, average_from=0)
    lookup = {velocity: index for index, velocity in enumerate(_D3Q19_VELOCITIES)}
    broken = []
    for velocity in _D3Q19_VELOCITIES:
        flipped = (-velocity[0], -velocity[1], velocity[2])  # wrongly reflects x as well
        broken.append(lookup[flipped])
    inst.reflect_y = tuple(broken)
    audit = bv.wall_slip_audit(inst)
    assert audit.reflection_tangential_mismatch > 0.0
    assert not audit.impermeable_and_frictionless


# -- flux helpers against closed forms --------------------------------------
def test_plane_mass_flux_matches_the_equilibrium_state():
    config = bv._boundary_only_config(
        nx=16, ny=10, nz=10, tau0=0.7, smagorinsky_cs=0.0, inflow=INFLOW
    )
    inst = bv.disable_disk(WakeLBM(config))
    uniform = inst._inlet_populations()
    inst.f = uniform.unsqueeze(0).expand(16, 10, 10, uniform.shape[-1]).clone()
    # rho = 1, u_x = U_in: the plane flux is U_in per cell
    assert bv.plane_mass_flux(inst, 0) == pytest.approx(INFLOW * 10 * 10, rel=1e-13)
    assert bv.plane_mass_flux(inst, -1) == pytest.approx(INFLOW * 10 * 10, rel=1e-13)
    # momentum flux adds the pressure term p = rho cs^2 per cell
    expected = (INFLOW**2 + 1.0 / 3.0) * 10 * 10
    assert bv.plane_axial_momentum_flux(inst, 0) == pytest.approx(expected, rel=1e-12)


def test_disable_disk_removes_the_thrust():
    config = forced_config()
    inst = bv.disable_disk(WakeLBM(config))
    inst.run(40)
    assert inst.thrust_impulse == 0.0
    assert inst.thrust_lattice() == 0.0


# -- C. open-boundary ledger --------------------------------------------------
def test_ledger_reports_positive_thrust_and_small_box_mass_drift():
    config = forced_config(nx=24, ny=12, nz=12)
    ledger = bv.open_boundary_ledger(config, steps=240, sample_every=5)
    assert ledger.thrust_rate > 0.0
    assert ledger.mass_drift_rate < 1e-3
    assert abs(ledger.flux_imbalance_relative) < 0.2
    assert ledger.samples[0].step == 0
    assert ledger.samples[-1].step == 240
    steps = [sample.step for sample in ledger.samples]
    assert steps == sorted(steps)


def test_ledger_of_a_uniform_stream_closes_the_mass_books():
    config = bv._boundary_only_config(
        nx=16, ny=10, nz=10, tau0=0.7, smagorinsky_cs=0.0, inflow=INFLOW
    )
    ledger = bv.open_boundary_ledger(config, steps=120, sample_every=10, initial_uniform=True)
    assert abs(ledger.mass_drift_rate) < 1e-12
    assert abs(ledger.flux_imbalance_relative) < 1e-12
    assert ledger.thrust_rate == 0.0
    fill = bv.fill_time_in_flow_throughs(ledger, tolerance=1e-3, window=3)
    assert fill["flow_throughs"] == pytest.approx(0.0, abs=1e-12)


def test_fill_time_is_measured_in_flow_throughs():
    config = forced_config(nx=20, ny=12, nz=12, radius=3.0)
    ledger = bv.open_boundary_ledger(config, steps=400, sample_every=10)
    fill = bv.fill_time_in_flow_throughs(ledger, tolerance=0.02, window=3)
    transit = config.nx / config.inflow_velocity
    assert fill["flow_throughs"] == pytest.approx(fill["steps"] / transit, rel=1e-12)
    assert fill["converged"] in (0.0, 1.0)


# -- D. race station audit ----------------------------------------------------
def test_race_station_audit_returns_the_published_quantities():
    config = forced_config()
    inst = WakeLBM(config)
    inst.run(200, average_from=100)
    rows = bv.race_station_audit(inst, [14, 16])
    assert [row.station for row in rows] == [14, 16]
    for row in rows:
        assert row.mass_flux > 0.0
        assert math.isfinite(row.excess_over_thrust)
        assert row.thrust == pytest.approx(inst.thrust_lattice())
    # the disk drives the race, so the excess flux must be positive downstream
    assert all(row.excess_momentum_flux > 0.0 for row in rows)


# -- E. m2949 tau0 = 1.4 honest-FAIL: resolution attribution ------------------
def test_tau0_1p4_fail_is_a_resolution_effect():
    """Doubling the lattice and halving k must bring the rate error inside 3 %."""
    coarse = shear_wave_decay(tau0=1.4, size=16, mode=1, steps=400)
    fine = shear_wave_decay(tau0=1.4, size=32, mode=1, steps=400)
    error_coarse = coarse.rate_measured / shear_wave_rate(1.4, 2.0 * math.pi / 16)[0] - 1.0
    error_fine = fine.rate_measured / shear_wave_rate(1.4, 2.0 * math.pi / 32)[0] - 1.0
    assert abs(error_coarse) > 0.03          # the published honest-FAIL, reproduced
    assert abs(error_fine) < 0.03            # and gone once the lattice resolves the mode
    assert abs(error_fine) < 0.5 * abs(error_coarse)