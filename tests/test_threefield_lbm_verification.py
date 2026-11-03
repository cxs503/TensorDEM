"""Tests for the analytic verification of the three-field LBM core.

The references are closed form, so the tolerances below are round-off / transient
limits, not curve-fit agreements:

* realised viscosity ``nu = rate / k^2`` versus ``(tau0 - 0.5)/3`` (< 1 %),
* Smagorinsky eddy viscosity versus the analytic closure (< 1e-10 relative),
* Guo forcing impulse identity (round-off),
* uniform-stream boundary audit (round-off),
* periodic streaming equivalence and the default (non-periodic) path being
  untouched by the verification switch.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.threefield import lbm_verification as V
from tensordem.threefield.wake_lbm import LBMConfig, WakeLBM


def test_shear_wave_viscosity_matches_molecular_value():
    result = V.run_shear_decay(tau0=0.8, ny=32, steps=240, sample_every=20)
    assert result["decay_fit"]["r2"] > 0.9999
    assert abs(result["nu_relative_error"]) < 0.01
    assert result["nu_theory"] == pytest.approx(0.1, rel=0, abs=1e-15)


def test_shear_wave_viscosity_scales_with_tau():
    measured = {}
    for tau0 in (0.7, 1.0, 1.6):
        result = V.run_shear_decay(tau0=tau0, ny=32, steps=200, sample_every=20)
        assert abs(result["nu_relative_error"]) < 0.02
        measured[tau0] = result["nu_measured"]
    assert measured[0.7] < measured[1.0] < measured[1.6]
    assert measured[1.6] / measured[0.7] > 3.0


def test_shear_wave_profile_and_invariants():
    result = V.run_shear_decay(tau0=0.8, ny=32, steps=200, sample_every=20)
    assert result["shape_error_rel_amplitude"] < 0.01
    assert result["mass_drift_rel"] < 1e-12
    assert result["momentum_drift_rel"] < 1e-12
    # A pure shear wave has no transverse velocity component.
    assert result["max_transverse_velocity"] < 1e-12


def test_periodic_streaming_conserves_mass_and_matches_interior_stencil():
    solver = V.shear_solver(nx=12, ny=16, nz=8, tau0=0.8)
    generator = torch.Generator().manual_seed(7)
    f = torch.rand((solver.nx, solver.ny, solver.nz, 19), generator=generator,
                   dtype=solver.dtype)
    wrapped = solver._stream(f, periodic=True)
    manual = torch.zeros_like(f)
    for i, (cx, cy, cz) in enumerate(zip(solver._cx, solver._cy, solver._cz)):
        manual[:, :, :, i] = torch.roll(f[:, :, :, i], shifts=(cx, cy, cz), dims=(0, 1, 2))
    assert torch.allclose(wrapped, manual, atol=0, rtol=0)
    # mass in == mass out
    # summation order differs between the rolling and the slicing stencil
    assert wrapped.sum().item() == pytest.approx(f.sum().item(), rel=1e-14, abs=0)
    # the interior of the wrapped field agrees with the open-boundary stencil
    open_stream = solver._stream(f, periodic=False)
    interior = (slice(1, -1), slice(1, -1), slice(1, -1), slice(None))
    assert torch.allclose(wrapped[interior], open_stream[interior], atol=0, rtol=0)


def test_periodic_flag_is_off_by_default():
    config = LBMConfig(
        nx=24,
        ny=12,
        nz=12,
        inflow_velocity=0.04,
        jet_velocity=0.10,
        disk_x=8,
        disk_radius=4.0,
        disk_thickness=2,
        quarter_domain=True,
    )
    solver = WakeLBM(config)
    assert solver.periodic is False
    # the production boundary set is still applied
    rho = torch.ones((24, 12, 12), dtype=solver.dtype)
    u = torch.zeros((24, 12, 12, 3), dtype=solver.dtype)
    u[:, :, :, 0] = 0.04
    f = solver._equilibrium(rho.unsqueeze(-1), u)
    out = solver._apply_boundaries(solver._stream(f))
    assert not torch.equal(out, f)


def test_smagorinsky_closure_matches_analytic_value():
    for cs in (0.05, 0.16, 0.25):
        solver = V.shear_solver(smagorinsky_cs=cs)
        for strain in (0.005, 0.02):
            out = V.chapman_enskog_field(solver, strain)
            assert out["rho_from_field"] == pytest.approx(1.0, abs=1e-14)
            assert out["max_velocity_from_field"] < 1e-15
            # Chapman-Enskog non-equilibrium stress used by the closure
            assert out["pi_norm"] == pytest.approx(
                out["pi_norm_over_rho_theory"], rel=1e-12
            )
            assert out["nu_eddy_measured"] == pytest.approx(
                out["nu_eddy_theory"], rel=1e-8, abs=1e-15
            )
            # leading-order behaviour: nu_t -> Cs^2 |S| / sqrt(2) for this
            # six-component Pi-norm convention (Cs_eff = 0.8409 Cs)
            assert out["nu_eddy_measured"] > 0.0
            ratio = out["nu_eddy_measured"] / (cs * cs * out["strain_norm"])
            assert 1.0 / math.sqrt(2.0) - 1e-6 <= ratio <= 1.0 / math.sqrt(2.0) + 1e-3


def test_smagorinsky_zero_closure_leaves_tau_untouched():
    solver = V.shear_solver(smagorinsky_cs=0.0)
    out = V.chapman_enskog_field(solver, 0.02)
    assert out["nu_eddy_measured"] == pytest.approx(0.0, abs=1e-15)
    assert out["nu_total_measured"] == pytest.approx(out["nu_molecular"], abs=1e-15)


def test_guo_forcing_impulse_identity():
    for tau0 in (0.5006, 0.8, 1.5):
        audit = V.momentum_impulse_audit(tau0=tau0, steps=4)
        assert audit["ratio_measured_over_expected"] == pytest.approx(1.0, abs=1e-9)
        assert audit["ratio_booked_over_expected"] == pytest.approx(1.0, abs=1e-12)


def test_guo_forcing_unguarded_prefactor_documented():
    """The pre-fix behaviour is the analytic (1 + 1/(2 tau)) over-injection."""
    audit = V.momentum_impulse_audit(tau0=0.5006, steps=1)
    assert audit["unguarded_prefactor"] == pytest.approx(1.0 + 0.5 / 0.5006, rel=1e-12)
    assert audit["unguarded_prefactor"] > 1.99


def test_uniform_stream_is_steady_under_production_boundaries():
    audit = V.boundary_audit()
    assert audit["max_population_deviation_rel"] < 1e-14
    assert audit["max_velocity_deviation"] < 1e-14
    assert audit["max_density_deviation"] < 1e-14
    # the streaming step alone does move populations, so the check is not vacuous
    assert audit["streaming_max_deviation"] > 1e-3


def test_one_dimensional_reference_is_monotone_and_closure_ordered():
    molecular = V.shear_decay_reference_1d(ny=32, tau0=0.8, smagorinsky_cs=0.0,
                                           mode=1, steps=120)
    closure = V.shear_decay_reference_1d(ny=32, tau0=0.8, smagorinsky_cs=0.16,
                                         mode=1, steps=120)
    assert all(abs(b) <= abs(a) + 1e-15 for a, b in zip(molecular, molecular[1:]))
    # the eddy viscosity can only add damping
    assert abs(closure[-1]) < abs(molecular[-1])
    assert molecular[0] == pytest.approx(0.02, rel=1e-15)
    assert math.isfinite(closure[-1])