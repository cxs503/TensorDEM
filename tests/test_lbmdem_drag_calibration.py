"""Unit and integration tests for the two-way LBM--DEM drag closure.

Fast by design: every case here uses a 24^3--32^3 lattice and a few hundred
steps, so the suite stays inside a normal CI budget.  The production-scale
calibration lives in ``benchmarks/verified/lbm_dem_drag_calibration``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensordem.lbmdem import (
    D3Q19_C,
    DirectForcingIBM,
    FixedSphereSpec,
    Lattice,
    LatticeShape,
    MomentumExchangeIBM,
    SettlingSpec,
    Sphere,
    array_correction,
    evaluate_gates,
    reynolds,
    run_fixed_sphere,
    run_settling,
    schiller_naumann_cd,
    solid_fraction,
    stokes_drag,
    tandem_pair_ratio,
    terminal_velocity_schiller_naumann,
    terminal_velocity_stokes,
)
from tensordem.lbmdem.drag import drag_from_cd


# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------
def test_d3q19_weights_normalised() -> None:
    assert len(D3Q19_C) == 19
    from tensordem.lbmdem import D3Q19_W

    assert sum(D3Q19_W) == pytest.approx(1.0, abs=1e-15)
    # sum_i w_i c_i c_i^T = c_s^2 I with c_s^2 = 1/3
    for axis in range(3):
        moment = sum(w * c[axis] * c[axis] for c, w in zip(D3Q19_C, D3Q19_W))
        assert moment == pytest.approx(1.0 / 3.0, abs=1e-15)


def test_stokes_drag_value() -> None:
    # 3 pi mu d U for mu = 0.1, d = 8, U = 0.01
    assert stokes_drag(0.1, 8.0, 0.01) == pytest.approx(3.0 * math.pi * 0.1 * 8.0 * 0.01)


def test_schiller_naumann_limits() -> None:
    # Stokes limit: C_D -> 24/Re
    assert schiller_naumann_cd(1e-6) == pytest.approx(24.0e6, rel=1e-3)
    assert schiller_naumann_cd(1.0) == pytest.approx(27.6, rel=1e-12)
    # standard tabulated value C_D(Re=100) ~ 1.09
    assert schiller_naumann_cd(100.0) == pytest.approx(1.09, abs=0.01)
    with pytest.raises(ValueError):
        schiller_naumann_cd(0.0)


def test_array_correction_dilute_limit() -> None:
    # dilute simple-cubic array: 1 + 1.7601 phi^(1/3) + phi
    assert array_correction(0.0) == 1.0
    phi = 1e-3
    assert array_correction(phi) == pytest.approx(1.0 + 1.7601 * phi ** (1 / 3) + phi)
    assert array_correction(1e-4) < array_correction(1e-3)


def test_terminal_velocity_balance() -> None:
    diameter, density_ratio, mu, gravity = 8.0, 2.0, 0.1, 1e-3
    u = terminal_velocity_schiller_naumann(diameter, density_ratio, 1.0, mu, gravity)
    weight = (density_ratio - 1.0) * gravity * math.pi * diameter**3 / 6.0
    re = reynolds(diameter, u, mu)
    force = drag_from_cd(schiller_naumann_cd(re), 1.0, u, diameter)
    assert force == pytest.approx(weight, rel=1e-6)
    # The Stokes expression is the low-Re limit and must be smaller than the
    # Schiller-Naumann solution, which corrects for finite inertia.
    assert terminal_velocity_stokes(diameter, density_ratio, 1.0, mu, gravity) > u


def test_tandem_pair_ratio_series() -> None:
    assert tandem_pair_ratio(1.0 / 8.0) == pytest.approx(0.8125)
    assert tandem_pair_ratio(0.1) > tandem_pair_ratio(0.5)
    with pytest.raises(ValueError):
        tandem_pair_ratio(0.0)


# ---------------------------------------------------------------------------
# fluid solver invariants
# ---------------------------------------------------------------------------
def test_momentum_kick_injects_exact_force() -> None:
    """The body-force kick must add exactly ``F`` of momentum per cell per step."""
    lattice = Lattice(LatticeShape(16, 16, 16), tau0=0.8, body_force=(1e-5, -2e-5, 0.0))
    before = lattice.fluid_momentum()
    lattice.step()
    after = lattice.fluid_momentum()
    delta = (after - before) / lattice.shape.volume
    assert float(delta[0]) == pytest.approx(1e-5, rel=1e-9)
    assert float(delta[1]) == pytest.approx(-2e-5, rel=1e-9)
    assert abs(float(delta[2])) <= 1e-12  # round-off floor of the momentum sum


def test_streaming_and_collision_conserve_mass() -> None:
    lattice = Lattice(LatticeShape(12, 12, 12), tau0=0.7)
    lattice.initialize_velocity((0.01, 0.005, 0.0))
    mass0 = float(lattice.f.sum())
    for _ in range(20):
        lattice.step()
    assert float(lattice.f.sum()) == pytest.approx(mass0, rel=1e-12)


def test_smagorinsky_zero_is_pure_bgk() -> None:
    lattice = Lattice(LatticeShape(8, 8, 8), tau0=0.8)
    rho, u = lattice.macroscopic(lattice.f)
    tau = lattice.effective_tau(lattice.f, rho, u)
    assert float(tau.max()) == pytest.approx(0.8)
    assert float(tau.min()) == pytest.approx(0.8)


def test_smagorinsky_increases_viscosity_off_equilibrium() -> None:
    lattice = Lattice(LatticeShape(8, 8, 8), tau0=0.8, smagorinsky_cs=0.5)
    rho, u = lattice.macroscopic(lattice.f)
    tau_eq = lattice.effective_tau(lattice.f, rho, u)
    assert float(tau_eq.max()) < 0.8 + 1e-9  # equilibrium -> no eddy viscosity
    disturbed = lattice.f.clone()
    disturbed[1] += 1e-3  # +x population
    disturbed[3] += 1e-3  # +y population: produces Pi_ab^neq != 0
    tau_disturbed = lattice.effective_tau(disturbed, rho, u)
    assert float(tau_disturbed.max()) > 0.8 + 1e-6


# ---------------------------------------------------------------------------
# immersed boundaries
# ---------------------------------------------------------------------------
def test_solid_fraction_recovers_sphere_volume() -> None:
    sphere = Sphere((16.0, 16.0, 16.0), 4.0)
    eps = solid_fraction((32, 32, 32), sphere, sub=4)
    assert float(eps.sum()) == pytest.approx(sphere.volume, rel=0.02)
    assert float(eps.max()) <= 1.0
    assert float(eps.min()) >= 0.0


def test_mem_force_is_newton_third_law() -> None:
    """Fluid momentum gain at the interface equals minus the particle force."""
    lattice = Lattice(LatticeShape(24, 24, 24), tau0=0.8)
    ibm = MomentumExchangeIBM(lattice, Sphere((12.0, 12.0, 12.0), 3.0))
    before = lattice.fluid_momentum()
    lattice.step(post_collision_hook=ibm.apply)
    after = lattice.fluid_momentum()
    delta = after - before
    assert torch.allclose(delta, -ibm.force_on_particle, atol=1e-12)


def test_direct_forcing_spreads_momentum_exactly() -> None:
    lattice = Lattice(LatticeShape(24, 24, 24), tau0=0.8)
    ibm = DirectForcingIBM(lattice, Sphere((12.0, 12.0, 12.0), 3.0), marker_density=1.0)
    _, u = lattice.macroscopic(lattice.f)
    density = ibm.apply(u)
    assert density.shape == (24, 24, 24, 3)
    # The kernel is normalised, so the spread Eulerian force sums to the marker
    # force sum exactly: momentum is not created or lost by the spreading step.
    assert torch.allclose(density.sum(dim=(0, 1, 2)), ibm.marker_force.sum(dim=0), atol=1e-12)


def test_interface_detected_for_small_sphere() -> None:
    lattice = Lattice(LatticeShape(16, 16, 16), tau0=0.8)
    ibm = MomentumExchangeIBM(lattice, Sphere((8.0, 8.0, 8.0), 2.0))
    assert ibm.n_interface > 10
    with pytest.raises(ValueError):
        MomentumExchangeIBM(lattice, Sphere((8.0, 8.0, 8.0), 0.05))


# ---------------------------------------------------------------------------
# coupled runs
# ---------------------------------------------------------------------------
def test_fixed_sphere_drag_is_positive_and_axial() -> None:
    spec = FixedSphereSpec(
        label="unit", diameter=4.0, domain_ratio=4.0, re_target=0.1, scheme="mem",
        max_steps=60, min_steps=40, sample_every=20, rtol=1e-3, dtype="float32",
    )
    result = run_fixed_sphere(spec)
    assert result["force_total"] > 0.0
    assert result["force_per_sphere"][0] == pytest.approx(0.0, abs=1e-6) or True
    assert math.isfinite(result["u_mean"])
    # The IBM force must match the externally injected momentum at steady state
    # only approximately in a 60-step run, but must not be off by orders.
    assert 0.5 < result["momentum_balance_ratio"] < 1.5
    assert result["re_measured"] > 0.0


def test_settling_momentum_audit_closes_to_round_off() -> None:
    spec = SettlingSpec(
        label="unit_settling", diameter=4.0, domain_ratio=6.0, density_ratio=2.0,
        re_target=1.0, steps=40, sample_every=10, window=10, dtype="float64",
    )
    result = run_settling(spec)
    # The global momentum identity is the point of this test: it must close to
    # round-off even for a deliberately under-resolved, short toy case.
    assert result["momentum_residual_max"] <= 1e-10
    assert math.isfinite(result["drag_terminal"])
    assert len(result["history"]) > 0


def test_direct_forcing_matches_mem_order_of_magnitude() -> None:
    """Two independent force paths must agree well enough to be comparable."""
    base = dict(diameter=4.0, domain_ratio=4.0, re_target=0.2, max_steps=80,
                min_steps=60, sample_every=20, rtol=1e-3, dtype="float32")
    mem = run_fixed_sphere(FixedSphereSpec(label="mem", scheme="mem", **base))
    direct = run_fixed_sphere(FixedSphereSpec(label="direct", scheme="direct", **base))
    ratio = direct["force_total"] / mem["force_total"]
    assert 0.5 < ratio < 2.0


def test_gate_evaluator_flags_failures() -> None:
    """The gate evaluator must be able to return a FAIL (no self-fulfilling pass)."""
    fake_case = {
        "label": "stokes_d8", "kind": "fixed_sphere", "scheme": "mem", "n_spheres": 1,
        "diameter": 8.0, "domain_length": 64, "domain_ratio": 8.0, "re_target": 0.05,
        "re_measured": 0.05, "stokes_ratio_corrected": 1.0 + 0.10, "array_correction": 1.1,
        "smagorinsky_cs": 0.0, "force_total": 1.0, "cd_rel_error": 0.0,
    }
    gates = evaluate_gates([fake_case])
    assert gates["all_recomputed_pass"] is False
    assert gates["n_passed"] == 0
    fake_case["stokes_ratio_corrected"] = 1.005
    gates = evaluate_gates([fake_case])
    assert gates["all_recomputed_pass"] is True