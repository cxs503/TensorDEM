"""Acceptance tests for the ice-structure four-ledger energy audit.

Covers the machine-checkable criteria of node P304W2-N06:

(a) kinetic energy error < 0.1 % vs the literal definition
(b) contact strain energy error < 0.5 % vs the spring-potential sum
(c) fracture energy error < 1 % vs the bond energy release (fine-grid quadrature)
(d) viscous dissipation error < 2 % vs numerical integration (and vs the
    analytic restitution loss for a single-sphere impact)
(e) energy residual eps_E < 1 % for an elastic (no-fracture) impact
(f) energy residual eps_E < 2 % for an impact with fracture
(g) negative control: zero velocity -> zero kinetic energy -> zero change
(h) timestep sensitivity: halving dt does not increase eps_E
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.ice_structure_energy_ledger import (
    CohesiveBondLaw,
    ContactLaw,
    IceStructureEnergyAudit,
    bond_dissipation_reference,
    build_bonded_lattice,
    contact_strain_energy_reference,
    kinetic_energy_reference,
    run_energy_audit_case,
    viscous_dissipation_reference,
)

RADIUS = 0.05
DENSITY = 900.0


def _elastic_case(dt: float = 1.0e-5, duration: float = 8.0e-3):
    return run_energy_audit_case(
        impact_velocity=(-2.0, 0.0, 0.0),
        contact=ContactLaw(k_normal=1.0e6, restitution=1.0),
        bond_law=CohesiveBondLaw(tensile_strength=4.0e5),
        dt=dt,
        duration=duration,
        gap=2.0e-3,
    )


def _fracture_case(dt: float = 1.0e-5, duration: float = 8.0e-3):
    return run_energy_audit_case(
        impact_velocity=(-2.0, 0.0, 0.0),
        contact=ContactLaw(k_normal=1.0e6, restitution=0.9),
        bond_law=CohesiveBondLaw(tensile_strength=2.0e5),
        dt=dt,
        duration=duration,
        gap=2.0e-3,
    )


# ---------------------------------------------------------------------------
# cohesive law admissibility
# ---------------------------------------------------------------------------
def test_cohesive_law_area_equals_fracture_energy():
    law = CohesiveBondLaw()
    assert law.separation > law.elastic_limit
    grid = np.linspace(0.0, law.separation, 200001)
    traction = np.where(
        grid <= law.elastic_limit,
        law.stiffness * grid,
        law.peak_force * (law.separation - grid)
        / (law.separation - law.elastic_limit),
    )
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = float(trapezoid(traction, grid))
    assert area == pytest.approx(law.surface_energy, rel=1e-6)


def test_cohesive_law_rejects_non_griffith_parameters():
    # sigma_t = 5e5 with k = 1e7, A = 1e-3 needs G_f > 12.5 J/m^2
    with pytest.raises(ValueError):
        CohesiveBondLaw(tensile_strength=5.0e5, fracture_energy=10.0)
    # admissible partner
    CohesiveBondLaw(tensile_strength=5.0e5, fracture_energy=20.0)


# ---------------------------------------------------------------------------
# (a) kinetic energy
# ---------------------------------------------------------------------------
def test_kinetic_energy_matches_definition():
    sim = _fracture_case(duration=3.0e-3)
    sim.omegas[0] = np.array([1.0, -2.0, 0.5])       # make E_k_rot non-trivial
    sim._refresh_state_ledger()
    ref = kinetic_energy_reference(sim.velocities, sim.omegas, sim.mass, sim.inertia)
    for key in ("E_k_trans", "E_k_rot", "E_k"):
        err = abs(getattr(sim.ledger, key) - ref[key]) / max(abs(ref[key]), 1e-30)
        assert err < 1.0e-3, f"{key} error {err:.3e}"
    assert sim.ledger.E_k_rot > 0.0


# ---------------------------------------------------------------------------
# (b) contact strain energy (normal + tangential + bond)
# ---------------------------------------------------------------------------
def test_contact_strain_energy_matches_spring_potential():
    sim = _fracture_case(duration=3.0e-3)
    ref = contact_strain_energy_reference(sim)
    for key in ("E_s_contact_normal", "E_s_bond", "E_s"):
        err = abs(getattr(sim.ledger, key) - ref[key]) / max(abs(ref[key]), 1e-30)
        assert err < 5.0e-3, f"{key} error {err:.3e}: {getattr(sim.ledger, key)} vs {ref[key]}"
    assert ref["E_s"] > 0.0


def test_tangential_spring_energy_is_tracked_in_oblique_impact():
    """Oblique sphere-wall impact: E_s must contain the tangential spring term."""
    lat = build_bonded_lattice(1, 1, 1, RADIUS, DENSITY)
    pos = lat["positions"].copy()
    pos[:, 0] += RADIUS + 1.0e-3 - pos[:, 0].min()
    sim = IceStructureEnergyAudit(
        positions=pos,
        velocities=np.array([[-1.5, 1.0, 0.0]]),
        radius=RADIUS,
        density=DENSITY,
        contact=ContactLaw(k_normal=1.0e6, restitution=0.95, friction=50.0),
        bond_law=CohesiveBondLaw(tensile_strength=1.0e30, fracture_energy=1.0e60),
        dt=2.0e-6,
        wall_plane_x=0.0,
    )
    sim.run_until(6.0e-3)
    peak_tangential = max(r["E_s_contact_tangential"] for r in sim.history)
    assert peak_tangential > 1.0e-3, "tangential spring never loaded"
    ref = contact_strain_energy_reference(sim)
    err = abs(sim.ledger.E_s_contact_tangential
              - ref["E_s_contact_tangential"]) / max(ref["E_s_contact_tangential"], 1e-30)
    assert err < 5.0e-3
    # rotation must have been excited by the tangential force
    assert np.abs(sim.omegas).max() > 0.0


# ---------------------------------------------------------------------------
# (c) fracture energy
# ---------------------------------------------------------------------------
def test_fracture_energy_matches_bond_release():
    sim = _fracture_case(duration=8.0e-3)
    assert sim.n_broken_total > 0, "no bond broke: the fracture ledger is untested"
    ref = bond_dissipation_reference(sim.bond_law, sim.bond_max_extension)
    err = abs(sim.ledger.E_f - ref["E_f_reference"]) / ref["E_f_reference"]
    assert err < 1.0e-2, f"E_f error {err:.3e}"


def test_no_bond_dissipates_more_than_its_fracture_energy():
    """Per-bond dissipation must be bounded by G_f * A (single cohesive law)."""
    sim = _fracture_case(duration=8.0e-3)
    law = sim.bond_law
    ref = bond_dissipation_reference(law, sim.bond_max_extension)
    assert ref["per_bond"].max() <= law.surface_energy * (1.0 + 1.0e-6)
    assert ref["per_bond"].sum() <= law.surface_energy * sim.bond_i.size
    # and the model may not break more bonds than it has
    assert sim.n_broken_total <= sim.bond_i.size


# ---------------------------------------------------------------------------
# (d) viscous dissipation
# ---------------------------------------------------------------------------
def test_viscous_dissipation_matches_numerical_integration():
    sim = _fracture_case(duration=8.0e-3)
    ref = viscous_dissipation_reference(sim.history)
    assert sim.ledger.E_v > 0.0
    err = abs(sim.ledger.E_v - ref) / ref
    assert err < 2.0e-2, f"E_v error {err:.3e}"


def test_viscous_dissipation_matches_restitution_analytic():
    """Single sphere vs wall: E_v must equal 0.5 m v0^2 (1 - e^2)."""
    restitution = 0.8
    lat = build_bonded_lattice(1, 1, 1, RADIUS, DENSITY)
    pos = lat["positions"].copy()
    pos[:, 0] += RADIUS + 5.0e-3 - pos[:, 0].min()
    v0 = -1.5
    sim = IceStructureEnergyAudit(
        positions=pos,
        velocities=np.array([[v0, 0.0, 0.0]]),
        radius=RADIUS,
        density=DENSITY,
        contact=ContactLaw(k_normal=1.0e6, restitution=restitution),
        bond_law=CohesiveBondLaw(tensile_strength=1.0e30, fracture_energy=1.0e60),
        dt=1.0e-6,
        wall_plane_x=0.0,
    )
    sim.run_until(1.2e-2)
    analytic = 0.5 * sim.mass * v0 ** 2 * (1.0 - restitution ** 2)
    err = abs(sim.ledger.E_v - analytic) / analytic
    assert err < 2.0e-2, f"E_v {sim.ledger.E_v:.6e} vs analytic {analytic:.6e}"
    # the rebound velocity gives back the restitution
    assert sim.velocities[0, 0] == pytest.approx(-v0 * restitution, rel=5e-2)


# ---------------------------------------------------------------------------
# (e) elastic closure
# ---------------------------------------------------------------------------
def test_elastic_impact_closes_below_one_percent():
    sim = _elastic_case()
    assert sim.n_broken_total == 0
    summary = sim.summary()
    assert summary["residual_relative_max"] < 1.0e-2, summary
    # the floe must have actually hit the wall
    peak_contact = max(r["E_s"] for r in sim.history)
    assert peak_contact > 0.05 * summary["E_total_0"]


# ---------------------------------------------------------------------------
# (f) fracture closure
# ---------------------------------------------------------------------------
def test_fracture_impact_closes_below_two_percent():
    sim = _fracture_case()
    assert sim.n_broken_total > 0
    summary = sim.summary()
    assert summary["residual_relative_max"] < 2.0e-2, summary
    assert summary["E_f"] > 0.0 and summary["E_v"] > 0.0


# ---------------------------------------------------------------------------
# (g) negative control
# ---------------------------------------------------------------------------
def test_zero_velocity_gives_zero_energy_and_zero_change():
    sim = run_energy_audit_case(
        impact_velocity=(0.0, 0.0, 0.0),
        contact=ContactLaw(k_normal=1.0e6, restitution=1.0),
        bond_law=CohesiveBondLaw(tensile_strength=4.0e5),
        dt=1.0e-5,
        duration=2.0e-3,
        gap=2.0e-3,
    )
    summary = sim.summary()
    assert summary["E_total_0"] == 0.0
    assert summary["E_k"] == 0.0
    assert summary["E_s"] == 0.0
    assert summary["E_f"] == 0.0
    assert summary["E_v"] == 0.0
    assert max(abs(r["E_total"]) for r in sim.history) == 0.0
    assert sim.n_broken_total == 0
    assert np.abs(sim.velocities).max() == 0.0


# ---------------------------------------------------------------------------
# (h) timestep sensitivity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["elastic", "fracture"])
def test_halving_dt_does_not_increase_residual(case):
    builder = _elastic_case if case == "elastic" else _fracture_case
    coarse = builder(dt=1.0e-5, duration=8.0e-3).summary()["residual_relative_max"]
    fine = builder(dt=5.0e-6, duration=8.0e-3).summary()["residual_relative_max"]
    assert fine <= coarse * (1.0 + 1.0e-3), (coarse, fine)


# ---------------------------------------------------------------------------
# structural invariants
# ---------------------------------------------------------------------------
def test_ledger_total_is_sum_of_four_ledgers():
    sim = _fracture_case(duration=4.0e-3)
    led = sim.ledger
    assert led.total == pytest.approx(led.E_k + led.E_s + led.E_f + led.E_v, rel=1e-12)


def test_history_rows_have_finite_ledgers():
    sim = _fracture_case(duration=4.0e-3)
    for row in sim.history:
        for key in ("E_k", "E_s", "E_f", "E_v", "E_total", "residual_relative"):
            assert math.isfinite(row[key]), (key, row)