"""Verification of the particle-chain thermal conduction operators.

Covers the contact law (Hertz patch -> constriction resistance), the 1D network
against the closed-form series solution, the 2D square lattice against its exact
two-face resistance, the implicit-Euler transient against the matrix
exponential (with its measured order of convergence), and the ice-bond
thermal-mechanical gate (two independent routes).

Acceptance target of the campaign node: particle-chain steady temperature within
5 % of the analytical solution; the tests below assert a much tighter 1e-9
because the 1D network can represent the exact profile.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.thermal.contact_chain import (
    BondGateResult,
    ChainCase,
    ChainError,
    ThermalChain,
    TransientChain,
    analytic_chain_temperature,
    bond_failure_gate,
    bond_temperature_analytic,
    chain_effective_conductivity,
    chain_resistance,
    contact_conductance,
    contact_constriction_resistance,
    hertz_contact_radius,
    ice_bond_tensile_strength,
    lattice_network_temperatures,
    sphere_bulk_resistance,
    square_lattice_resistance,
)


def test_hertz_contact_radius_matches_analytic_formula_and_scales_as_force_cube_root():
    force, radius, youngs, poisson = 20.0, 0.005, 9.0e9, 0.31
    e_star = youngs / (2.0 * (1.0 - poisson**2))
    expected = (3.0 * force * (radius / 2.0) / (4.0 * e_star)) ** (1.0 / 3.0)
    got = hertz_contact_radius(force, radius, youngs, poisson)
    assert got == pytest.approx(expected, rel=1e-14)
    # a_c ~ F^(1/3): an 8x load gives a 2x patch
    ratio = hertz_contact_radius(8.0 * force, radius, youngs, poisson) / got
    assert ratio == pytest.approx(2.0, rel=1e-12)


def test_constriction_resistance_is_one_over_four_k_a_and_scales_with_patch():
    a_c, k = 2.0e-4, 2.2
    assert contact_constriction_resistance(a_c, k) == pytest.approx(1.0 / (4.0 * k * a_c),
                                                                  rel=1e-15)
    assert contact_constriction_resistance(2.0 * a_c, k) == pytest.approx(
        0.5 * contact_constriction_resistance(a_c, k), rel=1e-15)
    assert contact_conductance(a_c, k) == pytest.approx(4.0 * k * a_c, rel=1e-15)
    # the constriction dominates the bulk of a sphere with a_c/R = 0.04
    assert contact_constriction_resistance(a_c, k) > 5.0 * sphere_bulk_resistance(
        0.005, k)


def test_chain_resistance_equals_sum_of_bulk_and_contact_resistances():
    case = ChainCase(n_spheres=8)
    closed = ((case.n_spheres - 1) * contact_constriction_resistance(
        case.contact_radius, case.conductivity)
        + case.n_spheres * sphere_bulk_resistance(case.radius, case.conductivity))
    assert case.resistance == pytest.approx(closed, rel=1e-14)
    assert chain_resistance(8, case.contact_radius, case.radius,
                            case.conductivity) == pytest.approx(closed, rel=1e-14)


def test_steady_chain_network_reproduces_the_closed_form_profile():
    """Campaign gate (a): chain steady temperature vs analytic (< 5 % required)."""
    case = ChainCase(n_spheres=10, cells_per_sphere=3)
    chain = ThermalChain(case)
    numeric = chain.solve_steady()
    analytic = chain.analytic_steady()
    span = float(np.ptp(analytic))
    rel = float(np.max(np.abs(numeric - analytic)) / span)
    assert rel < 1e-9, rel
    # heat flow of the solved field equals the closed-form value
    assert chain.measured_heat_flow(numeric) == pytest.approx(case.heat_flow, rel=1e-10)
    # the effective conductivity recovered from the field matches the series model
    assert chain.measured_effective_conductivity(numeric) == pytest.approx(
        case.effective_conductivity, rel=1e-10)
    # and the constriction costs more than 80 % of the conductivity
    assert case.conductivity_ratio < 0.2


def test_contact_temperature_jump_equals_heat_flow_times_constriction_resistance():
    case = ChainCase()
    chain = ThermalChain(case)
    t = chain.solve_steady()
    expected = case.heat_flow * contact_constriction_resistance(case.contact_radius,
                                                                case.conductivity)
    for index in range(case.n_spheres - 1):
        assert chain.contact_jump(t, index) == pytest.approx(expected, rel=1e-9)


def test_transient_matches_matrix_exponential_and_converges_first_order():
    case = ChainCase(n_spheres=8, cells_per_sphere=4, t_hot=272.15, t_cold=253.15)
    chain = TransientChain(case)
    tau = chain.relaxation_time()
    assert tau > 0.0
    errors = []
    for steps_per_tau in (400, 800):
        res = chain.march(dt=tau / steps_per_tau, n_steps=3 * steps_per_tau,
                          record_every=steps_per_tau)
        exact = chain.analytic_transient(res["time"])
        errors.append(float(np.max(np.abs(res["nodes"] - exact)) / np.ptp(exact)))
    assert errors[-1] < 2e-3
    order = math.log(errors[0] / errors[1]) / math.log(2.0)
    assert order == pytest.approx(1.0, abs=0.1)
    # the lumped ladder's slowest mode is within 15 % of the continuum slab
    cont = chain.continuum_relaxation_time()
    assert abs(tau - cont) / cont < 0.15


def test_square_lattice_two_face_resistance_matches_the_closed_form():
    nx, ny, r_link = 7, 5, 3.0
    exact = square_lattice_resistance(nx, ny, r_link)
    assert exact == pytest.approx((nx - 1) / ny * r_link, rel=1e-15)
    t = lattice_network_temperatures(nx, ny, r_link, 320.0, 280.0)
    q_total = ny * (t[1] - t[2]).mean() / r_link
    assert q_total == pytest.approx(40.0 / exact, rel=1e-10)
    # every column carries the same flow: the field is a pure function of x
    assert np.allclose(t, t[:, :1], rtol=0.0, atol=1e-9)


def test_ice_bond_strength_follows_glen_creep_temperature_dependence():
    cold = ice_bond_tensile_strength(263.15)
    warm = ice_bond_tensile_strength(273.15)
    assert cold == pytest.approx(1.0e6, rel=1e-12)
    assert warm < cold
    # Q = 60 kJ/mol, n = 3 over a 10 K step from -10 degC to 0 degC
    expected = math.exp((60.0e3 / (3.0 * 8.31446261815324)) * (1 / 273.15 - 1 / 263.15))
    assert warm / cold == pytest.approx(expected, rel=1e-12)
    assert 0.70 < warm / cold < 0.73


def test_bond_gate_two_routes_agree_on_the_failure_time():
    """Ice-bond thermal-mechanical gate: analytic vs simulated failure time."""
    case = ChainCase(n_spheres=6, radius=0.005, contact_force=20.0,
                     t_hot=273.15 - 0.5, t_cold=273.15 - 20.0)
    result = bond_failure_gate(case, bond_index=1, bond_radius=1e-4,
                               applied_force=0.042, n_steps=20000)
    assert isinstance(result, BondGateResult)
    assert result.fails
    assert math.isfinite(result.analytic_failure_time)
    assert result.failure_time_error < 0.05
    # the bond sits between the hot and cold ends, strictly inside the chain
    t_bond_hot = bond_temperature_analytic(case, 1, case.t_hot)
    assert case.t_cold < t_bond_hot < case.t_hot
    assert result.t_bond_critical < t_bond_hot
    # no load, no failure: capacity is always above a vanishing stress
    light = bond_failure_gate(case, bond_index=1, bond_radius=1e-4,
                              applied_force=1e-9, n_steps=50)
    assert math.isinf(light.simulated_failure_time) and not light.fails


@pytest.mark.parametrize("kwargs", [
    {"n_spheres": 1},
    {"cells_per_sphere": 0},
    {"radius": -0.005},
    {"conductivity": 0.0},
    {"t_hot": 250.0, "t_cold": 260.0},
])
def test_chain_case_rejects_unphysical_inputs(kwargs):
    with pytest.raises(ChainError):
        ChainCase(**kwargs)


def test_analytic_profile_helper_is_consistent_with_the_network_ordering():
    case = ChainCase(n_spheres=4, cells_per_sphere=2)
    profile = analytic_chain_temperature(case.n_spheres, case.contact_radius,
                                         case.radius, case.conductivity,
                                         case.cells_per_sphere, case.t_hot, case.t_cold)
    assert profile.shape == (case.n_spheres * (case.cells_per_sphere + 1),)
    assert profile[0] == pytest.approx(case.t_hot, rel=1e-15)
    assert profile[-1] == pytest.approx(case.t_cold, rel=1e-12)
    assert np.all(np.diff(profile) < 0.0)  # monotonically cooling
    assert chain_effective_conductivity(case.n_spheres, case.contact_radius,
                                        case.radius, case.conductivity) < case.conductivity

def test_thermal_api_contract_three_modes():
    from tensordem.thermal.contact_chain import thermal_chain_calculation

    steady = thermal_chain_calculation({"mode": "steady", "n_spheres": 6})
    assert steady["endpoint"] == "/api/v1/calculations/dem/thermal"
    assert steady["capability"] == "REAL"
    assert steady["steady"]["max_rel_error"] < 5e-3
    transient = thermal_chain_calculation({"mode": "transient", "n_spheres": 5,
                                          "cells_per_sphere": 2, "steps": 600})
    assert transient["transient"]["max_rel_error"] < 1e-2
    bond = thermal_chain_calculation({"mode": "bond", "n_spheres": 6,
                                      "t_hot": 272.65, "t_cold": 253.15})
    assert bond["bond_gate"]["fails"] is True
    assert bond["bond_gate"]["failure_time_rel_error"] < 0.05
    with pytest.raises(ChainError):
        thermal_chain_calculation({"mode": "nope"})
    with pytest.raises(ChainError):
        thermal_chain_calculation("not a mapping")
