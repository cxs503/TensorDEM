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


def test_lattice_internal_heating_matches_parabola():
    i = np.arange(8)
    expected = 0.5 * 3. * 2. * i * (7 - i)
    result = lattice_network_temperatures(8, 3, 2., 0., 0., nodal_power=3.)
    np.testing.assert_allclose(result, np.repeat(expected[:, None], 3, axis=1))


def test_layered_contact_lattice_series_heat_flow():
    rx = np.repeat(np.array([[1.], [4.], [2.], [8.]]), 3, axis=1)
    t = lattice_network_temperatures(5, 3, (rx, 0.5), 320., 290.)
    # Each parallel path has 15 K/W in series and carries 2 W.
    np.testing.assert_allclose(t[:, 0], [320., 318., 310., 306., 290.], atol=1e-12, rtol=0)
    np.testing.assert_allclose(-np.diff(t, axis=0) / rx, 2., atol=1e-12, rtol=0)


def test_unequal_contacts_drive_transverse_heat_flow():
    rx = np.array([[1., 3.], [3., 1.]])
    t = lattice_network_temperatures(3, 2, (rx, 1.), 10., 0.)
    # Interior balances are 7a - 3b = 30 and -3a + 7b = 10.
    np.testing.assert_allclose(t[1], [6., 4.])
    np.testing.assert_allclose(np.sum((10.-t[1])/rx[0]), np.sum(t[1]/rx[1]))


def test_edge_resistance_shape_and_positivity():
    for edges in ((np.ones((3, 2)), 1.), (1., 0.), (np.nan, 1.)):
        with pytest.raises(ChainError):
            lattice_network_temperatures(3, 2, edges, 10., 0.)


def test_insulated_unequal_capacity_pair_preserves_heat_and_relaxes_at_exact_rate():
    from tensordem.thermal import transient_contact_temperatures

    # Total heat is 12 J and equilibrium is 2 K. The temperature difference
    # decays at (1/R)*(1/C1+1/C2) = 1.5 per second.
    errors = []
    for steps in (10, 20, 40):
        time = np.linspace(0., 1., steps+1)
        temperatures = transient_contact_temperatures([.5], [2., 4.], time, [6., 0.])
        exact = 2 + np.exp(-1.5 * time[:, None]) * [4., -2.]
        errors.append(np.abs(temperatures - exact).max())
        np.testing.assert_allclose(temperatures @ [2., 4.], 12., rtol=0, atol=1e-13)
    assert all(a / b > 3.9 for a, b in zip(errors, errors[1:]))


def test_power_and_single_thermostat_on_nonuniform_times():
    from tensordem.thermal import transient_contact_temperatures

    time = [0., .05, .17, .5]
    # 2 W balances the link loss (9-5)/2. The thermostat overrides its
    # initial temperature and supplies the opposing reaction.
    for theta in (.5, 1.):
        temp = transient_contact_temperatures([2.], [3., 2.], time, [9., -1.],
                                             t_right=5., nodal_power=[2., 0.], theta=theta)
        np.testing.assert_allclose(temp, np.tile([9., 5.], (len(time), 1)), atol=1e-13)
    temp = transient_contact_temperatures([2.], [3., 2.], time, [9., 5.], nodal_power=[5., -2.])
    np.testing.assert_allclose((temp - temp[0]) @ [3., 2.], 3 * np.asarray(time), atol=1e-13)


@pytest.mark.parametrize("theta", [.5, 1.])
def test_sampled_massless_reservoir_and_nodal_energy_balance(theta):
    from tensordem.thermal import transient_contact_temperatures

    # One C=2 J/K node connected through R=.5 K/W to a ramp T_a=t.
    # T'=t-T, T(0)=0 gives T=t-1+exp(-t).
    errors = []
    for steps in (8, 16, 32):
        time = np.linspace(0., 1., steps+1)**1.2
        t = transient_contact_temperatures([.5], [0., 2.], time, [100., 0.],
                                           t_left=time, theta=theta)
        np.testing.assert_array_equal(t[:, 0], time)
        power = (time - t[:, 1]) / .5
        heat = np.diff(time) @ ((1-theta)*power[:-1] + theta*power[1:])
        assert abs(heat - 2*t[-1, 1]) < 1e-13
        errors.append(np.max(abs(t[:, 1] - (time-1+np.exp(-time)))))
    bound = 3.8 if theta == .5 else 1.9
    assert all(a/b > bound for a, b in zip(errors, errors[1:]))


def test_zero_capacity_requires_a_prescribed_reservoir():
    from tensordem.thermal import transient_contact_temperatures

    with pytest.raises(ChainError, match="positive capacity"):
        transient_contact_temperatures([1.], [0., 1.], [0., 1.], [0., 1.])
    with pytest.raises(ChainError, match="sampled at each time"):
        transient_contact_temperatures([1.], [0., 1.], [0., 1.], [0., 1.], t_left=[5.])


def test_cell_centered_slab_resistances_and_capacity_for_nonuniform_cells():
    from tensordem.thermal import cell_centered_slab_network

    links, capacity = cell_centered_slab_network(
        [.2, .5, .3], [2., 5., 1.], [3., 4., 2.], area=2.,
        contact_resistance=[.1, .2], films=(None, 4.))
    np.testing.assert_allclose(links, [.025, .1, .2, .2])
    np.testing.assert_allclose(capacity, [0., 1.2, 4., 1.2, 0.])


def test_one_cell_between_films_relaxes_to_conductance_weighted_ambient():
    from tensordem.thermal import cell_centered_slab_network, transient_contact_temperatures

    links, capacity = cell_centered_slab_network([.5], 2., 4., area=2., films=(8., None))
    time = np.linspace(0, .1, 101)
    result = transient_contact_temperatures(links, capacity, time, [0., 0., 0.], t_left=6., t_right=3.)
    # Link conductances 8 and 16 W/K, total capacity 4 J/K, equilibrium 4 K.
    expected = 4 * (1 - np.exp(-6*time))
    np.testing.assert_allclose(result[:, 1], expected, rtol=0, atol=4e-6)


def test_harmonic_contact_chain_matches_complex_single_store_response():
    from tensordem.thermal import harmonic_contact_temperatures

    left, right, drive = 1.+2j, -.3j, 1.-.5j
    for omega in (.05, 3., 50.):
        field = harmonic_contact_temperatures([.5, 2.], [0., 4., 0.], omega,
                                              t_left=left, t_right=right, nodal_power=[0., drive, 0.])
        expected = (2*left + .5*right + drive)/(2.5+4j*omega)
        np.testing.assert_allclose(field, [left, expected, right], atol=1e-14, rtol=0.)


def test_harmonic_insulated_chain_uniform_source_and_pinned_pair():
    from tensordem.thermal import harmonic_contact_temperatures

    field = harmonic_contact_temperatures([.4, .2], [1., 2., 3.], 2., nodal_power=[3., 6., 9.])
    np.testing.assert_allclose(field, -1.5j, atol=1e-14, rtol=0.)
    np.testing.assert_array_equal(harmonic_contact_temperatures([1.], 0., 2., t_left=1j, t_right=2.),
                                  [1j, 2.])
    with pytest.raises(ChainError):
        harmonic_contact_temperatures([1.], [0., 1.], 2.)


@pytest.mark.parametrize("theta", [.5, 1.])
def test_varying_nodal_heating_closes_each_store_on_nonuniform_steps(theta):
    from tensordem.thermal import transient_contact_temperatures

    time = np.array([0., .04, .15, .18, .4])
    r = np.array([.4, 1.2])
    capacity = np.array([0., 3., 2.])
    power = np.column_stack((0*time, 1+time**2, .2-time))
    t = transient_contact_temperatures(r, capacity, time, [1., .5, 2.], theta=theta,
                                       t_left=1+time, nodal_power=power)
    flux = -np.diff(t, axis=1)/r
    net = power[:, 1:]+np.column_stack((flux[:, 0]-flux[:, 1], flux[:, 1]))
    np.testing.assert_allclose(np.diff(t[:, 1:],axis=0)*capacity[1:],
                np.diff(time)[:, None]*((1-theta)*net[:-1]+theta*net[1:]), rtol=0, atol=2e-15)
    with pytest.raises(ChainError, match="power history"):
        transient_contact_temperatures(r, capacity, time, 0., t_left=0., nodal_power=power[:-1])


@pytest.mark.parametrize("theta", [.5, 1.])
def test_massless_driven_contact_matches_eliminated_network_at_every_instant(theta):
    from tensordem.thermal import transient_contact_temperatures

    r = np.array([.3, .7, .4])
    times = np.array([0., .02, .06, .13, .19, .3])
    wall = np.cos(5*times)
    injection = .7*np.sin(8*times)+.2
    input_power = np.zeros((len(times), 4))
    input_power[:, 1] = injection
    # Eliminate the junction between the left wall and the storage node.
    # The store receives r0/(r0+r1) of the junction's applied power.
    reduced_power = np.zeros((len(times), 3))
    reduced_power[:, 1] = r[0]/sum(r[:2])*injection
    reduced = transient_contact_temperatures([sum(r[:2]), r[2]], [0., 3., 0.],
                 times, [0., 2., 0.], t_left=wall, t_right=.2, nodal_power=reduced_power, theta=theta)
    full = transient_contact_temperatures(r, [0., 0., 3., 0.], times, [0., -500., 2., 0.],
                 t_left=wall, t_right=.2, nodal_power=input_power, theta=theta, allow_massless=True)
    np.testing.assert_allclose(full[:, [0, 2, 3]], reduced, atol=3e-14, rtol=0)
    expected = (wall/r[0]+reduced[:, 1]/r[1]+injection)/(1/r[0]+1/r[1])
    np.testing.assert_allclose(full[:, 1], expected, atol=3e-14, rtol=0)


def test_all_massless_series_chain_requires_reference_and_obeys_power_balance():
    from tensordem.thermal import harmonic_contact_temperatures, transient_contact_temperatures

    r = np.array([.3, .4, .8])
    fraction = np.r_[0., r.cumsum()]/r.sum()
    endpoints = np.array([.3+1j, 2.-.4j])
    expected = endpoints[0]+fraction*(endpoints[1]-endpoints[0])
    t = harmonic_contact_temperatures(r, 0., 3., t_left=endpoints[0], t_right=endpoints[1], allow_massless=True)
    np.testing.assert_allclose(t, expected, atol=1e-14, rtol=0)
    time = np.array([0., .13, .3])
    t = transient_contact_temperatures(r, 0., time, 0., t_left=1+time, t_right=2*time, allow_massless=True)
    np.testing.assert_allclose(t, (1+time[:, None])*(1-fraction)+2*time[:, None]*fraction)
    with pytest.raises(ChainError, match="prescribed temperature"):
        harmonic_contact_temperatures(r, 0., 3., allow_massless=True)
    with pytest.raises(ChainError, match="prescribed temperature"):
        transient_contact_temperatures(r, 0., time, 0., allow_massless=True)


def test_resolved_multiple_contacts_match_condensed_heterogeneous_stack():
    from tensordem.thermal import cell_centered_slab_network, harmonic_contact_temperatures

    args = ([.1, .25, .2, .45], [2., .8, 3., 1.], [3., 2., 4., 1.])
    options = dict(area=1.7, films=(4., 7.), contact_resistance=[.1, 0., .4])
    r, c = cell_centered_slab_network(*args, **options)
    expanded, stores = cell_centered_slab_network(*args, resolve_contacts=True, **options)
    ids = np.flatnonzero(stores)
    np.testing.assert_array_equal(ids, [1, 4, 5, 8])
    np.testing.assert_allclose(stores[ids], c[1:-1])
    wall = dict(t_left=.3+.7j, t_right=.1j)
    t = harmonic_contact_temperatures(r, c, 3., **wall)
    full = harmonic_contact_temperatures(expanded, stores, 3., allow_massless=True, **wall)
    np.testing.assert_allclose(full[ids], t[1:-1], atol=2e-14, rtol=0)
    flux = -np.diff(t)/r
    for cell, resistance in ((0, .1), (2, .4)):
        left = ids[cell]+1
        np.testing.assert_allclose(full[left]-full[left+1], resistance/1.7*flux[cell+1],
                                   atol=2e-14, rtol=0)


@pytest.mark.parametrize("contact", [0., .6])
@pytest.mark.parametrize("resolved", [False, True])
def test_interface_storage_splits_resistance_and_balances_complex_power(contact, resolved):
    from tensordem.thermal import cell_centered_slab_network, harmonic_contact_temperatures

    area = 2.
    links, c = cell_centered_slab_network([.2, .8], [1., 2.], [2., 4.], area=area,
                    contact_resistance=contact, resolve_contacts=resolved, interface_heat_capacity=3.)
    sheet, = np.flatnonzero(c == 6.)
    assert c.sum() == pytest.approx(area*(.2*2+.8*4+3))
    assert links.sum() == pytest.approx((.2+.8/2+contact)/area)
    t = harmonic_contact_temperatures(links, c, 3., t_left=1+.3j, t_right=.2j, allow_massless=True)
    q = -np.diff(t)/links
    np.testing.assert_allclose(q[sheet-1]-q[sheet], 3j*c[sheet]*t[sheet], atol=2e-14)
    # Independent three-store reduction and resistances to the sheet.
    r = np.array([.1, .1+contact/2, .2+contact/2, .2])/area
    exact = harmonic_contact_temperatures(r, [0., .8, 6., 6.4, 0.], 3., t_left=1+.3j, t_right=.2j)
    np.testing.assert_allclose(t[c > 0], exact[1:-1], atol=2e-14, rtol=0)
