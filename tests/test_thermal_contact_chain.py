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


def test_pinned_face_temperature_profiles_and_scalar_mixing():
    # In the two-row lattice, the interior row difference is halved
    # relative to the average of the prescribed end differences.
    left = np.array([6., 2.])
    t = lattice_network_temperatures(3, 2, 1., left, 0.)
    np.testing.assert_allclose(t, [left, [2.5, 1.5], [0., 0.]])
    np.testing.assert_allclose(lattice_network_temperatures(3, 2, 1., 0., left), t[::-1])
    for invalid in ([1.], [[1., 2.]], [1., np.inf], np.array([1j, 0j])):
        with pytest.raises(ChainError):
            lattice_network_temperatures(3, 2, 1., left, invalid)


@pytest.mark.parametrize("periodic", [False, True])
def test_lattice_constant_temperature_with_nonbinary_conductances(periodic):
    field = lattice_network_temperatures(97, 96, (1/1.5, 1/2.4), 300., 300., periodic_y=periodic)
    np.testing.assert_allclose(field, 300., atol=2e-13, rtol=0.)


def test_periodic_two_row_links_and_single_row_limit():
    # Two transverse links of conductance 1 and 3 connect the free pair.
    # With two axial links per particle, the resulting temperature difference is 4/10 K.
    ry = np.tile([1., 1/3], (3, 1))
    t = lattice_network_temperatures(3, 2, (1., ry), [6., 2.], 0., periodic_y=True)
    np.testing.assert_allclose(t[1], [2.2, 1.8])
    single = lattice_network_temperatures(4, 1, 1., 3., 0., periodic_y=True)
    np.testing.assert_allclose(single[:, 0], [3., 2., 1., 0.])
    with pytest.raises(ChainError):
        lattice_network_temperatures(3, 2, (1., ry[:, :1]), 1., 0., periodic_y=True)


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


@pytest.mark.parametrize("fraction", [0., .2, .85, 1.])
def test_asymmetric_sheet_heating_matches_two_branch_impedances(fraction):
    from tensordem.thermal import cell_centered_slab_network, harmonic_contact_temperatures

    omega, area, source = 3., 2., .8+.3j
    # Eliminate each adjacent cell with its heat capacity in parallel with
    # the path to the reservoir. Sum the two branch admittances at the sheet.
    half = np.array([.1, .2])/area
    side_r = np.array([fraction, 1-fraction])*.7/area
    cell_capacity = area*np.array([.2*2., .8*4.])
    z = side_r+half+1/(1/half+1j*omega*cell_capacity)
    expected = source/(sum(1/z)+1j*omega*3.*area)
    for resolve in (False, True):
        r, c = cell_centered_slab_network([.2,.8], [1.,2.], [2.,4.], area=area,
                 contact_resistance=.7, interface_heat_capacity=3., resolve_contacts=resolve,
                 left_resistance_fraction=fraction)
        sheet, = np.flatnonzero(c == 3.*area)
        power = np.zeros_like(c, dtype=complex)
        power[sheet] = source
        t = harmonic_contact_temperatures(r, c, omega, t_left=0., t_right=0.,
                                         nodal_power=power, allow_massless=True)
        np.testing.assert_allclose(t[sheet], expected, atol=1e-14, rtol=0)
        flux = -np.diff(t)/r
        np.testing.assert_allclose([-flux[sheet-1], flux[sheet]], expected/z, atol=1e-14, rtol=0)


def test_harmonic_interval_from_arbitrary_initial_state_with_link_heat():
    from tensordem.thermal import harmonic_contact_interval

    w, start, dt = 4., .23, .08
    r = np.array([.5, .8])
    c = 3.
    wall = np.array([.2+.3j, -.5j])
    source = .6-.2j
    t0 = 1.7
    mode = (sum(wall/r)+source)/(sum(1/r)+1j*w*c)
    phase = np.exp(1j*w*start)
    transient = t0-(mode*phase).real
    rate = sum(1/r)/c
    states = np.r_[wall[0], mode, wall[1]]
    exact = (states*phase*np.exp(1j*w*dt)).real
    exact[1] += transient*np.exp(-rate*dt)
    integral = (states*phase*np.expm1(1j*w*dt)/(1j*w)).real
    integral[1] -= transient*np.expm1(-rate*dt)/rate
    observed, heats = harmonic_contact_interval(r, [0., c, 0.], w, start, dt, t0,
                               t_left=wall[0], t_right=wall[1], nodal_power=[0., source, 0.])
    np.testing.assert_allclose(observed, exact, atol=2e-14, rtol=0)
    np.testing.assert_allclose(heats, -np.diff(integral)/r, atol=2e-14, rtol=0)


def test_constant_interval_resolves_sheet_source_and_storage_exactly():
    from tensordem.thermal import constant_contact_interval
    r = np.array([.3,.7,.5])
    c = [0.,0.,2.,0.]
    initial = [0.,999.,.8,0.]
    dt = .21
    end,q,mean = constant_contact_interval(r,c,dt,initial,t_left=1.,t_right=.4,
                        nodal_power=[0.,.6,.2,0.],allow_massless=True,return_interval_temperature=True)
    # Eliminating the sheet gives C*T' = (1+.3*.6-T)/1 + (.4-T)/.5 + .2.
    equilibrium = (1.+.3*.6+.4/.5+.2)/3
    decay = np.exp(-1.5*dt)
    expected = equilibrium+(.8-equilibrium)*decay
    integral = equilibrium*dt+(.8-equilibrium)*(-np.expm1(-1.5*dt))/1.5
    np.testing.assert_allclose(mean,[1.,1.-.3*(1.-integral/dt-.7*.6),integral/dt,.4],atol=1e-14)
    np.testing.assert_allclose(end,[1.,1.-.3*(1.-expected-.7*.6),expected,.4],atol=1e-14)
    np.testing.assert_allclose(q,[(1.-.7*.6)*dt-integral,(1.+.3*.6)*dt-integral,
                                  (integral-.4*dt)/.5],atol=1e-14)


def test_constant_interval_all_massless_and_subdivision():
    from tensordem.thermal import constant_contact_interval
    end,heat = constant_contact_interval([1.,2.],0.,.25,99.,t_left=2.,t_right=0.,
                                         nodal_power=[0.,1.,0.],allow_massless=True)
    np.testing.assert_allclose(end,[2.,2.,0.])
    np.testing.assert_allclose(heat,[0.,.25])
    opts = dict(t_left=.4,t_right=-.3,nodal_power=[0.,1.,.4,0.])
    def run(dt,u):
        return constant_contact_interval([.4,.12,.5],[0.,1.,2.,0.],dt,u,**opts)
    end,q = run(.25,1.)
    mid,a = run(.1,1.)
    split,b = run(.15,mid)
    np.testing.assert_allclose(end,split,atol=1e-14)
    np.testing.assert_allclose(q,a+b,atol=1e-14)


@pytest.mark.parametrize("dt",[1e-8,.37,6.])
def test_affine_contact_interval_massless_trace_and_storage(dt):
    from tensordem.thermal import affine_contact_interval
    # Manufacture particle T=1+2t+0.4 exp(-3t) and algebraic trace S.
    # Link resistances .25 and .75 on the left, .5 on the right, C=1.
    def reference(t):
        left,right,p_sheet = 2.+t,.7-.4*t,.3+.6*t
        particle = 1.+2*t+.4*np.exp(-3*t)
        sheet = .75*left+.25*particle+.1875*p_sheet
        particle_power = 2+3*(1+2*t)-left-2*right-.25*p_sheet
        return np.array([left,sheet,particle,right]),[0.,p_sheet,particle_power,0.]
    initial,p0 = reference(0.)
    expected,p1 = reference(dt)
    r = np.array([.25,.75,.5])
    end,q,mean = affine_contact_interval(r,[0.,0.,1.,0.],dt,initial,
                t_left=(initial[0],expected[0]),t_right=(initial[-1],expected[-1]),
                nodal_power_start=p0,nodal_power_end=p1,allow_massless=True,return_interval_temperature=True)
    avg,_ = reference(dt/2)
    avg[2] = 1+dt+.4*(-np.expm1(-3*dt))/(3*dt)
    avg[1] = .75*avg[0]+.25*avg[2]+.1875*(.3+.3*dt)
    np.testing.assert_allclose(end,expected,atol=3e-14,rtol=0)
    np.testing.assert_allclose(mean,avg,atol=3e-14,rtol=0)
    np.testing.assert_allclose(q,-dt*np.diff(avg)/r,atol=5e-14,rtol=0)


@pytest.mark.parametrize("films,expected_r,expected_c",[
    ((2.,4.),[.25,.0625,.0625,.125],[0.,4.,1.5,6.,0.]),
    ((None,4.),[.0625,.0625,.125],[4.,1.5,6.,0.]),
    ((None,None),[.0625,.0625],[4.,1.5,6.])])
def test_storing_surface_location_and_area_scaling(films,expected_r,expected_c):
    from tensordem.thermal import cell_centered_slab_network
    r,c = cell_centered_slab_network([.25],1.,3.,area=2.,films=films,wall_areal_heat_capacity=(2.,3.))
    np.testing.assert_allclose(r,expected_r)
    np.testing.assert_allclose(c,expected_c)


def test_harmonic_interval_massless_sheet_power_and_arbitrary_initial_temperature():
    from tensordem.thermal import harmonic_contact_interval
    w,t0,dt = 7.,.12,.19
    left,right,source = .4+.5j,-.2j,.7-.9j
    # Condense the massless sheet between .25 and .75 K/W links.
    store = (left+2*right+.25*source)/(3+1j*w)
    sheet = .75*left+.25*store+.1875*source
    amplitudes = np.array([left,sheet,store,right])*np.exp(1j*w*t0)
    initial = 1.6
    deviation = (initial-amplitudes[2].real)*np.array([0.,.25,1.,0.])
    exact = (amplitudes*np.exp(1j*w*dt)).real+deviation*np.exp(-3*dt)
    mean = (amplitudes*np.expm1(1j*w*dt)/(1j*w*dt)).real-deviation*np.expm1(-3*dt)/(3*dt)
    r = np.array([.25,.75,.5])
    end,q,avg = harmonic_contact_interval(r,[0.,0.,1.,0.],w,t0,dt,[0.,999.,initial,0.],
                    t_left=left,t_right=right,nodal_power=[0.,source,0.,0.],allow_massless=True,
                    return_interval_temperature=True)
    np.testing.assert_allclose(end,exact,atol=4e-14,rtol=0)
    np.testing.assert_allclose(avg,mean,atol=4e-14,rtol=0)
    np.testing.assert_allclose(q,-dt*np.diff(mean)/r,atol=4e-14,rtol=0)


@pytest.mark.parametrize("insulated",[(True,True),(True,False),(False,True)])
def test_affine_contact_insulated_ends_with_powered_massless_tails(insulated):
    from tensordem.thermal import affine_contact_interval

    # Manufacture a spatially varying linear ramp plus uniform acceleration.
    # With a pinned wall the quadratic coefficient must vanish there.
    r = np.array([.7,.3,.5,.8])
    c = np.array([0.,2.,0.,1.5,0.])
    u = np.array([.5,-.8,1.2,.2,.9])
    v = np.array([-.2,.3,.5,-.1,.4])
    a = .72 if all(insulated) else 0.
    dt = .36
    end = u+v*dt+a*dt**2
    mean = u+v*dt/2+a*dt**2/3
    def source(t):
        flux = -np.diff(u+t*v)/r
        return c*(v+2*a*t)+np.diff(np.r_[0.,flux,0.])
    walls = [None if free else (u[i],end[i]) for i,free in zip((0,-1),insulated)]
    initial = np.where(c>0,u,-789.)
    state,q,avg = affine_contact_interval(r,c,dt,initial,t_left=walls[0],t_right=walls[1],
                nodal_power_start=source(0),nodal_power_end=source(dt),allow_massless=True,
                return_interval_temperature=True)
    np.testing.assert_allclose(state,end,atol=2e-14,rtol=0.)
    np.testing.assert_allclose(avg,mean,atol=2e-14,rtol=0.)
    np.testing.assert_allclose(q,-dt*np.diff(mean)/r,atol=2e-14,rtol=0.)


def test_insulated_two_particle_decay_and_single_storage_limit():
    from tensordem.thermal import affine_contact_interval, constant_contact_interval
    c = np.array([1.5,3.])
    u = np.array([1.,-.2])
    dt,r = .48,.7
    equilibrium = c@u/c.sum()
    rate = (1/c).sum()/r
    state,q = constant_contact_interval([r],c,dt,u,t_left=None,t_right=None)
    expected = equilibrium+(u-equilibrium)*np.exp(-rate*dt)
    np.testing.assert_allclose(state,expected,atol=2e-15)
    np.testing.assert_allclose(q,c[0]*(u[0]-expected[0]),atol=2e-15)
    state,q,avg = affine_contact_interval([2.],[0.,3.],dt,[12.,.4],t_left=None,t_right=None,
             nodal_power_start=[1.,2.],nodal_power_end=[2.,4.],allow_massless=True,return_interval_temperature=True)
    final = .4+1.5*dt
    mean = .4+2*dt/3
    np.testing.assert_allclose(state,[final+4,final],atol=2e-15)
    np.testing.assert_allclose(avg,[mean+3,mean],atol=2e-15)
    np.testing.assert_allclose(q,[1.5*dt],atol=2e-15)


def test_insulated_harmonic_two_stores_with_transient_and_net_power():
    from tensordem.thermal import harmonic_contact_interval
    c = np.array([2.,3.])
    initial = np.array([.4,-.6])
    p = np.array([.8+.1j,-.3+.2j])
    start,dt,w,r = .17,.23,4.,.6
    phase = np.exp(1j*w*start)
    mean_phase = phase*np.expm1(1j*w*dt)/(1j*w*dt)
    rate = (1/c).sum()/r
    # The capacity-weighted mean and the difference solve independent scalar ODEs.
    common = p.sum()/(1j*w*c.sum())
    difference = (p[0]/c[0]-p[1]/c[1])/(rate+1j*w)
    drift = c@initial/c.sum()-(phase*common).real
    offset = initial[0]-initial[1]-(phase*difference).real
    sign = np.array([c[1],-c[0]])/c.sum()
    def state(z,relax):
        return drift+(z*common).real+sign*((z*difference).real+offset*relax)
    end,q,mean = harmonic_contact_interval([r],c,w,start,dt,initial,t_left=None,t_right=None,
                          nodal_power=p,return_interval_temperature=True)
    exact = state(phase*np.exp(1j*w*dt),np.exp(-rate*dt))
    avg = state(mean_phase,-np.expm1(-rate*dt)/(rate*dt))
    np.testing.assert_allclose(end,exact,atol=2e-15,rtol=0.)
    np.testing.assert_allclose(mean,avg,atol=2e-15,rtol=0.)
    np.testing.assert_allclose(q,dt*(avg[0]-avg[1])/r,atol=2e-15,rtol=0.)


@pytest.mark.parametrize("resolve",[False,True])
def test_stepped_area_resistances_and_storing_contact(resolve):
    from tensordem.thermal import cell_centered_slab_network
    r,c = cell_centered_slab_network([.6,.4],[2.,5.],[3.,4.],area=[1.,2.],
                 films=(2.,4.),contact_resistance=.8,interface_heat_capacity=.7,
                 left_resistance_fraction=.25,wall_areal_heat_capacity=(.1,.3),resolve_contacts=resolve)
    # Each half cell uses its local area; the interface has 1 m² overlap.
    expected = ([.5,.15,.15,.2,.6,.02,.02,.125] if resolve else
                [.5,.15,.35,.62,.02,.125])
    expected_c = ([0.,.1,1.8,0.,.7,0.,3.2,.6,0.] if resolve else
                  [0.,.1,1.8,.7,3.2,.6,0.])
    np.testing.assert_allclose(r,expected)
    np.testing.assert_allclose(c,expected_c)


def test_uniform_closed_chain_does_not_generate_heat_across_a_contact():
    from tensordem.thermal import affine_contact_interval
    c = np.linspace(.003,.1,64)
    initial = 1000.
    end,q,mean = affine_contact_interval(np.geomspace(.01,3.,63),c,.7,initial,
                        t_left=None,t_right=None,return_interval_temperature=True)
    np.testing.assert_array_equal(end,np.full_like(c,initial))
    np.testing.assert_array_equal(mean,end)
    np.testing.assert_array_equal(q,np.zeros(len(c)-1))


def test_insulated_low_frequency_uniform_rate_does_not_change_link_heat():
    from tensordem.thermal import harmonic_contact_interval, harmonic_contact_temperatures
    c = np.array([2., 3.])
    p = np.array([.7+.2j, -.7-.2j])
    rate, omega, dt = .4+.6j, 1e-16, .5
    opts = dict(t_left=None, t_right=None)
    relative, measured = harmonic_contact_temperatures([.6], c, omega, nodal_power=p+c*rate,
                                                        separate_mean=True, **opts)
    np.testing.assert_allclose(measured, rate, atol=2e-15)
    np.testing.assert_allclose(c@relative, 0., atol=2e-15)
    np.testing.assert_allclose(-np.diff(relative)/.6, p[:1], atol=2e-15)
    def interval(power):
        return harmonic_contact_interval([.6], c, omega, 0., dt, [.4, -.6],
                        nodal_power=power, return_interval_temperature=True, **opts)
    end, heat, avg = interval(p)
    driven, driven_heat, driven_avg = interval(p+c*rate)
    np.testing.assert_allclose(driven-end, rate.real*dt, atol=2e-15, rtol=0)
    np.testing.assert_allclose(driven_avg-avg, rate.real*dt/2, atol=2e-15, rtol=0)
    np.testing.assert_allclose(driven_heat, heat, atol=2e-15, rtol=0)


def test_zero_frequency_chain_anchors_balance_and_uniform_drift():
    from tensordem.thermal import harmonic_contact_interval, harmonic_contact_temperatures
    r, c = [.5, .3], np.array([2., 0., 1.])
    opts = dict(allow_massless=True, nodal_power=[1., .5, -1.5])
    t = harmonic_contact_temperatures(r, c, 0., **opts)
    np.testing.assert_allclose(-np.diff(t)/r, [1., 1.5], atol=1e-14)
    np.testing.assert_allclose(c@t, 0., atol=1e-14)
    opts["nodal_power"] = np.asarray(opts["nodal_power"])+.8*c
    with pytest.raises(ChainError, match="separate_mean"):
        harmonic_contact_temperatures(r, c, 0., **opts)
    drifted, rate = harmonic_contact_temperatures(r, c, 0., separate_mean=True, **opts)
    np.testing.assert_allclose(drifted, t, atol=1e-14)
    np.testing.assert_allclose(rate, .8, atol=1e-14)
    anchored = harmonic_contact_temperatures(r, c, 0., t_left=2., t_right=-1., **opts)
    expected = (.5+2/.5-1/.3)/(1/.5+1/.3)
    np.testing.assert_allclose(anchored, [2., expected, -1.])
    end, heat, mean = harmonic_contact_interval(r, c, 0., 4., .7, anchored.real,
        t_left=2., t_right=-1., return_interval_temperature=True, **opts)
    np.testing.assert_allclose(end, anchored)
    np.testing.assert_allclose(mean, anchored)
    np.testing.assert_allclose(heat, -.7*np.diff(anchored)/r)


@pytest.mark.parametrize("closed", [False, True])
def test_low_frequency_single_link_matches_analytic_relaxation(closed):
    from tensordem.thermal import low_frequency_contact_temperatures
    c = np.array([2., 3.])
    r = .6
    p = np.array([.2+.4j, .7-.3j])
    wall = None if closed else .8+.3j
    t, slope, rate = low_frequency_contact_temperatures([r], c, t_left=wall, nodal_power=p)
    if closed:
        relaxation = r/np.sum(1/c)
        difference = relaxation*(p[0]/c[0]-p[1]/c[1])
        expected = difference*np.array([c[1], -c[0]])/c.sum()
        np.testing.assert_allclose(rate, p.sum()/c.sum())
        np.testing.assert_allclose(t, expected)
        np.testing.assert_allclose(slope, -relaxation*expected)
    else:
        # A prescribed west wall fixes T0; the east store has time constant r*C.
        expected = wall+r*p[1]
        np.testing.assert_allclose(t, [wall, expected])
        np.testing.assert_allclose(slope, [0., -r*c[1]*expected])
        assert rate == 0.


def test_lattice_surface_exchange_combines_on_one_row():
    # Two axial links at 300 K, 3 W/K at 310 K and 1 W/K at 294 K.
    t = lattice_network_temperatures(3, 1, 1., 300., 300.,
                                     lower_exchange=([0., 3., 0.], [290., 310., 320.]),
                                     upper_exchange=(1., 294.))
    np.testing.assert_allclose(t[:, 0], [300., 304., 300.], rtol=0., atol=1e-13)


@pytest.mark.parametrize("exchange", [(-1., 300.), ([1., 2.], 300.), (1., np.nan), (1., 300j)])
def test_lattice_rejects_invalid_exchange(exchange):
    with pytest.raises(ChainError):
        lattice_network_temperatures(3, 2, 1., 300., 300., lower_exchange=exchange)


def test_periodic_lattice_rejects_surface_exchange():
    with pytest.raises(ChainError):
        lattice_network_temperatures(3, 2, 1., 300., 300., periodic_y=True, upper_exchange=(1., 300.))


def test_weak_chain_link_currents_at_large_temperature_offset():
    case = ChainCase(t_hot=305.+2**-24, t_cold=305., cells_per_sphere=64)
    chain = ThermalChain(case)
    temperature, currents = chain.solve_steady(return_link_currents=True)
    np.testing.assert_array_equal(temperature, chain.solve_steady())
    np.testing.assert_allclose(currents, case.heat_flow, rtol=1e-8, atol=0)


def test_steady_payload_resolves_one_ulp_wall_difference():
    import json
    from tensordem.thermal.contact_chain import thermal_chain_calculation

    payload = thermal_chain_calculation(dict(t_cold=305., t_hot=np.nextafter(305., np.inf),
                                             cells_per_sphere=64))
    json.dumps(payload, allow_nan=False)
    steady = payload["steady"]
    assert steady["measured_heat_flow_w"] == pytest.approx(steady["heat_flow_w"], rel=1e-8, abs=0)
    assert steady["measured_k_eff_w_m_k"] == pytest.approx(payload["contact_law"]["k_eff_w_m_k"], rel=1e-8)
    assert steady["contact_jump_k"] > 0


def test_transient_current_remains_resolved_at_one_ulp_drive():
    options = dict(n_spheres=4, cells_per_sphere=3)
    drop = np.spacing(305.)
    reference = TransientChain(ChainCase(t_hot=drop, t_cold=0., **options))
    chain = TransientChain(ChainCase(t_hot=305.+drop, t_cold=305., **options))
    dt = chain.relaxation_time()/32
    normalized = reference.march(dt, 32, record_every=4)
    actual = chain.march(dt, 32, record_every=4)
    np.testing.assert_array_equal(actual["hot_flux"], normalized["hot_flux"])
    np.testing.assert_array_equal(actual["nodes"], 305.+normalized["nodes"])


def test_transient_payload_temperature_error_is_offset_invariant():
    from tensordem.thermal.contact_chain import thermal_chain_calculation

    errors = []
    for base in (0., 305.):
        payload = thermal_chain_calculation(dict(mode="transient", n_spheres=4, cells_per_sphere=3,
                                                 t_cold=base, t_hot=base+2**-44, steps=256))
        errors.append(payload["transient"]["max_rel_error"])
    assert errors[0] == errors[1]
    assert 1e-3 < errors[0] < 3e-3


@pytest.mark.parametrize("theta", [.5, .75, 1.])
def test_integrated_link_heat_pinned_ends_and_massless_junction(theta):
    from tensordem.thermal.contact_chain import transient_contact_temperatures

    time = np.array([0., .02, .1])
    west, east = np.array([1., -.3, .7]), np.array([0., .2, -.4])
    for links in ([3.], [1., 2.]):
        field, heat = transient_contact_temperatures(
            links, 0., time, 0., t_left=west, t_right=east, theta=theta,
            allow_massless=True, return_link_heat=True)
        power = (west-east)/3
        expected = np.diff(time)*((1-theta)*power[:-1]+theta*power[1:])
        np.testing.assert_allclose(heat, np.broadcast_to(expected[:, None], heat.shape), atol=1e-14)
        assert field.shape == (3, len(links)+1) and heat.shape == (2, len(links))


def test_affine_link_time_moments_match_independent_rc_integrals():
    from scipy.integrate import quad
    from tensordem.thermal.contact_chain import affine_contact_interval

    # C=1 and two unit resistors. Left wall ramps as t, right wall stays zero.
    # dT/dt+2T=t, T(0)=1.
    for dt in (1e-6, .7, 25.):
        _, heat, moments = affine_contact_interval(
            [1., 1.], [0., 1., 0.], dt, [0., 1., 0.],
            t_left=(0., dt), t_right=(0., 0.), return_heat_moments=True)
        def temp(t):
            return t/2-.25+1.25*np.exp(-2*t)
        for j, actual in enumerate([heat, *moments]):
            expected = [quad(lambda t: t**j*(t-temp(t)), 0., dt)[0],
                        quad(lambda t: t**j*temp(t), 0., dt)[0]]
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=dt**(j+1)*2e-13)


@pytest.mark.parametrize("omega", [0., 1e-8, 2., 30.])
def test_harmonic_moments_insulated_massless_junction(omega, record_property):
    from scipy.integrate import quad
    from tensordem.thermal.contact_chain import harmonic_contact_interval

    # Two unit stores separated by two half-resistances and a massless source.
    # With D=T_left-T_right and junction input S, q_left=D-S/2, q_right=D+S/2.
    # Subtracting the two storage balances gives D'+2D=P_left-P_right.
    start, dt = .27, .8
    powers = np.array([.7+.2j, -.5+.4j, .1-.3j])
    difference = (powers[0]-powers[2])/(2+1j*omega)
    phase = np.exp(1j*omega*start)
    departure = .9-(difference*phase).real
    def flux(t, side):
        harmonic = (difference+side*powers[1]/2)*phase*np.exp(1j*omega*t)
        return harmonic.real+departure*np.exp(-2*t)
    _, heat, _, moments = harmonic_contact_interval(
        [.5, .5], [1., 0., 1.], omega, start, dt, [.9, 99., 0.],
        t_left=None, t_right=None, nodal_power=powers, allow_massless=True,
        return_interval_temperature=True, return_heat_moments=True)
    for j, actual in enumerate([heat, *moments]):
        expected = [quad(lambda t: t**j*flux(t, side), 0., dt)[0] for side in (-1, 1)]
        record_property(f"order_{j}_max_error", float(np.max(np.abs(actual-expected))))
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=0.)
