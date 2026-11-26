"""Verification of the interface conduction operators and particle network.

References that do not reuse the code under test:

* semi-infinite step response -- ``erfc(x / 2 sqrt(alpha t))``;
* constant surface flux -- ``Ts = Ti + 2 q sqrt(alpha t / pi) / k``;
* two-particle relaxation -- the analytic lumped exponential;
* particle network -- exact energy conservation and the energy-weighted
  equilibrium temperature.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.special import erfc

from tensordem.interface_thermal.conduction import (
    Conduction1D,
    Particle,
    ParticleThermalNetwork,
    contact_heat_transfer_coefficient,
    contact_thermal_resistance,
    maxwell_constriction_resistance,
    semi_infinite_surface_temperature,
    semi_infinite_temperature_profile,
    thermal_diffusivity,
)

ICE = dict(k_w_m_k=2.2, rho_kg_m3=917.0, cp_j_kg_k=2100.0)
ALPHA = thermal_diffusivity(ICE["k_w_m_k"], ICE["rho_kg_m3"], ICE["cp_j_kg_k"])


def test_semi_infinite_step_matches_erfc_solution():
    """Profile of a step-heated body against the exact erfc solution (<2 %)."""
    conductor = Conduction1D(length_m=0.06, cells=401, t_initial=-2.0, **ICE)
    history = conductor.run(60.0, 0.01, t_surface=0.0,
                            record_times=[5.0, 20.0, 60.0])
    for time, profile in zip(history["time"], history["profiles"]):
        exact = -2.0 + 2.0 * erfc(history["x"] / (2.0 * math.sqrt(ALPHA * time)))
        error = np.max(np.abs(profile - exact))
        assert error < 0.02 * 2.0, f"t={time}: max error {error:.4g} K"


def test_constant_flux_surface_temperature_matches_carslaw_jaeger():
    """Flash temperature under a constant flux against the exact solution."""
    flux = 2.0e4
    conductor = Conduction1D(length_m=0.06, cells=401, t_initial=-2.0, **ICE)
    history = conductor.run(60.0, 0.01, q_surface=flux,
                            record_times=[1.0, 10.0, 60.0])
    exact = semi_infinite_surface_temperature(np.asarray(history["time"]), ALPHA,
                                              ICE["k_w_m_k"], flux, -2.0)
    rise_exact = exact - (-2.0)
    rise_numeric = history["surface"] - (-2.0)
    relative = np.abs(rise_numeric - rise_exact) / rise_exact
    assert np.all(relative < 0.02), f"relative errors {relative}"


def test_adiabatic_conductor_conserves_enthalpy_exactly():
    conductor = Conduction1D(length_m=0.05, cells=101, t_initial=-4.0,
                             grading=1.05, **ICE)
    before = conductor.enthalpy_above_initial()
    conductor.run(30.0, 0.05)
    assert abs(conductor.enthalpy_above_initial() - before) < 1e-8


def test_dirichlet_conductor_reaches_linear_steady_state():
    conductor = Conduction1D(length_m=0.01, cells=201, k_w_m_k=2.2,
                             rho_kg_m3=917.0, cp_j_kg_k=2100.0, t_initial=0.0,
                             theta=1.0)
    # far end sealed -> steady state of a sealed slab with a fixed surface is
    # the uniform temperature, so drive a flux instead and compare to q L / k
    flux = 5.0e3
    conductor.run(4000.0, 2.0, q_surface=flux, t_far=0.0)
    expected = flux * conductor.length_m / conductor.k_w_m_k
    assert abs(conductor.t[0] - expected) / expected < 0.02
    assert np.all(np.diff(conductor.t) < 1e-9)  # monotone, no oscillation


@pytest.mark.parametrize("kwargs", [
    dict(theta=1.5), dict(cells=2), dict(grading=-1.0), dict(length_m=-1.0),
    dict(surface_capacity_j_m2_k=-1.0),
])
def test_conductor_rejects_unphysical_inputs(kwargs):
    with pytest.raises((ValueError, TypeError)):
        Conduction1D(**kwargs)


def test_step_rejects_nonpositive_dt():
    conductor = Conduction1D(cells=21)
    with pytest.raises(ValueError):
        conductor.step(0.0)


def test_cmy_contact_conductance_scales_with_pressure_and_roughness():
    base = contact_heat_transfer_coefficient(2.2, 45.0, 5e-5, 0.2, 1e6, 1e7)
    higher = contact_heat_transfer_coefficient(2.2, 45.0, 5e-5, 0.2, 2e6, 1e7)
    smoother = contact_heat_transfer_coefficient(2.2, 45.0, 2.5e-5, 0.2, 1e6, 1e7)
    assert higher > base > 0.0
    assert smoother == pytest.approx(2.0 * base, rel=1e-12)   # h_c ~ 1/sigma
    assert higher / base == pytest.approx(2.0 ** 0.95, rel=1e-12)


def test_maxwell_constriction_resistance_scales_inverse_with_radius():
    r1 = maxwell_constriction_resistance(2.2, 1e-3)
    r2 = maxwell_constriction_resistance(2.2, 2e-3)
    assert r1 / r2 == pytest.approx(2.0, rel=1e-12)
    assert contact_thermal_resistance(2.2, 45.0, 1e-3, 1e5) > r1


def test_particle_network_conserves_energy_and_reaches_equilibrium():
    particles = [Particle(0.01), Particle(0.01), Particle(0.01)]
    network = ParticleThermalNetwork(particles, [(0, 1), (1, 2)])
    network.reset([-10.0, -5.0, 0.0])
    energy = network.energy()
    # the contact conductance makes the relaxation time ~ R C ~ 1e2 s, so march
    # on that scale rather than guessing a small step
    conductance = float(network.conductances()[0])
    tau = network.capacity[0] / conductance
    for _ in range(400):
        network.step(0.5 * tau)
    assert abs(network.energy() - energy) / abs(energy) < 1e-12
    equilibrium = network.equilibrium_temperature()
    assert abs(network.temperature - equilibrium).max() < 1e-2 * 15.0


def test_particle_network_two_particle_relaxation_matches_analytic():
    particles = [Particle(0.01), Particle(0.01)]
    network = ParticleThermalNetwork(particles, [(0, 1)])
    network.reset([-10.0, 0.0])
    conductance = float(network.conductances()[0])
    capacity = particles[0].capacity_j_k
    tau = capacity / (2.0 * conductance)     # two equal capacities in series
    dt = 0.002 * tau
    time = 0.0
    for _ in range(2000):
        network.step(dt)
        time += dt
    analytic = 5.0 * math.exp(-time / tau)
    numeric = 0.5 * (network.temperature[1] - network.temperature[0])
    assert numeric == pytest.approx(analytic, rel=5e-3)

@pytest.mark.parametrize("pressure_ratio", [0., 1e-16, 1e-13])
def test_unloaded_and_weakly_loaded_particle_contact(pressure_ratio, record_property):
    # The two conductivities (2 and 6) give harmonic mean 3. Unit roughness,
    # slope and patch radius isolate the pressure dependence of the film.
    particles = [Particle(2., k, 1., 1.) for k in (2., 6.)]
    network = ParticleThermalNetwork(particles, [(0, 1)], contact_pressure_pa=pressure_ratio,
                                     hardness_pa=1., rms_roughness_m=1., rms_slope=1.,
                                     patch_radius_fraction=.5)
    film = 3.75*pressure_ratio**.95
    # Each bulk constriction is in series with the film of area pi.
    exact_g = math.pi*film / (1 + math.pi*film*(1/8 + 1/24))
    dt = 1e15
    capacity = 32*math.pi/3
    network.reset([1., -1.])
    exact_hot = 1/(1 + 2*dt*exact_g/capacity)
    temperatures = network.step(dt)
    error = float(np.max(abs(temperatures-[exact_hot, -exact_hot])))
    record_property("max_temperature_error_K", error)
    record_property("conductance_error_W_per_K", abs(float(network.conductances()[0])-exact_g))
    assert error < 2e-14
    assert network.conductances()[0] == pytest.approx(exact_g, rel=2e-15, abs=0.)


@pytest.mark.parametrize("exponents", [(-700, -600), (500, 600), (1023, -300)])
def test_particle_film_conductance_across_material_scales(exponents, record_property):
    from fractions import Fraction

    k1, k2 = [Fraction(2)**e for e in exponents]
    # Unit pressure ratio, slope, and roughness. Add the two half-material
    # resistivities in exact arithmetic before converting to binary64.
    film = float(Fraction(5, 4) / ((1/k1 + 1/k2)/2))
    # Unit patch radius, so the film area is pi and bulk spreading R=1/(4k).
    expected = 1/(float((1/k1 + 1/k2)/4) + (1/film)/math.pi)
    for left, right in ((float(k1), float(k2)), (float(k2), float(k1))):
        actual_film = contact_heat_transfer_coefficient(left, right, 1., 1., 1., 1.)
        assert actual_film == pytest.approx(film, rel=2e-15, abs=0.)
        pair = ParticleThermalNetwork([Particle(2., left), Particle(2., right)], [(0, 1)],
                                      contact_pressure_pa=1., hardness_pa=1., rms_roughness_m=1.,
                                      rms_slope=1., patch_radius_fraction=.5)
        actual = float(pair.conductances()[0])
        error = abs(actual-expected)/expected
        assert error < 2e-15
        record_property("relative_conductance_error_" + ("ascending" if left < right else "descending"), error)


@pytest.mark.parametrize("step_ratio", [1e8, 1e20])
@pytest.mark.parametrize("with_ambient", [False, True])
def test_stiff_disconnected_contact_pairs(step_ratio, with_ambient, record_property):
    # Two disconnected pairs, each with unequal capacities and its own
    # equilibrium temperature. Uniform h/C within each pair separates the
    # capacity-weighted mean and the temperature difference into two modes.
    particles = [Particle(1., rho_kg_m3=rho, cp_j_kg_k=1.) for rho in (1., 2., 3., 4.)]
    network = ParticleThermalNetwork(particles, [(0, 1), (2, 3)])
    initial = np.array([3., 9., -2., 2.])
    network.reset(initial)
    dt = step_ratio*network.capacity[0]/network.conductances()[0]
    ambient = np.array([1., 1., -3., -3.])
    coupling = .25 if with_ambient else 0.
    expected = []
    for i, g in zip((0, 2), network.conductances()):
        c1, c2 = network.capacity[i:i+2]
        t1, t2 = initial[i:i+2]
        mean = ((c1*t1+c2*t2)/(c1+c2) + coupling*ambient[i])/(1+coupling)
        jump = (t1-t2)/(1+coupling+dt*g*(1/c1+1/c2))
        expected.extend([mean+c2/(c1+c2)*jump, mean-c1/(c1+c2)*jump])
    options = dict(ambient=ambient, h_ambient_w_k=coupling*network.capacity/dt) if with_ambient else {}
    actual = network.step(dt, **options)
    error = float(np.max(abs(actual-expected)))
    record_property("max_temperature_error_K", error)
    assert error < 5e-14


@pytest.mark.parametrize("step_ratio", [1., 1e20])
def test_star_network_eliminated_junction(step_ratio, record_property):
    # The junction is eliminated first. Its three leaves share one symmetric
    # mode and two difference modes, with rates 4G/C and G/C respectively.
    star = ParticleThermalNetwork([Particle(1.) for _ in range(4)], [(0, i) for i in (1, 2, 3)])
    initial = np.array([9., 1., 2., 3.])
    star.reset(initial)
    dt = step_ratio*star.capacity[0]/star.conductances()[0]
    jump = (initial[0]-initial[1:].mean())/(1+4*step_ratio)
    expected = initial.mean() + np.r_[.75*jump, -.25*jump+(initial[1:]-2.)/(1+step_ratio)]
    error = float(np.max(abs(star.step(dt)-expected)))
    record_property("max_temperature_error_K", error)
    assert error < 5e-14


@pytest.mark.parametrize("step_ratio", [1e-8, 1., 1e20])
def test_signed_contact_heat_in_parallel_triangle(step_ratio, record_property):
    # Each side has two equal contacts listed in opposite directions. All
    # nonuniform modes have decay rate 6G/C. The ambient link adds h*dt/C.
    contacts = [(0, 1), (1, 0), (2, 1), (1, 2), (2, 0), (0, 2)]
    net = ParticleThermalNetwork([Particle(1., rho_kg_m3=1., cp_j_kg_k=1.) for _ in range(3)], contacts)
    initial = np.array([3., 8., -2.])
    ambient = np.array([1., -1., 4.])
    net.reset(initial)
    capacity = net.capacity[0]
    dt = step_ratio*capacity/net.conductances()[0]
    coupling = .25
    field, heat = net.step(dt, ambient=ambient, h_ambient_w_k=coupling*net.capacity/dt,
                           return_contact_heat=True)
    forcing = initial + coupling*ambient
    expected = np.array([forcing[i]-forcing[j] for i, j in contacts])*capacity*step_ratio/(1+coupling+6*step_ratio)
    error = float(np.max(abs(heat-expected))/max(abs(expected)))
    into_nodes = np.zeros(3)
    for (i, j), amount in zip(contacts, heat):
        into_nodes[i] -= amount
        into_nodes[j] += amount
    balance = max(abs(capacity*(field-initial)-coupling*capacity*(ambient-field)-into_nodes))
    record_property("relative_heat_error", error)
    record_property("max_particle_energy_residual_J", float(balance))
    assert error < 2e-14
    assert balance < 5e-14


def test_insulating_and_absent_contacts_report_no_heat():
    for contacts in ([], [(0, 1)]):
        net = ParticleThermalNetwork([Particle(1.), Particle(1.)], contacts, contact_pressure_pa=0.)
        net.reset([1., -2.])
        _, heat = net.step(1., return_contact_heat=True)
        np.testing.assert_array_equal(heat, np.zeros(len(contacts)))


def test_contact_heat_resolves_jumps_on_separate_temperature_datums():
    net = ParticleThermalNetwork([Particle(1.) for _ in range(4)], [(0, 1), (2, 3)])
    datums = np.array([300., 2.**40])
    jumps = np.spacing(datums)
    net.reset(np.column_stack((datums+jumps, datums)).ravel())
    ratio = 1e20
    _, heat = net.step(ratio*net.capacity[0]/net.conductances()[0], return_contact_heat=True)
    np.testing.assert_allclose(heat, net.capacity[::2]*jumps*ratio/(1+2*ratio), rtol=2e-15, atol=0.)
