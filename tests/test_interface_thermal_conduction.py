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