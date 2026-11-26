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


def test_particle_power_forces_contact_heat_reversal(record_property):
    net = ParticleThermalNetwork([Particle(1., rho_kg_m3=r, cp_j_kg_k=1.) for r in (1., 2.)], [(0, 1)])
    c1, c2 = net.capacity
    g = net.conductances()[0]
    tau = 1/(g*(1/c1+1/c2))
    initial = np.array([-1., 0.])
    # Applied heating and cooling give steady jump 2 K. From -1 K the
    # exact jump is 2-3 exp(-t/tau); its integral times G is contact heat.
    powers = np.array([c1, -c2])/tau
    end = 2*tau
    jump = 2-3*np.exp(-2.)
    mean = (net.capacity@initial + powers.sum()*end)/(c1+c2)
    exact = mean + np.array([c2, -c1])*jump/(c1+c2)
    exact_heat = g*(2*end + 3*tau*np.expm1(-2.))
    previous = None
    for steps in (16, 32, 64, 128):
        net.reset(initial)
        dt = end/steps
        amounts = []
        balance = 0.
        for _ in range(steps):
            before = net.temperature.copy()
            field, heat = net.step(dt, particle_power_w=powers, return_contact_heat=True)
            amounts.append(heat[0])
            residual = net.capacity*(field-before) - powers*dt + np.array([heat[0], -heat[0]])
            balance = max(balance, max(abs(residual)))
        assert amounts[0] < 0 < amounts[-1]
        errors = np.array([max(abs(field-exact)), abs(sum(amounts)-exact_heat)])
        record_property(f"temperature_error_{steps}_K", float(errors[0]))
        record_property(f"contact_heat_error_{steps}_J", float(errors[1]))
        record_property(f"max_particle_residual_{steps}_J", float(balance))
        assert balance < 2e-14
        if previous is not None:
            assert np.all(previous/errors > 1.9)
        previous = errors
    assert errors[0] < .005 and errors[1] < .02


def test_applied_power_heats_isolated_particle_and_rejects_invalid_loads():
    net = ParticleThermalNetwork([Particle(1.)], [])
    net.reset([2.])
    assert net.step(3., particle_power_w=[net.capacity[0]])[0] == pytest.approx(5.)
    for invalid in ([], [1., 2.], [np.inf]):
        with pytest.raises(ValueError, match="particle_power_w"):
            net.step(1., particle_power_w=invalid)
    assert net.temperature[0] == pytest.approx(5.)


@pytest.mark.parametrize("film_ratio", [1e-8, 1., 1e20])
def test_per_particle_ambient_heat_against_two_modes(film_ratio, record_property):
    # Equal particles with a common ambient. Both modes exchange heat with it;
    # the antisymmetric mode also decays through the contact at rate 2G/C.
    net = ParticleThermalNetwork([Particle(1., rho_kg_m3=1., cp_j_kg_k=1.) for _ in range(2)], [(0, 1)])
    cap = net.capacity[0]
    dt = cap/net.conductances()[0]
    ambient = np.full(2, 300.)
    net.reset([301., 298.])
    power = cap/dt*np.array([.5, -.25])
    forcing = net.temperature-ambient + dt*power/cap
    mean = forcing.mean()/(1+film_ratio)
    contrast = (forcing-forcing.mean())/(3+film_ratio)
    expected = -cap*film_ratio*(mean+contrast)
    initial = net.temperature.copy()
    field, contact, exchange = net.step(dt, ambient=ambient, h_ambient_w_k=np.full(2, film_ratio*cap/dt),
                                        particle_power_w=power, return_contact_heat=True, return_ambient_heat=True)
    error = float(max(abs((exchange-expected)/expected)))
    balance = float(max(abs(cap*(field-initial)-dt*power-exchange+contact[0]*np.array([1., -1.]))))
    record_property("max_relative_ambient_heat_error", error)
    record_property("max_particle_residual_J", balance)
    assert error < 3e-14
    assert balance < 3e-13


def test_ambient_heat_only_for_disconnected_particles():
    net = ParticleThermalNetwork([Particle(1.), Particle(1.)], [])
    net.reset([3., 7.])
    field, heat = net.step(1., return_ambient_heat=True)
    np.testing.assert_array_equal(field, [3., 7.])
    np.testing.assert_array_equal(heat, [0., 0.])
    _, heat = net.step(1., ambient=[4., 1.], h_ambient_w_k=[net.capacity[0], 0.], return_ambient_heat=True)
    np.testing.assert_allclose(heat, [.5*net.capacity[0], 0.])


def test_two_particle_contact_entropy_converges_to_isolated_entropy_gain(record_property):
    net = ParticleThermalNetwork([Particle(1., rho_kg_m3=r, cp_j_kg_k=1.) for r in (1.,2.)], [(0,1)])
    start = np.array([360., 270.])
    c1,c2 = net.capacity
    tau = 1/(net.conductances()[0]*(1/c1+1/c2))
    jump = (start[0]-start[1])*np.exp(-2.)
    mean = net.capacity@start/(c1+c2)
    exact = mean+np.array([c2,-c1])*jump/(c1+c2)
    exact_heat = c1*(start[0]-exact[0])
    exact_entropy = net.capacity@np.log(exact/start)
    previous = None
    for steps in (16,32,64,128):
        net.reset(start)
        heat = entropy = 0.
        for _ in range(steps):
            _, q, s = net.step(2*tau/steps, return_contact_heat=True, return_contact_entropy=True)
            heat += q[0]
            entropy += s[0]
            assert s[0] >= 0.
        errors = np.array([abs(heat-exact_heat), abs(entropy-exact_entropy)])
        record_property(f"heat_error_{steps}_J", errors[0])
        record_property(f"entropy_error_{steps}_J_per_K", errors[1])
        if previous is not None:
            assert np.all(previous/errors > 1.9)
        previous = errors


def test_stiff_contact_entropy_resolves_rounded_endpoint_jump(record_property):
    net = ParticleThermalNetwork([Particle(1.), Particle(1.)], [(1,0)])
    net.reset([300., 301.])
    ratio = 1.e20
    dt = ratio*net.capacity[0]/net.conductances()[0]
    field, entropy = net.step(dt, return_contact_entropy=True)
    assert field[0] == field[1]
    exact_jump = 1/(1+2*ratio)
    exact = (net.capacity[0]*ratio*exact_jump)*exact_jump/(300.5**2)
    error = abs(entropy[0]/exact-1)
    record_property("relative_entropy_error", error)
    assert error < 2e-15


def test_contact_entropy_requires_positive_absolute_temperature():
    net = ParticleThermalNetwork([Particle(1.)], [])
    net.reset([1.])
    with pytest.raises(ValueError, match="kelvin"):
        net.step(1., particle_power_w=[-2*net.capacity[0]], return_contact_entropy=True)
    np.testing.assert_array_equal(net.temperature, [1.])
    _, entropy = net.step(1., return_contact_entropy=True)
    assert entropy.shape == (0,)


@pytest.mark.parametrize("start", [270., 360.])
def test_reservoir_entropy_against_exponential_relaxation(start, record_property):
    net = ParticleThermalNetwork([Particle(1., rho_kg_m3=1., cp_j_kg_k=1.)], [])
    capacity = net.capacity[0]
    bath, h = 300., 2.5
    end = bath+(start-bath)*np.exp(-2.)
    exact = capacity*(np.log(end/start)-(end-start)/bath)
    previous = None
    for steps in (16,32,64,128):
        net.reset([start])
        total = 0.
        for _ in range(steps):
            _, entropy = net.step(2*capacity/(h*steps), ambient=[bath], h_ambient_w_k=[h],
                                  return_ambient_entropy=True)
            assert entropy[0] >= 0.
            total += entropy[0]
        error = abs(total-exact)
        record_property(f"entropy_error_{steps}_J_per_K", float(error))
        if previous is not None:
            assert previous/error > 1.9
        previous = error


def test_stiff_and_insulated_reservoir_entropy(record_property):
    net = ParticleThermalNetwork([Particle(1.), Particle(1.)], [])
    net.reset([302., 280.])
    capacity = net.capacity[0]
    ratio = 1e20
    field, contact, heat, contact_entropy, entropy = net.step(
        1., ambient=[300., 310.], h_ambient_w_k=[capacity*ratio, 0.],
        return_contact_heat=True, return_ambient_heat=True,
        return_contact_entropy=True, return_ambient_entropy=True)
    assert field[0] == 300. and heat[0] < 0
    assert len(contact) == len(contact_entropy) == 0
    assert entropy[1] == heat[1] == 0.
    gap = 2/(1+ratio)
    exact = capacity*(2-gap)*gap/(300.*(300.+gap))
    error = abs(entropy[0]/exact-1)
    record_property("stiff_relative_entropy_error", float(error))
    assert error < 2e-15
    for h,bath in ((-1.,300.), (1.,0.)):
        with pytest.raises(ValueError, match="ambient entropy"):
            net.step(1., ambient=[bath,bath], h_ambient_w_k=[h,h], return_ambient_entropy=True)
    np.testing.assert_array_equal(net.temperature, field)


def test_contact_and_two_reservoir_entropy_partition(record_property):
    from scipy.integrate import quad_vec

    particles = [Particle(1., rho_kg_m3=1., cp_j_kg_k=1.) for _ in range(2)]
    net = ParticleThermalNetwork(particles, [(0,1)])
    c, g = net.capacity[0], net.conductances()[0]
    start = np.array([280., 345.])
    baths = np.array([355., 275.])
    # With both reservoir conductances equal to the contact conductance G,
    # the mean and half-difference decay with rates G/C and 3G/C. The
    # hotter reservoir acts on the initially colder particle, reversing flow.
    signs = np.array([1., -1.])
    steady = baths.mean() + signs*(baths[0]-baths[1])/6
    common = (start-steady).mean()
    differential = start-steady-common
    def exact(s):
        return steady + common*np.exp(-s) + differential*np.exp(-3*s)

    def entropy_density(s):
        t = exact(s)
        # Dimensionless time s=t_physical*G/C. Rates below include dt/ds.
        return c*np.r_[(t[0]-t[1])**2/(t[0]*t[1]), (baths-t)**2/(baths*t)]

    end = 2.
    reference, _ = quad_vec(entropy_density, 0., end, epsabs=1e-14, epsrel=1e-13)
    ambient_heat = c*((baths-steady)*end + common*np.expm1(-end)
                      + differential*np.expm1(-3*end)/3)
    total = c*np.log(exact(end)/start).sum() - (ambient_heat/baths).sum()
    assert abs(reference.sum()-total) < 2e-14

    previous = None
    for steps in (16,32,64,128):
        net.reset(start)
        amounts = np.zeros(3)
        reservoir_heat = np.zeros(2)
        time_defect = 0.
        first_contact = None
        for _ in range(steps):
            before = net.temperature.copy()
            field, heat, exchange, contact_s, ambient_s = net.step(
                end*c/(steps*g), ambient=baths, h_ambient_w_k=[g,g],
                return_contact_heat=True, return_ambient_heat=True,
                return_contact_entropy=True, return_ambient_entropy=True)
            if first_contact is None:
                first_contact = heat[0]
            amounts += np.r_[contact_s, ambient_s]
            reservoir_heat += exchange
            # Endpoint quadrature uses C*dT/T_end for each particle.
            delta = field-before
            time_defect += c*np.sum(np.log1p(delta/before)-delta/field)
        assert first_contact < 0 < heat[0]
        state_entropy = c*np.log(field/start).sum() - (reservoir_heat/baths).sum()
        residual = abs(state_entropy-amounts.sum()-time_defect)
        errors = abs(amounts-reference)
        for name,value in zip(("contact", "hot_bath", "cold_bath"), errors):
            record_property(f"{name}_error_{steps}_J_per_K", float(value))
        record_property(f"entropy_balance_{steps}_J_per_K", float(residual))
        assert residual < 3e-14
        if previous is not None:
            assert np.all(previous/errors > 1.8)
        previous = errors
