"""Verification surface for Glen flow law / Norton power-law ice creep.

Covers the checks of mission IN13-W3-N02:

(a) Glen flow law tensor form: incompressibility, uniaxial consistency with
    Norton, rotation invariance/coaxiality, zero rate under hydrostatic stress.
(b) Norton power law ``edot = A sigma^n``: exponent recovery for n = 1, 3, 5.
(c) Constant-stress uniaxial creep with the three stages (primary -> steady ->
    tertiary) from the Kelvin + Norton + Rabotnov-damage model, checked against
    the closed-form solution and against Norton's steady rate.
(d) Constant-penetration contact force relaxation: the analytic single-bond
    relaxation reproduced by a DEM bond, and the bonded-sphere plate test fitted
    for its power-law exponent.
(e) Arrhenius temperature dependence ``A(T) = A0 exp(-Q/(RT))``.
"""
import math

import numpy as np
import pytest

from tensordem.ice_creep import (
    BondedSphereCreepLattice,
    GlenCreepLaw,
    LatticeCreepParameters,
    UniaxialCreepModel,
    arrhenius_fluidity_ratio,
    equivalent_strain_rate,
    glen_strain_rate,
    identify_creep_stages,
    norton_uniaxial_rate,
    power_law_fit,
    punch_relaxation,
    single_bond_relaxation_analytic,
    uniaxial_constant_stress,
    von_mises_stress,
)

R_GAS = 8.314462618
T0 = 273.15


# --------------------------------------------------------------------------- #
# (a) Glen flow law tensor form
# --------------------------------------------------------------------------- #
def test_glen_flow_law_is_incompressible_uniaxial_consistent():
    law = GlenCreepLaw()
    a = law.fluidity(-10.0)
    sigma = 1.0e6
    rate = glen_strain_rate(np.diag([sigma, 0.0, 0.0]), a, law.n)

    # trace-free: creep of ice does not change volume
    assert abs(np.trace(rate)) <= 1e-12 * abs(rate[0, 0])
    # uniaxial axial rate equals the Norton law exactly
    assert rate[0, 0] == pytest.approx(norton_uniaxial_rate(sigma, a, law.n),
                                       rel=1e-14)
    # lateral contraction is exactly -1/2 of the axial rate (n-independent)
    assert rate[1, 1] / rate[0, 0] == pytest.approx(-0.5, abs=1e-14)
    assert rate[2, 2] / rate[0, 0] == pytest.approx(-0.5, abs=1e-14)
    # equivalent strain rate equals A sigma_eq^n
    assert equivalent_strain_rate(rate) == pytest.approx(
        a * sigma ** law.n, rel=1e-13)


def test_glen_glaciological_normalisation_differs_by_two_thirds():
    """The literal mission expression carries the 2/3 normalisation factor."""
    law = GlenCreepLaw()
    a = law.fluidity(-10.0)
    sigma = np.diag([2.0e6, 0.0, 0.0])
    norton = glen_strain_rate(sigma, a, law.n, normalisation="norton")
    glacio = glen_strain_rate(sigma, a, law.n, normalisation="glaciological")
    assert glacio[0, 0] / norton[0, 0] == pytest.approx(2.0 / 3.0, rel=1e-14)


def test_glen_hydrostatic_stress_produces_no_creep():
    law = GlenCreepLaw()
    a = law.fluidity(-10.0)
    p = 5.0e6
    assert np.allclose(glen_strain_rate(-p * np.eye(3), a, law.n), 0.0, atol=0.0)
    assert von_mises_stress(-p * np.eye(3)) == pytest.approx(0.0, abs=1e-9)


def test_glen_is_objective_and_coaxial():
    law = GlenCreepLaw()
    a = law.fluidity(-5.0)
    rng = np.random.default_rng(20261109)
    m = rng.normal(size=(3, 3))
    sigma = 0.5 * (m + m.T) * 1.0e6
    rate = glen_strain_rate(sigma, a, law.n)
    dev = sigma - np.trace(sigma) / 3.0 * np.eye(3)
    # coaxial: the rate tensor is a positive multiple of the deviator
    ratio = rate[dev != 0.0] / dev[dev != 0.0]
    assert np.allclose(ratio, ratio[0], rtol=1e-12)
    # objective: rate(R sigma R^T) = R rate R^T
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotated = glen_strain_rate(q @ sigma @ q.T, a, law.n)
    assert np.allclose(rotated, q @ rate @ q.T, rtol=1e-12, atol=1e-18)
    # pure shear sanity: sigma_eq = sqrt(3) tau
    tau = np.zeros((3, 3))
    tau[0, 1] = tau[1, 0] = 1.0e5
    assert von_mises_stress(tau) == pytest.approx(math.sqrt(3.0) * 1.0e5,
                                                 rel=1e-14)


# --------------------------------------------------------------------------- #
# (b) Norton power law
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1.0, 3.0, 5.0])
def test_norton_exponent_recovered(n):
    law = GlenCreepLaw(a_ref=1.0e-24, n=n)
    a = law.fluidity(-10.0)
    sigmas = np.array([1.0e5, 3.0e5, 1.0e6, 3.0e6, 1.0e7])
    rates = norton_uniaxial_rate(sigmas, a, n)
    fit = power_law_fit(sigmas, rates)
    assert fit["exponent"] == pytest.approx(n, rel=1e-12)
    assert fit["prefactor"] == pytest.approx(a, rel=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-20)


def test_norton_sign_symmetry():
    law = GlenCreepLaw()
    a = law.fluidity(-10.0)
    assert norton_uniaxial_rate(-2.0e6, a, law.n) == pytest.approx(
        -norton_uniaxial_rate(2.0e6, a, law.n), rel=1e-15)


# --------------------------------------------------------------------------- #
# (c) three-stage uniaxial creep
# --------------------------------------------------------------------------- #
def test_three_stage_creep_matches_closed_form_and_norton():
    model = UniaxialCreepModel(kelvin_time_s=100.0, damage_exponent=4.0,
                               damage_time_s=5.0e5)
    sigma0 = 5.0e5
    t_r = model.time_to_rupture(sigma0)
    assert t_r == pytest.approx(model.damage_time_s / 5.0, rel=1e-12)

    t_end = 0.9 * t_r
    num = model.integrate_rk4(sigma0, t_end, n_steps=4000)
    # integrate_rk4 tracks the *inelastic* strain; the closed form adds the
    # instantaneous elastic part sigma0/E
    closed = model.strain(num["t"], sigma0)
    integrated = num["strain"] + sigma0 / model.young_modulus
    rel = np.abs(integrated - closed) / np.abs(closed)
    assert rel.max() < 1e-6              # RK4 vs closed form
    assert np.abs(num["damage"] - model.damage(num["t"], sigma0)).max() < 1e-9

    rate = model.strain_rate(num["t"], sigma0)
    stages = identify_creep_stages(num["t"], rate, plateau_tolerance=0.02)
    # the measured minimum rate sits where the decaying primary crosses the
    # growing tertiary, so it brackets A sigma^n from above by ~1 %
    norton_rate = model.law.fluidity(-10.0) * sigma0 ** model.law.n
    assert abs(stages["steady_rate"] / norton_rate - 1.0) < 0.02
    # primary exists: the initial rate exceeds the steady rate
    assert rate[1] > 3.0 * stages["steady_rate"]
    # tertiary exists and the peak rate is far above the steady rate
    assert stages["tertiary_start_time"] is not None
    assert stages["rate_ratio_peak_steady"] > 2.0
    # the steady window must be a genuine plateau, not a single point
    assert stages["steady_window_decades"] > 0.5
    # monotone strain with three regimes
    assert np.all(np.diff(num["strain"]) > 0.0)


# --------------------------------------------------------------------------- #
# (e) Arrhenius temperature dependence
# --------------------------------------------------------------------------- #
def test_arrhenius_matches_closed_form_and_recovers_q():
    q = 61500.0
    q_over_r = q / R_GAS
    law = GlenCreepLaw(q_over_r=q_over_r, t_ref_c=-10.0)
    temps_c = np.array([-2.0, -5.0, -10.0, -15.0, -20.0])
    a = np.array([law.fluidity(t) for t in temps_c])
    for t, ai in zip(temps_c, a):
        expected = law.a_ref * math.exp(-q_over_r * (1.0 / (t + T0)
                                                     - 1.0 / (-10.0 + T0)))
        assert ai == pytest.approx(expected, rel=1e-14)
    ratio = a[0] / a[-1]
    assert ratio == pytest.approx(
        arrhenius_fluidity_ratio(q_over_r, -2.0 + T0, -20.0 + T0), rel=1e-14)
    # fluidity decreases monotonically with cooling
    assert np.all(np.diff(a) < 0.0)
    # recover Q from a linear fit of ln A against 1/T
    inv_t = 1.0 / (temps_c + T0)
    coef = np.polyfit(inv_t, np.log(a), 1)
    assert -coef[0] * R_GAS == pytest.approx(q, rel=1e-9)
    # the fluidity ratio over 18 K is large
    assert ratio > 3.0


# --------------------------------------------------------------------------- #
# (d) contact force relaxation
# --------------------------------------------------------------------------- #
def _two_node_bond(n, *, a_ref, sigma0, spacing=0.02):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, spacing]])
    bonds = np.array([[0, 1]])
    params = LatticeCreepParameters(law=GlenCreepLaw(a_ref=a_ref, n=n),
                                    young_modulus=9.0e9)
    lat = BondedSphereCreepLattice(positions, bonds, radius=spacing / 2.0,
                                   params=params)
    area = math.pi * (spacing / 2.0) ** 2
    k = params.young_modulus * area / spacing
    for dof in (0, 1, 2, 3, 4, 5):
        lat.fixed[dof] = True
    lat.fixed_value[5] = -sigma0 * area / k      # fixed elongation
    return lat, area, k


@pytest.mark.parametrize("n,a_ref,sigma0,rel_tol",
                         [(1.0, 1.0e-14, 1.0e6, 5e-3),
                          (3.0, 1.0e-24, 7.5e4, 5e-3)])
def test_single_bond_relaxation_matches_analytic(n, a_ref, sigma0, rel_tol):
    # n = 1 is the linear dashpot: a visible relaxation needs a much larger
    # fluidity than the physical n = 3 value.  sigma0 for the n = 3 case keeps
    # the explicit step well inside the relaxation time tau.
    lat, area, k = _two_node_bond(n, a_ref=a_ref, sigma0=sigma0)
    lat.solve()
    f0 = abs(lat.bond_axial_force()[0])
    assert f0 == pytest.approx(sigma0 * area, rel=1e-5)   # penalty BC accuracy

    dt, steps = 20.0, 1000
    times, forces, ratio = [0.0], [f0], [lat.creep_step_ratio(dt)]
    for _ in range(steps):
        lat.step(dt)
        lat.solve()
        times.append(lat.time)
        forces.append(abs(lat.bond_axial_force()[0]))
        ratio.append(lat.creep_step_ratio(dt))
    assert max(ratio) < 0.5              # explicit step resolves the relaxation
    analytic = single_bond_relaxation_analytic(
        f0, k, area, lat.rest[0], lat.params.fluidity(), n, np.array(times))
    rel = np.abs(np.array(forces) - analytic) / analytic
    assert rel.max() < rel_tol            # explicit Euler with dt/tau << 1
    assert forces[-1] < 0.6 * f0          # the force genuinely relaxes


def test_punch_relaxation_is_a_decaying_power_law():
    params = LatticeCreepParameters(law=GlenCreepLaw(a_ref=1.0e-26, n=3.0))
    out = punch_relaxation(n_cells=4, spacing=0.02, params=params,
                           depth_fraction=1.0e-3, t_end=2.0e3, dt=2.0)
    t, force = out["t"], out["force"]
    assert out["n_bonds"] > 0
    assert out["creep_ratio"].max() < 0.5      # explicit step is resolved
    assert force[0] == pytest.approx(out["initial_force"], rel=1e-3)
    assert np.all(np.diff(force) <= 1e-6 * force[0])      # monotone relaxation
    assert force[-1] < 0.9 * force[0]                     # real relaxation
    fit = power_law_fit(t[1:], force[1:], x_min=t[-1] / 20.0)
    assert fit["r2"] > 0.9
    assert -2.0 < fit["exponent"] < -0.05
    # -1/(n+1) is one of the candidate exponents; record the distance
    assert abs(fit["exponent"] + 1.0 / (params.law.n + 1.0)) < 1.0


# --------------------------------------------------------------------------- #
# bonded-sphere uniaxial creep reproduces the Norton exponent
# --------------------------------------------------------------------------- #
def test_lattice_uniaxial_creep_exponent_matches_norton():
    """Constant-stress DEM creep: early-window rate scales as sigma^n.

    The window is kept short (10..40 s) on purpose: after that, load
    redistribution between axial and diagonal bonds slowly reduces the rate, so
    a long window biases the fitted exponent low (see docs, section 5).
    """
    n = 3.0
    params = LatticeCreepParameters(law=GlenCreepLaw(a_ref=1.0e-24, n=n),
                                    young_modulus=9.0e9)
    rates, sigmas = [], []
    for sigma0 in (5.0e5, 1.0e6, 2.0e6):
        out = uniaxial_constant_stress(sigma0=sigma0, n_cells=3, spacing=0.02,
                                       params=params, t_end=200.0, dt=10.0)
        assert out["creep_ratio"].max() < 0.5
        t, strain = out["t"], out["strain"]
        mask = (t >= 10.0) & (t <= 40.0)
        rates.append(float(np.polyfit(t[mask], strain[mask], 1)[0]))
        sigmas.append(sigma0)
    fit = power_law_fit(np.array(sigmas), np.array(rates))
    assert fit["r2"] > 0.999
    assert fit["exponent"] == pytest.approx(n, rel=0.05)
