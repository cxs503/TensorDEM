"""Verification surface for :mod:`tensordem.creep_constitutive`.

Campaign node IN69-N01 (ice creep constitutive + long-time contact relaxation).
Every claim the module makes is pinned here against a reference written
independently of it (closed forms derived in the module docstring and in
``docs/IN69_N01_creep_constitutive.md``):

* V1  equivalent measures      -- ``sigma_eq`` / ``eps_eq`` conventions
* V2  uniaxial reduction       -- ``eps_dot = A sigma**n`` exactly
* V3  convention control       -- the literal Glen form is off by 2/3 (negative)
* V4  work conjugacy           -- ``sigma:eps_dot_c = sigma_eq eps_dot_eq``
* V5  linear limit             -- ``n = 1`` gives the closed-form Newtonian update
* V6  radial-return consistency-- ``sigma_eq + 3 G dt A sigma_eq**n = sigma_tr``
* V7  relaxation, fixed total strain -- modulus ``3G``
* V8  relaxation, free lateral -- modulus ``E`` (and the 3G/E gap is *not* a bug)
* V9  exponential limit        -- ``n = 1`` relaxation
* V10 energy ledger            -- ``dU + int sigma:eps_dot_c dt = 0`` (fixed displacement)
* V11 creep work               -- ``int sigma:eps_dot_c = A sigma**(n+1) t``
* V12 primary stage            -- ``eps_p ~ t**(1/3)`` (emergent, fitted)
* V13 secondary stage          -- rate error ``< 3%``
* V14 tertiary stage           -- acceleration + Kachanov rupture time
* V15 three-stage closed form  -- per-stage sum vs the numeric history
* V16 pure Norton              -- no primary stage (negative control on V12)
* V17 Hertz contact relaxation -- ``F(t)/F0`` vs closed form ``< 5%`` (crit. c)
* V18 timestep convergence     -- first order in ``dt`` on the contact level
* V19 contact energy ledger    -- ``int F d(delta_c) = -dU``
* V20 contact ``n = 1`` limit  -- exponential overlap decay
* V21 input validation / V22 zero stress -- no creep without stress
"""
from __future__ import annotations

import math

import pytest
import torch

from tensordem.creep_constitutive import (
    GlenCreepParameters,
    HertzCreepContact,
    ThreeStageCreepParameters,
    creep_equivalent_strain_rate,
    creep_strain_rate,
    creep_stress_power,
    deviator,
    equivalent_strain,
    hertz_creep_relaxation,
    integrate_three_stage_creep,
    norton_relaxation,
    norton_strain,
    norton_strain_rate,
    radial_return,
    relaxation_histories,
    shear_modulus_from_young,
    three_stage_analytic,
    uniaxial_stress,
    von_mises,
)

ICE_A = 3.5e-7          # MPa^-3 s^-1, Glen at about -10 degC
ICE_N = 3.0
ICE_E = 9000.0          # MPa
ICE_NU = 0.3


def _glen(a: float = ICE_A, n: float = ICE_N) -> GlenCreepParameters:
    return GlenCreepParameters(A=a, n=n)


# ---------------------------------------------------------------------------
# V1 / V2 / V3 / V4 -- flow law and its conventions
# ---------------------------------------------------------------------------
def test_von_mises_and_equivalent_strain_conventions():
    assert von_mises(uniaxial_stress(2.5)).item() == pytest.approx(2.5, rel=1e-15)
    shear = torch.zeros(3, 3, dtype=torch.float64)
    shear[0, 1] = shear[1, 0] = 1.0
    assert von_mises(shear).item() == pytest.approx(math.sqrt(3.0), rel=1e-15)
    # incompressible uniaxial strain: eps_eq == axial strain
    e = torch.zeros(3, 3, dtype=torch.float64)
    e[0, 0], e[1, 1], e[2, 2] = 1e-3, -5e-4, -5e-4
    assert equivalent_strain(e).item() == pytest.approx(1e-3, rel=1e-15)
    # pressure is the mean normal stress, and the deviator is traceless
    s, p = deviator(uniaxial_stress(3.0))
    assert p.item() == pytest.approx(1.0, rel=1e-15)
    assert torch.diagonal(s).sum().item() == pytest.approx(0.0, abs=1e-18)


def test_uniaxial_flow_law_reduces_to_norton():
    params = _glen()
    for sigma in (0.1, 0.5, 1.0, 2.0):
        rate = creep_strain_rate(uniaxial_stress(sigma), params)
        expected = params.A * sigma ** params.n
        assert float(rate[0, 0]) == pytest.approx(expected, rel=1e-14)
        # creep is incompressible and the lateral rate is exactly -eps11/2
        assert float(torch.diagonal(rate).sum()) == pytest.approx(0.0, abs=1e-18)
        assert float(rate[1, 1]) == pytest.approx(-0.5 * expected, rel=1e-14)
        assert float(creep_equivalent_strain_rate(uniaxial_stress(sigma), params)) == (
            pytest.approx(expected, rel=1e-14)
        )


def test_glen_literal_normalization_is_two_thirds_negative_control():
    """Documents the convention trap: the literal form misses ``A sigma**n`` by 1/3."""
    literal = GlenCreepParameters(A=ICE_A, n=ICE_N, normalization="glen_literal")
    rate = float(creep_strain_rate(uniaxial_stress(1.0), literal)[0, 0])
    target = ICE_A
    assert rate / target - 1.0 == pytest.approx(-1.0 / 3.0, abs=1e-12)


def test_work_conjugacy_of_stress_power():
    params = _glen(n=2.5)
    states = [
        uniaxial_stress(1.7),
        torch.tensor([[1.0, 0.4, 0.0], [0.4, -0.2, 0.0], [0.0, 0.0, -0.8]], dtype=torch.float64),
        torch.tensor([[0.3, 0.9, 0.2], [0.9, 0.1, -0.4], [0.2, -0.4, -0.4]], dtype=torch.float64),
    ]
    for stress in states:
        seq = float(von_mises(stress))
        rate = creep_strain_rate(stress, params)
        eq = float(equivalent_strain(rate))
        assert eq == pytest.approx(params.A * seq ** params.n, rel=1e-13)
        assert float(creep_stress_power(stress, params)) == pytest.approx(seq * eq, rel=1e-13)


# ---------------------------------------------------------------------------
# V5 / V6 -- implicit radial return
# ---------------------------------------------------------------------------
def test_radial_return_n1_closed_form():
    params = _glen(n=1.0)
    g = shear_modulus_from_young(ICE_E, ICE_NU)
    dt = 10.0
    trial = uniaxial_stress(1.0)
    res = radial_return(trial, dt, params, shear_modulus=g)
    expected = 1.0 / (1.0 + 3.0 * g * dt * params.A)
    assert res["sigma_eq"] == pytest.approx(expected, rel=1e-12)
    assert res["dgamma"] == pytest.approx(dt * params.A * expected, rel=1e-12)
    # the hydrostatic part is untouched (creep is incompressible)
    assert float(torch.diagonal(res["stress"]).sum()) == pytest.approx(
        float(torch.diagonal(trial).sum()), rel=1e-14
    )


def test_radial_return_satisfies_implicit_consistency():
    params = _glen()
    g = shear_modulus_from_young(ICE_E, ICE_NU)
    trial = uniaxial_stress(1.0)
    dt = 5.0
    res = radial_return(trial, dt, params, shear_modulus=g)
    seq = res["sigma_eq"]
    assert seq + 3.0 * g * dt * params.A * seq ** params.n == pytest.approx(1.0, rel=1e-12)
    assert seq < 1.0
    assert float(torch.diagonal(res["d_creep_strain"]).sum()) == pytest.approx(0.0, abs=1e-18)
    # tiny dt -> almost elastic
    elastic = radial_return(trial, 1e-9, params, shear_modulus=g)
    assert elastic["sigma_eq"] == pytest.approx(1.0, rel=1e-8)


# ---------------------------------------------------------------------------
# V7 / V8 / V9 / V10 / V11 -- relaxation and the energy ledger
# ---------------------------------------------------------------------------
def test_relaxation_fixed_total_strain_matches_three_g():
    params = _glen()
    hist = relaxation_histories(
        1.0, 100.0, 0.05, params, young_modulus=ICE_E, constraint="fixed_total_strain"
    )
    g = shear_modulus_from_young(ICE_E, ICE_NU)
    assert hist["relaxation_modulus"] == pytest.approx(3.0 * g, rel=1e-15)
    err = max(
        abs(a - b) / b for a, b in zip(hist["sigma_eq"], hist["sigma_analytic"])
    )
    assert err < 1e-3
    # first-order convergence in dt (truncation, not a model mismatch)
    coarse = relaxation_histories(
        1.0, 100.0, 0.5, params, young_modulus=ICE_E, constraint="fixed_total_strain"
    )
    err_coarse = max(
        abs(a - b) / b for a, b in zip(coarse["sigma_eq"], coarse["sigma_analytic"])
    )
    assert err_coarse > err
    assert err_coarse / err == pytest.approx(10.0, rel=0.5)


def test_relaxation_free_lateral_matches_young_modulus():
    """The two 'constant strain' tests relax at 3G and E respectively -- 15% apart."""
    params = _glen()
    hist = relaxation_histories(
        1.0, 100.0, 0.05, params, young_modulus=ICE_E, constraint="free_lateral"
    )
    assert hist["relaxation_modulus"] == pytest.approx(ICE_E, rel=1e-15)
    err = max(
        abs(a - b) / b for a, b in zip(hist["sigma_eq"], hist["sigma_analytic"])
    )
    assert err < 1e-3
    # the axial stress of the free-lateral test equals the uniaxial closed form
    assert hist["sigma_11"][-1] == pytest.approx(
        norton_relaxation(100.0, 1.0, params, young_modulus=ICE_E), rel=1e-3
    )
    constrained = relaxation_histories(
        1.0, 100.0, 0.05, params, young_modulus=ICE_E, constraint="fixed_total_strain"
    )
    gap = abs(constrained["sigma_eq"][-1] - hist["sigma_eq"][-1]) / hist["sigma_eq"][-1]
    assert gap > 0.02  # documented trap: mixing the moduli is a >2% error


def test_relaxation_n1_is_exponential():
    params = _glen(n=1.0)
    hist = relaxation_histories(
        1.0, 100.0, 0.05, params, young_modulus=ICE_E, constraint="free_lateral"
    )
    for t, sigma in zip(hist["times"], hist["sigma_11"]):
        assert sigma == pytest.approx(math.exp(-ICE_E * params.A * t), rel=1e-2)


def test_relaxation_energy_ledger_closes():
    params = _glen()
    hist = relaxation_histories(
        1.0, 100.0, 0.05, params, young_modulus=ICE_E, constraint="free_lateral"
    )
    du = hist["elastic_energy"][-1] - hist["elastic_energy"][0]
    diss = hist["dissipation"][-1]
    assert du < 0.0
    assert diss > 0.0
    assert abs(du + diss) / abs(du) < 1e-2


def test_constant_stress_creep_work_matches_analytic():
    params = _glen()
    sigma, dt, n_steps = 1.2, 0.5, 200
    stress = uniaxial_stress(sigma)
    total = 0.0
    for _ in range(n_steps):
        rate = creep_strain_rate(stress, params)
        total += float((stress * rate).sum()) * dt
    expected = params.A * sigma ** (params.n + 1.0) * dt * n_steps
    assert total == pytest.approx(expected, rel=1e-13)
    # and the accumulated creep strain is exactly the Norton law
    assert norton_strain(dt * n_steps, sigma, params) == pytest.approx(
        norton_strain_rate(sigma, params) * dt * n_steps, rel=1e-15
    )


# ---------------------------------------------------------------------------
# V12 - V16 -- three-stage creep
# ---------------------------------------------------------------------------
def _slow_damage_params() -> ThreeStageCreepParameters:
    """Parameters whose three stages are separated in time (t_r = 3e7 s)."""
    return ThreeStageCreepParameters(
        glen=_glen(),
        primary_A=1e-6,
        primary_p=2.0 / 3.0,
        damage_B=1.0 / 3e7,
        damage_m=2.0,
        tertiary_q=3.0,
    )


def test_three_stage_primary_exponent_is_one_third():
    params = _slow_damage_params()
    hist = integrate_three_stage_creep(1.0, 1000.0, 1.0, params)
    ts, prim = hist["times"], hist["primary"]
    lo, hi = 1.0, 100.0
    xs = [math.log(t) for t in ts if lo <= t <= hi]
    ys = [math.log(v) for t, v in zip(ts, prim) if lo <= t <= hi]
    n = len(xs)
    slope = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / (
        n * sum(x * x for x in xs) - sum(xs) ** 2
    )
    assert slope == pytest.approx(1.0 / 3.0, rel=1e-2)


def test_three_stage_secondary_rate_within_three_percent():
    params = _slow_damage_params()
    hist = integrate_three_stage_creep(1.0, 1e5, 1.0, params)
    window = [(t, r) for t, r in zip(hist["times"], hist["rate"]) if t > 3e4]
    mean_rate = sum(r for _, r in window) / len(window)
    target = norton_strain_rate(1.0, params.glen)
    assert abs(mean_rate / target - 1.0) < 0.03


def test_three_stage_tertiary_accelerates_and_ruptures():
    params = ThreeStageCreepParameters(
        glen=_glen(),
        primary_A=1e-6,
        primary_p=2.0 / 3.0,
        damage_B=1e-3,
        damage_m=2.0,
        tertiary_q=3.0,
    )
    tr = params.rupture_time(1.0)
    assert tr == pytest.approx(1.0 / (3.0 * 1e-3), rel=1e-15)
    hist = integrate_three_stage_creep(1.0, tr, 0.01, params)
    assert hist["ruptured"] is True
    assert hist["times"][-1] == pytest.approx(tr, rel=1e-12)
    # acceleration: the rate near rupture is far above the secondary baseline
    sec = norton_strain_rate(1.0, params.glen)
    near = hist["rate"][-1]
    assert near / sec > 100.0
    mid = hist["rate"][len(hist["rate"]) // 2]
    assert mid > sec  # already past the pure secondary plateau


def test_three_stage_matches_per_stage_closed_form():
    params = ThreeStageCreepParameters(
        glen=_glen(),
        primary_A=1e-6,
        primary_p=2.0 / 3.0,
        damage_B=1e-3,
        damage_m=2.0,
        tertiary_q=3.0,
    )
    hist = integrate_three_stage_creep(1.0, 300.0, 0.01, params, young_modulus=ICE_E)
    for target in (10.0, 100.0, 250.0):
        i = min(range(len(hist["times"])), key=lambda k: abs(hist["times"][k] - target))
        ref = three_stage_analytic(hist["times"][i], 1.0, params, young_modulus=ICE_E)
        assert hist["strain"][i] == pytest.approx(ref["total"], rel=1e-2)


def test_pure_norton_has_no_primary_stage_negative_control():
    """With a vanishing primary coefficient the curve is straight from the start."""
    params = ThreeStageCreepParameters(
        glen=_glen(), primary_A=1e-18, primary_p=0.5, damage_B=1.0 / 3e9, damage_m=2.0
    )
    hist = integrate_three_stage_creep(1.0, 1e4, 1.0, params)
    ts = hist["times"]
    cs = hist["creep_strain"]
    lo, hi = 1.0, 1e4
    xs = [math.log(t) for t in ts if lo <= t <= hi]
    ys = [math.log(v) for t, v in zip(ts, cs) if lo <= t <= hi]
    n = len(xs)
    slope = (n * sum(x * y for x, y in zip(xs, ys)) - sum(xs) * sum(ys)) / (
        n * sum(x * x for x in xs) - sum(xs) ** 2
    )
    assert slope == pytest.approx(1.0, rel=2e-2)


# ---------------------------------------------------------------------------
# V17 - V20 -- two-particle Hertz contact relaxation
# ---------------------------------------------------------------------------
def _contact(params: GlenCreepParameters | None = None) -> HertzCreepContact:
    return HertzCreepContact.from_elastic(
        ICE_E, ICE_NU, 50.0, params or _glen()
    )


def test_hertz_relaxation_matches_closed_form_at_100s():
    """Criterion (c): F(t)/F0 at t = 100 s within 5% of the analytic solution."""
    contact = _contact()
    delta0 = 1e-4
    hist = contact.run(delta0, 100.0, 0.05)
    i = min(range(len(hist["times"])), key=lambda k: abs(hist["times"][k] - 100.0))
    num = hist["force_ratio"][i]
    ana = hertz_creep_relaxation(contact, hist["times"][i], delta0)["force_ratio"]
    assert abs(num - ana) / ana < 0.05
    assert abs(num - ana) / ana < 1e-2  # the achieved number is far tighter
    assert num < 0.95  # the force really relaxed


def test_hertz_relaxation_is_first_order_in_dt():
    contact = _contact()
    delta0 = 1e-4
    errs = {}
    for dt in (1.0, 0.2, 0.04):
        hist = contact.run(delta0, 100.0, dt)
        i = min(range(len(hist["times"])), key=lambda k: abs(hist["times"][k] - 100.0))
        ana = hertz_creep_relaxation(contact, hist["times"][i], delta0)["force_ratio"]
        errs[dt] = abs(hist["force_ratio"][i] - ana) / ana
    assert errs[0.04] < errs[0.2] < errs[1.0]
    assert errs[1.0] / errs[0.04] > 5.0


def test_hertz_relaxation_energy_ledger():
    contact = _contact()
    hist = contact.run(1e-4, 100.0, 0.05)
    du = hist["elastic_energy"][-1] - hist["elastic_energy"][0]
    work = hist["creep_work"][-1]
    assert hist["force"][-1] < hist["force"][0]
    assert du < 0.0 and work > 0.0
    assert abs(du + work) / abs(du) < 1e-2


def test_hertz_relaxation_n1_limit_is_exponential():
    params = _glen(n=1.0)
    contact = _contact(params)
    delta0 = 1e-4
    hist = contact.run(delta0, 100.0, 0.05)
    k = contact.sensitivity(delta0)
    for t, ratio in zip(hist["times"], hist["force_ratio"]):
        expected = (math.exp(-k * t)) ** 1.5
        assert ratio == pytest.approx(expected, rel=5e-3)


# ---------------------------------------------------------------------------
# V21 / V22 -- guards
# ---------------------------------------------------------------------------
def test_parameter_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        GlenCreepParameters(A=-1.0, n=3.0)
    with pytest.raises(ValueError):
        GlenCreepParameters(A=1e-7, n=0.0)
    with pytest.raises(ValueError):
        GlenCreepParameters(A=1e-7, n=3.0, normalization="glen")
    with pytest.raises(ValueError):
        radial_return(uniaxial_stress(1.0), 0.0, _glen(), shear_modulus=1.0)
    with pytest.raises(ValueError):
        radial_return(uniaxial_stress(1.0), 1.0, _glen(), shear_modulus=0.0)
    with pytest.raises(ValueError):
        _contact().run(0.0, 10.0, 0.1)
    with pytest.raises(ValueError):
        ThreeStageCreepParameters(glen=_glen(), primary_A=1e-6, primary_p=1.0)
    bad_q = ThreeStageCreepParameters(
        glen=_glen(), primary_A=1e-6, damage_m=2.0, tertiary_q=2.0
    )
    with pytest.raises(ValueError):
        bad_q.tertiary_strain(1.0, 1.0)


def test_zero_stress_produces_no_creep():
    params = _glen()
    zero = torch.zeros(3, 3, dtype=torch.float64)
    assert float(creep_strain_rate(zero, params).abs().sum()) == 0.0
    res = radial_return(zero, 1.0, params, shear_modulus=1.0)
    assert float(res["stress"].abs().sum()) == 0.0
    assert res["dgamma"] == 0.0
    assert norton_relaxation(10.0, 0.0, params, young_modulus=ICE_E) == 0.0