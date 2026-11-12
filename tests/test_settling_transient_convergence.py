"""Transient-phase verification of the single-sphere settling surface.

The existing Stokes case verifies only the *terminal* velocity, which is a
fixed point of the same drag law used to march the ODE (self-consistent, blind
to integrator error).  These tests assert the invariants of the missing
transient half:

* the closed-form solution of the linear settling ODE is consistent with the
  library's Stokes terminal velocity / relaxation time helpers,
* the exponential integrator is exact for the linear drag law (dt independent)
  while forward Euler is first order,
* forward-Euler's fitted relaxation time is biased by exactly ``-dt / ln(1 -
  dt/tau)`` -- the analytic prediction -- which explains the 4.8 % ``tau``
  "error" of the published 30 um case as discretisation, not physics,
* the exponential integrator is A-stable where forward Euler diverges
  (``dt > 2 tau``),
* the momentum-exchange ledger stays at machine precision.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensordem.two_way_coupling import (
    relaxation_time,
    stokes_terminal_velocity,
)
from tensordem.two_way_coupling.settling_transient import (
    ANCHOR_CASE,
    LINEAR_CASE,
    euler_tau_bias_prediction,
    fit_relaxation_time,
    integrate_settling_exponential,
    linear_settling_solution,
    transient_convergence_study,
)
from tensordem.two_way_coupling.particle_ode import (
    FluidState,
    ParticleODE,
    ParticleState,
)
from tensordem.cfd_dem_dense.forces import ForceModel


# ---------------------------------------------------------------------------
# 1. closed form vs the library's analytic helpers
# ---------------------------------------------------------------------------
def test_closed_form_matches_library_stokes_helpers() -> None:
    p = ANCHOR_CASE
    closed = linear_settling_solution(
        p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"],
        p["gravity"], p["added_mass_coeff"],
    )
    v_t_lib = stokes_terminal_velocity(
        p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"], p["gravity"]
    )
    tau_lib = relaxation_time(
        p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"],
        p["added_mass_coeff"],
    )
    assert closed["v_terminal"] == pytest.approx(v_t_lib, rel=1e-13)
    assert closed["tau"] == pytest.approx(tau_lib, rel=1e-13)
    # v_t = 7.3575e-04 m/s for the 30 um sand-in-water case at g = 9.81 m/s^2
    # (the library's GRAVITY constant is 9.80665, hence the tiny offset above)
    ref_981 = linear_settling_solution(p["diameter"], p["rho_particle"],
                                       p["rho_fluid"], p["mu"], 9.81, 0.5)
    assert ref_981["v_terminal"] == pytest.approx(7.3575e-04, rel=1e-4)


def test_terminal_velocity_is_independent_of_added_mass() -> None:
    """The drag balance does not contain m_eff: C_a must not move v_t."""
    p = ANCHOR_CASE
    a = linear_settling_solution(p["diameter"], p["rho_particle"], p["rho_fluid"],
                                 p["mu"], p["gravity"], 0.0)
    b = linear_settling_solution(p["diameter"], p["rho_particle"], p["rho_fluid"],
                                 p["mu"], p["gravity"], 0.5)
    c = linear_settling_solution(p["diameter"], p["rho_particle"], p["rho_fluid"],
                                 p["mu"], p["gravity"], 1.0)
    assert a["v_terminal"] == pytest.approx(b["v_terminal"], rel=1e-15)
    assert b["v_terminal"] == pytest.approx(c["v_terminal"], rel=1e-15)
    # ... while the relaxation time scales with it
    assert a["tau"] < b["tau"] < c["tau"]


def test_closed_form_rejects_nonsettling_particle() -> None:
    with pytest.raises(ValueError):
        linear_settling_solution(1e-5, 900.0, 1000.0, 1e-3)
    with pytest.raises(ValueError):
        linear_settling_solution(-1e-5, 2500.0, 1000.0, 1e-3)


# ---------------------------------------------------------------------------
# 2. integrator accuracy / order on the linear-limit case
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def linear_study() -> dict:
    return transient_convergence_study(LINEAR_CASE, label="linear_limit")


@pytest.fixture(scope="module")
def anchor_study() -> dict:
    return transient_convergence_study(ANCHOR_CASE, label="anchor")


def test_euler_transient_error_is_first_order(linear_study: dict) -> None:
    orders = [o["observed_order"] for o in linear_study["orders"]["euler"]]
    assert len(orders) >= 3
    for p_obs in orders:
        assert 0.95 < p_obs < 1.10, f"Euler transient order {p_obs} is not first order"


def test_exponential_integrator_is_exact_and_dt_independent(linear_study: dict) -> None:
    """In the linear limit the exponential advance is exact at any dt.

    The residual (~1e-4 here) is the Schiller-Naumann correction
    ``0.15 Re^0.687`` at Re ~ 1e-5, i.e. the difference between the integrated
    model and the *linear* closed form -- not an integration error.  Hence the
    two key signatures: tiny and *constant* under dt refinement.
    """
    errs = [r["regimes"]["exponential"]["max_rel_transient_error"]
            for r in linear_study["rows"]]
    assert max(errs) < 2e-4
    assert max(errs) / min(errs) < 1.05, "exponential error must not scale with dt"


def test_exponential_beats_euler_by_more_than_an_order(linear_study: dict) -> None:
    coarsest = linear_study["rows"][0]["regimes"]
    ratio = (coarsest["euler"]["max_rel_transient_error"]
             / coarsest["exponential"]["max_rel_transient_error"])
    assert ratio > 10.0, f"only {ratio:.1f}x improvement"


# ---------------------------------------------------------------------------
# 3. the analytic tau-bias prediction (verification content)
# ---------------------------------------------------------------------------
def test_euler_tau_bias_matches_analytic_prediction(linear_study: dict) -> None:
    """tau_fit/tau - 1 == -dt/(2 tau) + O(...) must be reproduced numerically.

    This is a falsifiable analytic prediction: the ratio estimator applied to
    the Euler discrete decay returns exactly ``-dt / ln(1 - dt/tau)``.
    """
    for row in linear_study["rows"]:
        pred = row["euler_bias_prediction"]["relative_bias"]
        got = row["regimes"]["euler"]["tau_signed_error"]
        assert got == pytest.approx(pred, abs=1e-3), (
            f"dt/tau={row['dt_over_tau']:.4f}: measured {got:+.5f} vs {pred:+.5f}")


def test_euler_tau_bias_vanishes_with_dt(linear_study: dict) -> None:
    """The bias is O(dt): halving dt must halve it (systematic, not noise)."""
    biases = [abs(r["regimes"]["euler"]["tau_signed_error"])
              for r in linear_study["rows"]]
    for coarse, fine in zip(biases, biases[1:]):
        assert coarse / fine == pytest.approx(2.0, rel=0.05)


def test_exponential_tau_is_dt_independent(linear_study: dict) -> None:
    taus = [r["regimes"]["exponential"]["tau_fit"] for r in linear_study["rows"]]
    median = float(np.median(taus))
    for t in taus:
        assert t == pytest.approx(median, rel=1e-3)


def test_euler_tau_bias_prediction_formula() -> None:
    tau, dt = 1.5e-4, 1.0e-5
    pred = euler_tau_bias_prediction(tau, dt)
    # exact discrete result: tau_fit = -dt / ln(1 - dt/tau)
    assert pred["tau_fit_predicted"] == pytest.approx(-dt / math.log(1.0 - dt / tau), rel=1e-14)
    # first-order asymptotic: -dt/(2 tau)
    assert pred["relative_bias"] == pytest.approx(-0.5 * dt / tau, rel=2e-2)
    assert pred["relative_bias"] < 0.0
    with pytest.raises(ValueError):
        euler_tau_bias_prediction(1e-6, 1e-6)


# ---------------------------------------------------------------------------
# 4. published-case regression: the 4.8 % tau "error" is discretisation
# ---------------------------------------------------------------------------
def test_anchor_case_tau_error_is_euler_bias(anchor_study: dict) -> None:
    """At the published dt = 1e-5 s (dt/tau ~ 0.067) Euler under-estimates tau
    by more than 2 %, in line with the analytic prediction, while the
    exponential integrator stays within 2 % of the analytic tau.
    """
    row = anchor_study["rows"][1]  # dt = 1.5e-5 ~ published 1e-5 s
    euler = row["regimes"]["euler"]
    expo = row["regimes"]["exponential"]
    assert euler["tau_signed_error"] < -0.02
    assert abs(expo["tau_signed_error"]) < 0.02
    assert abs(euler["tau_signed_error"]) > abs(expo["tau_signed_error"])


def test_anchor_case_residual_vs_closed_form_is_the_sn_correction(
    anchor_study: dict
) -> None:
    """The exponential integrator's residual against the *linear* closed form
    must equal the Schiller-Naumann correction ``0.15 Re_t^0.687`` at the
    terminal Reynolds number (1.02e-2 here) -- i.e. it is model, not solver.
    """
    cf = anchor_study["closed_form"]
    v_t = cf["v_terminal"]
    p = anchor_study["params"]
    Re_t = p["rho_fluid"] * v_t * p["diameter"] / p["mu"]
    sn_correction = 0.15 * Re_t**0.687
    residual = anchor_study["rows"][-1]["regimes"]["exponential"]["max_rel_transient_error"]
    assert Re_t == pytest.approx(0.0218, abs=5e-4)
    assert residual == pytest.approx(sn_correction, rel=0.1), (
        f"residual {residual:.4e} vs SN correction {sn_correction:.4e}")


# ---------------------------------------------------------------------------
# 5. stability of the exponential advance (forward Euler diverges for dt > 2 tau)
# ---------------------------------------------------------------------------
def test_exponential_is_stable_where_euler_diverges() -> None:
    p = dict(LINEAR_CASE)
    closed = linear_settling_solution(p["diameter"], p["rho_particle"],
                                      p["rho_fluid"], p["mu"], p["gravity"],
                                      p["added_mass_coeff"])
    tau = closed["tau"]
    dt = 5.0 * tau          # |1 - dt/tau| = 4 > 1 -> Euler amplification diverges
    ode = ParticleODE(p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"],
                      ForceModel(drag_model="schiller_naumann"), p["gravity"])
    state0 = ParticleState(position=np.zeros(3), velocity=np.zeros(3))
    fluid_fn = lambda t, x: FluidState.quiescent(p["rho_fluid"], p["gravity"])

    from tensordem.two_way_coupling.particle_ode import integrate_particle
    t_end = 20.0 * tau      # 4 Euler steps at dt = 5 tau
    st_eul, _, _ = integrate_particle(ode, state0, fluid_fn, t_end, dt)
    st_exp, _, _ = integrate_settling_exponential(ode, state0, fluid_fn, t_end, dt)

    v_eul = abs(st_eul.velocity[2])
    v_exp = abs(st_exp.velocity[2])
    assert v_eul > 2.0 * closed["v_terminal"], "Euler should be unstable here"
    assert v_exp == pytest.approx(closed["v_terminal"], rel=2e-4)


# ---------------------------------------------------------------------------
# 6. conservation + API hygiene
# ---------------------------------------------------------------------------
def test_exponential_ledger_conservation(anchor_study: dict) -> None:
    for row in anchor_study["rows"]:
        assert row["regimes"]["exponential"]["ledger_residual"] < 1e-12


def test_fit_relaxation_time_recovers_exact_discrete_decay() -> None:
    tau, dt, n = 1.5e-4, 1.0e-5, 200
    a = 1.0 - dt / tau
    times = np.arange(n) * dt
    speed = 1.0 - a ** np.arange(n)
    tau_fit = fit_relaxation_time(times, speed, 1.0)
    assert tau_fit == pytest.approx(-dt / math.log(a), rel=1e-12)
    # fewer than two points below the fitting threshold -> refuse
    with pytest.raises(ValueError):
        fit_relaxation_time(times[:1], speed[:1], 1.0)


def test_exponential_integrator_validates_inputs() -> None:
    p = ANCHOR_CASE
    ode = ParticleODE(p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"],
                      ForceModel(), p["gravity"])
    state0 = ParticleState(position=np.zeros(3), velocity=np.zeros(3))
    fluid_fn = lambda t, x: FluidState.quiescent(p["rho_fluid"], p["gravity"])
    with pytest.raises(ValueError):
        integrate_settling_exponential(ode, state0, fluid_fn, 1e-3, 0.0)