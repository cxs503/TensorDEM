"""Transient (relaxation-phase) settling verification surface.

Motivation
----------
The existing Stokes settling surface (:mod:`.stokes_terminal`) verifies only the
*terminal* velocity.  In the Stokes regime the terminal velocity is a fixed
point of the same drag law that is used to march the ODE, so that gate is
largely self-consistent: it cannot see a time-integration error.  The transient
phase ``v(t) = v_t (1 - exp(-t/tau))`` however *is* sensitive to the integrator,
and the forward-Euler march used by :func:`integrate_particle` is only first
order accurate and biases the fitted relaxation time ``tau`` by
``-dt/(2 tau)`` (leading order).  For the 30 um reference case
(``tau = 1.5e-4 s``, ``dt = 1e-5 s``) that bias is ``-3.3 %`` -- the same order
as the 4.8 % ``tau`` discrepancy that was reported as a tolerance-passing
number.

This module supplies the missing pieces:

1. :func:`linear_settling_solution` -- the *closed-form* solution of the linear
   (Stokes-drag) particle equation of motion with added mass and buoyancy.
   This is an independent baseline: it is derived by hand from

   .. math::

       (m_p + C_a \\rho_f V_p) \\frac{dv}{dt}
           = (m_p - \\rho_f V_p) g - 3 \\pi \\mu d \\, v

   giving ``v(t) = v_t (1 - e^{-t/tau})`` with

   .. math::

       v_t = \\frac{(\\rho_p - \\rho_f) g d^2}{18 \\mu}, \\qquad
       \\tau = \\frac{(\\rho_p + C_a \\rho_f) d^2}{18 \\mu}

   Note ``v_t`` is independent of ``C_a`` (added mass slows the approach but
   not the asymptote) -- a non-trivial identity that the numerical march must
   reproduce.

2. :func:`integrate_settling_exponential` -- a linearly-implicit *exponential*
   integrator with the same signature/semantics as
   :func:`~.particle_ode.integrate_particle`.  Per axis it freezes the drag
   coefficient ratio ``k = -a_drag / u_rel`` over one step and advances the
   velocity by the exact solution of the frozen linear ODE,

   .. math::

       v_{n+1} = v_\\infty + (v_n - v_\\infty) e^{-k \\Delta t}

   which is *exact* for the linear drag law (any ``dt``) and unconditionally
   stable for stiff (small-particle) drag, unlike forward Euler.

3. :func:`transient_convergence_study` -- the study driver: per ``dt`` it runs
   both integrators and reports the transient error against the closed form,
   the observed order of accuracy, the fitted relaxation time and the analytic
   prediction of the Euler ``tau`` bias.

4. :func:`fit_relaxation_time` -- the ``tau`` estimator, numerically identical
   to the one inside :func:`~.stokes_terminal.run_stokes_verification` so that
   the two surfaces are directly comparable.

Nothing here carries ship / ice / other domain constants.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

import numpy as np

from ..cfd_dem_dense.forces import GRAVITY
from .particle_ode import (
    FluidState,
    MomentumLedger,
    ParticleODE,
    ParticleState,
)

__all__ = [
    "linear_settling_solution",
    "fit_relaxation_time",
    "euler_tau_bias_prediction",
    "integrate_settling_exponential",
    "transient_convergence_study",
]


# ---------------------------------------------------------------------------
# 1. Closed-form baseline
# ---------------------------------------------------------------------------
def linear_settling_solution(
    diameter: float,
    rho_particle: float,
    rho_fluid: float,
    mu: float,
    gravity: float = GRAVITY,
    added_mass_coeff: float = 0.5,
) -> dict:
    """Closed-form solution of the linear (Stokes) settling ODE.

    Derivation (hand, see module docstring)::

        m_eff dv/dt = (m_p - rho_f V_p) g - 3 pi mu d v
        m_eff       = m_p + C_a rho_f V_p
        v(t)        = v_t (1 - exp(-t / tau))
        v_t         = (rho_p - rho_f) g d^2 / (18 mu)
        tau         = (rho_p + C_a rho_f) d^2 / (18 mu)
        k           = 1 / tau  (drag rate, [1/s])

    The terminal velocity is independent of the added-mass coefficient: the
    drag balance does not contain ``m_eff``.

    Returns a dict of the scalars plus a ``velocity(t)`` callable.
    """
    if diameter <= 0.0:
        raise ValueError("diameter must be positive")
    if mu <= 0.0:
        raise ValueError("mu must be positive")
    if added_mass_coeff < 0.0:
        raise ValueError("added_mass_coeff must be non-negative")
    if rho_particle <= rho_fluid:
        raise ValueError("particle must be denser than the fluid to settle")

    v_t = (rho_particle - rho_fluid) * gravity * diameter**2 / (18.0 * mu)
    tau = (rho_particle + added_mass_coeff * rho_fluid) * diameter**2 / (18.0 * mu)
    k = 1.0 / tau

    def velocity(t):
        t = np.asarray(t, dtype=float)
        return v_t * (1.0 - np.exp(-k * t))

    return {
        "v_terminal": float(v_t),
        "tau": float(tau),
        "drag_rate_k": float(k),
        "velocity": velocity,
    }


def fit_relaxation_time(times: Sequence[float], speed: Sequence[float], v_inf: float,
                        fraction: float = 0.9) -> float:
    """Estimate ``tau`` from a settling curve (same estimator as the library).

    Uses the points with ``v < fraction * v_inf`` and the ratio estimator
    ``tau = -sum(t) / sum(ln(1 - v/v_inf))``, which is the estimator used
    inside :func:`~.stokes_terminal.run_stokes_verification`.  For the exact
    discrete decay ``v_n = v_inf (1 - a^n)`` it returns exactly
    ``-dt / ln(a)``, which makes it a sharp probe of integrator bias.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(speed, dtype=float)
    mask = v < fraction * v_inf
    if np.count_nonzero(mask) <= 1:
        raise ValueError("not enough points below the fitting threshold")
    t_fit = t[mask]
    ratio = np.clip(1.0 - v[mask] / v_inf, 1e-15, 1.0)
    denom = float(np.sum(np.log(ratio)))
    if denom == 0.0:
        raise ValueError("degenerate settling curve (no logarithmic decay)")
    return float(-np.sum(t_fit) / denom)


def euler_tau_bias_prediction(tau: float, dt: float) -> dict:
    """Analytic prediction of forward-Euler's fitted ``tau`` bias.

    Forward Euler on ``v' = (v_inf - v)/tau`` gives ``v_n = v_inf (1 - a^n)``
    with ``a = 1 - dt/tau``.  The ratio estimator of
    :func:`fit_relaxation_time` therefore returns ``tau_fit = -dt / ln(a)``
    exactly, i.e. a first-order bias

        tau_fit / tau = 1 - dt/(2 tau) + O((dt/tau)^2).

    The same ``a`` also flips sign at ``dt = 2 tau`` (loss of stability).
    """
    x = dt / tau
    if x >= 1.0:
        raise ValueError("dt >= tau: forward Euler decay is no longer contractive")
    tau_fit = -dt / math.log(1.0 - x)
    return {
        "dt": float(dt),
        "dt_over_tau": float(x),
        "a": float(1.0 - x),
        "tau_fit_predicted": float(tau_fit),
        "relative_bias": float(tau_fit / tau - 1.0),
        "first_order_estimate": float(-0.5 * x),
        "stable": bool(x < 1.0),
    }


# ---------------------------------------------------------------------------
# 2. Exponential (linearly implicit) integrator
# ---------------------------------------------------------------------------
def integrate_settling_exponential(
    ode: ParticleODE,
    state0: ParticleState,
    fluid_fn: Callable[[float, np.ndarray], FluidState],
    t_end: float,
    dt: float,
    ledger: MomentumLedger | None = None,
) -> tuple[ParticleState, list[dict], MomentumLedger]:
    """Exponential integrator for the point-particle EOM.

    Same contract as :func:`~.particle_ode.integrate_particle`: it drives the
    *same* force assembly (``ode.compute_forces``), so every closure /
    switching model is exercised unchanged; only the time advance differs.

    Per step and per axis the drag channel is frozen as ``a_drag = k (u_f - v)``
    with ``k`` taken from the current force evaluation.  The remaining channels
    (gravity, pressure gradient, explicit added mass) are frozen as ``a_rest``.
    The frozen ODE is linear, so its exact solution is used::

        v_inf    = u_f + a_rest / k
        v_{n+1}  = v_inf + (v_n - v_inf) exp(-k dt)
        x_{n+1}  = x_n + v_inf dt + (v_n - v_inf)(1 - exp(-k dt)) / k

    Properties: exact for the linear (Stokes) drag law at any ``dt``; A-stable
    for the drag stiffness (forward Euler is not, for ``dt > 2 tau``); axes
    with zero relative velocity fall back to forward Euler.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if ledger is None:
        ledger = MomentumLedger()

    state = state0.copy()
    history: list[dict] = []
    n_steps = int(math.ceil(t_end / dt))
    m_eff = float(ode.m_eff)

    for i in range(n_steps):
        t_now = i * dt
        fluid = fluid_fn(t_now, state.position)

        accel, force_info = ode.compute_forces(state.velocity, fluid)
        reaction = ode.compute_reaction(force_info)

        f_l1 = sum(np.linalg.norm(f) for f in
                   [force_info["drag"], force_info["pressure"],
                    force_info["added_mass_explicit"]])
        ledger.accumulate(
            drag_p=float(np.sum(force_info["drag"])),
            drag_f=float(-np.sum(force_info["drag"])),
            pressure_p=float(np.sum(force_info["pressure"])),
            pressure_f=float(-np.sum(force_info["pressure"])),
            added_mass_p=float(np.sum(force_info["added_mass_explicit"])),
            added_mass_f=float(-np.sum(force_info["added_mass_explicit"])),
            force_l1=f_l1,
        )

        history.append({
            "t": t_now,
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "acceleration": accel.copy(),
            "Re": force_info["Re"],
            "drag": force_info["drag"].copy(),
            "pressure": force_info["pressure"].copy(),
            "added_mass": force_info["added_mass_explicit"].copy(),
            "gravity": force_info["gravity"].copy(),
            "reaction": reaction.copy(),
        })

        # Predictor probe: one extra force evaluation at the Euler-predicted
        # velocity.  For a linear drag law the secant estimate of the drag rate
        # below is then *exact at the very first step*, where ``u_rel == 0``
        # would otherwise leave the rate undetermined.
        v = state.velocity
        u_f = np.asarray(fluid.velocity, dtype=float)
        v_pred = v + np.asarray(accel, dtype=float) * dt
        accel_p, fi_p = ode.compute_forces(v_pred, fluid)
        a_drag = np.asarray(fi_p["drag"], dtype=float) / m_eff
        a_rest = np.asarray(accel_p, dtype=float) - a_drag
        u_rel = u_f - v_pred

        v_new = np.empty_like(v)
        for ax in range(v.size):
            scale = max(abs(u_f[ax]), abs(v_pred[ax]), 1e-12)
            if abs(u_rel[ax]) > 1e-14 * scale:
                k = a_drag[ax] / u_rel[ax]           # > 0: drag acts along u_rel
            else:
                k = 0.0
            if k <= 0.0:
                # no resolvable drag on this axis -> forward Euler
                v_new[ax] = v[ax] + accel[ax] * dt
                continue
            v_inf = u_f[ax] + a_rest[ax] / k
            decay = math.exp(-k * dt)
            v_new[ax] = v_inf + (v[ax] - v_inf) * decay
            state.position[ax] += v_inf * dt + (v[ax] - v_inf) * (1.0 - decay) / k

        state.velocity = v_new

    return state, history, ledger


# ---------------------------------------------------------------------------
# 3. Study driver
# ---------------------------------------------------------------------------
#: Reference case used by the case pool (micro-scale sand in water).
ANCHOR_CASE = dict(
    diameter=3.0e-5,
    rho_particle=2500.0,
    rho_fluid=1000.0,
    mu=1.0e-3,
    gravity=GRAVITY,
    added_mass_coeff=0.5,
)

#: Linear-limit scrape: Re ~ 1e-5, so the Schiller-Naumann correction
#: ``0.15 Re^0.687`` is < 1e-4 and the closed form is a rigorous baseline.
LINEAR_CASE = dict(ANCHOR_CASE, diameter=3.0e-6)


def _run_case(params: dict, dt: float, t_end: float, integrator: str):
    """March one case with the requested integrator; return (t, v_z) arrays."""
    model_kwargs = dict(
        drag_model="schiller_naumann",
        pressure_gradient=True,
        added_mass=True,
        added_mass_coefficient=params["added_mass_coeff"],
        fluid_reaction=True,
    )
    from ..cfd_dem_dense.forces import ForceModel  # local import keeps module import light

    ode = ParticleODE(
        params["diameter"], params["rho_particle"], params["rho_fluid"],
        params["mu"], ForceModel(**model_kwargs), params["gravity"],
    )
    state0 = ParticleState(position=np.zeros(3), velocity=np.zeros(3))
    fluid_fn = lambda t, x: FluidState.quiescent(params["rho_fluid"], params["gravity"])

    if integrator == "euler":
        from .particle_ode import integrate_particle
        _, hist, ledger = integrate_particle(ode, state0, fluid_fn, t_end, dt)
    elif integrator == "exponential":
        _, hist, ledger = integrate_settling_exponential(ode, state0, fluid_fn, t_end, dt)
    else:
        raise ValueError(f"unknown integrator {integrator!r}")

    times = np.array([h["t"] for h in hist], dtype=float)
    speed = np.abs(np.array([h["velocity"][2] for h in hist], dtype=float))
    return times, speed, ledger, ode


def transient_convergence_study(
    params: dict | None = None,
    dts: Iterable[float] | None = None,
    dt_over_tau: Iterable[float] = (0.2, 0.1, 0.05, 0.025, 0.0125),
    t_end: float | None = None,
    label: str = "case",
) -> dict:
    """Transient convergence study: forward Euler vs exponential integrator.

    For every ``dt`` the study reports

    * ``max_rel_transient_error`` of each integrator against the closed-form
      solution (max over all recorded steps of ``|v - v_exact| / v_t``),
    * the fitted relaxation time ``tau_fit`` and its error against the
      analytic ``tau``,
    * the analytic forward-Euler bias prediction ``-dt/(2 tau)``.

    The observed order of accuracy is computed between consecutive ``dt``.
    """
    p = dict(ANCHOR_CASE if params is None else params)
    closed = linear_settling_solution(
        p["diameter"], p["rho_particle"], p["rho_fluid"], p["mu"],
        p["gravity"], p["added_mass_coeff"],
    )
    v_t = closed["v_terminal"]
    tau = closed["tau"]
    t_stop = t_end if t_end is not None else 12.0 * tau

    # ``dt`` is normalised by the physical relaxation time so that the same
    # study is meaningful for particles of any size (tau ~ d^2).  Explicit
    # ``dts`` override the normalised sweep.
    if dts is None:
        dts = [float(r) * tau for r in dt_over_tau]

    rows = []
    for dt in dts:
        exact_v = closed["velocity"]
        row = {"dt": float(dt), "dt_over_tau": float(dt / tau), "regimes": {}}
        for name in ("euler", "exponential"):
            times, speed, ledger, ode = _run_case(p, dt, t_stop, name)
            err = float(np.max(np.abs(speed - exact_v(times))) / v_t)
            try:
                tau_fit = fit_relaxation_time(times, speed, v_t)
            except ValueError:
                tau_fit = float("nan")
            # Same estimator but normalised by the *numerical* terminal speed,
            # exactly as run_stokes_verification does; needed to compare with
            # the published case numbers without estimator ambiguity.
            try:
                tau_fit_lib = fit_relaxation_time(times, speed, float(speed[-1]))
            except ValueError:
                tau_fit_lib = float("nan")
            row["regimes"][name] = {
                "v_terminal_numerical": float(speed[-1]),
                "terminal_error_vs_closed_form": float(abs(speed[-1] - v_t) / v_t),
                "max_rel_transient_error": err,
                "tau_fit": float(tau_fit),
                "tau_error_vs_closed_form": float(abs(tau_fit - tau) / tau) if np.isfinite(tau_fit) else float("nan"),
                "tau_signed_error": float(tau_fit / tau - 1.0) if np.isfinite(tau_fit) else float("nan"),
                "tau_fit_library_estimator": float(tau_fit_lib),
                "tau_signed_error_library_estimator": (
                    float(tau_fit_lib / tau - 1.0) if np.isfinite(tau_fit_lib) else float("nan")),
                "ledger_residual": float(ledger.residual),
            }
        try:
            row["euler_bias_prediction"] = euler_tau_bias_prediction(tau, dt)
        except ValueError:
            row["euler_bias_prediction"] = {"dt": float(dt), "dt_over_tau": float(dt / tau),
                                            "stable": False, "relative_bias": float("nan")}
        rows.append(row)

    def _order(key: str) -> list[dict]:
        out = []
        for prev, cur in zip(rows, rows[1:]):
            e0 = prev["regimes"][key]["max_rel_transient_error"]
            e1 = cur["regimes"][key]["max_rel_transient_error"]
            ratio = (prev["dt"] / cur["dt"])
            if e1 > 0.0 and e0 > 0.0:
                p_obs = math.log(e0 / e1) / math.log(ratio)
            else:
                p_obs = float("nan")
            out.append({
                "dt_coarse": prev["dt"], "dt_fine": cur["dt"],
                "error_coarse": e0, "error_fine": e1,
                "refinement_ratio": float(ratio),
                "observed_order": float(p_obs),
            })
        return out

    return {
        "label": label,
        "params": p,
        "closed_form": {
            "v_terminal": v_t,
            "tau": tau,
            "drag_rate_k": closed["drag_rate_k"],
        },
        "t_end": float(t_stop),
        "rows": rows,
        "orders": {"euler": _order("euler"), "exponential": _order("exponential")},
        "summary": {
            "euler_order": float(np.nanmedian([o["observed_order"]
                                              for o in _order("euler")])) if len(rows) > 1 else float("nan"),
            "euler_tau_error_at_coarsest": rows[0]["regimes"]["euler"]["tau_error_vs_closed_form"],
            "euler_tau_error_at_finest": rows[-1]["regimes"]["euler"]["tau_error_vs_closed_form"],
            "exponential_tau_error_at_finest": rows[-1]["regimes"]["exponential"]["tau_error_vs_closed_form"],
            "exponential_max_transient_error_at_finest": rows[-1]["regimes"]["exponential"]["max_rel_transient_error"],
            "euler_max_transient_error_at_finest": rows[-1]["regimes"]["euler"]["max_rel_transient_error"],
        },
    }