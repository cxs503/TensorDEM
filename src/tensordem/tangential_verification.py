"""Analytic verification of the contact models in :mod:`tensordem.contact_tangential`.

Two independent anchors are implemented here; both compare a *time-stepped*
solution against a closed-form reference, so they verify the numerical use of
the models (not just the algebra of a single formula):

Anchor R - restitution mapping
    ``restitution_to_damping`` maps a target coefficient of restitution ``e``
    to a normal dashpot coefficient ``gamma_n`` through
    ``e = exp(-pi*zeta/sqrt(1-zeta^2))``.  Here the underlying mass-spring-
    dashpot impact is integrated with the same semi-implicit Euler scheme the
    solver uses, and the measured rebound ratio is compared with the target
    ``e``.  A correct mapping gives ``|e_meas - e| / e << 1`` and the error
    shrinks with the time step.

Anchor T - Coulomb stick-slip hysteresis
    For a contact under constant normal force ``F_n`` whose tangential
    displacement is driven harmonically with amplitude ``a``, the
    Cundall-Strack loop dissipates

        W = 4 * mu * F_n * (a - mu * F_n / k_t)     (a >= mu * F_n / k_t)

    per cycle.  The measured loop area (trapezoidal quadrature of
    ``F_t . d(delta)`` over the last of several cycles) is compared with that
    expression.  A negative control repeats the loop while storing the raw
    (unclamped) displacement state, which yields a single-valued force curve
    and therefore zero hysteresis: that is the failure mode the state
    projection in ``compute_tangential_force`` prevents.

All quantities are SI.  The problems are single-contact and tiny, so the whole
campaign is a CPU-bound, sub-minute job.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .contact_tangential import (
    compute_tangential_force,
    integrate_tangential_displacement,
    restitution_to_damping,
    tangential_cycle_dissipated_work,
    damping_to_restitution,
)

DTYPE = torch.float64


# ===================================================================
# Anchor R - coefficient of restitution
# ===================================================================

@dataclass(frozen=True)
class RestitutionCase:
    """One impact case: target e plus normal contact parameters."""

    e_target: float
    k_n: float = 1000.0
    m_eff: float = 1.0
    steps_per_period: int = 4000
    v0: float = 1.0


def simulate_rebound(case: RestitutionCase) -> Dict[str, float]:
    """Integrate one damped impact and measure the rebound coefficient.

    Model: ``m_eff * x_ddot + gamma_n * x_dot + k_n * x = 0`` with
    ``x(0) = 0`` (contact just made), ``x_dot(0) = v0 > 0`` (compression
    positive).  Semi-implicit (symplectic) Euler is used, matching the
    integrator of the DEM core.  The contact ends when ``x`` returns to zero
    with ``x_dot < 0``; the rebound velocity is linearly interpolated at that
    crossing and ``e_meas = |x_dot| / v0``.
    """
    k_n = case.k_n
    m_eff = case.m_eff
    omega_n = math.sqrt(k_n / m_eff)
    period = 2.0 * math.pi / omega_n
    dt = period / float(case.steps_per_period)
    gamma_n = restitution_to_damping(case.e_target, k_n, m_eff)

    x = 0.0
    v = case.v0
    x_prev = x
    v_prev = v
    t = 0.0
    n_steps = 0
    max_steps = 50 * case.steps_per_period

    e_meas = float("nan")
    t_end = float("nan")
    while n_steps < max_steps:
        a = (-k_n * x - gamma_n * v) / m_eff
        v = v + a * dt          # semi-implicit Euler: velocity first ...
        x = x + v * dt          # ... then position with the new velocity
        t += dt
        n_steps += 1
        # detect contact release: x returns to zero going in -x after compression
        if n_steps > 1 and v < 0.0 and x <= 0.0 and x_prev > 0.0:
            # linear interpolation of v at x = 0
            frac = x_prev / (x_prev - x)
            v_reb = v_prev + frac * (v - v_prev)
            e_meas = abs(v_reb) / case.v0
            t_end = t
            break
        x_prev, v_prev = x, v

    return {
        "e_target": case.e_target,
        "e_measured": e_meas,
        "rel_error": (abs(e_meas - case.e_target) / case.e_target
                      if e_meas == e_meas else float("nan")),
        "gamma_n": gamma_n,
        "dt": dt,
        "steps": n_steps,
        "contact_time": t_end,
        "zeta": gamma_n / (2.0 * math.sqrt(k_n * m_eff)),
    }


def restitutions_to_damping_check(pairs: Sequence[tuple[float, float, float]] | None = None) -> List[Dict[str, float]]:
    """Round-trip check ``e -> gamma_n -> e`` for a few (e, k_n, m_eff) triples."""
    if pairs is None:
        pairs = [(0.3, 1000.0, 1.0), (0.5, 1000.0, 1.0),
                 (0.8, 500.0, 2.0), (1.0, 1000.0, 1.0)]
    out = []
    for e, k_n, m_eff in pairs:
        g = restitution_to_damping(e, k_n, m_eff)
        e_back = damping_to_restitution(g, k_n, m_eff)
        out.append({"e": e, "k_n": k_n, "m_eff": m_eff, "gamma_n": g,
                    "e_roundtrip": e_back, "abs_error": abs(e_back - e)})
    return out


# ===================================================================
# Anchor T - Coulomb stick-slip loop
# ===================================================================

@dataclass(frozen=True)
class StickSlipCase:
    """One harmonic tangential drive case."""

    amplitude: float           # m, peak tangential displacement
    mu: float = 0.3
    normal_force: float = 100.0  # N
    k_t: float = 1000.0          # N/m
    steps_per_cycle: int = 4000
    cycles: int = 3

    @property
    def a_stick(self) -> float:
        return self.mu * self.normal_force / self.k_t


def simulate_stick_slip(case: StickSlipCase, clamp_state: bool = True) -> Dict[str, object]:
    """Drive a single contact harmonically and measure the hysteresis loop.

    The tangential displacement is prescribed as ``delta(t) = a sin(2 pi t / T)``
    and the force history is produced *only* through the library functions
    ``integrate_tangential_displacement`` and ``compute_tangential_force``.

    Args:
        case:        case parameters.
        clamp_state: if True, feed the projected (``F_t / k_t``) state back in
                     (physically correct).  If False, feed the raw, unclamped
                     displacement back in; this is the negative control that
                     removes the hysteresis.

    Returns:
        dict with analytic work, measured work, relative error, loop samples and
        the sliding fraction of the cycle.
    """
    normal = torch.tensor([[1.0, 0.0]], dtype=DTYPE)
    f_n = torch.tensor([case.normal_force], dtype=DTYPE)
    period = 1.0  # arbitrary: the model is rate independent
    dt = period / float(case.steps_per_cycle)
    n_steps = case.steps_per_cycle * case.cycles

    state = torch.zeros((1, 2), dtype=DTYPE)
    f_prev = 0.0
    work_total = 0.0
    work_by_cycle: List[float] = []
    work_cycle = 0.0
    delta_hist: List[float] = []
    force_hist: List[float] = []
    sliding_steps = 0

    for n in range(n_steps):
        t0 = n * dt
        t1 = (n + 1) * dt
        delta0 = case.amplitude * math.sin(2.0 * math.pi * t0 / period)
        delta1 = case.amplitude * math.sin(2.0 * math.pi * t1 / period)
        dd = delta1 - delta0

        v_rel = torch.tensor([[0.0, dd / dt]], dtype=DTYPE)
        disp_guess = integrate_tangential_displacement(state, v_rel, normal, dt)
        f_t, clamped_state = compute_tangential_force(
            normal, disp_guess, f_n, case.k_t, case.mu,
        )
        f_val = float(f_t[0, 1].item())
        work_cycle += 0.5 * (f_prev + f_val) * dd
        f_prev = f_val

        raw_limit = abs(float(disp_guess[0, 1].item())) * case.k_t
        if raw_limit > case.mu * case.normal_force + 1e-12:
            sliding_steps += 1

        delta_hist.append(delta1)
        force_hist.append(f_val)

        state = clamped_state if clamp_state else disp_guess

        if (n + 1) % case.steps_per_cycle == 0:
            work_by_cycle.append(work_cycle)
            work_total += work_cycle
            work_cycle = 0.0

    analytic = tangential_cycle_dissipated_work(
        case.mu, case.normal_force, case.k_t, case.amplitude,
    )
    measured = work_by_cycle[-1] if work_by_cycle else float("nan")
    if analytic > 0.0:
        rel_error = abs(measured - analytic) / analytic
    else:
        rel_error = abs(measured)

    return {
        "amplitude": case.amplitude,
        "mu": case.mu,
        "normal_force": case.normal_force,
        "k_t": case.k_t,
        "a_stick": case.a_stick,
        "clamped_state": clamp_state,
        "work_analytic": analytic,
        "work_measured": measured,
        "work_by_cycle": work_by_cycle,
        "rel_error": rel_error,
        "sliding_fraction": sliding_steps / float(n_steps),
        "delta_history": delta_hist,
        "force_history": force_hist,
        "steps_per_cycle": case.steps_per_cycle,
    }


# ===================================================================
# Campaign
# ===================================================================

def run_campaign(
    e_targets: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    steps_per_period: Sequence[int] = (1000, 2000, 4000, 8000),
    amplitudes: Sequence[float] = (0.01, 0.03, 0.05, 0.1, 0.2, 0.4),
    steps_per_cycle: Sequence[int] = (500, 1000, 2000, 4000),
) -> Dict[str, object]:
    """Run both anchors and return a JSON-serialisable result tree.

    Includes the discretisation series for both anchors (dt refinement) and the
    unclamped-state negative control for the friction anchor.
    """
    # --- Anchor R: restitution, dt refinement at e = 0.5, sweep at finest dt
    rest_convergence = []
    for spp in steps_per_period:
        r = simulate_rebound(RestitutionCase(e_target=0.5, steps_per_period=spp))
        rest_convergence.append({"steps_per_period": spp, "dt": r["dt"],
                                 "e_measured": r["e_measured"],
                                 "rel_error": r["rel_error"]})

    finest = max(steps_per_period)
    rest_sweep = []
    for e in e_targets:
        r = simulate_rebound(RestitutionCase(e_target=e, steps_per_period=finest))
        rest_sweep.append({
            "e_target": e, "e_measured": r["e_measured"],
            "rel_error": r["rel_error"], "gamma_n": r["gamma_n"],
            "zeta": r["zeta"], "contact_time": r["contact_time"],
            "dt": r["dt"],
        })

    # --- Anchor T: stick-slip loop, amplitude sweep at the finest step count
    finest_cycle = max(steps_per_cycle)
    slip_sweep = []
    slip_negative = []
    for a in amplitudes:
        case = StickSlipCase(amplitude=a, steps_per_cycle=finest_cycle)
        pos = simulate_stick_slip(case, clamp_state=True)
        neg = simulate_stick_slip(case, clamp_state=False)
        slip_sweep.append({k: v for k, v in pos.items()
                           if k not in ("delta_history", "force_history")})
        slip_negative.append({k: v for k, v in neg.items()
                              if k not in ("delta_history", "force_history")})

    slip_convergence = []
    for spc in steps_per_cycle:
        r = simulate_stick_slip(StickSlipCase(amplitude=0.2, steps_per_cycle=spc),
                                clamp_state=True)
        slip_convergence.append({"steps_per_cycle": spc,
                                 "work_measured": r["work_measured"],
                                 "work_analytic": r["work_analytic"],
                                 "rel_error": r["rel_error"]})

    # loop samples for plotting (a = 0.2, last two cycles)
    loops = {}
    for a in (0.05, 0.1, 0.2):
        r = simulate_stick_slip(StickSlipCase(amplitude=a, steps_per_cycle=finest_cycle,
                                             cycles=3), clamp_state=True)
        n = r["steps_per_cycle"]
        keep = int(2 * n)
        stride = max(1, keep // 500)   # decimate for artifact size; plots unaffected
        loops[f"{a:g}"] = {
            "delta": r["delta_history"][-keep::stride],
            "force": r["force_history"][-keep::stride],
            "work_analytic": r["work_analytic"],
            "work_measured": r["work_measured"],
        }

    return {
        "anchor_restitution": {
            "mapping_roundtrip": restitutions_to_damping_check(),
            "convergence": rest_convergence,
            "sweep": rest_sweep,
            "worst_rel_error": max(abs(r["rel_error"]) for r in rest_sweep),
            "worst_rel_error_convergence": max(abs(r["rel_error"]) for r in rest_convergence
                                               if r["steps_per_period"] == finest),
        },
        "anchor_stick_slip": {
            "sweep": slip_sweep,
            "negative_control_unclamped": slip_negative,
            "convergence": slip_convergence,
            "loops": loops,
            "worst_rel_error": max(
                (r["rel_error"] for r in slip_sweep if r["work_analytic"] > 0.0),
                default=0.0,
            ),
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke run
    import json

    print(json.dumps(run_campaign(steps_per_period=(2000,),
                                  steps_per_cycle=(1000,)), indent=2)[:2000])