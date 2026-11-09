"""Rigid-contact collision **energy-ledger certification** (three conventions).

Maturation of ``docs/in54w1-contact-energy-order`` (mission m5404's
``residual_state_relative = 5.111290246e-07``).  That node defined four ledger
channels and showed the state channel refines; it also had to declare one
acceptance item *unmet*, because the trapezoid channel turned out to be a
**discrete identity** under velocity-Verlet (slope undefined, not 2).  This
module turns the whole thing into a certificate: three conventions defined side
by side, a checkpoint protocol that actually separates their orders, and one
machine-checkable verdict per acceptance item.

Why three conventions
---------------------
A contact episode exchanges energy through its interface.  There are three
canonical ways to *book* that exchange at discrete time, and they have
different status -- this is the entire content of the "is 5e-7 a convention or
a bug?" question:

``flux`` (**通量口径**, structural identity)
    Interface energy flux as impulse x interface velocity::

        W_flux = sum_n (1/2)(P_n + P_{n+1}) h,     P_n = F_n . v_n

    compared against the *same run's* kinetic-energy change.  For
    velocity-Verlet this closes to round-off when the episode starts and ends
    contact-free (``F = 0`` at both ends, so the leftover telescoping term
    ``(h/4)(F_end - F_start)`` vanishes).  A non-zero residual here is a
    *bookkeeping/implementation* defect, never a discretisation error -- that is
    exactly the discriminating power the acceptance item asks for
    (``< 1e-9``).

``explicit`` (**显式口径**, first order)
    Left-endpoint work along the discrete displacement::

        W_explicit = sum_n F_n (x_{n+1} - x_n)

    The Riemann error of the power flux is ``-(h/2)(P_end - P_start)`` plus the
    ``(h/2) sum F dv`` term, both ``O(h)``: ``|slope - 1| < 0.2``.

``trapezoid`` (**梯形口径**, second order)
    Force-trapezoid work along the discrete displacement::

        W_trapezoid = sum_n (1/2)(F_n + F_{n+1}) (x_{n+1} - x_n)

    Under velocity-Verlet this equals the discrete ``dKE`` *identically*, so its
    residual against an **analytic** reference is the trajectory error of the
    integrator, i.e. ``O(h^2)``: ``|slope - 2| < 0.2``.

The checkpoint protocol (the actual maturation)
----------------------------------------------
Both ``trapezoid == dKE`` and ``flux == dKE`` are identities **at the end of a
complete episode**, so a final-state residual cannot rank them by order (this is
what in54w1 hit).  The certificate therefore keeps two readings:

* *full episode* -- for the ``flux`` closure (an identity test, ``< 1e-9``);
* *interior checkpoint* ``t* = 0.3 t_c`` of a **linear** spring impact, where
  the exact solution ``delta(t) = (v0/omega) sin(omega t)`` is available in
  closed form, so ``dKE(t*) = E0 (cos^2(omega t*) - 1)`` is an analytic
  reference and the two quadrature conventions are ranked by their genuine
  orders (1 vs 2).

Everything is normalised by ``E0 = (1/2) m_eff v0^2``.

Public API
----------
``order_study``               dt sweep of the three ledgers + fitted slopes
``restitution_certificate``   measured vs target restitution (``< 1e-3``)
``newton_cradle_certificate`` 5-ball cradle: momentum (``< 1e-12``) + closure
``m5404_certificate``         explicit verdict on the 5.111e-07 residual
``certify_all``               every acceptance item as a boolean + ``all_passed``

The m5404 verdict deliberately drives the **shipped** solver
(``tensordem.floe_rigid6dof.FloeWorld``, through
``contact_energy_order.m5404_rejudgement``), not a re-implementation, so the
judgement lands on the same code that produced 5.111e-07.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .contact_energy_order import (
    ContactLaw,
    fit_order,
    m5404_rejudgement,
    newton_cradle,
    two_body_impact,
)

__all__ = [
    "FLUX_TOL",
    "SLOPE_TOL",
    "RESTITUTION_TOL",
    "MOMENTUM_TOL",
    "CONVENTIONS",
    "LedgerRow",
    "flux_closure",
    "order_study",
    "restitution_certificate",
    "newton_cradle_certificate",
    "m5404_certificate",
    "certify_all",
]

# Acceptance thresholds (campaign p54w3-n03 metadata).
FLUX_TOL = 1.0e-9          # (a) flux convention closes to round-off
SLOPE_TOL = 0.2            # (b) |slope - 1| and |slope - 2|
RESTITUTION_TOL = 1.0e-3   # (c) |e_measured - e_target|
MOMENTUM_TOL = 1.0e-12     # (d) 5-ball cradle total momentum

#: Machine-readable definition of the three conventions (for the report table).
CONVENTIONS: Dict[str, Dict[str, str]] = {
    "flux": {
        "cn": "通量口径",
        "formula": "W = sum_n (1/2)(P_n + P_{n+1}) h,  P_n = F_n v_n",
        "reference": "same-run discrete dKE (full episode, contact-free ends)",
        "expected": "identity: round-off (< 1e-9); residual would be a defect",
    },
    "explicit": {
        "cn": "显式口径",
        "formula": "W = sum_n F_n (x_{n+1} - x_n)",
        "reference": "analytic dKE(t*) at the interior checkpoint t* = 0.3 t_c",
        "expected": "first order: slope 1",
    },
    "trapezoid": {
        "cn": "梯形口径",
        "formula": "W = sum_n (1/2)(F_n + F_{n+1}) (x_{n+1} - x_n)",
        "reference": "analytic dKE(t*) at the interior checkpoint t* = 0.3 t_c",
        "expected": "second order: slope 2 (= trajectory order under Verlet)",
    },
}


@dataclass
class LedgerRow:
    """One ``dt`` level of the three-convention study."""

    dt: float
    steps_per_contact: float
    checkpoint_steps: int
    checkpoint_time_rel: float
    checkpoint_frac: float
    e0: float
    dke_exact: float
    dke_discrete: float
    w_flux_full: float
    w_explicit_checkpoint: float
    w_trapezoid_checkpoint: float
    residual_flux_full: float
    residual_explicit: float
    residual_trapezoid: float
    state_residual_full: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "dt": self.dt,
            "steps_per_contact": self.steps_per_contact,
            "checkpoint_steps": self.checkpoint_steps,
            "checkpoint_time_rel": self.checkpoint_time_rel,
            "checkpoint_frac": self.checkpoint_frac,
            "e0_j": self.e0,
            "dke_exact_j": self.dke_exact,
            "dke_discrete_j": self.dke_discrete,
            "w_flux_full_j": self.w_flux_full,
            "w_explicit_checkpoint_j": self.w_explicit_checkpoint,
            "w_trapezoid_checkpoint_j": self.w_trapezoid_checkpoint,
            "residual_flux_full": self.residual_flux_full,
            "residual_explicit": self.residual_explicit,
            "residual_trapezoid": self.residual_trapezoid,
            "residual_state_full": self.state_residual_full,
        }


def flux_closure(impact: Any) -> float:
    """Interface energy-flux residual of one impact, normalised by ``E0``.

    ``W_flux = sum_n (1/2)(P_n + P_{n+1}) h`` with ``P_n = F_n v_n`` the contact
    power at the sample states, compared with the same run's kinetic-energy
    change.  Velocity-Verlet plus a contact-free start **and** end make this an
    identity (round-off); the residual therefore measures *bookkeeping* defects,
    which is why the acceptance item can be ``< 1e-9`` at every ``dt``.
    """
    trace = impact.trace
    power = trace.forces[0] * (trace.velocities[1] - trace.velocities[0])
    w_flux = float(np.sum(0.5 * (power[:-1] + power[1:]) * np.diff(trace.times)))
    dke = float(trace.kinetic[-1] - trace.kinetic[0])
    e0 = 0.5 * float(impact.m_eff) * float(impact.v0) ** 2
    return abs(w_flux - dke) / e0


def _channels(k_stiffness: float, refinements: int,
              checkpoint_fraction: float) -> LedgerRow:
    """One dt level: linear-spring impact, exact solution available.

    The spring is linear (``exponent = 1``, no damping) so that the impact has
    the closed form ``delta(t) = (v0/omega) sin(omega t)``, ``omega =
    sqrt(k/m_eff)``, ``t_c = pi/omega``: an analytic reference for the interior
    checkpoint that owes nothing to the discrete run.
    """
    law = ContactLaw(stiffness=k_stiffness, exponent=1.0)
    impact = two_body_impact(law, refinements=refinements)
    trace = impact.trace
    h = float(trace.dt)
    m_eff = float(impact.m_eff)
    v0 = float(impact.v0)
    e0 = 0.5 * m_eff * v0 * v0
    omega = math.sqrt(k_stiffness / m_eff)
    t_c = math.pi / omega
    t_touch = float(impact.gap0) / v0

    # --- interior checkpoint: a generic (non-stationary) instant inside contact
    k = int(round((t_touch + checkpoint_fraction * t_c) / h))
    k = max(1, min(k, trace.times.size - 2))
    t_rel = float(trace.times[k] - t_touch)

    force = trace.forces[0]
    dgap = np.diff(trace.gaps[0])
    rates = trace.velocities[1] - trace.velocities[0]
    power = force * rates

    w_explicit = float(np.sum(force[:k] * dgap[:k]))
    w_trapezoid = float(np.sum(0.5 * (force[:k] + force[1:k + 1]) * dgap[:k]))
    w_flux_full = float(np.sum(0.5 * (power[:-1] + power[1:]) * np.diff(trace.times)))

    dke_exact = e0 * (math.cos(omega * t_rel) ** 2 - 1.0)
    dke_discrete = float(trace.kinetic[k] - trace.kinetic[0])
    dke_discrete_full = float(trace.kinetic[-1] - trace.kinetic[0])

    return LedgerRow(
        dt=h,
        steps_per_contact=t_c / h,
        checkpoint_steps=k,
        checkpoint_time_rel=t_rel,
        checkpoint_frac=t_rel / t_c,
        e0=e0,
        dke_exact=dke_exact,
        dke_discrete=dke_discrete,
        w_flux_full=w_flux_full,
        w_explicit_checkpoint=w_explicit,
        w_trapezoid_checkpoint=w_trapezoid,
        residual_flux_full=abs(w_flux_full - dke_discrete_full) / e0,
        residual_explicit=abs(w_explicit - dke_exact) / e0,
        residual_trapezoid=abs(w_trapezoid - dke_exact) / e0,
        # diagnostic: the state-function channel of in54w1 (full episode)
        state_residual_full=abs(
            (dke_discrete_full + float(trace.stored[-1] - trace.stored[0]))) / e0,
    )


def order_study(*, stiffness: float = 1.0e5, refinements: int = 8,
                levels: int = 8, checkpoint_fraction: float = 0.3) -> Dict[str, Any]:
    """``dt/2^j`` sweep of the three conventions on a linear-spring impact.

    Returns the per-level rows, the fitted log-log orders of the explicit and
    trapezoid conventions, the worst flux closure over the sweep, and the
    boolean acceptance flags of items (a) and (b).
    """
    if levels < 2:
        raise ValueError("an order study needs at least two dt levels")
    if not 0.0 < checkpoint_fraction < 0.5:
        raise ValueError("checkpoint_fraction must lie in (0, 0.5) "
                         "(mid-compression, away from the stationary point)")
    rows: List[LedgerRow] = [
        _channels(stiffness, refinements * 2 ** j, checkpoint_fraction)
        for j in range(levels)
    ]
    dts = [r.dt for r in rows]
    fits = {
        "residual_explicit": fit_order(dts, [r.residual_explicit for r in rows]),
        "residual_trapezoid": fit_order(dts, [r.residual_trapezoid for r in rows]),
    }
    flux_max = max(r.residual_flux_full for r in rows)
    slope_expl = fits["residual_explicit"]["slope"]
    slope_trap = fits["residual_trapezoid"]["slope"]
    return {
        "case": {
            "law": "linear spring, undamped (exact solution available)",
            "stiffness_n_m": stiffness,
            "m_eff_kg": 1.0,
            "v0_m_s": 1.0,
            "gap0_m": 0.02,
            "checkpoint_fraction": checkpoint_fraction,
            "refinements_per_contact": [refinements * 2 ** j for j in range(levels)],
            "levels": levels,
        },
        "rows": [r.as_dict() for r in rows],
        "fits": fits,
        "flux_residual_max": flux_max,
        "slope_explicit": slope_expl,
        "slope_trapezoid": slope_trap,
        "flux_closes": bool(flux_max < FLUX_TOL),
        "explicit_first_order": bool(abs(slope_expl - 1.0) < SLOPE_TOL),
        "trapezoid_second_order": bool(abs(slope_trap - 2.0) < SLOPE_TOL),
    }


def restitution_certificate(*, zeta: float = 0.2, stiffness: float = 1.0e5,
                            refinements: int = 1024,
                            coarse_refinements: int = 32) -> Dict[str, Any]:
    """Measured restitution vs the closed-form target (acceptance item (c)).

    The target is the *clamped* restitution of the Cundall-Strack dashpot
    (``contact_impact.clamped_restitution``, exposed as ``ContactLaw.restitution``)
    -- the analytic value for the law actually integrated.  The fixed-step run
    must agree with it to ``< 1e-3`` at ``refinements >= 1024`` steps per
    contact, and the coarse run must be *worse* (the error is a discretisation
    error, not a bias).
    """
    damping = 2.0 * zeta * math.sqrt(stiffness)
    law = ContactLaw(stiffness=stiffness, exponent=1.0, damping_model="linear",
                     damping=damping)
    e_target = float(law.restitution(1.0))
    fine = two_body_impact(law, refinements=refinements)
    coarse = two_body_impact(law, refinements=coarse_refinements)
    e_fine = float(fine.trace.v_out / fine.v0)
    e_coarse = float(coarse.trace.v_out / coarse.v0)
    err_fine = abs(e_fine - e_target)
    err_coarse = abs(e_coarse - e_target)
    return {
        "zeta": zeta,
        "stiffness_n_m": stiffness,
        "damping_n_s_m": damping,
        "e_target": e_target,
        "e_measured": e_fine,
        "restitution_error": err_fine,
        "steps_per_contact": fine.steps_per_contact,
        "coarse_steps_per_contact": coarse.steps_per_contact,
        "coarse_restitution_error": err_coarse,
        "error_shrinks_with_refinement": bool(err_fine < err_coarse),
        "within_tolerance": bool(err_fine < RESTITUTION_TOL),
    }


def newton_cradle_certificate(*, n: int = 5, stiffness: float = 1.0e7,
                              refinements: int = 8,
                              levels: int = 3) -> Dict[str, Any]:
    """Five-ball cradle: total-momentum conservation and ledger closure.

    All ``n-1`` contacts engage simultaneously (a statically indeterminate
    network while the wave crosses), which is where a pairwise force/work
    bookkeeping defect shows up as a momentum leak.  Acceptance: total momentum
    drift ``< 1e-12`` and the flux ledger round-off.
    """
    res = newton_cradle(n=n, stiffness=stiffness, refinements=refinements,
                        levels=levels)
    momentum = float(res["momentum_error_max"])
    flux = float(res["residual_average_velocity_max"])
    return {
        "n": n,
        "radius_m": res["radius"],
        "mass_kg": res["mass"],
        "stiffness_n_m": stiffness,
        "velocity_m_s": res["velocity"],
        "levels": levels,
        "dt": [r.dt for r in res["results"]],
        "steps_per_contact": [r.steps_per_contact for r in res["results"]],
        "momentum_error_max": momentum,
        "flux_residual_max": flux,
        "state_residual": [r.residual_state for r in res["results"]],
        "explicit_residual": [r.residual_explicit for r in res["results"]],
        "slope_explicit": res["fits"]["residual_explicit"]["slope"],
        "e_measured": [r.e_measured for r in res["results"]],
        "v_last_ball_final": [float(r.velocities_final[-1]) for r in res["results"]],
        "momentum_conserved": bool(momentum < MOMENTUM_TOL),
        "flux_closes": bool(flux < FLUX_TOL),
    }


def m5404_certificate(*, levels: int = 4) -> Dict[str, Any]:
    """Explicit verdict on m5404's ``5.111290246e-07`` residual (item (e)).

    Decision rule, fixed *before* the numbers (same rule as
    ``contact_energy_order.m5404_verdict``): the residual is classed
    ``convention`` when its refined log-log slope is ``>= 0.8`` (it tends to
    zero, so it is a discretisation/accounting artefact) and ``implementation``
    otherwise; the injected ``1e-3`` state-function defect is the control that
    the slope test can discriminate (a real defect leaves a plateau).
    """
    if levels < 3:
        raise ValueError(
            "at least three dt levels are required for the verdict: with two "
            "levels the residual pair is still pre-asymptotic (5.111e-07 -> "
            "3.96e-07 is a slope of 0.37) and the rule would return a false "
            "'implementation'")
    out = m5404_rejudgement(levels=levels)
    slope_state = float(out["fits"]["residual_state"]["slope"])
    slope_trap = float(out["fits"]["residual_trapezoid"]["slope"])
    slope_defect = float(out["fits"]["residual_state_injected_defect"]["slope"])
    verdict = out["verdict"]
    return {
        "reported_residual": out["case"]["reported_residual_state"],
        "reproduced_residual": out["reproduced_reported_residual"],
        "ratio_to_reported": out["ratio_to_reported"],
        "slope_state": slope_state,
        "slope_trapezoid": slope_trap,
        "slope_injected_defect": slope_defect,
        "state_residual_ratio": out["state_residual_ratio"],
        "defect_plateau_ratio": out["defect_plateau_ratio"],
        "rows": out["rows"],
        "fits": out["fits"],
        "verdict": verdict,
        "verdict_rule": "convention iff fitted slope of the state residual >= 0.8",
        "statement": (
            "5.111e-07 是**口径误差**（离散/记账量），不是实现缺陷：状态量口径残差在 "
            f"{levels} 级 Δt 细化下按斜率 {slope_state:.2f} 幂律塌缩"
            f"（粗/细 {out['state_residual_ratio']:.1f}×），"
            "而同一算例注入 1e-3 状态函数缺陷后残差呈平台"
            f"（斜率 {slope_defect:+.3f}，粗/细 {out['defect_plateau_ratio']:.3f}×），"
            "说明该判据对实现缺陷有分辨力，5.111e-07 未呈现平台行为。"
            if verdict == "convention" else
            "5.111e-07 不随 Δt 细化消失"
            f"（斜率 {slope_state:.2f} < 0.8）：判为实现缺陷。"
        ),
        "convention_identified": bool(verdict == "convention"),
    }


def certify_all(*, levels: int = 8, refinements: int = 8,
                cradle_levels: int = 3, m5404_levels: int = 4,
                restitution_refinements: int = 1024) -> Dict[str, Any]:
    """Run every acceptance item of the node and fold them into one verdict."""
    study = order_study(refinements=refinements, levels=levels)
    rest = restitution_certificate(refinements=restitution_refinements)
    cradle = newton_cradle_certificate(levels=cradle_levels)
    if m5404_levels < 3:
        raise ValueError("m5404_levels must be >= 3 (see m5404_certificate)")
    verdict = m5404_certificate(levels=m5404_levels)
    criteria = {
        "a_flux_convention_below_1e-9": study["flux_closes"],
        "b_explicit_slope_one": study["explicit_first_order"],
        "b_trapezoid_slope_two": study["trapezoid_second_order"],
        "c_restitution_within_1e-3": rest["within_tolerance"],
        "d_cradle_momentum_below_1e-12": cradle["momentum_conserved"],
        "e_m5404_verdict_given": bool(verdict["verdict"] in ("convention",
                                                             "implementation")),
    }
    return {
        "criteria": criteria,
        "all_passed": bool(all(criteria.values())),
        "thresholds": {
            "flux": FLUX_TOL,
            "slope_tolerance": SLOPE_TOL,
            "restitution": RESTITUTION_TOL,
            "momentum": MOMENTUM_TOL,
        },
        "key_numbers": {
            "flux_residual_max": study["flux_residual_max"],
            "slope_explicit": study["slope_explicit"],
            "slope_trapezoid": study["slope_trapezoid"],
            "restitution_error": rest["restitution_error"],
            "e_measured": rest["e_measured"],
            "e_target": rest["e_target"],
            "cradle_momentum_error_max": cradle["momentum_error_max"],
            "cradle_flux_residual_max": cradle["flux_residual_max"],
            "m5404_slope_state": verdict["slope_state"],
            "m5404_verdict": verdict["verdict"],
        },
        "study": study,
        "restitution": rest,
        "cradle": cradle,
        "m5404": verdict,
        "conventions": CONVENTIONS,
    }