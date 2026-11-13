"""Restitution calibration for the Hertz + Tsuji normal dashpot.

Why this module exists
======================
``tensordem.dem3d`` maps a requested coefficient of restitution ``e`` to a
damping ratio with the *linear* spring-dashpot result

    zeta(e) = |ln e| / sqrt(pi^2 + ln^2 e),        e = exp(-pi zeta / sqrt(1 - zeta^2))

and then builds the pair dashpot from the local Hertz secant stiffness,

    gamma_n(delta) = 2 zeta sqrt(m_eff k_s(delta)),  k_s(delta) = (4/3) E* sqrt(R*) sqrt(delta)

i.e. ``gamma ~ delta^(1/4)`` (Tsuji-type non-linear damper).  The linear mapping
is exact only for a constant dashpot, so the realised restitution ``e_real`` does
*not* equal the requested ``e``.  Measured on the engine
(``benchmarks/case_pair_collision_hertz``, ``docs/dem3d-hertz-restitution``):

    e_requested = 0.5  ->  e_real = 0.5823   (+16.3 %)
    e_requested = 0.8  ->  e_real = 0.8259   (+ 3.2 %)

so ``restitution_coefficient`` was not a physical parameter in the Hertz branch.
This module provides the missing inverse map: given the requested ``e`` it
returns the damping ratio that the *Hertz* dashpot must carry to deliver it.

Similarity reduction (parameter free)
=====================================
With ``delta = D u``, ``t = T tau``, ``D`` the elastic peak overlap and
``T = sqrt(m_eff / (k sqrt(D)))``, ``k = (4/3) E* sqrt(R*)``, the equation of
relative motion reduces to

    u'' = -u^(3/2) - 2 zeta u^(1/4) u',      u(0) = 0, u'(0) = sqrt(4/5)

with no material or velocity left in it (energy of the elastic limit gives
``u'(0) = sqrt(4/5)`` exactly).  Hence

    e_real = G(zeta)

is a *universal* curve: independent of radius, density, Young's modulus and
impact speed.  That is what makes a single calibration valid for every impact
speed -- and it is the property measured in the velocity sweep of the benchmark.

Public API
==========
``hertz_restitution_from_zeta``   G(zeta), the universal Hertz restitution curve.
``hertz_damping_ratio_for_restitution``  zeta* with G(zeta*) = e_requested.
``delivered_restitution_legacy``  G(zeta_linear(e)) -- diagnoses the legacy bias.
``linear_restitution_from_zeta`` / ``zeta_from_linear_restitution``  the exact
linear spring-dashpot pair (used for the sanity check that the calibration is
the identity there).

Accuracy: the reference integrator is 4th-order RK4 in dimensionless time with a
cubic-Hermite event location at separation; the default step ``DT`` keeps
|G - G_exact| < 1e-7 (see ``tests/test_contact_calibration.py``).
"""

from __future__ import annotations

import math
from functools import lru_cache

__all__ = [
    "IMPACT_RATE",
    "DT",
    "hertz_restitution_from_zeta",
    "hertz_damping_ratio_for_restitution",
    "delivered_restitution_legacy",
    "linear_restitution_from_zeta",
    "zeta_from_linear_restitution",
    "calibration_table",
]

#: Dimensionless impact rate u'(0) = sqrt(4/5) of the similarity ODE.
IMPACT_RATE = math.sqrt(4.0 / 5.0)

#: Default dimensionless RK4 step.  G moves by <1e-7 between DT and DT/16.
DT = 1.0e-4

#: Empirical inflation of the damping ratio caused by the one-sided (no-tension)
#: Hertz dashpot; only used to seed the secant iteration.
_ZETA_SEED_INFLATION = 1.118

#: Safety cap on dimensionless time (<< 1/DT cycle) for over-damped impacts.
_MAX_TAU = 50.0


# ---------------------------------------------------------------------------
# Universal Hertz restitution curve G(zeta)
# ---------------------------------------------------------------------------
def _accel(u: float, w: float, zeta: float, clamp: bool = True) -> float:
    """u'' = -max(u^(3/2) + 2 zeta u^(1/4) u', 0).

    The clip is the unilateral-contact projection ``F_n = max(k_s delta - gamma
    delta_dot, 0)`` of the solver: near detachment the dashpot alone would pull
    (negative normal force), and the solver does not allow tension.  Dropping the
    clip raises the delivered restitution by ~0.8 % -- it is exactly the
    difference between a two-sided linear dashpot and the solver's one-sided one,
    so it must be modelled explicitly (see tests/test_contact_calibration.py).
    """
    if u <= 0.0:
        return 0.0
    f = u * math.sqrt(u) + 2.0 * zeta * u ** 0.25 * w
    if clamp:
        return -f if f > 0.0 else 0.0
    return -f


def _integrate(zeta: float, dt: float, clamp: bool = True) -> float:
    """RK4 integration of the similarity ODE for one ``zeta``."""
    w0 = IMPACT_RATE
    u, w = 0.0, w0
    seen_contact = False
    n_max = int(_MAX_TAU / dt) + 1
    for _ in range(n_max):
        a1 = _accel(u, w, zeta, clamp)
        k1u, k1w = w, a1
        u2 = u + 0.5 * dt * k1u
        w2 = w + 0.5 * dt * k1w
        k2u, k2w = w2, _accel(u2, w2, zeta, clamp)
        u3 = u + 0.5 * dt * k2u
        w3 = w + 0.5 * dt * k2w
        k3u, k3w = w3, _accel(u3, w3, zeta, clamp)
        u4 = u + dt * k3u
        w4 = w + dt * k3w
        k4u, k4w = w4, _accel(u4, w4, zeta, clamp)
        u_new = u + dt * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
        w_new = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        if u > 0.0:
            seen_contact = True
        if seen_contact and u_new <= 0.0 and w_new < 0.0:
            return min(1.0, max(0.0, abs(_hermite_exit_rate(u, w, u_new, w_new, dt)) / w0))
        u, w = u_new, w_new
    return 0.0


def hertz_restitution_from_zeta(zeta: float, *, dt: float = DT,
                                clamp: bool = True) -> float:
    """Universal restitution of the Hertz spring + Tsuji dashpot.

    Integrates ``u'' = -u^(3/2) - 2 zeta u^(1/4) u'`` from ``u'(0) = sqrt(4/5)``
    and returns ``|u'(exit)| / u'(0)``.

    Parameters
    ----------
    zeta : float
        Damping ratio in ``gamma_n = 2 zeta sqrt(m_eff k_s(delta))``, ``zeta >= 0``.
    dt : float
        Dimensionless RK4 step (default ``DT``; ``tests/test_contact_calibration``
        shows the drift from ``dt/4`` is below 1e-8).
    clamp : bool
        Keep the solver's unilateral (no-tension) projection
        ``F = max(k_s delta - gamma delta_dot, 0)``.  ``clamp=False`` gives the
        two-sided dashpot of linear theory, retained only for the comparison in
        ``tests/test_contact_calibration.py`` (the clip lifts ``e`` by ~0.8 %).

    Returns
    -------
    float
        Restitution in ``[0, 1]``; exactly 1 for ``zeta == 0`` and 0 for an
        over-damped impact that never separates.
    """
    if not math.isfinite(zeta) or zeta < 0.0:
        raise ValueError("zeta must be finite and non-negative")
    if zeta == 0.0:
        return 1.0
    return _integrate(zeta, dt, clamp)


def _hermite_exit_rate(u0: float, w0: float, u1: float, w1: float, dt: float) -> float:
    """Rate w at the theta where the cubic Hermite interpolant of u vanishes."""
    # Hermite basis on [0, 1]:  u(theta) = h00 u0 + h10 dt w0 + h01 u1 + h11 dt w1
    def u_of(theta: float) -> float:
        t2 = theta * theta
        t3 = t2 * theta
        return ((2 * t3 - 3 * t2 + 1) * u0 + (t3 - 2 * t2 + theta) * dt * w0
                + (-2 * t3 + 3 * t2) * u1 + (t3 - t2) * dt * w1)

    def du_of(theta: float) -> float:
        t2 = theta * theta
        return ((6 * t2 - 6 * theta) * u0 + (3 * t2 - 4 * theta + 1) * dt * w0
                + (-6 * t2 + 6 * theta) * u1 + (3 * t2 - 2 * theta) * dt * w1)

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if u_of(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-14:
            break
    theta = 0.5 * (lo + hi)
    return du_of(theta) / dt


# ---------------------------------------------------------------------------
# Linear spring-dashpot pair (exact in the linear branch)
# ---------------------------------------------------------------------------
def linear_restitution_from_zeta(zeta: float) -> float:
    """Restitution of a linear spring-dashpot pair, ``exp(-pi zeta / sqrt(1-zeta^2))``."""
    if not math.isfinite(zeta) or zeta < 0.0:
        raise ValueError("zeta must be finite and non-negative")
    if zeta >= 1.0:
        return 0.0
    return math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta ** 2))


def zeta_from_linear_restitution(e: float) -> float:
    """Damping ratio of a linear spring-dashpot pair delivering restitution ``e``."""
    if not (0.0 < e <= 1.0):
        raise ValueError("restitution must be in (0, 1]")
    abs_ln_e = -math.log(e)
    return abs_ln_e / math.sqrt(math.pi ** 2 + abs_ln_e ** 2)


# ---------------------------------------------------------------------------
# Inverse map: requested e -> damping ratio
# ---------------------------------------------------------------------------
@lru_cache(maxsize=256)
def hertz_damping_ratio_for_restitution(e_requested: float,
                                        *, tol: float = 1.0e-10) -> float:
    """Damping ratio ``zeta*`` with ``G(zeta*) == e_requested``.

    Secant iteration on the monotone universal curve, started from the linear
    mapping.  Cached, because ``dem3d`` asks for it on every force evaluation.
    """
    if not (0.0 < e_requested <= 1.0):
        raise ValueError("restitution must be in (0, 1]")
    if e_requested == 1.0:
        return 0.0

    z0 = zeta_from_linear_restitution(e_requested)
    f0 = hertz_restitution_from_zeta(z0) - e_requested
    z1 = z0 * _ZETA_SEED_INFLATION if f0 > 0.0 else z0 * (1.0 - 0.05)
    f1 = hertz_restitution_from_zeta(z1) - e_requested
    for _ in range(80):
        if abs(f1) <= tol:
            return z1
        denom = f1 - f0
        if denom == 0.0:
            break
        z2 = z1 - f1 * (z1 - z0) / denom
        # Keep the iterate physical; fall back to bisection-style damping.
        if not (0.0 < z2 < 1.0):
            z2 = 0.5 * (z0 + z1) if z0 < z1 else 0.5 * (z1 + z0)
        f2 = hertz_restitution_from_zeta(z2) - e_requested
        z0, f0, z1, f1 = z1, f1, z2, f2
    return z1


def delivered_restitution_legacy(e_requested: float) -> float:
    """Restitution the *legacy* (linear-theory) mapping actually delivers.

    ``G(zeta_linear(e_requested))``.  Used as the predicted baseline bias; it
    reproduces the engine measurement ``e=0.5 -> 0.5823`` to ~1e-4.
    """
    return hertz_restitution_from_zeta(zeta_from_linear_restitution(e_requested))


def calibration_table(e_values=(0.3, 0.5, 0.7, 0.8, 0.9, 0.95)) -> list[dict]:
    """Diagnostics for a set of requested restitution coefficients."""
    rows = []
    for e in e_values:
        zeta_lin = zeta_from_linear_restitution(e)
        zeta_cal = hertz_damping_ratio_for_restitution(e)
        legacy = hertz_restitution_from_zeta(zeta_lin)
        calibrated = hertz_restitution_from_zeta(zeta_cal)
        rows.append({
            "restitution_requested": e,
            "zeta_linear_mapping": zeta_lin,
            "restitution_legacy_delivered": legacy,
            "restitution_legacy_rel_error": abs(legacy - e) / e,
            "zeta_calibrated": zeta_cal,
            "zeta_inflation_factor": zeta_cal / zeta_lin,
            "restitution_calibrated_delivered": calibrated,
            "restitution_calibrated_rel_error": abs(calibrated - e) / e,
        })
    return rows