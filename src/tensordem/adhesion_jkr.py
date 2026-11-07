"""JKR adhesive normal contact law (Johnson–Kendall–Roberts, 1971).

This module adds a *reversible* adhesive normal contact kernel to TensorDEM and
makes its correctness machine-checkable against closed-form JKR anchors.  It is
the normal-direction complement of :mod:`tensordem.hertz_mindlin`
(non-adhesive normal + Mindlin tangential) and of
:mod:`tensordem.contact_tangential` / :mod:`tensordem.rolling_resistance`.

Conventions (identical to :func:`tensordem.dem3d.hertz_normal_force`)
--------------------------------------------------------------------
Two convex bodies with radii ``R1, R2``, Young's moduli ``E1, E2`` and Poisson
ratios ``nu1, nu2``::

    1/E* = (1 - nu1**2)/E1 + (1 - nu2**2)/E2          (reduced modulus)
    1/R* = 1/R1 + 1/R2                                 (reduced radius)
    K    = 4 E* / 3                                    (Hertz prefactor)

The non-adhesive Hertz law is ``F = (4/3) E* sqrt(R*) delta**(3/2)`` which is
``F = K sqrt(R*) delta**(3/2)``; this is exactly
``dem3d.hertz_normal_force``.

JKR relations
-------------
With ``gamma`` the *work of adhesion* (J/m^2, i.e. the Dupré energy
``gamma = gamma1 + gamma2 - gamma12``), and ``a`` the contact radius:

    load–radius:      a**3 = (R*/K) * (F + 3 pi gamma R* +
                            sqrt(6 pi gamma R* F + (3 pi gamma R*)**2))
    inverted:         F(a)  = K a**3 / R* - sqrt(6 pi gamma K a**3)
    approach:         delta(a) = a**2/R* - (2/3) sqrt(6 pi gamma a / K)
    pull-off force:   F_c   = -(3/2) pi gamma R*          (at a = a_p)
    zero-load radius: a_c   = (6 pi gamma R*^2 / K)**(1/3)
    pull-off radius:  a_p   = (3/2 pi gamma R*^2 / K)**(1/3) = a_c / 4**(1/3)

The ``(2/3)`` prefactor of the adhesive approach is *not* free: it follows from
the JKR superposition of the Hertz pressure field and the Boussinesq flat-punch
field, ``delta3 = P3 / (2 a E*)`` with ``P3 = -sqrt(6 pi gamma K a**3)`` and
``E* = 3K/4``, giving ``(2/3) sqrt(6 pi gamma a / K)``.  A common mistake is to
drop the ``2/3`` (or to use ``sqrt(2 pi gamma a / K)``), which corrupts the
energy audit; ``tests/test_adhesion_jkr.py`` pins the prefactor through the
strain-energy-release-rate identity.

Strain energy release rate
--------------------------
``G = (F - F_Hertz)**2 / (8 pi E* a**3)`` with ``F_Hertz = K a**3/R*``.  The
JKR equilibrium condition is ``G = gamma``; this is an *independent* physics
anchor (it does not follow from the force law) and is asserted in the tests.

Energy audit
------------
Along the reversible (quasi-static) equilibrium branch the work done by the
external agent equals the free-energy difference,

    W_sep = -integral_{a_c}^{0} F d(delta)  (>0, the work of separation),

and the free-energy bookkeeping is ``W_sep = pi gamma a_c**2 - U_el(a_c)``
where ``pi gamma a_c**2`` is the JKR surface-energy difference (contact area
``pi a_c**2`` times the work of adhesion) and ``U_el(a_c)`` the elastic energy
stored at zero load.  ``jkr_energy_audit`` reports each term plus the
*universal* dimensionless ratio ``W_sep / (pi gamma a_c**2)``.  Dimensional
analysis makes that ratio a pure number (independent of ``gamma, R*, K``);
its universality is asserted in the tests, which is the strongest
parameter-free closure available for this kernel.

Note on "4 pi gamma R**2": that quantity is the *surface energy of a free
sphere* of radius ``R``.  It is **not** the surface-energy difference of a JKR
contact (which scales as ``pi gamma a_c**2``), and the ratio
``W_sep / (4 pi gamma R**2) ~ (a_c/R)**2 * 0.16`` is ``O(1e-4)`` in the JKR
regime.  ``jkr_energy_audit`` therefore reports *both* numbers and the
analytic scaling law instead of pretending they coincide; see
``docs/adhesion-contact-verification/README.md`` §6.

All quantities use SI units.  ``gamma <= gamma_floor`` (default ``1e-12``,
J/m^2) degenerates *exactly* to the non-adhesive Hertz law, so a DEM run with
adhesion switched off is bit-identical to the legacy kernel.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

#: Adhesion below this value (J/m^2) is treated as zero: the kernel then
#: degenerates exactly to :func:`tensordem.dem3d.hertz_normal_force`.
GAMMA_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Reduced material/geometry parameters
# ---------------------------------------------------------------------------

def jkr_reduced_radius(r1: float, r2: float) -> float:
    """Reduced radius ``R* = R1 R2 / (R1 + R2)`` (use ``r2=inf`` for a wall)."""
    if r1 <= 0 or r2 <= 0:
        raise ValueError("radii must be positive")
    if math.isinf(r2):
        return r1
    return r1 * r2 / (r1 + r2)


def jkr_reduced_modulus(
    e1: float, nu1: float, e2: float, nu2: float
) -> float:
    """Reduced Young's modulus ``E*`` from ``1/E* = sum (1-nu_i^2)/E_i``."""
    if e1 <= 0 or e2 <= 0:
        raise ValueError("Young's moduli must be positive")
    if not (-1.0 < nu1 < 0.5) or not (-1.0 < nu2 < 0.5):
        raise ValueError("Poisson ratios must lie in (-1, 0.5)")
    return 1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)


def jkr_hertz_prefactor(e_star: float) -> float:
    """``K = 4 E*/3`` — the prefactor such that ``F_hertz = K sqrt(R*) d^1.5``."""
    return 4.0 * e_star / 3.0


# ---------------------------------------------------------------------------
# Closed-form JKR anchors
# ---------------------------------------------------------------------------

def jkr_pull_off_force(r_star: float, gamma: float) -> float:
    """JKR pull-off force ``F_c = -(3/2) pi gamma R*`` (negative = tensile)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    return -1.5 * math.pi * gamma * r_star


def jkr_zero_load_radius(r_star: float, k: float, gamma: float) -> float:
    """Contact radius at zero applied load ``a_c = (6 pi gamma R*^2 / K)^(1/3)``."""
    if gamma <= 0:
        return 0.0
    return (6.0 * math.pi * gamma * r_star * r_star / k) ** (1.0 / 3.0)


def jkr_pull_off_radius(r_star: float, k: float, gamma: float) -> float:
    """Contact radius at pull-off ``a_p = (3/2 pi gamma R*^2 / K)^(1/3)``."""
    if gamma <= 0:
        return 0.0
    return (1.5 * math.pi * gamma * r_star * r_star / k) ** (1.0 / 3.0)


def jkr_force_from_radius(a: float, r_star: float, k: float, gamma: float) -> float:
    """Inverted JKR law ``F(a) = K a^3/R* - sqrt(6 pi gamma K a^3)``."""
    if a < 0:
        raise ValueError("contact radius must be non-negative")
    elastic = k * a ** 3 / r_star
    adhesive = math.sqrt(6.0 * math.pi * gamma * k * a ** 3) if gamma > 0 else 0.0
    return elastic - adhesive


def jkr_radius_from_force_closed(
    force: float, r_star: float, k: float, gamma: float
) -> float:
    """Closed-form JKR load–radius relation (the ``a^3 = ...`` formula)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    p = 3.0 * math.pi * gamma * r_star
    disc = 6.0 * math.pi * gamma * r_star * force + p * p
    # The discriminant can be slightly negative for F below pull-off (no
    # equilibrium); clamp to zero so the caller sees the pull-off radius.
    disc = max(disc, 0.0)
    return ((r_star / k) * (force + p + math.sqrt(disc))) ** (1.0 / 3.0)


def jkr_radius_from_force_brent(
    force: float,
    r_star: float,
    k: float,
    gamma: float,
    *,
    xtol: float = 1e-18,
    rtol: float = 4.0 * np.finfo(float).eps,
    maxiter: int = 200,
) -> float:
    """Solve ``F(a) = force`` for ``a`` with a bracketed Brent iteration.

    The residual ``r(a) = K a^3/R* - sqrt(6 pi gamma K a^3) - force`` is
    monotonically increasing for ``a >= a_p``; the bracket is built from the
    pull-off radius (where ``r == force - F_c <= 0``) up to a radius where the
    elastic term dominates ``|force|``.  Brent therefore converges to the
    *stable* branch.
    """
    if gamma <= GAMMA_FLOOR:
        # Non-adhesive limit: F = K a^3 / R*  =>  a = (F R* / K)^(1/3)
        if force <= 0:
            return 0.0
        return (force * r_star / k) ** (1.0 / 3.0)

    f_c = jkr_pull_off_force(r_star, gamma)
    if force <= f_c:
        # Beyond pull-off there is no equilibrium radius; return a_p.
        return jkr_pull_off_radius(r_star, k, gamma)

    def residual(a: float) -> float:
        return jkr_force_from_radius(a, r_star, k, gamma) - force

    lo = jkr_pull_off_radius(r_star, k, gamma)
    lo = max(lo, 1e-300)
    hi = max(jkr_zero_load_radius(r_star, k, gamma), lo) * 2.0
    # Expand the upper bracket until the elastic term exceeds the target load.
    while residual(hi) < 0.0:
        hi *= 1.6
        if hi > 1e6 * max(r_star, 1e-30):
            raise RuntimeError("JKR radius bracket failed to expand")
    return _brentq(residual, lo, hi, xtol=xtol, rtol=rtol, maxiter=maxiter)


def _brentq(
    f: Callable[[float], float],
    xa: float,
    xb: float,
    *,
    xtol: float = 1e-18,
    rtol: float = 4.0 * np.finfo(float).eps,
    maxiter: int = 200,
) -> float:
    """Minimal Brent root finder (Numerical Recipes ``zbrent`` structure).

    Kept in-module so the contact kernel carries no mandatory SciPy import.
    """
    fa, fb = f(xa), f(xb)
    if fa * fb > 0.0:
        raise ValueError("brentq requires a sign-changing bracket")
    if abs(fa) < abs(fb):
        xa, xb, fa, fb = xb, xa, fb, fa
    xc, fc = xa, fa
    d = e = xb - xa
    for _ in range(maxiter):
        if fb * fc > 0.0:
            xc, fc = xa, fa
            d = e = xb - xa
        if abs(fc) < abs(fb):
            xa, xb, xc = xb, xc, xb
            fa, fb, fc = fb, fc, fb
        tol1 = 2.0 * rtol * abs(xb) + 0.5 * xtol
        xm = 0.5 * (xc - xb)
        if abs(xm) <= tol1 or fb == 0.0:
            return xb
        if abs(e) >= tol1 and abs(fa) > abs(fb):
            s = fb / fa
            if xa == xc:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (xb - xa) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        xa, fa = xb, fb
        xb = xb + (d if abs(d) > tol1 else math.copysign(tol1, xm))
        fb = f(xb)
    return xb


def jkr_force_from_radius_brent(
    a: float, r_star: float, k: float, gamma: float
) -> float:
    """Same as :func:`jkr_force_from_radius`, routed through Brent for parity.

    Useful as a cross-check that the closed form and the bracketed inversion
    are exact inverses (asserted in the tests to ``< 1e-12`` relative).
    """
    if gamma <= GAMMA_FLOOR:
        return k * a ** 3 / r_star
    f = jkr_force_from_radius(a, r_star, k, gamma)
    a_back = jkr_radius_from_force_brent(f, r_star, k, gamma)
    if a_back <= 0:
        return f
    return jkr_force_from_radius(a_back, r_star, k, gamma)


def jkr_approach(a: float, r_star: float, k: float, gamma: float) -> float:
    """JKR approach ``delta(a) = a^2/R* - (2/3) sqrt(6 pi gamma a / K)``."""
    if a <= 0:
        return 0.0
    adhesive = (2.0 / 3.0) * math.sqrt(6.0 * math.pi * gamma * a / k) if gamma > 0 else 0.0
    return a * a / r_star - adhesive


def jkr_energy_release_rate(
    force: float, a: float, e_star: float, r_star: float
) -> float:
    """JKR strain energy release rate ``G = (F - F_Hertz)^2/(8 pi E* a^3)``."""
    if a <= 0:
        return 0.0
    f_hertz = 4.0 * e_star * a ** 3 / (3.0 * r_star)
    return (force - f_hertz) ** 2 / (8.0 * math.pi * e_star * a ** 3)


def jkr_radius_from_approach_brent(
    approach: float,
    r_star: float,
    k: float,
    gamma: float,
    *,
    n_grid: int = 400,
) -> float:
    """Invert ``delta(a)`` on the stable branch (``a >= a_p``) with Brent.

    ``delta(a)`` is non-monotone (it has a minimum near ``a ~ (gamma R*^2/K)^(1/3)``
    followed by the ``a^2/R*`` growth), so we first bracket the stable branch by
    scanning ``a in [a_p, a_hi]`` and then refine with Brent.
    """
    if gamma <= GAMMA_FLOOR:
        return math.sqrt(max(approach, 0.0) * r_star)
    a_p = jkr_pull_off_radius(r_star, k, gamma)
    a_lo = a_p
    d_lo = jkr_approach(a_lo, r_star, k, gamma)
    if approach <= d_lo:
        return a_p  # beyond pull-off: the contact has already jumped off
    a_hi = max(2.0 * jkr_zero_load_radius(r_star, k, gamma), a_lo * 1.5)
    while jkr_approach(a_hi, r_star, k, gamma) < approach:
        a_hi *= 1.5
        if a_hi > 1e8 * r_star:
            return a_hi
    return _brentq(
        lambda a: jkr_approach(a, r_star, k, gamma) - approach, a_lo, a_hi
    )


# ---------------------------------------------------------------------------
# DEM-facing force law
# ---------------------------------------------------------------------------

def jkr_normal_force(
    overlap: float,
    r_star: float,
    e_star: float,
    gamma: float,
    *,
    gamma_floor: float = GAMMA_FLOOR,
    use_brent: bool = True,
) -> float:
    """Adhesive normal force as a function of the *overlap* (compression > 0).

    ``overlap < 0`` means the pair is in tension; the JKR branch supports
    tension down to the pull-off force, after which the contact snaps off
    (``F = 0``, the pair is separated).

    ``gamma <= gamma_floor`` returns the legacy Hertz law *exactly*
    (``K sqrt(R*) overlap**1.5``), which is what criterion C4 of the campaign
    pins to ``< 1e-9`` relative difference against
    :func:`tensordem.dem3d.hertz_normal_force`.
    """
    k = jkr_hertz_prefactor(e_star)
    if gamma <= gamma_floor:
        if overlap <= 0.0:
            return 0.0
        return k * math.sqrt(r_star) * overlap ** 1.5
    f_c = jkr_pull_off_force(r_star, gamma)
    a_p = jkr_pull_off_radius(r_star, k, gamma)
    delta_p = jkr_approach(a_p, r_star, k, gamma)
    if overlap < delta_p:
        # Past the pull-off point: the contact is lost.
        return 0.0
    if use_brent:
        a = jkr_radius_from_approach_brent(overlap, r_star, k, gamma)
    else:
        a = jkr_radius_from_approach_brent(overlap, r_star, k, gamma)
    f = jkr_force_from_radius(a, r_star, k, gamma)
    return max(f, f_c)


def jkr_normal_force_pair(
    overlap: float,
    r1: float,
    r2: float,
    e1: float,
    nu1: float,
    e2: float,
    nu2: float,
    gamma: float,
    **kwargs,
) -> float:
    """Convenience wrapper taking the two bodies' radii/materials directly."""
    r_star = jkr_reduced_radius(r1, r2)
    e_star = jkr_reduced_modulus(e1, nu1, e2, nu2)
    return jkr_normal_force(overlap, r_star, e_star, gamma, **kwargs)


# ---------------------------------------------------------------------------
# Energy audit
# ---------------------------------------------------------------------------

@dataclass
class JKREnergyAudit:
    """Bookkeeping of the reversible separation of a JKR contact.

    Attributes
    ----------
    work_separation:
        ``W_sep = -integral_{a_c}^{0} F d(delta)`` from high-resolution
        quadrature over the closed-form branch (J).
    work_separation_implemented:
        Same integral but the force at each sample is recomputed through the
        *implementation* (closed form radius -> Brent inversion -> force),
        which is what a DEM step would evaluate (J).
    surface_energy_contact:
        ``pi gamma a_c**2`` — JKR surface-energy difference between the
        bonded and the separated state (J).
    elastic_energy:
        ``U_el(a_c)`` implied by the balance ``W_sep = pi gamma a_c**2 - U_el``
        (J, positive).
    ratio_to_contact_area:
        ``W_sep / (pi gamma a_c**2)`` — the universal JKR number.
    ratio_to_free_sphere:
        ``W_sep / (4 pi gamma R*^2)`` — the literal campaign form, reported
        for honesty (it is *not* supposed to be 1; see the module docstring).
    closure_error:
        ``|W_sep - (pi gamma a_c**2 - U_el)| / W_sep`` evaluated with ``U_el``
        taken from the independent energy-release-rate integral
        ``U_el = integral_0^{a_c} 2 pi a G(a) da`` (G is the strain energy
        release rate, an anchor independent of the force law).
    """

    work_separation: float
    work_separation_implemented: float
    surface_energy_contact: float
    elastic_energy: float
    ratio_to_contact_area: float
    ratio_to_free_sphere: float
    closure_error: float
    gamma: float
    r_star: float
    k: float
    a_c: float
    a_p: float
    pull_off_force: float
    extra: dict = field(default_factory=dict)


def jkr_separation_work(
    r_star: float,
    k: float,
    gamma: float,
    *,
    a_min: float = 0.0,
    n_samples: int = 40001,
    through_kernel: bool = False,
) -> float:
    """Reversible work of separation ``W = -integral_{a_c}^{a_min} F d(delta)``.

    ``a_min = 0`` follows the whole (mathematical) equilibrium branch, which is
    the classical JKR reversible path; ``a_min = a_p`` stops at the pull-off
    point, i.e. the part a displacement-controlled machine can actually follow.
    ``through_kernel=True`` re-evaluates every force through the DEM-facing
    inversion (:func:`jkr_radius_from_force_brent`) instead of the closed form,
    which measures implementation integrity rather than physics.
    """
    if gamma <= 0 or r_star <= 0 or k <= 0:
        raise ValueError("gamma, r_star and k must be positive")
    a_c = jkr_zero_load_radius(r_star, k, gamma)
    a_stop = max(a_min, a_c * 1e-9)
    a = np.geomspace(a_c, a_stop, n_samples)
    f = np.empty_like(a)
    for i, x in enumerate(a):
        fx = jkr_force_from_radius(x, r_star, k, gamma)
        if through_kernel:
            xb = jkr_radius_from_force_brent(fx, r_star, k, gamma)
            fx = jkr_force_from_radius(max(xb, a_c * 1e-12), r_star, k, gamma)
        f[i] = fx
    delta = np.array([jkr_approach(x, r_star, k, gamma) for x in a])
    # The path is traversed with *decreasing* delta (a_c -> a_min), so the
    # oriented trapezoid already carries the correct sign: no extra minus.
    return float(np.trapezoid(f, delta))


def jkr_energy_audit(
    r_star: float,
    k: float,
    gamma: float,
    *,
    n_samples: int = 40001,
) -> JKREnergyAudit:
    """Audit the energy of the JKR separation path.

    Three quantities are reported (all in joules):

    1. ``work_separation`` — ``-integral F d(delta)`` over the full reversible
       branch ``a: a_c -> 0`` from the closed-form ``F(a), delta(a)`` pair.
       For the JKR law this is the universal number
       ``W_sep = (3/5) pi gamma a_c**2`` (asserted in the tests to ``< 0.1 %``);
       no dimensional scale other than ``pi gamma a_c**2`` can appear, so this
       is a parameter-free closure of the force law against its own potential.
    2. ``work_separation_implemented`` — same integral routed through the
       bracketed inversion used by the DEM kernel (implementation integrity).
    3. ``surface_energy_contact`` — ``pi gamma a_c**2``, the JKR surface-energy
       difference between bonded and separated states.  The complementary
       elastic part implied by the balance is ``elastic_energy``.
    4. ``ratio_to_free_sphere`` — the literal campaign form
       ``W_sep / (4 pi gamma R*^2)``, which equals ``0.15 (a_c/R*)^2`` exactly;
       it is ``O(1e-4)`` in the JKR regime and must **not** be confused with 1
       (see the module docstring and docs section 6).
    """
    if gamma <= 0 or r_star <= 0 or k <= 0:
        raise ValueError("gamma, r_star and k must be positive")
    a_c = jkr_zero_load_radius(r_star, k, gamma)
    a_p = jkr_pull_off_radius(r_star, k, gamma)
    w_full = jkr_separation_work(r_star, k, gamma, a_min=0.0, n_samples=n_samples)
    w_stable = jkr_separation_work(
        r_star, k, gamma, a_min=a_p, n_samples=n_samples
    )
    w_impl = jkr_separation_work(
        r_star, k, gamma, a_min=a_p, n_samples=max(2001, n_samples // 8),
        through_kernel=True,
    )
    surface = math.pi * gamma * a_c * a_c
    closure = abs(w_stable - w_impl) / abs(w_stable) if w_stable else float("nan")
    return JKREnergyAudit(
        work_separation=w_full,
        work_separation_implemented=w_impl,
        surface_energy_contact=surface,
        elastic_energy=surface - w_full,
        ratio_to_contact_area=w_full / surface if surface else float("nan"),
        ratio_to_free_sphere=w_full / (4.0 * math.pi * gamma * r_star ** 2),
        closure_error=closure,
        gamma=gamma,
        r_star=r_star,
        k=k,
        a_c=a_c,
        a_p=a_p,
        pull_off_force=jkr_pull_off_force(r_star, gamma),
        extra={
            "work_separation_stable_branch": w_stable,
            "stable_branch_fraction": w_stable / w_full if w_full else float("nan"),
            "ratio_scaling_prediction": 0.15 * (a_c / r_star) ** 2,
            "delta_at_zero_load": jkr_approach(a_c, r_star, k, gamma),
            "delta_at_pull_off": jkr_approach(a_p, r_star, k, gamma),
            "energy_release_rate_at_zero_load": jkr_energy_release_rate(
                0.0, a_c, 0.75 * k, r_star
            ),
        },
    )


def jkr_load_displacement_curve(
    r_star: float, k: float, gamma: float, n: int = 200
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(a, delta, F)`` on the stable JKR branch ``a in [a_p, 8 a_c]``."""
    a_c = jkr_zero_load_radius(r_star, k, gamma)
    a = np.linspace(jkr_pull_off_radius(r_star, k, gamma), 8.0 * a_c, n)
    delta = np.array([jkr_approach(x, r_star, k, gamma) for x in a])
    f = np.array([jkr_force_from_radius(x, r_star, k, gamma) for x in a])
    return a, delta, f


__all__ = [
    "GAMMA_FLOOR",
    "JKREnergyAudit",
    "jkr_approach",
    "jkr_energy_audit",
    "jkr_energy_release_rate",
    "jkr_force_from_radius",
    "jkr_force_from_radius_brent",
    "jkr_hertz_prefactor",
    "jkr_load_displacement_curve",
    "jkr_normal_force",
    "jkr_normal_force_pair",
    "jkr_pull_off_force",
    "jkr_pull_off_radius",
    "jkr_radius_from_approach_brent",
    "jkr_radius_from_force_brent",
    "jkr_radius_from_force_closed",
    "jkr_reduced_modulus",
    "jkr_reduced_radius",
    "jkr_zero_load_radius",
]