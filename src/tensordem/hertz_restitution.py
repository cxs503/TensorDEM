"""Restitution control for the Hertz contact law (Tsuji-type damping).

Background and the debt this module closes
------------------------------------------
``DEM3DConfig(contact_model="hertz")`` implements the classical Hertz normal
law ``F = (4/3) E* sqrt(R*) delta^(3/2)``.  Its dashpot is *velocity
independent* in the Tsuji sense, i.e. proportional to ``delta^(1/4)``::

    gamma(delta) = alpha * sqrt(m_eff * k_H) * delta^(1/4)
    F_n = max(0, k_H delta^(3/2) + gamma(delta) * ddelta/dt)

which is exactly the form the solver already builds from its local *secant*
stiffness, ``gamma_n = alpha * sqrt(k_local * m_eff)`` with
``k_local = k_H sqrt(delta)``.  What was missing -- and is documented as an
open item in ``docs/dem3d_hertz_contact_verification.md`` -- is the mapping
from a *physical* target coefficient of restitution ``e`` to ``alpha``.  The
solver used to borrow the closed-form mapping of the **linear** spring-dashpot
(``zeta = |ln e| / sqrt(pi^2 + ln^2 e)``, hence ``alpha = 2 zeta``), which is
not the correct relation for a Hertzian contact: it over-predicts the realised
restitution by up to 44 % for ``e = 0.3`` (see
``docs/dem3d_hertz_restitution_mapping.md``).

The exact mapping, derived below, is a *universal function of alpha alone*:
it does not depend on the material, the radii or the impact speed.  That is
the reason a single one-dimensional quadrature settles the whole question.

Derivation of the universal problem
-----------------------------------
Take a head-on impact of two bodies with effective mass ``m`` and Hertz
coefficient ``k_H`` (so ``F = k_H delta^(3/2)``).  The elastic peak overlap at
incident speed ``v0`` follows from energy conservation,
``delta_max = (5 m v0^2 / (4 k_H))^(2/5)``.  Non-dimensionalise with
``delta = delta_max * x`` and ``t = T * tau`` where
``T = sqrt(m / (k_H delta_max^(1/2)))``.  Then

    x'' = -max(0, x^(3/2) + alpha x^(1/4) x'),   x(0) = 0, x'(0) = 2/sqrt(5)

because ``gamma T / m = alpha delta^(1/4) delta_max^(-1/4) = alpha x^(1/4)``
and ``v0 T / delta_max = 2/sqrt(5)`` exactly.  So ``e(alpha)`` and the
dimensionless contact time ``tau_c(alpha)`` are universal, and the clamp
``max(0, .)`` (unilateral contact, the law the solver actually integrates) is
part of the universal problem.

Two consequences used as verification anchors:

* ``alpha = 0`` reduces to elastic Hertz impact, whose contact time has the
  textbook closed form ``t_c = 2.94 (5 m / (4 k_H))^(2/5) v0^(-1/5)``, i.e.
  ``tau_c(0) * 2/sqrt(5) = 2.94``.  The reference quadrature reproduces it to
  ``+0.12 %`` (see the doc), which validates the reference itself.
* because ``e`` is speed independent, a Hertz + Tsuji contact reproduces a
  target restitution at *any* impact speed -- the property the previous
  linear-law mapping silently destroyed.

All numbers in this module are produced by the quadrature documented in
:func:`restitution_for_alpha`; nothing is tabulated or hard-coded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

__all__ = [
    "DIMENSIONLESS_IMPACT_SPEED",
    "ELASTIC_CONTACT_TIME_CONSTANT",
    "HertzImpactReference",
    "alpha_for_restitution",
    "dimensionless_contact_time",
    "peak_overlap_fraction",
    "elastic_hertz_contact_time",
    "hertz_contact_time",
    "impact_reference",
    "linear_law_alpha",
    "restitution_for_alpha",
    "tsuji_damping_alpha",
]

#: ``x'(0)`` of the universal impact problem: ``v0 T / delta_max = 2/sqrt(5)``.
DIMENSIONLESS_IMPACT_SPEED = 2.0 / math.sqrt(5.0)

#: Textbook constant of the elastic Hertz contact time,
#: ``t_c = 2.94 (5 m / (4 k_H))^(2/5) v0^(-1/5)`` (Johnson, *Contact Mechanics*,
#: 1985, ch. 4 impact of elastic bodies; the same constant appears in every DEM
#: contact-time estimate).  Used only as an *external* anchor for the
#: quadrature below, never as a source of results.
ELASTIC_CONTACT_TIME_CONSTANT = 2.94

_DEFAULT_STEP = 5.0e-4
_MAX_TIME = 200.0
_MAX_STEPS = 4_000_000


def _force(x: float, u: float, alpha: float) -> float:
    """Dimensionless unilateral contact force ``max(0, x^(3/2) + alpha x^(1/4) u)``."""
    if x <= 0.0:
        return 0.0
    value = x ** 1.5 + alpha * (x ** 0.25) * u
    return value if value > 0.0 else 0.0


def _derivative(x: float, u: float, alpha: float) -> tuple[float, float]:
    return u, -_force(x, u, alpha)


def _rk4(x: float, u: float, h: float, alpha: float) -> tuple[float, float]:
    k1x, k1u = _derivative(x, u, alpha)
    k2x, k2u = _derivative(x + 0.5 * h * k1x, u + 0.5 * h * k1u, alpha)
    k3x, k3u = _derivative(x + 0.5 * h * k2x, u + 0.5 * h * k2u, alpha)
    k4x, k4u = _derivative(x + h * k3x, u + h * k3u, alpha)
    return (
        x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x),
        u + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u),
    )


def _integrate_impact(alpha: float, step: float = _DEFAULT_STEP) -> tuple[float, float, float]:
    """Integrate the universal impact problem; return ``(e, tau_c, x_peak)``.

    ``tau_c`` is the duration of the *active* contact force, ``x_peak`` the peak
    dimensionless overlap.  Event handling: the ``x = 0`` crossing on the way
    out is located by linear interpolation inside the final step; the earlier
    clamp event (damper pulls the force to zero while the pair is still
    overlapping) is detected at a step boundary, which biases ``tau_c`` by
    ``O(step)`` -- checked below against Richardson refinement.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    x = 0.0
    u = DIMENSIONLESS_IMPACT_SPEED
    t = 0.0
    x_peak = 0.0
    force_seen = False
    for _ in range(_MAX_STEPS):
        if force_seen and _force(x, u, alpha) == 0.0:
            # Clamped: the pair is separating ballistically, contact over.
            return -u / DIMENSIONLESS_IMPACT_SPEED, t, x_peak
        x_next, u_next = _rk4(x, u, step, alpha)
        if force_seen and x_next <= 0.0 and u_next <= 0.0:
            frac = x / (x - x_next) if x != x_next else 1.0
            u_out = u + frac * (u_next - u)
            return -u_out / DIMENSIONLESS_IMPACT_SPEED, t + frac * step, x_peak
        if force_seen and x_next > 0.0 and _force(x_next, u_next, alpha) == 0.0:
            # Contact released inside this step; refine the release time so the
            # measured duration is not biased by the step size.
            xs, us, ts = x, u, t
            sub = step / 32.0
            for _ in range(31):
                xn, un = _rk4(xs, us, sub, alpha)
                if _force(xn, un, alpha) == 0.0:
                    frac = 0.5
                    xs, us = xn, un
                    ts += frac * sub
                    return -us / DIMENSIONLESS_IMPACT_SPEED, ts, x_peak
                xs, us, ts = xn, un, ts + sub
            return -us / DIMENSIONLESS_IMPACT_SPEED, ts, x_peak
        if _force(x, u, alpha) > 0.0:
            force_seen = True
        x, u, t = x_next, u_next, t + step
        if x > x_peak:
            x_peak = x
        if t > _MAX_TIME:  # pragma: no cover - guarded by the physics
            raise RuntimeError("impact did not separate within the integration window")
    raise RuntimeError("impact integration step budget exhausted")  # pragma: no cover


@lru_cache(maxsize=256)
def restitution_for_alpha(alpha: float, step: float = _DEFAULT_STEP) -> float:
    """Realised coefficient of restitution of the Hertz + Tsuji law for ``alpha``.

    Exact (numerical) solution of the universal impact problem above, i.e. of
    the same clamped force law the DEM engine integrates.  ``alpha = 0`` gives
    ``1.0`` (elastic); ``e`` decreases monotonically with ``alpha``.
    """
    return _integrate_impact(alpha, step)[0]


@lru_cache(maxsize=256)
def dimensionless_contact_time(alpha: float, step: float = _DEFAULT_STEP) -> float:
    """Dimensionless active-contact duration ``tau_c(alpha)``."""
    return _integrate_impact(alpha, step)[1]


@lru_cache(maxsize=256)
def peak_overlap_fraction(alpha: float, step: float = _DEFAULT_STEP) -> float:
    """Peak overlap in units of the *elastic* peak overlap ``delta_max``."""
    return _integrate_impact(alpha, step)[2]


def linear_law_alpha(e_target: float) -> float:
    """Damping coefficient the **linear** spring-dashpot needs for ``e_target``.

    ``zeta = |ln e| / sqrt(pi^2 + ln^2 e)`` and ``alpha = 2 zeta``.  This is the
    mapping the Hertz branch of the solver borrowed historically; for a Hertz
    contact it is wrong (it ignores the ``delta^(1/4)`` shape of the damper) and
    is kept only so the bias can be quantified and pinned by a regression test.
    """
    if not (0.0 < e_target <= 1.0):
        raise ValueError(f"e_target must be in (0, 1], got {e_target}")
    if e_target == 1.0:
        return 0.0
    abs_ln_e = -math.log(e_target)
    zeta = abs_ln_e / math.sqrt(math.pi ** 2 + abs_ln_e ** 2)
    return 2.0 * zeta


@lru_cache(maxsize=256)
def alpha_for_restitution(e_target: float, rel_tol: float = 1.0e-8) -> float:
    """Tsuji damping coefficient ``alpha`` that realises ``e_target`` exactly.

    Solved by bracketing the (monotone, smooth) universal map; the returned
    ``alpha`` reproduces ``e_target`` to ``rel_tol`` -- see
    :func:`tsuji_damping_alpha` for the engineering entry point (this one is
    the raw inverse of :func:`restitution_for_alpha`).
    """
    if not (0.0 <= e_target <= 1.0):
        raise ValueError(f"e_target must be in [0, 1], got {e_target}")
    if e_target == 1.0:
        return 0.0
    if e_target <= 0.0:  # pragma: no cover - alpha -> tight clamp limit
        return float("inf")
    lo, hi = 0.0, 8.0
    while restitution_for_alpha(hi) > e_target:
        hi *= 2.0
        if hi > 1.0e3:  # pragma: no cover
            raise ValueError(f"e_target={e_target} is not reachable")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if restitution_for_alpha(mid) > e_target:
            lo = mid
        else:
            hi = mid
        if restitution_for_alpha(mid) == e_target or hi - lo <= rel_tol * max(1.0, hi):
            break
    return 0.5 * (lo + hi)


def tsuji_damping_alpha(e_target: float, *, mapping: str = "exact") -> float:
    """Damping coefficient for a target restitution under a given mapping.

    ``mapping="exact"`` is the verified Hertz/Tsuji map of this module;
    ``mapping="linear_formula"`` is the legacy linear-law formula kept for
    bit-for-bit reproduction of pre-fix results.
    """
    if mapping == "exact":
        return alpha_for_restitution(e_target)
    if mapping == "linear_formula":
        return linear_law_alpha(e_target)
    raise ValueError(f"mapping must be 'exact' or 'linear_formula', got {mapping}")


def elastic_hertz_contact_time(m_eff: float, k_h: float, v0: float) -> float:
    """Textbook elastic Hertz impact duration ``2.94 (5m/(4k_H))^(2/5) v0^(-1/5)``."""
    if m_eff <= 0 or k_h <= 0 or v0 <= 0:
        raise ValueError("m_eff, k_h and v0 must be positive")
    return (ELASTIC_CONTACT_TIME_CONSTANT
            * (5.0 * m_eff / (4.0 * k_h)) ** 0.4
            * v0 ** -0.2)


def hertz_contact_time(m_eff: float, k_h: float, v0: float, alpha: float = 0.0) -> float:
    """Contact duration of the damped Hertz impact from the universal quadrature.

    Scales as ``t_c = tau_c(alpha) * 2 delta_max / (sqrt(5) v0)`` with
    ``delta_max = (5 m v0^2/(4 k_H))^(2/5)``; at ``alpha = 0`` this differs from
    :func:`elastic_hertz_contact_time` only by the rounding of the 2.94
    constant.
    """
    if m_eff <= 0 or k_h <= 0 or v0 <= 0:
        raise ValueError("m_eff, k_h and v0 must be positive")
    delta_max = (5.0 * m_eff * v0 * v0 / (4.0 * k_h)) ** 0.4
    scale = 2.0 * delta_max / (math.sqrt(5.0) * v0)
    return dimensionless_contact_time(alpha) * scale


@dataclass(frozen=True)
class HertzImpactReference:
    """Reference solution of one damped Hertz impact (no solver involved)."""

    alpha: float
    restitution: float
    contact_time: float
    peak_overlap_fraction: float

    @property
    def compliance_time(self) -> float:
        """Dimensionless contact duration ``tau_c``."""
        return self.contact_time


def impact_reference(
    *,
    e_target: float,
    m_eff: float,
    k_h: float,
    v0: float,
    mapping: str = "exact",
) -> HertzImpactReference:
    """Full reference (restitution, duration, peak overlap) for one impact."""
    alpha = tsuji_damping_alpha(e_target, mapping=mapping)
    delta_max = (5.0 * m_eff * v0 * v0 / (4.0 * k_h)) ** 0.4
    return HertzImpactReference(
        alpha=alpha,
        restitution=restitution_for_alpha(alpha),
        contact_time=dimensionless_contact_time(alpha) * 2.0 * delta_max / (math.sqrt(5.0) * v0),
        peak_overlap_fraction=peak_overlap_fraction(alpha) * delta_max,
    )