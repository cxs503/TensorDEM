"""Particle adhesion contacts for TensorDEM: JKR normal law + pendular liquid bridge.

Two physical ingredients are implemented here, both as *law* modules (no
integrator is embedded; ``adhesive_pile2d`` shows one way to drive them):

1. **JKR adhesive elastic contact** (Johnson, Kendall & Roberts 1971).

   Contact radius / load / overlap relations for an adhesive Hertzian pair
   (effective radius ``R*``, reduced modulus ``E*``, surface energy ``gamma``)::

       a^3 = (3 R* / 4 E*) [ F + 3 pi gamma R* + sqrt(6 pi gamma R* F + (3 pi gamma R*)^2) ]
       F(a) = 4 E* a^3 / (3 R*) - sqrt(8 pi gamma E* a^3)
       delta(a) = a^2 / R* - sqrt(2 pi gamma a / E*)

   The **pull-off (detachment) force** is the theory anchor of this module::

       F_pull = (3/2) pi gamma R*

   For a sphere of radius ``R`` against a rigid plane ``R* = R`` so
   ``F_pull = 1.5 pi gamma R``; for two identical spheres (R1 = R2 = R) the
   effective radius is ``R* = R/2`` and the same expression gives
   ``F_pull = 0.75 pi gamma R``.  Both are the *same* ``1.5 pi gamma R*`` law;
   the convention is spelled out in ``docs/adhesion_jkr_capillary.md`` §1 so
   that the numeric checks are unambiguous.

   ``gamma -> 0`` degenerates to Hertz::

       F_Hertz = (4/3) E* sqrt(R*) delta^(3/2),   a = sqrt(R* delta)

2. **Pendular liquid bridge / capillary force** by numerical solution of the
   axisymmetric Young-Laplace (Laplace) equation in the meridian plane.

   The free surface of a bridge between two equal spheres of radius R with a
   surface gap ``2 h`` is integrated from the neck (waist) at ``z = 0`` up to the
   contact line on the upper sphere.  With arc length ``s``, radius ``r(s)``,
   height ``z(s)`` and tangent angle ``phi(s)`` (measured from the r axis)::

       dr/ds = cos(phi),  dz/ds = sin(phi),  dphi/ds = -(dP/gamma) - sin(phi)/r

   where ``dP = p_gas - p_liquid >= 0`` is the capillary suction.  The two
   unknowns (``dP`` and the neck radius ``r_n``) are fixed by two conditions at
   the contact line ``(r_c, z_c) = (R sin(beta), h + R (1 - cos(beta)))``::

       r(z_c) = r_c        and        phi(z_c) = beta + theta

   with ``beta`` the filling angle and ``theta`` the contact angle.

   Two *independent* expressions for the attractive axial capillary force are
   evaluated from the same converged profile::

       F_neck    = 2 pi gamma r_n + dP pi r_n^2               (cut at the waist)
       F_contact = 2 pi gamma r_c sin(phi_c) + dP pi r_c^2    (force on the sphere)

   Their agreement is a strong internal validation of the shooting solution.
   The classical small-bridge limit is::

       F_cap -> 2 pi gamma R cos(theta)      (bridge volume -> 0, gap -> 0)

   which is the analytic anchor used by the test suite.

3. **Interface with the superquadric contact (P96-N3)**: ``adhesive_contact_force``
   consumes a ``SuperquadricContactResult``-compatible object (``overlap`` and
   ``effective_radius``) so the JKR / capillary terms stack on top of the true
   shape contact of :mod:`tensordem.superquadric` without modifying it.

Sign conventions / units
------------------------
Forces named ``F_pull`` / ``F_cap`` are *magnitudes of attraction* (positive).
The JKR normal force returned by :func:`jkr_normal_force` is signed (positive =
repulsion, negative = tension).  All quantities are SI (m, N, Pa, J/m^2).

References
----------
- Johnson, Kendall, Roberts, Proc. R. Soc. A 324 (1971) 301.
- Maugis, "Contact, Adhesion and Rupture of Elastic Solids", Springer 2000, ch. 4.
- Lian, Thornton, Adams, J. Colloid Interface Sci. 161 (1993) 138.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

__all__ = [
    "JKRState",
    "LiquidBridgeResult",
    "reduced_modulus",
    "jkr_pull_off_force",
    "jkr_force_from_contact_radius",
    "jkr_overlap_from_contact_radius",
    "jkr_contact_radius_from_force",
    "jkr_pull_off_numerical",
    "jkr_hertz_relative_error",
    "jkr_normal_force",
    "hertz_normal_force",
    "capillary_force_analytic",
    "solve_liquid_bridge",
    "capillary_force_laplace",
    "capillary_force_ratio_table",
    "adhesive_contact_force",
    "ADHESION_EVIDENCE",
]

DTYPE = torch.float64

ADHESION_EVIDENCE = {
    "jkr": "REAL: JKR closed-form force/displacement relations + numeric shooting/bisection",
    "capillary": "REAL: axisymmetric Young-Laplace ODE shooting (Laplace equation solved numerically)",
    "hertz_limit": "REAL: analytic gamma->0 degeneration of the same code path",
    "repose": "REAL: 2D DEM pile simulation (tensordem.adhesive_pile2d) with the above laws",
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def reduced_modulus(
    young_modulus: float,
    poisson_ratio: float,
    *,
    rigid_counterpart: bool = False,
    young_modulus_2: Optional[float] = None,
    poisson_ratio_2: Optional[float] = None,
) -> float:
    """Reduced Young's modulus ``E*``.

    ``1/E* = (1 - nu_1^2)/E_1 + (1 - nu_2^2)/E_2``.  With ``rigid_counterpart``
    the second term vanishes; otherwise pass the second material.

    Returns:
        E* in Pa.
    """
    if not (math.isfinite(young_modulus) and young_modulus > 0.0):
        raise ValueError("young_modulus must be finite and positive")
    if not (math.isfinite(poisson_ratio) and 0.0 <= poisson_ratio < 0.5):
        raise ValueError("poisson_ratio must be in [0, 0.5)")
    inv = (1.0 - poisson_ratio ** 2) / young_modulus
    if not rigid_counterpart:
        if young_modulus_2 is None:
            young_modulus_2 = young_modulus
            poisson_ratio_2 = poisson_ratio if poisson_ratio_2 is None else poisson_ratio_2
        if poisson_ratio_2 is None:
            poisson_ratio_2 = poisson_ratio
        if not (math.isfinite(young_modulus_2) and young_modulus_2 > 0.0):
            raise ValueError("young_modulus_2 must be finite and positive")
        if not (math.isfinite(poisson_ratio_2) and 0.0 <= poisson_ratio_2 < 0.5):
            raise ValueError("poisson_ratio_2 must be in [0, 0.5)")
        inv += (1.0 - poisson_ratio_2 ** 2) / young_modulus_2
    return 1.0 / inv


def effective_radius(radius_a: float, radius_b: Optional[float] = None) -> float:
    """Effective radius ``R*`` with ``1/R* = 1/R_a + 1/R_b``.

    ``radius_b = None`` (or ``inf``) means a rigid plane, i.e. ``R* = R_a``.
    """
    if not (math.isfinite(radius_a) and radius_a > 0.0):
        raise ValueError("radius_a must be finite and positive")
    if radius_b is None or radius_b == math.inf:
        return float(radius_a)
    if not (math.isfinite(radius_b) and radius_b > 0.0):
        raise ValueError("radius_b must be finite and positive")
    return 1.0 / (1.0 / radius_a + 1.0 / radius_b)


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(DTYPE) if value.dtype != DTYPE else value
    return torch.tensor(float(value), dtype=DTYPE)


# ---------------------------------------------------------------------------
# JKR: closed-form relations
# ---------------------------------------------------------------------------

def jkr_pull_off_force(gamma: float, r_eff: float) -> float:
    """JKR pull-off (detachment) force magnitude ``F_pull = (3/2) pi gamma R*``.

    This is the theory anchor.  For a sphere-plane contact ``R* = R`` it equals
    ``1.5 pi gamma R``.

    Args:
        gamma: surface energy (work of adhesion / 2), J/m^2, >= 0.
        r_eff: effective radius R*, m.

    Returns:
        Pull-off force magnitude in N (>= 0).
    """
    if not (math.isfinite(gamma) and gamma >= 0.0):
        raise ValueError("gamma must be finite and nonnegative")
    if not (math.isfinite(r_eff) and r_eff > 0.0):
        raise ValueError("r_eff must be finite and positive")
    return 1.5 * math.pi * gamma * r_eff


def jkr_force_from_contact_radius(a, e_star: float, r_eff: float, gamma: float):
    """JKR normal force as a function of contact radius.

    ``F(a) = 4 E* a^3 / (3 R*) - sqrt(8 pi gamma E* a^3)``

    Signed: positive = repulsion, negative = tension.
    """
    a_t = _as_tensor(a).clamp_min(0.0)
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")
    rep = 4.0 * e_star * a_t ** 3 / (3.0 * r_eff)
    att = torch.sqrt(8.0 * math.pi * gamma * e_star * a_t ** 3)
    return rep - att


def jkr_overlap_from_contact_radius(a, e_star: float, r_eff: float, gamma: float):
    """JKR normal overlap ``delta(a) = a^2/R* - sqrt(2 pi gamma a / E*)``."""
    a_t = _as_tensor(a).clamp_min(0.0)
    elastic = a_t ** 2 / r_eff
    adhesive = torch.sqrt(2.0 * math.pi * gamma * a_t / e_star)
    return elastic - adhesive


def jkr_contact_radius_from_force(force, e_star: float, r_eff: float, gamma: float):
    """Invert the JKR load relation: contact radius for a given load F.

    ``a^3 = (3 R* / 4 E*) [ F + 3 pi gamma R* + sqrt(6 pi gamma R* F + (3 pi gamma R*)^2) ]``

    The radicand is degenerate (zero) exactly at the pull-off load
    ``F = -1.5 pi gamma R*``; for ``F < F_pull`` no equilibrium contact exists
    and ``nan`` is returned.
    """
    f_t = _as_tensor(force)
    jkr_scale = 3.0 * math.pi * gamma * r_eff
    disc = 6.0 * math.pi * gamma * r_eff * f_t + jkr_scale ** 2
    out = torch.full_like(f_t, float("nan"))
    ok = disc >= 0.0
    bracket = f_t + jkr_scale + torch.sqrt(torch.clamp(disc, min=0.0))
    cube = 3.0 * r_eff * bracket / (4.0 * e_star)
    out = torch.where(ok & (cube >= 0.0), torch.clamp(cube, min=0.0) ** (1.0 / 3.0), out)
    return out


def jkr_pull_off_numerical(
    e_star: float,
    r_eff: float,
    gamma: float,
    *,
    n_scan: int = 20001,
) -> dict[str, float]:
    """Pull-off force obtained *numerically* from the JKR branch.

    Independent of :func:`jkr_pull_off_force`: the contact radius is scanned on
    a log grid, ``F(a)`` is evaluated and its minimum (most tensile value) is
    located; the minimiser is then polished with a parabolic fit on the three
    neighbouring samples.  This is the number compared against
    ``1.5 pi gamma R*`` in the acceptance test F1.

    Returns:
        dict with ``a_c``, ``delta_c``, ``f_pull`` (magnitude), ``f_pull_theory``,
        ``rel_error``.
    """
    if gamma <= 0.0:
        return {
            "a_c": 0.0,
            "delta_c": 0.0,
            "f_pull": 0.0,
            "f_pull_theory": 0.0,
            "rel_error": 0.0,
        }
    a_theory = (9.0 * math.pi * gamma * r_eff ** 2 / (8.0 * e_star)) ** (1.0 / 3.0)
    lo = math.log10(a_theory * 0.05)
    hi = math.log10(a_theory * 20.0)
    a = torch.logspace(lo, hi, n_scan, dtype=DTYPE)
    f = jkr_force_from_contact_radius(a, e_star, r_eff, gamma)
    k = int(torch.argmin(f))
    k = min(max(k, 1), n_scan - 2)
    # parabolic refinement in log(a)
    x = torch.log(a[k - 1:k + 2])
    y = f[k - 1:k + 2]
    try:
        coeff = torch.linalg.solve(
            torch.stack([x ** 2, x, torch.ones_like(x)], dim=1), y)
        x_star = float(-coeff[1] / (2.0 * coeff[0]))
        f_star = float(coeff[0] * x_star ** 2 + coeff[1] * x_star + coeff[2])
        a_star = math.exp(x_star)
    except Exception:  # pragma: no cover - singular parabola fallback
        a_star = float(a[k])
        f_star = float(f[k])
    f_theory = -jkr_pull_off_force(gamma, r_eff)
    rel = abs(abs(f_star) - abs(f_theory)) / abs(f_theory) if f_theory else 0.0
    delta_star = float(jkr_overlap_from_contact_radius(
        torch.tensor(a_star, dtype=DTYPE), e_star, r_eff, gamma))
    # overlap at which the displacement-driven law detaches: the minimum of
    # delta(a), i.e. the point where d(delta)/da = 0 (a_m), not a_c.
    a_m = ((r_eff / 4.0) * math.sqrt(2.0 * math.pi * gamma / e_star)) ** (2.0 / 3.0)
    delta_detach = float(jkr_overlap_from_contact_radius(
        torch.tensor(a_m, dtype=DTYPE), e_star, r_eff, gamma))
    return {
        "a_c": a_star,
        "delta_c": delta_star,
        "delta_detach": delta_detach,
        "a_detach": a_m,
        "f_pull": abs(f_star),
        "f_pull_theory": abs(f_theory),
        "rel_error": rel,
    }


def hertz_normal_force(overlap, e_star: float, r_eff: float):
    """Non-adhesive Hertz normal force ``(4/3) E* sqrt(R*) delta^(3/2)`` (tension-free)."""
    d = _as_tensor(overlap).clamp_min(0.0)
    return (4.0 / 3.0) * e_star * math.sqrt(r_eff) * d ** 1.5


@dataclass
class JKRState:
    """Output of the displacement-driven JKR law (one entry per input overlap)."""

    overlap: torch.Tensor
    force: torch.Tensor
    contact_radius: torch.Tensor
    attached: torch.Tensor
    pressure_curvature: float = 0.0

    @property
    def tensile(self) -> torch.Tensor:
        return self.force < 0.0


def jkr_normal_force(
    overlap,
    e_star: float,
    r_eff: float,
    gamma: float,
    *,
    n_bisect: int = 120,
    return_state: bool = False,
):
    """Displacement-driven JKR normal force (signed: + repulsion, - tension).

    For ``gamma == 0`` the Hertz law is used exactly, so the ``gamma -> 0``
    control is analytic rather than a small-``gamma`` approximation.

    For ``gamma > 0`` the JKR overlap relation ``delta(a)`` is inverted for the
    *stable* branch ``a > a_m`` (``a_m`` = the overlap minimum); the returned
    force is ``F(a)``.  If ``delta < delta_c`` (no equilibrium: the contact
    detaches) the pull-off force ``-1.5 pi gamma R*`` is returned and the
    ``attached`` flag is False - the standard DEM treatment of the JKR jump.

    Args:
        overlap: signed normal overlap (``> 0`` compression, ``< 0`` gap), scalar
            or tensor, metres.
        e_star: reduced Young's modulus, Pa.
        r_eff: effective radius R*, m.
        gamma: surface energy (J/m^2); ``0`` gives pure Hertz.

    Returns:
        force tensor (or :class:`JKRState` when ``return_state``).
    """
    delta = _as_tensor(overlap)
    if not (math.isfinite(e_star) and e_star > 0.0):
        raise ValueError("e_star must be finite and positive")
    if not (math.isfinite(r_eff) and r_eff > 0.0):
        raise ValueError("r_eff must be finite and positive")
    if not (math.isfinite(gamma) and gamma >= 0.0):
        raise ValueError("gamma must be finite and nonnegative")

    if gamma == 0.0:
        force = hertz_normal_force(delta, e_star, r_eff)
        a = torch.sqrt(torch.clamp(delta, min=0.0) * r_eff)
        attached = delta > 0.0
        if return_state:
            return JKRState(delta, force, a, attached)
        return force

    # characteristic JKR scales
    a_m = ((r_eff / 4.0) * math.sqrt(2.0 * math.pi * gamma / e_star)) ** (2.0 / 3.0)
    delta_c = float(jkr_overlap_from_contact_radius(
        torch.tensor(a_m, dtype=DTYPE), e_star, r_eff, gamma))
    a_c = (9.0 * math.pi * gamma * r_eff ** 2 / (8.0 * e_star)) ** (1.0 / 3.0)

    # brackets on the stable branch a >= a_m
    lo = torch.full_like(delta, a_m * (1.0 + 1e-12))
    hi = torch.full_like(
        delta, max(2.0 * (a_m + math.sqrt(max(delta_c, 0.0) + 1e-30) * math.sqrt(r_eff)), a_c * 4.0))

    def delta_of(a_t):
        return jkr_overlap_from_contact_radius(a_t, e_star, r_eff, gamma)

    # expand hi until it brackets the target overlap
    for _ in range(200):
        need = delta_of(hi) < delta
        if not bool(need.any()):
            break
        hi = torch.where(need, hi * 2.0, hi)
        lo = torch.where(need, lo * 1.5, lo)

    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        low = delta_of(mid) < delta
        lo = torch.where(low, mid, lo)
        hi = torch.where(low, hi, mid)
    a = 0.5 * (lo + hi)

    force = jkr_force_from_contact_radius(a, e_star, r_eff, gamma)
    attached = delta >= delta_c
    f_pull = -jkr_pull_off_force(gamma, r_eff)
    force = torch.where(attached, force, torch.full_like(force, f_pull))
    a = torch.where(attached, a, torch.full_like(a, a_c))
    if return_state:
        return JKRState(delta, force, a, attached)
    return force


def jkr_hertz_relative_error(
    e_star: float,
    r_eff: float,
    gamma: float,
    *,
    delta_over_radius: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2),
) -> dict[str, object]:
    """Max relative deviation of the JKR law from Hertz for a given ``gamma``.

    Used for the ``gamma -> 0`` negative control (F3): the JKR branch is
    evaluated with a small surface energy and compared with the analytic Hertz
    force over a range of compressive overlaps.

    Returns:
        dict with ``gamma``, ``max_rel_error``, ``per_overlap`` (list of
        ``(overlap, f_jkr, f_hertz, rel_error)``).
    """
    rows = []
    worst = 0.0
    for frac in delta_over_radius:
        delta = frac * r_eff
        f_jkr = float(jkr_normal_force(delta, e_star, r_eff, gamma))
        f_hertz = float(hertz_normal_force(delta, e_star, r_eff))
        rel = abs(f_jkr - f_hertz) / abs(f_hertz)
        worst = max(worst, rel)
        rows.append((delta, f_jkr, f_hertz, rel))
    return {"gamma": gamma, "max_rel_error": worst, "per_overlap": rows}


# ---------------------------------------------------------------------------
# Liquid bridge: analytic anchor
# ---------------------------------------------------------------------------

def capillary_force_analytic(gamma: float, radius: float, contact_angle_deg: float) -> float:
    """Small-bridge capillary force ``F = 2 pi gamma R cos(theta)`` (attraction, N).

    Valid for a pendular bridge of vanishing volume between two spheres of radius
    ``R`` (or a sphere and a plane) whose contact angle is ``theta``.
    """
    if not (math.isfinite(gamma) and gamma > 0.0):
        raise ValueError("gamma must be finite and positive")
    if not (math.isfinite(radius) and radius > 0.0):
        raise ValueError("radius must be finite and positive")
    if not (0.0 <= contact_angle_deg < 90.0):
        raise ValueError("contact_angle_deg must lie in [0, 90)")
    return 2.0 * math.pi * gamma * radius * math.cos(math.radians(contact_angle_deg))


# ---------------------------------------------------------------------------
# Liquid bridge: numerical Young-Laplace solution
# ---------------------------------------------------------------------------

@dataclass
class LiquidBridgeResult:
    """Converged pendular-bridge solution (all quantities dimensionless-scaled back to SI)."""

    radius: float
    gap: float
    contact_angle_deg: float
    filling_angle_deg: float
    gamma: float
    suction: float           # dP = p_gas - p_liquid, Pa
    neck_radius: float       # r_n, m
    contact_radius: float    # r_c, m
    contact_slope_deg: float  # phi_c, deg
    volume: float            # m^3
    force_neck: float        # N (attraction)
    force_contact: float     # N (attraction)
    residual: float = 0.0            # max |shooting residual|
    residual_angle: float = 0.0      # contact-angle residual (rad)
    residual_volume: float = 0.0     # relative volume residual
    volume_rel_error: float = 0.0    # |V_solved - V_target| / V_target
    converged: bool = True
    n_ode_steps: int = 0
    profile_z: np.ndarray = field(default_factory=lambda: np.zeros(0))
    profile_r: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def force(self) -> float:
        """Capillary attraction averaged over the two equivalent expressions."""
        return 0.5 * (self.force_neck + self.force_contact)

    @property
    def consistency(self) -> float:
        """Relative gap between ``F_neck`` and ``F_contact`` (numerical quality)."""
        scale = max(abs(self.force_neck), abs(self.force_contact), 1e-300)
        return abs(self.force_neck - self.force_contact) / scale

    @property
    def analytic_force(self) -> float:
        return capillary_force_analytic(self.gamma, self.radius, self.contact_angle_deg)

    @property
    def rel_error_vs_analytic(self) -> float:
        return abs(self.force - self.analytic_force) / self.analytic_force


def _integrate_to_sphere(dp_hat: float, x_n: float, gap_over_R: float,
                        theta: float, n_phi: int = 2000):
    """March the Young-Laplace profile from the neck until it meets the sphere.

    Dimensionless variables: ``x = r/R``, ``z = z/R``, ``dp_hat = dP R / gamma``
    and ``phi`` the tangent angle measured from the +r axis, used as the
    independent variable (it decreases monotonically from ``pi/2`` at the neck)::

        dx/dphi = -cos(phi) / (dp_hat + sin(phi)/x)
        dz/dphi = -sin(phi) / (dp_hat + sin(phi)/x)

    The upper sphere is centred at ``(0, h/R + 1)`` with radius ``R``.  The march
    stops at the first crossing of that sphere (linear interpolation in ``phi``
    of the sphere's implicit function).

    Returns ``(x_c, z_c, phi_c, xs, zs)`` or ``None`` if the profile reaches the
    axis / the angle floor without touching the sphere.
    """
    phi_floor = 1e-3
    h = (phi_floor - 0.5 * math.pi) / n_phi
    x = x_n
    z = 0.0
    z_centre = gap_over_R + 1.0

    def rhs(x, phi):
        den = dp_hat + math.sin(phi) / x
        if not math.isfinite(den) or den <= 1e-14:
            return None
        return -math.cos(phi) / den, -math.sin(phi) / den

    def f_inside(x, z):  # < 0 outside, > 0 inside the upper sphere
        return 1.0 - (x * x + (z - z_centre) ** 2)

    xs = [x]
    zs = [z]
    phi = 0.5 * math.pi
    for _ in range(n_phi):
        k1 = rhs(x, phi)
        if k1 is None:
            return None
        k2 = rhs(x + 0.5 * h * k1[0], phi + 0.5 * h)
        k3 = rhs(x + 0.5 * h * k2[0], phi + 0.5 * h) if k2 is not None else None
        if k2 is None or k3 is None:
            return None
        k4 = rhs(x + h * k3[0], phi + h)
        if k4 is None:
            return None
        x_new = x + h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        z_new = z + h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        phi_new = phi + h
        if not (math.isfinite(x_new) and math.isfinite(z_new)) or x_new <= 0.0:
            return None
        f_new = f_inside(x_new, z_new)
        if f_new >= 0.0:
            f_old = f_inside(x, z)
            if f_old >= 0.0:
                return None  # started inside the solid: invalid geometry
            t = f_old / (f_old - f_new)
            return (x + t * (x_new - x), z + t * (z_new - z),
                    phi + t * h, np.array(xs), np.array(zs))
        x, z, phi = x_new, z_new, phi_new
        xs.append(x)
        zs.append(z)
    return None


def _bridge_volume_over_R3(xs: np.ndarray, zs: np.ndarray, gap_over_R: float) -> float:
    """Bridge volume (in units of R^3) from the meridional profile of the free surface."""
    h = 0.5 * gap_over_R
    z_abs = np.abs(zs)
    r_sph = np.where(
        z_abs > h,
        np.sqrt(np.clip(1.0 - (h + 1.0 - z_abs) ** 2, 0.0, None)),
        0.0,
    )
    area = np.pi * np.clip(xs ** 2 - r_sph ** 2, 0.0, None)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return 2.0 * float(trapz(area, zs))


def _bridge_residual(x: np.ndarray, gap_over_R: float, theta: float,
                     volume_target: float, n_phi: int) -> np.ndarray:
    """Residuals of the (neck radius, suction) shooting problem.

    Conditions: (i) the interface meets the solid at the contact angle
    ``phi_c = beta + theta``; (ii) the bridge volume equals ``volume_target``
    (dimensionless, in units of R^3).
    """
    x_n, dp_hat = float(x[0]), float(x[1])
    if not (1e-7 < x_n < 0.9) or not (math.isfinite(dp_hat) and dp_hat > 1e-4):
        return np.array([1e2, 1e2])
    out = _integrate_to_sphere(dp_hat, x_n, gap_over_R, theta, n_phi=n_phi)
    if out is None:
        return np.array([1e2, 1e2])
    x_c, z_c, phi_c, xs, zs = out
    beta = math.asin(min(max(x_c, 0.0), 1.0))
    res_angle = phi_c - (beta + theta)
    vol = _bridge_volume_over_R3(xs, zs, gap_over_R)
    res_vol = (vol - volume_target) / volume_target
    return np.array([res_angle, res_vol])


def solve_liquid_bridge(
    radius: float,
    *,
    gap: float = 0.0,
    contact_angle_deg: float = 0.0,
    volume_over_R3: float = 1.0e-5,
    gamma: float = 0.072,
    n_phi: int = 1600,
    n_phi_scan: int = 300,
    tol: float = 1e-6,
) -> LiquidBridgeResult:
    """Solve the axisymmetric Young-Laplace equation for a pendular bridge.

    The bridge is specified by its **liquid volume** (dimensionless
    ``volume_over_R3``), the gap between the two equal spheres of radius
    ``radius``, the surface tension ``gamma`` and the contact angle ``theta``.
    This is the physically well-posed parameterisation: for a given volume the
    neck radius ``r_n`` and the capillary suction ``dP = p_gas - p_liquid`` are
    the two shooting unknowns.

    Conditions enforced at the contact line:

    * the interface meets the sphere at the contact angle, ``phi_c = beta + theta``
      (``beta`` = filling angle of the contact circle);
    * the revolution volume of the liquid equals the prescribed value.

    Args:
        radius: sphere radius R, m.
        gap: surface-to-surface separation, m (0 = touching spheres).
        contact_angle_deg: contact angle theta, deg, in [0, 90).
        volume_over_R3: dimensionless bridge volume in units of R^3.
        gamma: liquid surface tension, N/m.
        n_phi: angular steps of the converged profile.
        n_phi_scan: angular steps used inside the coarse seeding scan.
        tol: tolerance on the shooting residuals.

    Returns:
        :class:`LiquidBridgeResult`.
    """
    from scipy.optimize import least_squares

    for name, value in (("radius", radius), ("gamma", gamma),
                        ("volume_over_R3", volume_over_R3)):
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"{name} must be finite and positive")
    if not (math.isfinite(gap) and gap >= 0.0):
        raise ValueError("gap must be finite and nonnegative")
    if not (0.0 <= contact_angle_deg < 90.0):
        raise ValueError("contact_angle_deg must lie in [0, 90)")
    theta = math.radians(contact_angle_deg)
    gap_over_R = gap / radius

    # --- coarse seeding -----------------------------------------------------
    seeds: list[tuple[float, float, float]] = []
    for x_n in np.logspace(-4.0, -0.7, 16):
        for dp_hat in np.logspace(-1.0, 4.0, 22):
            res = _bridge_residual(np.array([x_n, dp_hat]), gap_over_R, theta,
                                   volume_over_R3, n_phi_scan)
            seeds.append((float(np.linalg.norm(res)), float(x_n), float(dp_hat)))
    seeds.sort(key=lambda item: item[0])

    best = None
    for cost0, x_n0, dp0 in seeds[:8]:
        try:
            sol = least_squares(
                lambda x: _bridge_residual(x, gap_over_R, theta, volume_over_R3, n_phi),
                np.array([x_n0, dp0]),
                bounds=([1e-8, 1e-3], [0.95, 1e6]),
                xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400,
            )
        except Exception:  # pragma: no cover - solver robustness
            continue
        cost = float(np.max(np.abs(sol.fun)))
        if best is None or cost < best[0]:
            best = (cost, sol.x)
        if cost < tol:
            break

    if best is None:
        raise RuntimeError("liquid bridge shooting failed for all initial guesses")

    cost, (x_n, dp_hat) = best
    volume_target_si = volume_over_R3 * radius ** 3
    resid = _bridge_residual(best[1], gap_over_R, theta, volume_over_R3, n_phi)
    cost_angle, cost_volume = float(resid[0]), float(resid[1])
    out = _integrate_to_sphere(dp_hat, x_n, gap_over_R, theta, n_phi=n_phi)
    if out is None:  # pragma: no cover - resolution fallback for thin bridges
        for n_try in (n_phi // 2, n_phi_scan, n_phi_scan // 2):
            out = _integrate_to_sphere(dp_hat, x_n, gap_over_R, theta,
                                       n_phi=max(int(n_try), 50))
            if out is not None:
                break
    if out is None:  # pragma: no cover
        raise RuntimeError("profile integration failed at the converged point")
    x_c, z_c, phi_c, xs, zs = out

    r_n = x_n * radius
    suction = dp_hat * gamma / radius
    beta = math.asin(min(max(x_c, 0.0), 1.0))
    r_c = radius * x_c
    volume = _bridge_volume_over_R3(xs, zs, gap_over_R) * radius ** 3

    # --- two independent force expressions ---------------------------------
    force_neck = 2.0 * math.pi * gamma * r_n + suction * math.pi * r_n ** 2
    force_contact = (2.0 * math.pi * gamma * r_c * math.sin(phi_c)
                     + suction * math.pi * r_c ** 2)

    return LiquidBridgeResult(
        radius=radius,
        gap=gap,
        contact_angle_deg=contact_angle_deg,
        filling_angle_deg=math.degrees(beta),
        gamma=gamma,
        suction=suction,
        neck_radius=r_n,
        contact_radius=r_c,
        contact_slope_deg=math.degrees(phi_c),
        volume=volume,
        force_neck=force_neck,
        force_contact=force_contact,
        residual=cost,
        residual_angle=abs(float(cost_angle)),
        residual_volume=abs(float(cost_volume)),
        volume_rel_error=abs(volume - volume_target_si) / volume_target_si,
        converged=bool(abs(cost_angle) < 5.0e-3 and abs(cost_volume) < 2.0e-2),
        n_ode_steps=n_phi,
        profile_z=zs * radius,
        profile_r=xs * radius,
    )


def capillary_force_laplace(
    radius: float,
    *,
    gap: float = 0.0,
    contact_angle_deg: float = 0.0,
    volume_over_R3: float = 1.0e-5,
    gamma: float = 0.072,
    n_phi: int = 1600,
) -> float:
    """Capillary attraction from the numerical Laplace solution (N)."""
    return solve_liquid_bridge(
        radius, gap=gap, contact_angle_deg=contact_angle_deg,
        volume_over_R3=volume_over_R3, gamma=gamma, n_phi=n_phi,
    ).force


def capillary_force_ratio_table(
    radius: float,
    *,
    gap: float = 0.0,
    contact_angle_deg: float = 0.0,
    volumes_over_R3: tuple[float, ...] = (1e-7, 1e-6, 1e-5, 1e-4),
    gamma: float = 0.072,
    n_phi: int = 1600,
) -> list[dict[str, float]]:
    """Scan the bridge volume against the analytic small-bridge limit.

    Returns one dict per volume with the converged solution, the two force
    expressions, their consistency and the relative error against
    ``2 pi gamma R cos(theta)``.
    """
    rows = []
    for vol in volumes_over_R3:
        res = solve_liquid_bridge(
            radius, gap=gap, contact_angle_deg=contact_angle_deg,
            volume_over_R3=vol, gamma=gamma, n_phi=n_phi)
        rows.append({
            "volume_over_R3": vol,
            "volume_over_R3_solved": res.volume / radius ** 3,
            "filling_angle_deg": res.filling_angle_deg,
            "force": res.force,
            "force_neck": res.force_neck,
            "force_contact": res.force_contact,
            "consistency": res.consistency,
            "analytic": res.analytic_force,
            "rel_error": res.rel_error_vs_analytic,
            "suction_over_gamma_R": res.suction * radius / gamma,
            "neck_radius_over_R": res.neck_radius / radius,
            "contact_slope_deg": res.contact_slope_deg,
            "residual": res.residual,
            "converged": res.converged,
        })
    return rows


# ---------------------------------------------------------------------------
# Interface with the superquadric contact (P96-N3)
# ---------------------------------------------------------------------------

def adhesive_contact_force(
    contact,
    e_star: float,
    *,
    gamma_solid: float = 0.0,
    gamma_liquid: float = 0.0,
    contact_angle_deg: float = 0.0,
    use_jkr: bool = True,
) -> dict[str, float]:
    """Stack adhesion (JKR + capillary) on a superquadric contact result.

    The input is any object exposing ``overlap`` (m) and ``effective_radius``
    (m) - i.e. a :class:`tensordem.superquadric.SuperquadricContactResult`; only
    those two scalars are used, so the P96-N3 true-shape contact detector stays
    untouched and this function works for spheres, blocks and intermediate
    shapes alike.

    The returned normal force is *signed*: ``hertz + jkr_delta - f_pull`` for a
    JKR contact, i.e. the adhesive term shifts the force curve in tension.

    Args:
        contact: object with ``overlap`` and ``effective_radius``.
        e_star: reduced Young's modulus E*, Pa.
        gamma_solid: dry surface energy for the JKR term, J/m^2.
        gamma_liquid: liquid surface tension for the capillary term, N/m.
        contact_angle_deg: contact angle for the capillary term.
        use_jkr: include the JKR term.

    Returns:
        dict with ``f_hertz``, ``f_jkr``, ``f_pull``, ``f_capillary``,
        ``f_normal`` (signed total) and ``r_eff``.
    """
    overlap = float(getattr(contact, "overlap"))
    r_eff = float(getattr(contact, "effective_radius"))
    if r_eff <= 0.0:
        raise ValueError("contact effective_radius must be positive")
    f_hertz = float(hertz_normal_force(overlap, e_star, r_eff))
    f_pull = jkr_pull_off_force(gamma_solid, r_eff) if gamma_solid > 0 else 0.0
    f_jkr = float(jkr_normal_force(overlap, e_star, r_eff, gamma_solid)) if use_jkr else f_hertz
    f_cap = (capillary_force_analytic(gamma_liquid, r_eff, contact_angle_deg)
             if gamma_liquid > 0.0 else 0.0)
    # JKR law already contains the adhesive shift; the capillary term is added
    # as an independent tensile contribution (liquid bridge acts in parallel).
    f_normal = f_jkr - f_cap
    return {
        "overlap": overlap,
        "r_eff": r_eff,
        "f_hertz": f_hertz,
        "f_jkr": f_jkr,
        "f_pull": f_pull,
        "f_capillary": f_cap,
        "f_normal": f_normal,
    }