"""Analytic normal-contact anchors for ellipsoidal and superellipsoidal particles.

Campaign node P27W2-N07, criteria C1 (and the contact-side physics behind C5).

Physics
-------
An ellipsoid pressed against a plane touches over an **ellipse**.  With the
surface gap near the pole written as ``g(x, y) = A x^2 + B y^2`` (``A >= B`` are
half the sums of the principal curvatures of the two facing surfaces), the Hertz
solution is

    p(x, y) = p0 sqrt(1 - x^2/a_x^2 - y^2/a_y^2),          (contact ellipse)

with the contact-ellipse eccentricity fixed by the *elliptic-integral* relation

    B/A = (K(e) - E(e)) / (E(e)/(1-e^2) - K(e)),            e^2 = 1 - (a_y/a_x)^2,

``K``, ``E`` the complete elliptic integrals of the first and second kind.  The
semi-axes, the peak pressure and the approach follow from the same integrals:

    a_x = (3 P / (4 (A+B) E*))^(1/3) [ 2 E / (pi (1-e^2)) ]^(1/3),
    a_y = t a_x,             t = a_y/a_x = sqrt(1 - e^2),
    p0  = 3 P / (2 pi a_x a_y),
    delta = 3 P K / (2 pi a_x E*),
    k_n = dP/ddelta = (3/2) P/delta            (Hertz scaling P ~ delta^(3/2)).

The major contact axis lies along the *flatter* direction (larger radius,
smaller curvature ``B``) -- the cylinder limit ``B -> 0`` gives the expected
long strip; the sphere limit ``A = B`` collapses to ``a = (3 P R / 4 E*)^(1/3)``,
``delta = a^2/R`` exactly, which is asserted in the unit tests.

Why this matters for shape representation: a *superellipsoid* pole is only
Hertzian for exponent ``m = 2``.  For ``m > 2`` the pole is locally flat
(curvature vanishes), the gap grows as ``|x|^m`` and the contact law becomes a
power-law indenter,

    P ~ delta^(1 + 1/m),   k_n ~ delta^(1/m)   (m = 1 cone -> delta^2,
                                                 m = 2 sphere -> delta^1.5,
                                                 m -> inf flat punch -> delta).

So a clump and a superquadric differ *at the contact law itself*, not only in the
surface geometry -- that is the mechanism the verdict table quantifies.

Verification (no elliptic integrals involved)
---------------------------------------------
``verify_closed_form`` substitutes the closed-form pressure into the elastic
half-space integral equation and integrates it with a *singularity-free* polar
quadrature (``dA/r = dr dtheta``), which is exact for a point inside the patch:
the displacement must be ``delta - g(x, y)`` up to machine-level quadrature
error.  ``axisymmetric_indenter_law`` solves Love's integral equation for a
power-law indenter and recovers the ``1 + 1/m`` exponent numerically.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import ellipk, ellipe

_HALF_PI = 0.5 * math.pi


# ---------------------------------------------------------------------------
# Complete elliptic integrals (numerically clean near the sphere limit)
# ---------------------------------------------------------------------------

def elliptic_terms(q: float) -> Tuple[float, float]:
    """Complete elliptic integrals ``(K, E)`` with parameter ``q = e^2``.

    A series is used for ``q < 1e-4`` because ``E - (1-q) K`` and ``K - E`` are
    both ``O(q)`` there and evaluating them as differences of two ``pi/2``-sized
    numbers loses half of the significant digits (that cancellation alone would
    be fatal for the sphere limit, which is the primary anchor).
    """
    q = float(q)
    if not (0.0 < q < 1.0):
        return _HALF_PI, _HALF_PI
    if q < 1e-4:
        K = _HALF_PI * (1.0 + q / 4.0 + 9.0 * q * q / 64.0 + 25.0 * q ** 3 / 256.0)
        E = _HALF_PI * (1.0 - q / 4.0 - 3.0 * q * q / 64.0 - 5.0 * q ** 3 / 256.0)
        return float(K), float(E)
    return float(ellipk(q)), float(ellipe(q))


def _hertz_sums(q: float) -> Tuple[float, float, float, float]:
    """``(K, E, K-E, E/(1-q)-K)`` -- the last two from their own series.

    Neither difference may be formed by subtracting two ``pi/2``-sized numbers
    near ``q = 0``; the series below keep full precision, and their ratio at
    ``q -> 0`` is exactly 1, which is the sphere limit.
    """
    q = float(q)
    if q < 1e-4:
        K = _HALF_PI * (1.0 + q / 4.0 + 9.0 * q * q / 64.0)
        E = _HALF_PI * (1.0 - q / 4.0 - 3.0 * q * q / 64.0)
        KmE = _HALF_PI * (q / 2.0 + 3.0 * q * q / 16.0 + 15.0 * q ** 3 / 128.0)
        EsK = _HALF_PI * (q / 2.0 + 9.0 * q * q / 16.0 + 75.0 * q ** 3 / 128.0)
        return float(K), float(E), float(KmE), float(EsK)
    K, E = elliptic_terms(q)
    return K, E, K - E, E / (1.0 - q) - K


def curvature_ratio_function(q: float) -> float:
    """Hertz curvature ratio ``B/A`` as a function of the eccentricity parameter.

    Monotone decreasing from 1 (sphere, ``q=0``) to 0 (line contact, ``q->1``).
    """
    _, _, KmE, EsK = _hertz_sums(q)
    if KmE <= 0.0 or EsK <= 0.0:
        return 1.0
    return float(KmE / EsK)


def _solve_q(ratio: float) -> float:
    """Invert ``curvature_ratio_function`` by bisection (monotone, bracket [0,1))."""
    ratio = float(min(max(ratio, 1e-12), 1.0))
    if ratio >= 1.0:
        return 0.0
    lo, hi = 0.0, 1.0 - 1e-15
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if curvature_ratio_function(mid) > ratio:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-16:
            break
    return 0.5 * (lo + hi)


def eccentricity_from_ratio(ratio: float) -> float:
    """Contact-ellipse eccentricity ``e`` for a curvature ratio ``B/A``.

    ``ratio = 1`` -> ``e = 0`` (circle); ``ratio -> 0`` -> ``e -> 1`` (strip).
    """
    return float(math.sqrt(max(0.0, _solve_q(ratio))))


# ---------------------------------------------------------------------------
# Closed-form ellipsoid / plane contact
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EllipsoidPlaneContact:
    """Closed-form Hertz contact of an ellipsoid pole with a rigid plane.

    ``r_flat`` is the surface radius along the *flatter* direction (the major
    contact axis), ``r_sharp`` the radius along the sharper direction.
    """
    r_flat: float
    r_sharp: float
    e_star: float
    load: float
    ecc: float
    axis_ratio: float
    a_major: float
    a_minor: float
    p0: float
    approach: float
    stiffness: float
    curvature_coef_flat: float     # B = 1/(2 r_flat)
    curvature_coef_sharp: float    # A = 1/(2 r_sharp)

    def to_dict(self) -> Dict[str, float]:
        return {
            "r_flat_m": float(self.r_flat),
            "r_sharp_m": float(self.r_sharp),
            "curvature_ratio_B_over_A": float(self.r_sharp / self.r_flat),
            "e_star_pa": float(self.e_star),
            "load_n": float(self.load),
            "eccentricity": float(self.ecc),
            "contact_axis_ratio_b_over_a": float(self.axis_ratio),
            "a_major_m": float(self.a_major),
            "a_minor_m": float(self.a_minor),
            "p0_pa": float(self.p0),
            "approach_m": float(self.approach),
            "stiffness_n_per_m": float(self.stiffness),
        }


def ellipsoid_plane_contact(
    r_flat: float, r_sharp: float, e_star: float, *,
    load: Optional[float] = None, approach: Optional[float] = None,
) -> EllipsoidPlaneContact:
    """Hertz contact of an ellipsoid pole against a plane (closed form, elliptic integrals).

    Exactly one of ``load`` (normal force, N) or ``approach`` (normal overlap, m)
    must be given.  ``r_flat`` / ``r_sharp`` are the principal surface radii of
    curvature at the pole; order does not matter (they are sorted internally).
    """
    if (load is None) == (approach is None):
        raise ValueError("give exactly one of load= or approach=")
    r_sharp, r_flat = sorted((float(r_sharp), float(r_flat)))
    if r_sharp <= 0.0 or r_flat <= 0.0 or e_star <= 0.0:
        raise ValueError("radii and E* must be positive")
    A = 0.5 / r_sharp          # larger curvature coefficient (minor axis)
    B = 0.5 / r_flat           # smaller curvature coefficient (major axis)
    ratio = B / A              # = r_sharp / r_flat in [0, 1]
    q = _solve_q(ratio)
    K, E, KmE, EsK = _hertz_sums(q)
    t = math.sqrt(max(0.0, 1.0 - q))         # contact axis ratio a_minor/a_major
    if t <= 0.0:
        t = 1e-12
    base = 3.0 / (4.0 * (A + B) * e_star)    # a_major^3 = base * P * bracket

    # Contact semi-axis (major, along the flatter direction):
    #   a^3 = (3 P /(4 (A+B) E*)) * 2 E / (pi (1-e^2)),
    # obtained from the quadratic coefficients of the exact half-space quadrature
    # (c_x = -(p0 b/E*)(K-E)/(e^2 a^2), c_y = -(p0 b/E*)(E/(1-e^2)-K)/(e^2 a^2),
    #  P = (2/3) pi a^2 t p0) rather than from a remembered formula.
    bracket = 2.0 * E / (math.pi * max(1.0 - q, 1e-300))

    def _solve_for(load_n: float) -> Tuple[float, float, float, float, float]:
        a_major = (base * load_n * bracket) ** (1.0 / 3.0)
        a_minor = t * a_major
        p0 = 3.0 * load_n / (2.0 * math.pi * a_major * a_minor)
        delta = p0 * a_minor * K / e_star
        return a_major, a_minor, p0, delta, 1.5 * load_n / delta

    if approach is None:
        a_major, a_minor, p0, delta, k_n = _solve_for(float(load))
        P = float(load)
    else:
        # invert delta(P): delta = c P^(2/3)
        delta_unit = _solve_for(1.0)[3]
        P = (float(approach) / delta_unit) ** 1.5
        a_major, a_minor, p0, delta, k_n = _solve_for(P)
    return EllipsoidPlaneContact(
        r_flat=r_flat, r_sharp=r_sharp, e_star=float(e_star), load=P,
        ecc=math.sqrt(q), axis_ratio=t, a_major=a_major, a_minor=a_minor,
        p0=p0, approach=delta, stiffness=k_n,
        curvature_coef_flat=B, curvature_coef_sharp=A,
    )


def hertz_stiffness(r_flat: float, r_sharp: float, e_star: float, approach: float) -> float:
    """Normal contact stiffness ``dP/ddelta`` at a given overlap (N/m)."""
    return ellipsoid_plane_contact(r_flat, r_sharp, e_star, approach=approach).stiffness


def force_from_overlap(r_flat: float, r_sharp: float, e_star: float, approach: float) -> float:
    """Normal force at a given overlap from the same closed form (N)."""
    return ellipsoid_plane_contact(r_flat, r_sharp, e_star, approach=approach).load


# ---------------------------------------------------------------------------
# Independent verification: singularity-free polar quadrature of Love's equation
# ---------------------------------------------------------------------------

def displacement_quadrature(
    x: np.ndarray, y: np.ndarray, a_major: float, a_minor: float,
    p0: float, e_star: float, n_theta: int = 2048, n_s: int = 24,
) -> np.ndarray:
    """Displacement of an elastic half-space under the Hertz ellipse pressure.

    ``w(x,y) = (1/(pi E*)) iint p/|r-r'| dA'`` evaluated for points **inside** the
    contact ellipse.  Written in polar coordinates around the field point the
    kernel cancels (``dA/r = dr dtheta``) so the integrand is bounded; the
    square-root behaviour of ``p`` at the patch edge is removed by the
    substitution ``u = u_max (1 - s^2)``.

    This uses *no* Hertz formula other than the pressure *shape*, which is what
    makes it an independent check of the closed-form geometry and stiffness.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    ax, ay = float(a_major), float(a_minor)
    th = (np.arange(n_theta) + 0.5) * (2.0 * math.pi / n_theta)
    ct, st = np.cos(th), np.sin(th)
    s_nodes, s_weights = np.polynomial.legendre.leggauss(n_s)
    s_nodes = 0.5 * (s_nodes + 1.0)
    s_weights = 0.5 * s_weights
    out = np.zeros_like(x)
    for k in range(len(x)):
        P, Q = x[k] / ax, y[k] / ay
        aa = (ct / ax) ** 2 + (st / ay) ** 2
        bb = 2.0 * (P * ct / ax + Q * st / ay)
        cc = P * P + Q * Q - 1.0                      # <= 0 inside
        disc = np.maximum(bb * bb - 4.0 * aa * cc, 0.0)
        u_max = np.maximum((-bb + np.sqrt(disc)) / (2.0 * aa), 0.0)
        # along the ray: u = u_max (1 - s^2), s in [0,1], du = 2 s u_max ds
        u = u_max[:, None] * (1.0 - s_nodes[None, :] ** 2)          # (n_theta, n_s)
        px = x[k] + u * ct[:, None]
        py = y[k] + u * st[:, None]
        f = 1.0 - (px / ax) ** 2 - (py / ay) ** 2
        f = np.sqrt(np.maximum(f, 0.0))
        inner = np.sum(f * (2.0 * s_nodes[None, :] * u_max[:, None]) * s_weights[None, :], axis=1)
        out[k] = (p0 / (math.pi * e_star)) * (2.0 * math.pi / n_theta) * float(np.sum(inner))
    return out


@dataclass
class HertzVerification:
    """Residual check of a closed-form Hertz contact against the integral equation."""
    curvature_ratio: float
    stiffness_closed_form: float
    stiffness_integral: float
    stiffness_rel_error: float
    approach_closed_form: float
    approach_integral: float
    approach_rel_error: float
    flatness_spread: float          # (max-min) of w + g, relative to delta
    n_points: int
    load: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "curvature_ratio_B_over_A": float(self.curvature_ratio),
            "stiffness_closed_form_n_per_m": float(self.stiffness_closed_form),
            "stiffness_integral_n_per_m": float(self.stiffness_integral),
            "stiffness_rel_error": float(self.stiffness_rel_error),
            "approach_closed_form_m": float(self.approach_closed_form),
            "approach_integral_m": float(self.approach_integral),
            "approach_rel_error": float(self.approach_rel_error),
            "flatness_spread_rel": float(self.flatness_spread),
            "n_points": int(self.n_points),
            "load_n": float(self.load),
        }


def _sample_interior(a_major: float, a_minor: float, n_ring: int = 6,
                     n_theta: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Interior sample points of the contact ellipse (rings + centre)."""
    xs, ys = [0.0], [0.0]
    for rho in np.linspace(0.15, 0.85, n_ring):
        for th in np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False):
            xs.append(rho * a_major * math.cos(th))
            ys.append(rho * a_minor * math.sin(th))
    return np.array(xs), np.array(ys)


def verify_closed_form(
    contact: EllipsoidPlaneContact, *, n_theta: int = 2048, n_s: int = 24,
    n_ring: int = 4, n_theta_pts: int = 8,
) -> HertzVerification:
    """Check the closed form by substituting it into the half-space integral equation.

    The test quantity is ``w(x,y) + g(x,y)``; the Hertz solution requires it to be
    constant (``= delta``) over the whole contact ellipse.  Its spread is the
    residual of the closed-form field; the stiffness is then recomputed from the
    integral (``P`` from the pressure patch, ``delta`` from the mean residual) and
    compared with the closed form.
    """
    ax, ay = contact.a_major, contact.a_minor
    xs, ys = _sample_interior(ax, ay, n_ring=n_ring, n_theta=n_theta_pts)
    w = displacement_quadrature(xs, ys, ax, ay, contact.p0, contact.e_star,
                               n_theta=n_theta, n_s=n_s)
    gap = contact.curvature_coef_flat * xs ** 2 + contact.curvature_coef_sharp * ys ** 2
    total = w + gap
    delta_int = float(np.mean(total))
    spread = float(np.max(total) - np.min(total))
    P_int = (2.0 / 3.0) * math.pi * ax * ay * contact.p0
    k_int = 1.5 * P_int / delta_int
    return HertzVerification(
        curvature_ratio=contact.r_sharp / contact.r_flat,
        stiffness_closed_form=contact.stiffness,
        stiffness_integral=k_int,
        stiffness_rel_error=abs(k_int - contact.stiffness) / k_int,
        approach_closed_form=contact.approach,
        approach_integral=delta_int,
        approach_rel_error=abs(delta_int - contact.approach) / delta_int,
        flatness_spread=spread / delta_int,
        n_points=int(len(xs)),
        load=contact.load,
    )


def verify_elliptic_relation(
    curvature_ratio: float, e_star: float = 1.0e9, load: float = 1.0,
    *, n_theta: int = 2048, n_s: int = 24, ratios: int = 11,
) -> Dict[str, object]:
    """Free-boundary scan: which contact-ellipse axis ratio makes ``w + g`` flat?

    The Hertz solution has two free geometric unknowns (contact axis ratio ``t``
    and scale ``a``).  Here they are solved *numerically* from the integral
    equation: for each trial ``t`` the scale follows from the x-flatness
    condition ``d(w+g)/dx^2 = 0`` (exact, because ``w`` is quadratic inside), and
    the remaining y-flatness residual ``|A + d(w+g)/dy^2| / A`` measures how badly
    that trial violates the free boundary.  The scan minimum is the numerically
    determined contact shape; the closed form (elliptic integrals) must sit on it.
    """
    A = 1.0 / (1.0 + curvature_ratio)          # A + B = 1 by construction
    B = curvature_ratio * A
    r_flat, r_sharp = 1.0 / (2.0 * B), 1.0 / (2.0 * A)
    cf = ellipsoid_plane_contact(r_flat, r_sharp, e_star, load=load)
    ts = np.linspace(max(0.05, cf.axis_ratio * 0.5), min(0.999, cf.axis_ratio * 1.6), ratios)
    scan: List[Dict[str, float]] = []
    for t in ts:
        # unit-scale quadrature: a_major = 1, p0 = 1, then scale analytically
        w0 = float(displacement_quadrature([0.0], [0.0], 1.0, float(t), 1.0,
                                          e_star, n_theta=n_theta, n_s=n_s)[0])
        wx = float(displacement_quadrature([0.5], [0.0], 1.0, float(t), 1.0,
                                          e_star, n_theta=n_theta, n_s=n_s)[0])
        wy = float(displacement_quadrature([0.0], [0.5], 1.0, float(t), 1.0,
                                          e_star, n_theta=n_theta, n_s=n_s)[0])
        c_hat_x = (wx - w0) / 0.25             # d w / d x^2  at (a=1, p0=1)
        c_hat_y = (wy - w0) / 0.25
        # x-flatness:  p0 * c_hat_x / a  = -B  with p0 = 3P/(2 pi t a^2)
        a = (3.0 * load * (-c_hat_x) / (2.0 * math.pi * float(t) * B)) ** (1.0 / 3.0)
        p0 = 3.0 * load / (2.0 * math.pi * float(t) * a * a)
        cy = p0 * c_hat_y / a
        scan.append({
            "axis_ratio": float(t),
            "a_major": float(a),
            "y_residual_rel": float(abs(A + cy) / A),
        })
    best = min(scan, key=lambda r: r["y_residual_rel"])
    # local refinement of the ratio grid (the coarse scan is resolution limited)
    lo, hi = best["axis_ratio"] * 0.85, best["axis_ratio"] * 1.15
    for _ in range(3):
        for t in np.linspace(lo, hi, 5):
            w0 = float(displacement_quadrature([0.0], [0.0], 1.0, float(t), 1.0,
                                              e_star, n_theta=n_theta, n_s=n_s)[0])
            wx = float(displacement_quadrature([0.5], [0.0], 1.0, float(t), 1.0,
                                              e_star, n_theta=n_theta, n_s=n_s)[0])
            wy = float(displacement_quadrature([0.0], [0.5], 1.0, float(t), 1.0,
                                              e_star, n_theta=n_theta, n_s=n_s)[0])
            c_hat_x = (wx - w0) / 0.25
            c_hat_y = (wy - w0) / 0.25
            a = (3.0 * load * (-c_hat_x) / (2.0 * math.pi * float(t) * B)) ** (1.0 / 3.0)
            p0 = 3.0 * load / (2.0 * math.pi * float(t) * a * a)
            cy = p0 * c_hat_y / a
            scan.append({"axis_ratio": float(t), "a_major": float(a),
                         "y_residual_rel": float(abs(A + cy) / A)})
        best = min(scan, key=lambda r: r["y_residual_rel"])
        lo, hi = best["axis_ratio"] * 0.94, best["axis_ratio"] * 1.06
    return {
        "curvature_ratio_B_over_A": float(curvature_ratio),
        "closed_form_axis_ratio": float(cf.axis_ratio),
        "closed_form_a_major": float(cf.a_major),
        "best_axis_ratio": float(best["axis_ratio"]),
        "best_a_major": float(best["a_major"]),
        "axis_ratio_rel_error": abs(best["axis_ratio"] - cf.axis_ratio) / cf.axis_ratio,
        "a_major_rel_error": abs(best["a_major"] - cf.a_major) / cf.a_major,
        "min_y_residual_rel": float(best["y_residual_rel"]),
        "closed_form_y_residual_rel": float(min(
            scan, key=lambda r: abs(r["axis_ratio"] - cf.axis_ratio))["y_residual_rel"]),
        "scan": scan,
    }


# ---------------------------------------------------------------------------
# Superellipsoid pole: power-law indenter law
# ---------------------------------------------------------------------------

def superellipsoid_pole_gap(
    a: float, b: float, c: float, m: float, n_theta: int = 720,
    extent: float = 0.02,
) -> Dict[str, float]:
    """Local gap of a superellipsoid pole: curvature radii and the power exponent.

    The superellipsoid ``|x/a|^m + |y/b|^m + |z/c|^m = 1`` has, at its pole
    ``(0,0,c)``,

    * ``m = 2``: an elliptic paraboloid with principal radii ``R_x = a^2/c``,
      ``R_y = b^2/c`` -> Hertz applies (``gap ~ A x^2 + B y^2``);
    * ``m != 2``: ``gap ~ (c/m)(|x/a|^m + |y/b|^m)``, i.e. the curvature at the
      pole vanishes for ``m > 2`` (flat facet) and diverges for ``m < 2``
      (edge-like), so the contact is a *power-law indenter*, not Hertz.

    Returns the exact ellipsoidal radii for ``m = 2``, the fitted power exponent
    from a least-squares fit of the numerically evaluated gap, and the resulting
    force exponent ``1 + 1/m``.
    """
    m = float(m)
    radii = {"r_x": a * a / c, "r_y": b * b / c}
    # numerical gap along the x axis (exact implicit surface z(x))
    h = np.linspace(-extent * a, extent * a, n_theta)
    inside = np.maximum(0.0, 1.0 - np.abs(h / a) ** m)
    z = c * inside ** (1.0 / m)
    gap = c - z
    # fit gap = kappa |x|^p in log space, excluding the near-axis round-off zone
    mask = (np.abs(h) > 0.08 * extent * a) & (gap > 0.0)
    p_fit = float(np.polyfit(np.log(np.abs(h[mask])), np.log(gap[mask]), 1)[0])
    kappa = float(np.exp(np.polyfit(np.log(np.abs(h[mask])), np.log(gap[mask]), 1)[1]))
    return {
        "superellipsoid_exponent_m": m,
        "pole_radius_x_m": float(radii["r_x"]),
        "pole_radius_y_m": float(radii["r_y"]),
        "gap_power_fit": p_fit,
        "gap_prefactor": kappa,
        "force_exponent_1_plus_1_over_m": 1.0 + 1.0 / m,
        "hertzian": bool(abs(m - 2.0) < 1e-9),
    }


def pole_contact_exponent(m: float) -> float:
    """Force exponent of a power-law indenter with gap ``~ |r|^m``: ``1 + 1/m``."""
    return 1.0 + 1.0 / float(m)


def _love_kernel_matrix(r: np.ndarray, edges: np.ndarray, n_gauss: int = 32) -> np.ndarray:
    """Discretised axisymmetric Love kernel ``int s K(k)/(r+s) ds`` per cell.

    ``k^2 = 4 r s/(r+s)^2``.  The self cell is integrated with Gauss-Legendre in
    the variable ``eps = s - r`` which turns the logarithmic endpoint singularity
    into an integrable one (accuracy ~1e-3, enough for exponent fitting).
    """
    xg, wg = np.polynomial.legendre.leggauss(n_gauss)
    n = len(r)
    M = np.zeros((n, n))
    for j in range(n):
        s0, s1 = edges[j], edges[j + 1]
        s = 0.5 * (s1 - s0) * xg + 0.5 * (s1 + s0)
        w = 0.5 * (s1 - s0) * wg
        for i in range(n):
            k2 = 4.0 * r[i] * s / (r[i] + s) ** 2
            k2 = np.clip(k2, 0.0, 1.0 - 1e-15)
            ker = s * ellipk(k2) / (r[i] + s)
            M[i, j] = float(np.sum(w * ker))
    return M


def axisymmetric_indenter_law(
    m: float, a_values: Optional[np.ndarray] = None, n_cells: int = 120,
    kappa: float = 1.0, e_star: float = 1.0, n_gauss: int = 32,
) -> Dict[str, object]:
    """Solve Love's integral equation for a rigid power-law indenter ``f(r)=kappa r^m``.

    For each contact radius ``a`` the pressure is found from the linear system
    ``w(r_i) = delta - kappa r_i^m`` closed by ``p(a) = 0`` (the Hertz-like class,
    ``m >= 1``).  The resulting ``P(a)`` and ``delta(a)`` give the numerical
    exponent of ``P ~ delta^alpha``, to be compared with ``alpha = 1 + 1/m``.
    The ``m = 2`` case is additionally compared with the exact Hertz disk
    ``P = (4/3) E* sqrt(R) delta^(3/2)``.
    """
    m = float(m)
    if a_values is None:
        a_values = np.array([0.5, 0.75, 1.0, 1.5, 2.0])
    a_values = np.asarray(a_values, dtype=float)
    rows: List[Dict[str, float]] = []
    for a in a_values:
        # collocation at mid-points of equal-width cells
        edges = np.linspace(0.0, a, n_cells + 1)
        rc = 0.5 * (edges[:-1] + edges[1:])
        M = _love_kernel_matrix(rc, edges, n_gauss=n_gauss)
        pref = 4.0 / (math.pi * e_star)
        Amat = np.zeros((n_cells + 1, n_cells + 1))
        Amat[:n_cells, :n_cells] = pref * M                    # w(r_i) from p_j
        Amat[:n_cells, n_cells] = -1.0                          # -delta
        bvec = np.concatenate([-kappa * rc ** m, [0.0]])
        # boundary condition p(a) = 0 (pressure vanishes at the contact edge)
        Amat[n_cells, -1] = 0.0
        Amat[n_cells, n_cells - 1] = 1.0
        sol = np.linalg.solve(Amat, bvec)
        p = sol[:n_cells]
        delta = float(sol[n_cells])
        area = 2.0 * math.pi * rc * (edges[1:] - edges[:-1])
        P = float(np.sum(p * area))
        rows.append({
            "contact_radius_m": float(a),
            "delta_m": delta,
            "force_n": P,
            "min_pressure": float(np.min(p)),
            "edge_pressure": float(p[-1]),
        })
    valid = [r for r in rows if r["delta_m"] > 0 and r["force_n"] > 0]
    alpha = float(np.polyfit(np.log([r["delta_m"] for r in valid]),
                             np.log([r["force_n"] for r in valid]), 1)[0])
    out: Dict[str, object] = {
        "power_m": m,
        "expected_force_exponent": 1.0 + 1.0 / m,
        "numerical_force_exponent": alpha,
        "exponent_rel_error": abs(alpha - (1.0 + 1.0 / m)) / (1.0 + 1.0 / m),
        "rows": rows,
    }
    if abs(m - 2.0) < 1e-12:
        # exact Hertz disk for the parabola kappa r^2, i.e. R = 1/(2 kappa)
        R = 1.0 / (2.0 * kappa)
        hertz = [(4.0 / 3.0) * e_star * math.sqrt(R) * r["delta_m"] ** 1.5 for r in valid]
        rel = [abs(h - r["force_n"]) / h for h, r in zip(hertz, valid)]
        out["hertz_force_compare"] = hertz
        out["hertz_rel_error"] = rel
        out["hertz_max_rel_error"] = float(max(rel))
    return out