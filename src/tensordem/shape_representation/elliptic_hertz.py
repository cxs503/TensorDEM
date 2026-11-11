"""Elliptical Hertz contact: closed form plus an independent quadrature reference.

Campaign node P40-N10 (TensorDEM).  The module was dropped from the tree by the
merge gate while its acceptance tests survived; it is restored here to the same
contract (see ``docs/SHAPE_REPRESENTATION_MODULE_RESTORATION.md``).

Two implementations of the *same* contact problem are kept apart on purpose:

``elliptic_hertz_quadrature``
    The **reference**.  For a trial contact ellipse ``(a_tight, t·a_tight)`` the
    half-space displacement under the Hertz pressure ``p0·sqrt(1 - x²/a² - y²/b²)``
    is obtained by direct quadrature of the Boussinesq kernel
    ``w = (1/(pi E*)) iint p/|r-r'| dA'`` (:func:`displacement_quadrature`).  Inside
    the contact the compatibility condition ``w = delta - (A x² + B y²)/2`` fixes the
    axis ratio ``t`` (from ``A/B`` alone, independent of the load), then the semi-axes,
    the peak pressure and the approach follow from linearity in ``p0`` and in
    ``p0 a / E*``.  **No elliptic integral and no Hertz formula is used.**

``elliptic_hertz``
    The **closed form**, delegated to :func:`ellipsoid_plane_contact` (complete
    elliptic integrals), so that the library keeps exactly one implementation of the
    elliptic-integral Hertz geometry.

Conventions
-----------
``A >= B > 0`` are the *curvature coefficients* of the body pressed against a rigid
plane, i.e. the pole gap is ``z = (A x² + B y²)/2``.  For a sphere of radius ``R``
both equal ``1/R`` and the classical Hertz law is recovered exactly:

    delta = (9 F² / (16 E*² R))^(1/3),    k_n = 2 E* sqrt(R delta).

``A > B`` gives an elongated contact ellipse: the semi-axis in the *tight* (large
curvature) direction is the short one (``a_tight``), the semi-axis in the *flat*
direction the long one (``a_flat``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .hertz_ellipsoid import displacement_quadrature, ellipsoid_plane_contact

__all__ = [
    "complete_elliptic_KE",
    "EllipticHertzResult",
    "elliptic_hertz",
    "elliptic_hertz_quadrature",
    "equivalent_radius_hertz",
]


# ---------------------------------------------------------------------------
# Complete elliptic integrals
# ---------------------------------------------------------------------------

def complete_elliptic_KE(k: float) -> Tuple[float, float]:
    """Complete elliptic integrals ``(K(k), E(k))`` for modulus ``k`` (``k = e``).

    ``K(k) = int_0^{pi/2} (1 - k² sin²θ)^(-1/2) dθ`` and
    ``E(k) = int_0^{pi/2} (1 - k² sin²θ)^(1/2) dθ``, evaluated with the
    arithmetic-geometric mean::

        K = pi / (2 AGM(1, sqrt(1-k²))),      E = K (1 - sum_{n>=0} 2^{n-1} c_n²)

    with ``c_0 = k`` and ``c_{n+1} = (a_n - b_n)/2``.  The leading ``n = 0`` term
    ``c_0²/2 = k²/2`` is *part* of the sum: dropping it produces
    ``E(0.9) = 2.0953`` instead of the tabulated ``1.1717`` (a defect of the
    pre-gate implementation, pinned by its own test and corrected here).

    Checked against Abramowitz & Stegun table 17.1 (``k = 0.9``:
    ``K = 2.2805491384227703``, ``E = 1.1716970527816143``) and against the
    Legendre relation below.
    """
    k = float(k)
    if not 0.0 <= k < 1.0:
        raise ValueError("modulus k must satisfy 0 <= k < 1")
    a, b = 1.0, math.sqrt(max(0.0, 1.0 - k * k))
    c = k
    total = 0.5 * c * c                     # n = 0 term: 2^(-1) c_0^2
    for n in range(1, 64):
        c_next = 0.5 * (a - b)
        a, b = 0.5 * (a + b), math.sqrt(a * b)
        total += (2.0 ** (n - 1)) * c_next * c_next
        if abs(c_next) < 1e-17:
            break
    big_k = math.pi / (2.0 * a)
    big_e = big_k * (1.0 - total)
    return big_k, big_e


def legendre_relation(k: float) -> float:
    """Legendre relation ``K E' + E K' - K K' = pi/2`` residual (independent check).

    ``K' = K(sqrt(1-k²))`` etc.  Exactly ``pi/2`` for the correct integrals, so the
    returned value is the residual and should be at round-off level.
    """
    k = float(k)
    if not 0.0 < k < 1.0:
        raise ValueError("the Legendre relation needs 0 < k < 1 (K(0)=pi/2, K(1)=inf)")
    kp = math.sqrt(max(0.0, 1.0 - k * k))
    K, E = complete_elliptic_KE(k)
    Kp, Ep = complete_elliptic_KE(kp)
    return K * Ep + E * Kp - K * Kp


# ---------------------------------------------------------------------------
# Closed form
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EllipticHertzResult:
    """Contact of an ellipsoid pole against a rigid plane under a normal load."""

    A: float                 # curvature coefficient, tight direction (1/m)
    B: float                 # curvature coefficient, flat direction (1/m)
    e_star: float            # reduced modulus (Pa)
    load: float              # normal force (N)
    a_tight: float           # contact semi-axis, tight direction (m)
    a_flat: float            # contact semi-axis, flat direction (m)
    p0: float                # peak pressure (Pa)
    delta: float             # normal approach (m)
    stiffness: float         # dF/ddelta at the given load (N/m)

    @property
    def axis_ratio(self) -> float:
        return self.a_tight / self.a_flat

    def as_dict(self) -> dict:
        return {
            "curvature_coef_A": float(self.A),
            "curvature_coef_B": float(self.B),
            "e_star_pa": float(self.e_star),
            "load_n": float(self.load),
            "a_tight_m": float(self.a_tight),
            "a_flat_m": float(self.a_flat),
            "p0_pa": float(self.p0),
            "delta_m": float(self.delta),
            "stiffness_n_per_m": float(self.stiffness),
        }


def _check(A: float, B: float, e_star: float, load: float) -> Tuple[float, float]:
    A, B = float(A), float(B)
    if B > A:
        A, B = B, A
    if B <= 0.0 or A <= 0.0:
        raise ValueError("curvature coefficients must be positive")
    if e_star <= 0.0 or load <= 0.0:
        raise ValueError("E* and load must be positive")
    return A, B


def elliptic_hertz(A: float, B: float, e_star: float, load: float) -> EllipticHertzResult:
    """Closed-form Hertz contact of an ellipsoid pole against a rigid plane.

    Delegates to :func:`tensordem.shape_representation.hertz_ellipsoid.ellipsoid_plane_contact`,
    which owns the single elliptic-integral implementation of this geometry in the
    library (``A = 1/R_tight`` -> surface radius ``1/A``).
    """
    A, B = _check(A, B, e_star, load)
    res = ellipsoid_plane_contact(1.0 / B, 1.0 / A, float(e_star), load=float(load))
    return EllipticHertzResult(
        A=A, B=B, e_star=float(e_star), load=float(load),
        a_tight=res.a_minor, a_flat=res.a_major, p0=res.p0,
        delta=res.approach, stiffness=res.stiffness,
    )


# ---------------------------------------------------------------------------
# Independent reference: Boussinesq quadrature (no elliptic integrals)
# ---------------------------------------------------------------------------

def _unit_quadratic(t: float, n_theta: int, n_s: int) -> Tuple[float, float, float]:
    """Fit ``w = w0 + kx X² + ky Y²`` inside the unit ellipse ``(t, 1)``.

    Unit pressure and unit modulus are used (the problem is linear in ``p0`` and in
    ``1/E*``), so the fit coefficients are pure geometry.
    """
    frac = np.array([0.25, 0.50, 0.75])
    xs = np.concatenate([[0.0], frac * t, np.zeros_like(frac)])
    ys = np.concatenate([[0.0], np.zeros_like(frac), frac])
    w = displacement_quadrature(xs, ys, t, 1.0, 1.0, 1.0, n_theta=n_theta, n_s=n_s)
    w0 = float(w[0])
    sx = (frac * t) ** 2
    sy = frac ** 2
    dw = w - w0
    kx = float(np.sum(sx * dw[1:4]) / np.sum(sx * sx))
    ky = float(np.sum(sy * dw[4:]) / np.sum(sy * sy))
    return w0, kx, ky


def _axis_ratio(A: float, B: float, n_theta: int, n_s: int) -> Tuple[float, float, float]:
    """Solve for the contact axis ratio ``t = a_tight/a_flat`` from ``|kx|/|ky| = A/B``."""
    target = A / B
    if abs(target - 1.0) < 1e-12:
        w0, kx, ky = _unit_quadratic(1.0, n_theta, n_s)
        return 1.0, w0, 0.5 * (kx + ky)

    def resid(t: float) -> float:
        _, kx, ky = _unit_quadratic(t, n_theta, n_s)
        return abs(kx) / abs(ky) - target

    lo, hi = 1e-4, 1.0
    # resid(1) = 0 (circular symmetry) and resid decreases towards 1
    if resid(lo) < 0.0:
        raise RuntimeError("quadrature is not monotone in the axis ratio")
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if resid(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi / lo - 1.0 < 1e-10:
            break
    t = math.sqrt(lo * hi)
    w0, kx, ky = _unit_quadratic(t, n_theta, n_s)
    return t, w0, kx


def elliptic_hertz_quadrature(
    A: float, B: float, e_star: float, load: float, *,
    n_theta: int = 4096, n_s: int = 32,
) -> EllipticHertzResult:
    """Reference solution of the same contact problem by direct Boussinesq quadrature."""
    A, B = _check(A, B, e_star, load)
    E_star = float(e_star)
    P = float(load)
    t, w0, kx = _axis_ratio(A, B, n_theta, n_s)

    # compatibility w = delta - (A x^2 + B y^2)/2 with c_x = p0 kx /(E* a) = -A/2
    p0 = (3.0 * P * E_star ** 2 * A ** 2 / (8.0 * math.pi * t * kx * kx)) ** (1.0 / 3.0)
    a_flat = -2.0 * p0 * kx / (E_star * A)
    a_tight = t * a_flat
    delta = (p0 * a_flat / E_star) * w0
    if delta <= 0.0:
        raise RuntimeError("non-positive approach from the quadrature reference")
    return EllipticHertzResult(
        A=A, B=B, e_star=E_star, load=P, a_tight=a_tight, a_flat=a_flat,
        p0=p0, delta=delta, stiffness=1.5 * P / delta,
    )


def equivalent_radius_hertz(A: float, B: float, e_star: float, load: float) -> EllipticHertzResult:
    """Isotropic (equivalent-radius) shortcut: the error source quantified by P40-N10.

    One sphere with ``1/R_eq = (A + B)/2`` is used instead of the elliptical patch.
    Exact for ``A = B``; the deviation from :func:`elliptic_hertz_quadrature` at
    ``A/B = 8`` is the representation error the node reports.
    """
    A, B = _check(A, B, e_star, load)
    r_eq = 2.0 / (A + B)
    delta = (9.0 * float(load) ** 2 / (16.0 * float(e_star) ** 2 * r_eq)) ** (1.0 / 3.0)
    stiffness = 2.0 * float(e_star) * math.sqrt(r_eq * delta)
    a_tight = math.sqrt(r_eq * delta)       # circle of the same approach
    return EllipticHertzResult(
        A=A, B=B, e_star=float(e_star), load=float(load),
        a_tight=a_tight, a_flat=a_tight,
        p0=float(load) / (math.pi * a_tight ** 2),
        delta=delta, stiffness=stiffness,
    )