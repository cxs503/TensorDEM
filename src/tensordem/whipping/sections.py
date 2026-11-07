"""Two-dimensional added-mass coefficients by a boundary element method.

For vertical hull-girder vibration the water is in the high-frequency regime
(``omega^2 B / (2 g) >> 1``), where the linear free-surface condition
``dphi/dz = (omega^2/g) phi`` degenerates to the Dirichlet condition
``phi = 0``: the free surface behaves like a rigid wall and the added mass
tends to its classical "infinite frequency" value.

The wetted section is discretised into straight panels carrying a constant
source strength ``sigma`` with the 2-D Green function ``G = ln(r)/(2 pi)``.
The Dirichlet condition on the still-water line is imposed *exactly* (not
discretely) by the odd mirror image of the section about the waterline with
negated source strengths: the mirrored sources make ``phi`` vanish identically
on ``z = 0`` and the paired strengths sum to zero, so the potential decays in
the far field without an extra constraint.  For a deeply submerged section
(``free_surface=False``) the body is closed and the zero-total-strength
constraint is enforced through one Lagrange multiplier row/column.

Panel integrals are evaluated in closed form (the logarithmic single-layer and
the ``1/r`` double-layer integrals over a straight segment both have analytic
antiderivatives), so no quadrature error enters the influence matrix; the only
error is the panel discretisation.  The added mass follows from the Bernoulli
reaction

    m_a = -rho int_body phi n_z ds ,

with ``n_z`` the vertical component of the outward (fluid-side) normal and the
Neumann data ``dphi/dn = n_z`` for unit vertical body velocity (``phi`` is then
negative where ``n_z > 0``).

Verification anchors (see ``tests/test_whipping_added_mass.py``):

* full ellipse of semi-axes ``a`` (horizontal), ``b`` (vertical), deep:
  ``m_a = rho pi a^2``, independent of ``b``;
* semi-ellipse / semicircle with the flat face on the still-water line:
  ``m_a = rho pi a^2 / 2``.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "ellipse_section",
    "box_section",
    "section_normals",
    "added_mass_per_length",
    "added_mass_ellipse_reference",
    "added_mass_semielipse_reference",
]


def section_normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Panel midpoints, unit normals (pointing away from the body) and lengths."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or p.shape[0] < 3:
        raise ValueError("section points must be an (M, 2) array with M >= 3")
    if not np.isfinite(p).all():
        raise ValueError("section points must be finite")
    seg = p[1:] - p[:-1]
    length = np.linalg.norm(seg, axis=1)
    if np.any(length <= 1e-12):
        raise ValueError("section contains a zero-length segment")
    tangent = seg / length[:, None]
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    mid = 0.5 * (p[1:] + p[:-1])
    centroid = p.mean(axis=0)
    if np.mean(np.sum((mid - centroid) * normal, axis=1)) < 0.0:
        normal = -normal
    return mid, normal, length


def _panel_integrals(x: np.ndarray, n: np.ndarray,
                     a_all: np.ndarray, b_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form single- and double-layer integrals of straight source panels.

    ``x`` is one collocation point, ``n`` the unit normal there, and
    ``a_all``/``b_all`` the ``(M, 2)`` panel end points.  Returns arrays of
    length ``M`` with ``int G ds`` and ``int dG/dn ds`` for unit strength.
    """
    t_vec = b_all - a_all
    length = np.linalg.norm(t_vec, axis=1)
    tang = t_vec / length[:, None]
    d = x[None, :] - a_all
    u = np.sum(d * tang, axis=1)
    perp2 = np.sum(d * d, axis=1) - u * u
    eps = 1e-18 * np.maximum(length * length, 1.0)
    perp2 = np.maximum(perp2, eps)
    root = np.sqrt(perp2)
    c1 = d @ n
    c2 = tang @ n

    def single(s: np.ndarray) -> np.ndarray:
        # antiderivative of 0.5 log((s-u)^2 + perp2)
        w = s - u
        return 0.5 * (w * np.log(w * w + perp2) - 2.0 * w + 2.0 * root * np.arctan(w / root))

    def double(s: np.ndarray) -> np.ndarray:
        # integrand (c1 - c2 s) / ((s-u)^2 + perp2) with s measured in metres
        w = s - u
        return (c1 - c2 * u) * np.arctan(w / root) / root \
            - 0.5 * c2 * np.log(w * w + perp2)

    pot = (single(length) - single(np.zeros_like(length))) / (2.0 * math.pi)
    dip = (double(length) - double(np.zeros_like(length))) / (2.0 * math.pi)
    return pot, dip


def _added_mass(points: np.ndarray, rho: float, free_surface: bool) -> dict:
    mid, normal, length = section_normals(points)
    a_all, b_all = points[:-1], points[1:]
    n_panels = mid.shape[0]

    if free_surface:
        mirrored_a = np.stack([a_all[:, 0], -a_all[:, 1]], axis=1)
        mirrored_b = np.stack([b_all[:, 0], -b_all[:, 1]], axis=1)

    a_mat = np.zeros((n_panels, n_panels))
    for i in range(n_panels):
        _, dip = _panel_integrals(mid[i], normal[i], a_all, b_all)
        if free_surface:
            _, dip_m = _panel_integrals(mid[i], normal[i], mirrored_a, mirrored_b)
            dip = dip - dip_m
        dip[i] = 0.0  # straight self panel: principal value vanishes
        a_mat[i, :] = dip
    # outgoing-normal jump of the single-layer normal derivative (+sigma/2)
    np.fill_diagonal(a_mat, np.diag(a_mat) + 0.5)

    rhs = normal[:, 1].copy()

    if free_surface:
        sigma = np.linalg.solve(a_mat, rhs)
    else:
        aug = np.zeros((n_panels + 1, n_panels + 1))
        aug[:n_panels, :n_panels] = a_mat
        aug[:n_panels, n_panels] = 1.0
        aug[n_panels, :n_panels] = length
        sol = np.linalg.solve(aug, np.concatenate([rhs, [0.0]]))
        sigma = sol[:n_panels]

    phi = np.zeros(n_panels)
    for i in range(n_panels):
        pot, _ = _panel_integrals(mid[i], normal[i], a_all, b_all)
        if free_surface:
            pot_m, _ = _panel_integrals(mid[i], normal[i], mirrored_a, mirrored_b)
            pot = pot - pot_m
        pot[i] = _self_potential(length[i])
        phi[i] = float(sigma @ pot)

    m_a = -rho * float(np.sum(phi * normal[:, 1] * length))
    return {
        "added_mass_per_length": m_a,
        "panel_count": int(n_panels),
        "min_phi": float(phi.min()),
        "max_phi": float(phi.max()),
        "rhs_residual": float(
            np.linalg.norm(a_mat @ sigma - rhs) / max(np.linalg.norm(rhs), 1e-300)
        ),
    }


def _self_potential(length: float) -> float:
    """``int_{-L/2}^{L/2} ln|s| / (2 pi) ds = (L / pi) (ln(L/2) - 1)``."""
    h = 0.5 * length
    # int_{-h}^{h} ln|s| ds = 2 h (ln h - 1)
    return (2.0 * h * (math.log(h) - 1.0)) / (2.0 * math.pi)


def added_mass_per_length(points, rho: float, *, free_surface: bool = True) -> dict:
    """Added mass per unit length of a 2-D section (kg/m).

    ``points`` is the wetted contour.  For ``free_surface=True`` the contour
    must be open, running from the port waterline point around the keel to the
    starboard waterline point (both on ``z = 0``); for ``free_surface=False``
    the contour must be closed and deep.
    """
    if not math.isfinite(rho) or rho <= 0.0:
        raise ValueError("rho must be finite and positive")
    p = np.asarray(points, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("section points must be finite")
    if free_surface:
        tol = 1e-9 * max(1.0, float(np.abs(p).max()))
        if abs(p[0, 1]) > tol or abs(p[-1, 1]) > tol:
            raise ValueError("open free-surface contour must start and end on z = 0")
    else:
        if np.linalg.norm(p[0] - p[-1]) > 1e-9 * max(1.0, float(np.abs(p).max())):
            raise ValueError("deep section contour must be closed")
    return _added_mass(p, rho, free_surface)


# --------------------------------------------------------------------------- #
# section generators + analytic references
# --------------------------------------------------------------------------- #
def ellipse_section(a: float, b: float, n: int = 240, *, deep: bool = True) -> np.ndarray:
    """Ellipse with semi-axes ``a`` (horizontal) and ``b`` (vertical).

    ``deep=True`` returns the closed contour (repeated end point);
    ``deep=False`` the lower half, with the flat face on the still-water line.
    """
    if not (a > 0 and b > 0):
        raise ValueError("semi-axes must be positive")
    if deep:
        th = np.linspace(0.0, 2.0 * math.pi, n + 1)
        return np.stack([a * np.cos(th), b * np.sin(th)], axis=1)
    th = np.linspace(math.pi, 2.0 * math.pi, n + 1)
    pts = np.stack([a * np.cos(th), b * np.sin(th)], axis=1)
    pts[0, 1] = 0.0
    pts[-1, 1] = 0.0
    return pts


def box_section(breadth: float, draft: float, n_per_side: int = 40) -> np.ndarray:
    """Box section (flat bottom, vertical sides) with the deck on ``z = 0``."""
    if not (breadth > 0 and draft > 0):
        raise ValueError("breadth and draft must be positive")
    b = 0.5 * breadth
    left = np.stack([np.full(n_per_side, -b), np.linspace(0.0, -draft, n_per_side)], axis=1)
    bottom = np.stack(
        [np.linspace(-b, b, n_per_side + 1)[1:], np.full(n_per_side, -draft)], axis=1)
    right = np.stack([np.full(n_per_side, b), np.linspace(-draft, 0.0, n_per_side + 1)[1:]], axis=1)
    return np.vstack([left, bottom, right])


def added_mass_ellipse_reference(rho: float, a: float) -> float:
    """Exact deep-water heave added mass of a full ellipse: ``rho pi a^2``."""
    return rho * math.pi * a * a


def added_mass_semielipse_reference(rho: float, a: float) -> float:
    """Exact high-frequency added mass of a semi-ellipse on the waterline: ``rho pi a^2/2``."""
    return rho * math.pi * a * a / 2.0