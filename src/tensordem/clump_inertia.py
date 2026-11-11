"""Analytic mass properties of a rigid multi-sphere (clump) union.

Motivation
----------
``tensordem.particle.clump.ClumpParticle`` fills ``mass``/``inertia`` from a
*voxel* quadrature (``voxel_mass_properties``, ``grid=72``).  A voxel estimate is
a numerical reference only: there is no closed-form value to compare it with for
overlapping spheres, so an error in the voxel integrand (one-sided sampling,
overlap double counting, grid too coarse for a slender body) can hide behind an
apparent agreement with a second voxel run of similar resolution.

This module supplies the missing *analytic* reference.  Every member sphere is
integrated in closed form, and the union is obtained by inclusion-exclusion:

    U = sum_k B_k  -  sum_{k<l} (B_k n B_l)  +  sum_{k<l<m} (B_k n B_l n B_m) - ...

Pair intersections (lenses) are computed exactly from spherical-cap moments.
Triple intersections are *not* elementary in closed form; for clumps whose
members form a chain (the usual DEM construction) they are either empty or a
thin sliver, so they are evaluated numerically in a small local box and the
residual fraction is reported (``third_order_fraction``).  When that fraction is
below ``triple_tol`` the returned properties are analytic to that tolerance;
above it the result is flagged ``exact=False`` and the diagnostic is exposed so
a caller can refine the grid instead of silently trusting the number.

Conventions
-----------
* All moments are returned for the *union*, about the union centre of mass, in
  the frame of the input member centres (``com`` is reported as well).
* ``inertia`` is the second-moment matrix ``I = density * int (|r|^2 1 - r r^T) dV``
  integrated over the union (the tensor used by the rigid-body equations), i.e.
  the tensor whose diagonal is ``I_xx = int (y^2+z^2) dm``.
* SI units throughout; ``density`` is solid density (mass per union volume).

The cap and lens formulas are textbook (Landau & Lifshitz, *Mechanics*, §33 for
the tensor definition; the cap moments are elementary integrals) and are
exercised against the exact sphere, the tangent dumbbell and a high-resolution
voxel quadrature in ``tests/test_clump_inertia_analytic.py``.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

__all__ = [
    "sphere_moments",
    "sphere_cap_moments",
    "lens_moments",
    "union_mass_properties",
    "compare_with_voxel",
]

_FOUR_PI_OVER_3 = 4.0 / 3.0 * math.pi


# ---------------------------------------------------------------------------
# Closed-form pieces (local frame: axis = +x, origin = sphere centre)
# ---------------------------------------------------------------------------
def _cap_volume(r: float, h: float) -> float:
    return math.pi / 3.0 * (2.0 * r**3 - 3.0 * r * r * h + h**3)


def _cap_first_moment(r: float, h: float) -> float:
    """int_{cap} x dV."""
    return math.pi * (r**4 / 4.0 - r * r * h * h / 2.0 + h**4 / 4.0)


def _cap_second_axial(r: float, h: float) -> float:
    """int_{cap} x^2 dV."""
    return r * r * (r**3 - h**3) / 3.0 - (r**5 - h**5) / 5.0
    # (pi factored by the caller through the bracket below)


def _cap_second_axial_pi(r: float, h: float) -> float:
    return math.pi * _cap_second_axial(r, h)


def _cap_transverse(r: float, h: float) -> float:
    """int_{cap} (y^2 + z^2) dV."""
    prim = r**4 * r - 2.0 / 3.0 * r * r * r**3 + r**5 / 5.0
    prim -= r**4 * h - 2.0 / 3.0 * r * r * h**3 + h**5 / 5.0
    return math.pi / 2.0 * prim


def sphere_cap_moments(r: float, h: float) -> dict:
    """Exact moments of the cap ``{x >= h}`` of a sphere of radius ``r``.

    Returns ``volume``, ``first_axial`` = int x dV, ``second_axial`` = int x^2 dV,
    ``transverse`` = int (y^2+z^2) dV, all about the *sphere centre* with the cut
    axis along +x.  ``h == -r`` gives the whole sphere, ``h == +r`` gives zero.
    """
    if not (r > 0.0):
        raise ValueError("radius must be positive")
    if h <= -r:
        return {
            "volume": _FOUR_PI_OVER_3 * r**3,
            "first_axial": 0.0,
            "second_axial": 4.0 * math.pi * r**5 / 15.0,
            "transverse": 8.0 * math.pi * r**5 / 15.0,
        }
    if h >= r:
        return {"volume": 0.0, "first_axial": 0.0, "second_axial": 0.0, "transverse": 0.0}
    return {
        "volume": _cap_volume(r, h),
        "first_axial": _cap_first_moment(r, h),
        "second_axial": _cap_second_axial_pi(r, h),
        "transverse": _cap_transverse(r, h),
    }


def sphere_moments(center: Sequence[float], radius: float, density: float) -> dict:
    """Exact properties of one sphere about the requested frame origin."""
    c = np.asarray(center, dtype=float).reshape(3)
    m = density * _FOUR_PI_OVER_3 * radius**3
    i_own = 0.4 * m * radius * radius
    d = c  # offset of the sphere centre from the frame origin
    inertia = i_own * np.eye(3) + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {"volume": _FOUR_PI_OVER_3 * radius**3, "mass": m,
            "first": m * c, "inertia": inertia}


# ---------------------------------------------------------------------------
# Pair intersection (lens) -- exact
# ---------------------------------------------------------------------------
def lens_moments(c1: Sequence[float], r1: float, c2: Sequence[float], r2: float,
                 density: float = 1.0) -> dict | None:
    """Exact moments of ``B(c1,r1) n B(c2,r2)`` about the frame origin.

    Returns ``None`` when the balls do not intersect.  Handles containment.
    The result has keys ``volume``, ``mass``, ``first`` (3-vector, = mass*com),
    ``inertia`` (3x3 about the frame origin) and ``com``.
    """
    a = np.asarray(c1, dtype=float).reshape(3)
    b = np.asarray(c2, dtype=float).reshape(3)
    dvec = b - a
    d = float(np.linalg.norm(dvec))
    if d >= r1 + r2:  # disjoint (touching measure-zero intersection)
        return None
    if d <= abs(r2 - r1):  # containment
        inner_c, inner_r = (b, r2) if r2 <= r1 else (a, r1)
        return sphere_moments(inner_c, inner_r, density)
    u = dvec / d
    x0 = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)  # split plane, measured from c1

    def _as_frame1(mom: dict, shift: float) -> dict:
        """Shift cap moments along +u by ``shift`` (moments were about a centre)."""
        v = mom["volume"]
        m1 = mom["first_axial"] + shift * v
        m2 = mom["second_axial"] + 2.0 * shift * mom["first_axial"] + shift * shift * v
        return {"volume": v, "first_axial": m1, "second_axial": m2,
                "transverse": mom["transverse"]}

    cap1 = _as_frame1(sphere_cap_moments(r1, x0), 0.0)
    full2 = _as_frame1(sphere_cap_moments(r2, -r2), d)
    hole2 = _as_frame1(sphere_cap_moments(r2, x0 - d), d)  # {x >= x0} of sphere 2
    parts = [cap1, full2]
    signs = [1.0, 1.0, -1.0]
    parts.append(hole2)

    volume = sum(s * p["volume"] for s, p in zip(signs, parts))
    first_x = sum(s * p["first_axial"] for s, p in zip(signs, parts))
    second_x = sum(s * p["second_axial"] for s, p in zip(signs, parts))
    transverse = sum(s * p["transverse"] for s, p in zip(signs, parts))
    if volume <= 0.0:
        return None
    xbar = first_x / volume
    axial = transverse  # about the axis u (through the origin)
    second_about_com = second_x - volume * xbar * xbar
    perp = second_about_com + 0.5 * transverse
    com = a + xbar * u
    # assemble the tensor in the global frame: build a rotation taking x-> u
    rot = _axis_frame(u)
    inertia_local = np.diag([axial, perp, perp]) * density
    inertia_com = rot @ inertia_local @ rot.T
    m = density * volume
    # express the tensor about the *frame origin* (the same frame in which
    # ``sphere_moments`` reports), so that the inclusion-exclusion sums are
    # parallel-axis consistent.
    inertia = inertia_com + m * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
    return {"volume": volume, "mass": m, "first": m * com, "inertia": inertia,
            "com": com, "inertia_com": inertia_com}


def _axis_frame(u: np.ndarray) -> np.ndarray:
    """Rotation matrix whose first column is the unit vector ``u``."""
    u = u / np.linalg.norm(u)
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(u, helper))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(u, helper)
    e2 /= np.linalg.norm(e2)
    e3 = np.cross(u, e2)
    return np.column_stack([u, e2, e3])


# ---------------------------------------------------------------------------
# Local numerical intersection (used only for order >= 3 corrections)
# ---------------------------------------------------------------------------
def _intersection_moments_numeric(centers: np.ndarray, radii: np.ndarray,
                                 density: float, spacing: float) -> dict:
    """Voxel moments of the common intersection of the given balls."""
    lo = (centers - radii[:, None]).min(axis=0)
    hi = (centers + radii[:, None]).max(axis=0)
    n = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 1)
    axes = [lo[k] + (np.arange(n[k]) + 0.5) * (hi[k] - lo[k]) / n[k] for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    inside = np.ones(X.shape, dtype=bool)
    for c, r in zip(centers, radii):
        inside &= ((X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2) <= r * r
    if not inside.any():
        return {"volume": 0.0, "mass": 0.0, "first": np.zeros(3),
                "inertia": np.zeros((3, 3))}
    dx = np.diff(axes[0])[0]
    dy = np.diff(axes[1])[0]
    dz = np.diff(axes[2])[0]
    dv = dx * dy * dz
    pts = np.stack([X[inside], Y[inside], Z[inside]], axis=1)
    com = pts.mean(axis=0)
    d = pts - com
    inertia = density * dv * (np.eye(3) * float(np.sum(d * d)) - d.T @ d)
    vol = dv * pts.shape[0]
    return {"volume": vol, "mass": density * vol, "first": density * vol * com,
            "inertia": inertia, "com": com}


def _combine(parts: Iterable[Tuple[float, dict]]) -> dict:
    vol = 0.0
    first = np.zeros(3)
    inertia = np.zeros((3, 3))
    for s, p in parts:
        vol += s * p["volume"]
        first += s * p["first"]
        inertia += s * p["inertia"]
    m = sum(s * p["mass"] for s, p in parts)
    com = first / m if m != 0 else np.zeros(3)
    return {"volume": vol, "mass": m, "first": first, "com": com,
            "inertia": inertia}


# ---------------------------------------------------------------------------
# Union by inclusion-exclusion
# ---------------------------------------------------------------------------
def union_mass_properties(centers: Sequence[Sequence[float]], radii: Sequence[float],
                          density: float = 1.0, *, max_order: int = 3,
                          triple_tol: float = 1e-3) -> dict:
    """Analytic union mass properties of a multi-sphere clump.

    Parameters
    ----------
    centers, radii : member geometry (body frame, any origin -- the returned
        ``com`` is in the same frame).
    density : solid density.
    max_order : inclusion-exclusion order (2 = pair lenses only, 3 = pair lenses
        plus numerically evaluated triple intersections).
    triple_tol : if the third-order correction exceeds this fraction of the union
        volume the result is flagged ``exact=False`` (the caller should refine).

    Returns
    -------
    dict with ``volume``, ``mass``, ``com``, ``inertia`` (about the COM),
    ``n_members``, ``order``, ``third_order_fraction``, ``exact``,
    ``min_overlap`` (deepest pairwise relative overlap) and ``max_order_used``.
    """
    c = np.asarray(centers, dtype=float).reshape(-1, 3)
    r = np.asarray(radii, dtype=float).reshape(-1)
    if c.shape[0] != r.shape[0] or c.shape[0] == 0:
        raise ValueError("centers and radii must be non-empty and the same length")
    if np.any(r <= 0.0):
        raise ValueError("all radii must be positive")
    n = c.shape[0]
    # move to an origin near the geometric centre for conditioning
    origin = c.mean(axis=0)
    cs = c - origin

    parts: List[Tuple[float, dict]] = []
    for k in range(n):
        parts.append((1.0, sphere_moments(cs[k], r[k], density)))

    pairs: List[Tuple[int, int]] = []
    min_gap = math.inf
    for k in range(n):
        for l in range(k + 1, n):
            gap = float(np.linalg.norm(cs[k] - cs[l])) - (r[k] + r[l])
            min_gap = min(min_gap, gap)
            lens = lens_moments(cs[k], r[k], cs[l], r[l], density)
            if lens is not None:
                parts.append((-1.0, lens))
                pairs.append((k, l))

    union0 = _combine(parts)
    third_vol = 0.0
    triples: List[Tuple[int, ...]] = []
    if max_order >= 3 and n >= 3:
        spacing = float(r.min()) / 45.0
        for k in range(n):
            for l in range(k + 1, n):
                if (k, l) not in pairs:
                    continue
                for m in range(l + 1, n):
                    sel = (k, l, m)
                    sub_c = cs[list(sel)]
                    sub_r = r[list(sel)]
                    # cheap reject: a triple intersection needs all three pair lenses
                    needed = [s for s in ((k, l), (k, m), (l, m))]
                    if any(s not in pairs for s in needed):
                        continue
                    lo = (sub_c - sub_r[:, None]).max(axis=0)
                    hi = (sub_c + sub_r[:, None]).min(axis=0)
                    if np.any(hi <= lo):
                        continue
                    tri = _intersection_moments_numeric(sub_c, sub_r, density, spacing)
                    if tri["volume"] <= 1e-12 * union0["volume"]:
                        continue
                    parts.append((1.0, tri))
                    triples.append(sel)
                    third_vol += tri["volume"]

    total = _combine(parts)
    com = total["com"] + origin
    # parallel-axis shift of the tensor from the intermediate origin to the COM
    shift = total["com"]
    inertia = total["inertia"] - total["mass"] * (
        np.dot(shift, shift) * np.eye(3) - np.outer(shift, shift)
    )
    frac = third_vol / total["volume"] if total["volume"] > 0 else 0.0
    return {
        "volume": float(total["volume"]),
        "mass": float(total["mass"]),
        "com": com,
        "inertia": inertia,
        "n_members": int(n),
        "max_order_used": int(max_order),
        "n_pairs": len(pairs),
        "n_triples": len(triples),
        "third_order_fraction": float(frac),
        "exact": bool(frac <= triple_tol),
        "min_gap": float(min_gap) if np.isfinite(min_gap) else 0.0,
    }


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------
def compare_with_voxel(particle, grid: int = 144) -> dict:
    """Confront ``particle.inertia`` (voxel, production) with the analytic value.

    ``particle`` is any object exposing ``members`` (each with ``center`` and
    ``radius``), ``density`` and ``inertia``.  Also runs a second, finer voxel
    quadrature as an independent check of the analytic tensor.
    """
    from tensordem.particle.clump import voxel_mass_properties

    centers = np.array([m.center for m in particle.members], dtype=float)
    radii = np.array([m.radius for m in particle.members], dtype=float)
    analytic = union_mass_properties(centers, radii, particle.density)
    ref = np.asarray(particle.inertia, dtype=float)
    scale = float(np.max(np.abs(np.diag(analytic["inertia"]))))
    err = float(np.max(np.abs(np.diag(ref) - np.diag(analytic["inertia"]))) / scale)
    fine = voxel_mass_properties(particle.members, particle.density, grid=grid)
    err_fine = float(
        np.max(np.abs(np.diag(fine["inertia"]) - np.diag(analytic["inertia"]))) / scale
    )
    err_mass = abs(float(particle.mass) - analytic["mass"]) / analytic["mass"]
    return {
        "analytic_inertia_diag": np.diag(analytic["inertia"]).tolist(),
        "production_inertia_diag": np.diag(ref).tolist(),
        "fine_voxel_inertia_diag": np.diag(fine["inertia"]).tolist(),
        "rel_err_production": err,
        "rel_err_fine_voxel": err_fine,
        "rel_err_mass": err_mass,
        "analytic": analytic,
        "grid": grid,
    }