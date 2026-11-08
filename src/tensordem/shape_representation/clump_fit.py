"""Greedy covering multi-sphere clump fit and its convergence in ``N``.

Campaign node P27W2-N07, criterion C2 (clump convergence) and the geometry half
of C5 (when may a clump replace the true shape).

The fit used here is a *shell covering*: candidate centres are the true-shape
surface points pulled inward along the inward normal by one target sphere radius
(the radius that a sphere must have to cover the boundary with ``N`` spheres),
the ``N`` centres are chosen by farthest-point sampling (Poisson-disk like), and
each radius is then the exact distance to the surface so that the clump is
inscribed (zero protrusion) unless a global inflation is requested.

The point of the module is not the fit itself but the *measured* error: callers
get ``clump_convergence`` which returns, for ``N = 8, 32, 128``, the two-sided
surface deviation of the clump and the fitted convergence exponent, so the
verdict can state at which ``N`` the representation error stops mattering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..clump_geometry import ClumpFit, geometric_error, sample_surface
from ..superquadric_kernel import implicit_local, signed_distance_batch


def _interior_candidates(a: float, b: float, c: float, n1: float, n2: float,
                         n_cand: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random interior points of the superquadric (rejection sampling)."""
    out = []
    while sum(len(o) for o in out) < n_cand:
        pts = rng.uniform(-1.0, 1.0, size=(4 * n_cand, 3)) * np.array([a, b, c])
        pts = pts[implicit_local(pts, a, b, c, n1, n2) < 0.0]
        if len(pts):
            out.append(pts)
    return np.vstack(out)[:n_cand]


def fit_clump_cover(
    a: float, b: float, c: float, n1: float, n2: float, n_spheres: int,
    *, seed: int = 0, n_surface: int = 2500, offset_fraction: float = 0.5,
    inflate: Optional[float] = None, n_mc: int = 6000,
) -> ClumpFit:
    """Shell-covering multi-sphere fit (inscribed spheres, optional inflation).

    Candidates are the true-shape surface points pulled inward along the inward
    normal by the depth that lets ``N`` spheres cover the boundary
    (``t = 0.5 sqrt(A/(pi N))``, capped by the local inscribed distance), the
    ``N`` centres are chosen by farthest-point sampling, and each radius is the
    exact distance to the surface.  Because every sphere is inscribed the clump
    has zero protrusion; a single global inflation ``alpha`` can be requested
    (``inflate="volume"``) to restore the true volume/mass.
    """
    rng = np.random.default_rng(seed)
    pts, normals = sample_surface(a, b, c, n1, n2,
                                  n_u=int(math.sqrt(n_surface) * 2),
                                  n_v=int(math.sqrt(n_surface)))
    area = _parametric_area(a, b, c, n1, n2)
    r_target = 0.5 * math.sqrt(area / (math.pi * max(n_spheres, 1)))
    t0 = min(offset_fraction * r_target, 0.4 * min(a, b, c))
    first = pts - t0 * normals
    r_local = np.abs(signed_distance_batch(first, a, b, c, n1, n2))
    t = np.minimum(t0, np.maximum(r_local, 1e-9))
    cand = pts - t[:, None] * normals
    inside = implicit_local(cand, a, b, c, n1, n2) < -1e-9
    cand = cand[inside]
    if len(cand) < n_spheres:
        cand = np.vstack([cand, _interior_candidates(a, b, c, n1, n2,
                                                     n_spheres - len(cand), rng)])
    # farthest-point sampling for an even covering
    idx = [int(np.argmax(np.linalg.norm(cand, axis=1)))]
    d = np.linalg.norm(cand - cand[idx[0]], axis=1)
    for _ in range(int(n_spheres) - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(cand - cand[i], axis=1))
    centres = cand[np.array(idx)]
    radii = np.maximum(np.abs(signed_distance_batch(centres, a, b, c, n1, n2)), 1e-9)
    # optional global inflation: measured against a Monte-Carlo volume sample
    alpha = 1.0
    lo, hi = -np.array([a, b, c]), np.array([a, b, c])
    mc = rng.uniform(lo, hi, size=(8 * n_mc, 3))
    mc = mc[implicit_local(mc, a, b, c, n1, n2) < 0.0][:n_mc]
    if inflate == "volume":
        for _ in range(6):
            cov = float(np.mean(_in_any(mc, centres, radii * alpha)))
            if cov <= 1e-9:
                alpha *= 1.6
                continue
            alpha *= cov ** (-1.0 / 3.0)
    elif inflate is not None:
        alpha = float(inflate)
    radii = radii * alpha
    pts_dev, normals_dev = sample_surface(a, b, c, n1, n2,
                                          n_u=int(math.sqrt(n_surface) * 2),
                                          n_v=int(math.sqrt(n_surface)))
    two_sided, _, missing, protrusion = _surface_deviation(
        a, b, c, n1, n2, centres, radii, pts_dev, normals_dev)
    vol_ratio = float(np.mean(_in_any(mc, centres, radii))) / max(
        float(np.mean(implicit_local(mc, a, b, c, n1, n2) < 0.0)), 1e-30)
    return ClumpFit(offsets=centres, radii=radii, eps=alpha - 1.0,
                    two_sided_max=two_sided, volume_ratio=vol_ratio,
                    n_spheres=len(centres), missing_max=missing,
                    protrusion_max=protrusion)


def _in_any(pts: np.ndarray, centres: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Boolean mask of points inside the union of spheres."""
    m = np.zeros(len(pts), dtype=bool)
    for off, r in zip(centres, radii):
        m |= np.linalg.norm(pts - off, axis=1) <= r
    return m


def _parametric_area(a: float, b: float, c: float, n1: float, n2: float,
                     nu: int = 160, nv: int = 80) -> float:
    """Surface area of the superquadric by parametric quadrature."""
    eta = np.linspace(-math.pi / 2 + 1e-6, math.pi / 2 - 1e-6, nv)
    om = np.linspace(-math.pi + 1e-6, math.pi - 1e-6, nu)
    E, O = np.meshgrid(eta, om, indexing="ij")
    ce, se = np.cos(E), np.sin(E)
    co, so = np.cos(O), np.sin(O)
    e2, e1 = 2.0 / n2, 2.0 / n1
    x = a * np.sign(ce) * np.abs(ce) ** e1 * np.sign(co) * np.abs(co) ** e2
    y = b * np.sign(ce) * np.abs(ce) ** e1 * np.sign(so) * np.abs(so) ** e2
    z = c * np.sign(se) * np.abs(se) ** e1
    P = np.stack([x, y, z], axis=-1)
    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    cr = np.cross(du, dv)
    return float(np.sum(np.linalg.norm(cr, axis=-1)))


def _surface_deviation(a: float, b: float, c: float, n1: float, n2: float,
                       centres: np.ndarray, radii: np.ndarray,
                       pts: np.ndarray, normals: np.ndarray):
    """Two-sided surface deviation of a clump (missing / protruding material)."""
    d = np.linalg.norm(pts[:, None, :] - centres[None, :, :], axis=2) - radii[None, :]
    shell = np.abs(d)
    nearest = np.argmin(shell, axis=1)
    dist = shell[np.arange(len(pts)), nearest]
    # missing material: true surface outside every sphere
    gap = np.min(d, axis=1)
    missing = float(np.max(np.maximum(gap, 0.0)))
    # clump -> true surface
    rng = np.random.default_rng(11)
    samples = []
    for off, r in zip(centres, radii):
        dirs = rng.standard_normal((16, 3))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1), 1e-30)[:, None]
        samples.append(off + r * dirs)
    sp = np.vstack(samples)
    sd = np.abs(signed_distance_batch(sp, a, b, c, n1, n2))
    protrusion = float(np.max(sd))
    two_sided = max(float(np.max(dist)), protrusion)
    return two_sided, float(np.mean(dist)), missing, protrusion


def _volume_ratio(a: float, b: float, c: float, n1: float, n2: float,
                  centres: np.ndarray, radii: np.ndarray,
                  rng: np.random.Generator, n_mc: int = 20000) -> float:
    """Monte-Carlo union volume over true volume."""
    lo = -np.array([a, b, c])
    hi = np.array([a, b, c])
    pts = rng.uniform(lo, hi, size=(n_mc, 3))
    box = float(np.prod(hi - lo))
    inside_true = implicit_local(pts, a, b, c, n1, n2) < 0.0
    inside_any = np.zeros(len(pts), dtype=bool)
    for off, r in zip(centres, radii):
        inside_any |= np.linalg.norm(pts - off, axis=1) <= r
    vt = float(np.mean(inside_true)) * box
    vu = float(np.mean(inside_any)) * box
    return vu / vt if vt > 0 else float("nan")


def clump_convergence(
    a: float, b: float, c: float, n1: float, n2: float,
    n_list: Sequence[int] = (8, 32, 128), *, seed: int = 0,
    n_samples: int = 400, inflation: Optional[str] = None,
) -> Dict[str, object]:
    """Surface deviation versus the number of spheres, for two independent fitters.

    ``covering`` is the shell fitter of this module (zero protrusion); ``kmeans``
    is :func:`tensordem.clump_geometry.fit_clump` (volume-matched, used by the
    DEM channels).  Reporting both is deliberate: the convergence claim must not
    depend on one fitting heuristic.
    """
    from ..clump_geometry import fit_clump

    rows: List[Dict[str, object]] = []
    for n in n_list:
        fit_shell = fit_clump_cover(a, b, c, n1, n2, int(n), seed=seed, inflate=inflation)
        err_shell = geometric_error(a, b, c, n1, n2, fit_shell, n_samples=n_samples).to_dict()
        fit_km = fit_clump(a, b, c, n1, n2, n_spheres=int(n), seed=seed)
        err_km = geometric_error(a, b, c, n1, n2, fit_km, n_samples=n_samples).to_dict()
        rows.append({
            "n_spheres": int(n),
            "covering": err_shell,
            "kmeans": err_km,
            "covering_two_sided_fit_m": float(fit_shell.two_sided_max),
            "covering_volume_ratio": float(fit_shell.volume_ratio),
            "kmeans_volume_ratio": float(fit_km.volume_ratio),
        })
    def _slope(key: str) -> float:
        ns = np.array([r["n_spheres"] for r in rows], dtype=float)
        em = np.array([max(r[key]["max_distance_m"], 1e-12) for r in rows], dtype=float)
        return float(np.polyfit(np.log(ns), np.log(em), 1)[0]) if len(rows) > 1 else float("nan")
    return {
        "shape": {"a": a, "b": b, "c": c, "n1": n1, "n2": n2},
        "rows": rows,
        "convergence_exponent_covering": _slope("covering"),
        "convergence_exponent_kmeans": _slope("kmeans"),
        "relative_error_covering": {str(r["n_spheres"]): r["covering"]["max_distance_ratio"] for r in rows},
        "relative_error_kmeans": {str(r["n_spheres"]): r["kmeans"]["max_distance_ratio"] for r in rows},
        "normalisation": "max two-sided deviation / smallest half-extent",
    }
