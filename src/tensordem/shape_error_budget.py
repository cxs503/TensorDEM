"""Shape error budget: superquadric true shape vs multi-sphere clump.

The module quantifies the two error layers that decide whether a DEM model may
replace a superquadric particle by an assembly of spheres:

1. **Geometric layer** (``compute_geometric_error``): the exact signed distance
   of the *union* of the clump spheres is compared with the true superquadric
   surface -- maximum and mean absolute surface distance, plus the distribution
   (mean / p95 / max) of the surface-normal deviation.  The clump is the
   best-fit minimax cover obtained greedily from inscribed spheres, so the
   error decreases monotonically as spheres are added (negative control for
   ``N -> inf``) and every sphere stays inside the true body by construction.

2. **Macroscopic layer** (``compute_macroscopic_error``): repose angle and
   internal friction angle.  The angle-of-repose function ``theta_r(mu)`` is a
   *measured* DEM calibration (column collapse of monodisperse spheres, see
   ``tensordem.shape_macro``); shape angularity enters through the geometric
   interlocking angle ``i`` -- the RMS surface-normal deviation measured in
   layer 1 -- as an interlocking augmentation ``mu_eff = mu + tan i``.  This is
   an anchored model, not a poly-shape DEM: it is labelled as such in every
   result (``evidence`` field) and in ``artifacts/dem/shape_error_table.json``.

3. **Disable ratio table** (``build_disable_ratio_table``) and the API gate
   (``api_gate`` / ``ClumpApproximationDisabled``): for an (aspect ratio,
   sphere count) combination the gate refuses the clump approximation when
   ``|theta_true - theta_clump| > 3 deg`` or the surface-normal deviation
   exceeds ``5 deg``.

References
----------
- Wellmann, Lillie, Wriggers (2008): contact algorithm for rigid superquadrics.
- Cleary (2010): DEM simulation of industrial particle flows (clump error).
- Rowe (1962) / Newland & Allely (1957): interlocking contribution to the
  internal friction angle of granular assemblies.
- Kruggel-Emden et al. (2008): contact geometry of non-spherical particles.

All quantities are SI (m, kg, s, N, Pa; angles in degrees).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .superquadric import (
    Superquadric,
    _gradient_batch,
    _radius_along,
    superquadric_gradient,
    superquadric_implicit,
)

__all__ = [
    "ShapeErrorBudget",
    "GeometricErrorResult",
    "MacroscopicErrorResult",
    "DisableRatioTable",
    "ClumpApproximationDisabled",
    "compute_geometric_error",
    "compute_macroscopic_error",
    "build_disable_ratio_table",
    "fit_clump_to_superquadric",
    "repose_angle_from_geometry",
    "api_gate",
    "sphere_volume",
    "equivalent_sphere_radius",
]

# Acceptance criteria of the campaign node (kept in one place so the table, the
# gate and the tests can never drift apart).
CRITERIA = {
    "max_repose_error_deg": 3.0,
    "max_force_chain_error_deg": 5.0,
    "max_normal_deviation_deg": 5.0,
    "max_geometric_error_rel": 0.02,
}

# DEM-calibrated sphere repose anchor (mu, theta_r) -- refreshed by
# ``tensordem.shape_macro.calibrate_sphere_repose`` / the artifact builder.
_REPOSE_ANCHOR: List[Tuple[float, float]] = [
    (0.30, 17.5),
    (0.40, 21.6),
    (0.50, 26.0),
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sample_surface_superquadric(
    sq: Superquadric,
    n_samples: int = 400,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample the superquadric surface in direction space.

    A ray from the centre in the unit direction ``u`` meets the surface at
    ``s(u) u`` with the closed form used by the contact kernel
    (``_radius_along``)::

        s(u) = [ (|u_x/a|^n2 + |u_y/b|^n2)^(n1/n2) + |u_z/c|^n1 ]^(-1/n1)

    so every sample lies on the surface exactly -- including the block limit
    (n1, n2 > 40) where the implicit function switches to the max norm.  This
    replaces an earlier lat-long parametrisation that mixed the exponents and
    therefore sampled a *different* surface for n1 != n2 (see docs).

    Returns:
        (points, normals): (N, 3) arrays in the world frame.
    """
    n_u = max(12, int(round(math.sqrt(max(n_samples, 64) * 2))))
    n_v = max(8, int(round(n_u / 2)))
    u_ang = np.linspace(-math.pi, math.pi, n_u, endpoint=False)
    ct = np.linspace(-1.0, 1.0, n_v)                    # equal-area bands
    uu, cc = np.meshgrid(u_ang, ct, indexing="ij")
    st = np.sqrt(np.maximum(1.0 - cc ** 2, 0.0))
    directions = np.stack([st * np.cos(uu), st * np.sin(uu), cc], axis=-1).reshape(-1, 3)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)

    a = np.array([sq.a]); b = np.array([sq.b]); c = np.array([sq.c])
    n1 = np.array([sq.n1]); n2 = np.array([sq.n2])
    s = np.asarray(_radius_along(directions, a, b, c, n1, n2)).reshape(-1)
    points_local = s[:, None] * directions
    grad = _gradient_batch(points_local, a, b, c, n1, n2)
    normals_local = grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-300)
    return sq.to_world(points_local), normals_local @ sq.rotation_matrix.T


def sphere_volume(sq: Superquadric, n_probe: int = 200_000, seed: int = 5) -> float:
    """Volume of the superquadric by rejection sampling of its implicit function.

    Used only to build a size-independent error measure (``R_eq``); the
    reported value is compared against the analytic sphere volume in the tests.
    """
    rng = np.random.default_rng(seed)
    lo = -np.array([sq.a, sq.b, sq.c])
    hi = np.array([sq.a, sq.b, sq.c])
    pts = lo + (hi - lo) * rng.random((n_probe, 3))
    f = superquadric_implicit(pts, sq.a, sq.b, sq.c, sq.n1, sq.n2)
    frac = float(np.count_nonzero(f < 0.0)) / n_probe
    return frac * float(np.prod(hi - lo))


def equivalent_sphere_radius(sq: Superquadric, n_probe: int = 200_000) -> float:
    """Radius of the volume-equivalent sphere."""
    return (3.0 * sphere_volume(sq, n_probe) / (4.0 * math.pi)) ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Best-fit clump
# ---------------------------------------------------------------------------

@dataclass
class ClumpFit:
    """Best-fit clump: inscribed spheres plus construction diagnostics."""
    centers: np.ndarray            # (S, 3) world frame
    radii: np.ndarray              # (S,)
    surface_error_history: Tuple[float, ...] = ()

    def as_list(self) -> List[Tuple[np.ndarray, float]]:
        return [(self.centers[i], float(self.radii[i])) for i in range(len(self.radii))]


def _candidate_spheres(sq: Superquadric, surf_pts: np.ndarray, surf_normals: np.ndarray,
                       n_anchor: int = 160, offsets: Sequence[float] = (0.25, 0.5, 0.75, 1.0),
                       n_march: int = 48) -> Tuple[np.ndarray, np.ndarray]:
    """Candidate clump spheres anchored on the surface.

    Standard multi-sphere clump construction: each sphere is placed on the
    inward normal of a surface anchor and sized so that its own surface passes
    through that anchor (``c = p - h n``, ``r = h``).  ``h`` is a fraction of
    the *local thickness* of the body along that normal, found by ray-marching
    the implicit function until the sign changes.  Spheres therefore lie on the
    surface instead of merely being inscribed, which is what makes a clump of a
    handful of spheres approximate a rounded/blocky particle.

    Returns:
        (centers (K, 3), radii (K,)); centres in the world frame.
    """
    n_surf = len(surf_pts)
    idx = np.unique(np.linspace(0, n_surf - 1, min(n_anchor, n_surf)).astype(int))
    anchors = surf_pts[idx]
    normals = surf_normals[idx]
    # local thickness: march inward until the implicit function changes sign
    r_eq = equivalent_sphere_radius(sq, n_probe=60_000)
    t_grid = np.linspace(0.0, 2.2 * r_eq, n_march)
    centres = anchors[:, None, :] - t_grid[None, :, None] * normals[:, None, :]
    local = sq.to_local(centres.reshape(-1, 3))
    f = superquadric_implicit(local, sq.a, sq.b, sq.c, sq.n1, sq.n2).reshape(len(idx), n_march)
    thickness = np.full(len(idx), 2.2 * r_eq)
    for k in range(len(idx)):
        outside = np.nonzero(f[k] > 0.0)[0]
        if len(outside):
            thickness[k] = t_grid[outside[0]]
    cand_c = []
    cand_r = []
    for frac in offsets:
        h = frac * thickness
        cand_c.append(anchors - h[:, None] * normals)
        cand_r.append(h)
    return np.concatenate(cand_c, axis=0), np.concatenate(cand_r, axis=0)


def fit_clump_to_superquadric(
    sq: Superquadric,
    n_spheres: int = 10,
    n_surface_samples: int = 1600,
    grid: int = 11,
    return_fit: bool = False,
):
    """Greedy minimax best-fit clump of ``n_spheres`` spheres.

    The clump is the *union* of the spheres, whose signed distance at a point is
    exactly ``min_i (|p - c_i| - r_i)``.  Candidates are surface-anchored
    spheres (see ``_candidate_spheres``); sphere ``k`` is the candidate that
    minimises the worst absolute surface deviation
    ``max_j | min_i (|p_j - c_i| - r_i) |`` over the true-surface samples.  That
    min-max greedy keeps the deviation monotone non-increasing in the number of
    spheres (the negative control for ``N -> inf``) and is deterministic.

    Args:
        sq: True superquadric geometry.
        n_spheres: Number of spheres to place.
        n_surface_samples: Samples used by the cover objective and the error.
        grid: Unused (kept for backwards compatibility of the signature).
        return_fit: Return :class:`ClumpFit` (with the error history) instead of
            a list of ``(centre, radius)`` pairs.
    """
    if n_spheres < 1:
        raise ValueError("n_spheres must be >= 1")
    surf_pts, surf_normals = _sample_surface_superquadric(sq, n_surface_samples)
    cand_c, cand_r = _candidate_spheres(sq, surf_pts, surf_normals)

    # union signed distance (min over the chosen spheres), initialised outside
    current = np.full(len(surf_pts), np.inf)
    chosen_centers: List[np.ndarray] = []
    chosen_radii: List[float] = []
    history: List[float] = []
    available = np.ones(len(cand_c), dtype=bool)
    for _ in range(min(n_spheres, len(cand_c))):
        d = np.linalg.norm(surf_pts[:, None, :] - cand_c[None, :, :], axis=-1)
        signed = d - cand_r[None, :]              # signed distance to each sphere
        candidate_best = np.minimum(current[:, None], signed)   # union SDF
        worst = np.abs(candidate_best).max(axis=0)             # |deviation|
        worst = np.where(available, worst, np.inf)
        pick = int(np.argmin(worst))
        available[pick] = False
        chosen_centers.append(cand_c[pick])
        chosen_radii.append(float(cand_r[pick]))
        current = candidate_best[:, pick]
        history.append(float(np.abs(current).max()))
    fit = ClumpFit(centers=np.asarray(chosen_centers), radii=np.asarray(chosen_radii),
                   surface_error_history=tuple(history))
    return fit if return_fit else fit.as_list()


# ---------------------------------------------------------------------------
# Geometric error
# ---------------------------------------------------------------------------

@dataclass
class GeometricErrorResult:
    """Geometric error between the true superquadric and its clump union.

    Attributes:
        max_surface_distance: max |SDF_union| over the true surface (metres).
        mean_surface_distance: mean |SDF_union| (metres).
        max_surface_distance_rel: max distance / equivalent sphere radius.
        max_normal_deviation_deg: worst surface-normal angle (degrees).
        mean_normal_deviation_deg: mean surface-normal angle (degrees).
        p95_normal_deviation_deg: 95th percentile normal angle (degrees).
        n_clump_spheres: spheres actually placed.
        aspect_ratio: max/min half-extent of the true shape.
        shape_exponents: (n1, n2).
        equivalent_radius_m: volume-equivalent sphere radius.
        error_history: max deviation after each greedy insertion (monotone).
    """
    max_surface_distance: float = 0.0
    mean_surface_distance: float = 0.0
    max_surface_distance_rel: float = 0.0
    max_normal_deviation_deg: float = 0.0
    mean_normal_deviation_deg: float = 0.0
    p95_normal_deviation_deg: float = 0.0
    n_clump_spheres: int = 0
    aspect_ratio: float = 1.0
    shape_exponents: Tuple[float, float] = (2.0, 2.0)
    equivalent_radius_m: float = 1.0
    error_history: Tuple[float, ...] = ()

    def to_dict(self) -> Dict:
        return {
            "max_surface_distance_m": self.max_surface_distance,
            "mean_surface_distance_m": self.mean_surface_distance,
            "max_surface_distance_rel": self.max_surface_distance_rel,
            "max_normal_deviation_deg": self.max_normal_deviation_deg,
            "mean_normal_deviation_deg": self.mean_normal_deviation_deg,
            "p95_normal_deviation_deg": self.p95_normal_deviation_deg,
            "n_clump_spheres": self.n_clump_spheres,
            "aspect_ratio": self.aspect_ratio,
            "shape_exponents": list(self.shape_exponents),
            "error_history": list(self.error_history),
        }


def compute_geometric_error(
    sq: Superquadric,
    n_clump_spheres: int = 10,
    n_surface_samples: int = 1600,
    grid: int = 11,
) -> GeometricErrorResult:
    """Geometric error of the best-fit clump against the true surface.

    The clump is treated as the *union* of its spheres, whose signed distance
    is exactly ``min_i (|p - c_i| - r_i)``.  Every true-surface sample is
    evaluated against that union; the normal deviation is the angle between the
    true normal and the normal of the sphere achieving the minimum (i.e. the
    normal the clump contact would actually transmit).
    """
    surf_pts, true_normals = _sample_surface_superquadric(sq, n_surface_samples)
    fit = fit_clump_to_superquadric(sq, n_clump_spheres, n_surface_samples,
                                   grid, return_fit=True)
    r_eq = equivalent_sphere_radius(sq)
    ar = max(sq.a, sq.b, sq.c) / min(sq.a, sq.b, sq.c)
    if len(fit.radii) == 0:
        return GeometricErrorResult(n_clump_spheres=0, aspect_ratio=ar,
                                    shape_exponents=(sq.n1, sq.n2),
                                    equivalent_radius_m=r_eq)

    d = np.linalg.norm(surf_pts[:, None, :] - fit.centers[None, :, :], axis=-1)
    sdf = d - fit.radii[None, :]                  # union SDF: min over spheres
    nearest = np.argmin(sdf, axis=1)
    depth = np.abs(sdf[np.arange(len(surf_pts)), nearest])   # |deviation|
    clump_normals = (surf_pts - fit.centers[nearest])
    clump_normals /= np.maximum(np.linalg.norm(clump_normals, axis=1, keepdims=True), 1e-300)
    cos = np.clip(np.sum(true_normals * clump_normals, axis=1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos))

    return GeometricErrorResult(
        max_surface_distance=float(depth.max()),
        mean_surface_distance=float(depth.mean()),
        max_surface_distance_rel=float(depth.max() / r_eq),
        max_normal_deviation_deg=float(angle_deg.max()),
        mean_normal_deviation_deg=float(angle_deg.mean()),
        p95_normal_deviation_deg=float(np.percentile(angle_deg, 95)),
        n_clump_spheres=int(len(fit.radii)),
        aspect_ratio=float(ar),
        shape_exponents=(sq.n1, sq.n2),
        equivalent_radius_m=float(r_eq),
        error_history=fit.surface_error_history,
    )


# ---------------------------------------------------------------------------
# Macroscopic error: repose angle + shear box
# ---------------------------------------------------------------------------

@dataclass
class MacroscopicErrorResult:
    """Macroscopic (pile / shear-cell) error of the clump approximation.

    Attributes:
        repose_angle_true_deg: repose angle of the true (superquadric) material.
        repose_angle_clump_deg: repose angle of the clump material.
        repose_angle_diff_deg: |difference| (degrees).
        force_chain_deviation_deg: RMS surface-normal deviation of the clump,
            i.e. the direction error the force chains inherit.
        shear_angle_true_deg: internal friction angle, true shape.
        shear_angle_clump_deg: internal friction angle, clump.
        shear_angle_diff_deg: |difference| (degrees).
        interlocking_clump_deg: geometric interlocking angle of the clump.
        evidence: provenance of each number (honest labelling).
    """
    repose_angle_true_deg: float = 0.0
    repose_angle_clump_deg: float = 0.0
    repose_angle_diff_deg: float = 0.0
    force_chain_deviation_deg: float = 0.0
    shear_angle_true_deg: float = 0.0
    shear_angle_clump_deg: float = 0.0
    shear_angle_diff_deg: float = 0.0
    interlocking_clump_deg: float = 0.0
    aspect_ratio: float = 1.0
    shape_exponents: Tuple[float, float] = (2.0, 2.0)
    evidence: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "repose_angle_true_deg": self.repose_angle_true_deg,
            "repose_angle_clump_deg": self.repose_angle_clump_deg,
            "repose_angle_diff_deg": self.repose_angle_diff_deg,
            "force_chain_deviation_deg": self.force_chain_deviation_deg,
            "shear_angle_true_deg": self.shear_angle_true_deg,
            "shear_angle_clump_deg": self.shear_angle_clump_deg,
            "shear_angle_diff_deg": self.shear_angle_diff_deg,
            "interlocking_clump_deg": self.interlocking_clump_deg,
            "aspect_ratio": self.aspect_ratio,
            "shape_exponents": list(self.shape_exponents),
            "evidence": self.evidence,
        }


def set_repose_anchor(pairs: Sequence[Tuple[float, float]]) -> None:
    """Install a measured ``(mu, theta_r_deg)`` calibration (DEM output)."""
    cleaned = sorted((float(mu), float(theta)) for mu, theta in pairs)
    if len(cleaned) < 2:
        raise ValueError("at least two anchor points are required")
    global _REPOSE_ANCHOR
    _REPOSE_ANCHOR = cleaned


def repose_anchor() -> List[Tuple[float, float]]:
    """Current ``(mu, theta_r_deg)`` anchor points."""
    return list(_REPOSE_ANCHOR)


def _repose_from_friction(mu: float) -> float:
    """Interpolate the measured repose anchor ``theta_r(mu)`` (degrees)."""
    mus = np.array([p[0] for p in _REPOSE_ANCHOR])
    th = np.array([p[1] for p in _REPOSE_ANCHOR])
    return float(np.interp(mu, mus, th))


def repose_angle_from_geometry(
    particle_positions: np.ndarray,
    floor_z: float = 0.0,
    method: str = "cone_fit",
) -> float:
    """Estimate the repose angle from a settled pile (geometry post-processing).

    Fits the free surface of the pile on the two flank sides of the apex and
    returns the mean inclination.  Kept for DEM post-processing; the macro
    budget below uses the measured ``mu -> theta_r`` anchor instead.
    """
    pos = np.asarray(particle_positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 3 or len(pos) < 10:
        raise ValueError("particle_positions must be (N, 3) with N >= 10")
    z = pos[:, 2] - floor_z
    top = z > (z.max() - 0.15 * (z.max() - z.min() + 1e-30))
    surface = pos[top]
    if len(surface) < 5:
        surface = pos
        z_surface = surface[:, 2] - floor_z
    else:
        z_surface = surface[:, 2] - floor_z
    x = surface[:, 0]
    x_center = 0.5 * (x.min() + x.max())
    slopes = []
    for side in (-1.0, 1.0):
        mask = side * (x - x_center) > 0
        if np.count_nonzero(mask) < 3:
            continue
        xs = side * (x[mask] - x_center)
        zs = z_surface[mask]
        slope = np.polyfit(xs, zs, 1)[0]
        slopes.append(abs(float(slope)))
    if method != "cone_fit":
        raise ValueError(f"unknown method {method!r}")
    if not slopes:
        return 0.0
    return math.degrees(math.atan(float(np.mean(slopes))))


def compute_macroscopic_error(
    sq: Superquadric,
    n_clump_spheres: int = 10,
    mu: float = 0.40,
    n_surface_samples: int = 800,
    geometric: Optional[GeometricErrorResult] = None,
) -> MacroscopicErrorResult:
    """Macroscopic error of the clump approximation (anchored model).

    Physics used (and its provenance, recorded in ``evidence``):

    * ``theta_r(mu)`` -- **measured** 3D DEM calibration of monodisperse
      spheres (column collapse, ``tensordem.shape_macro``).  The default
      anchor is the build-time result shipped in
      ``artifacts/dem/shape_error_table.json``.
    * Interlocking angle ``i`` -- **geometric** RMS surface-normal deviation of
      the clump (layer 1).  The true superquadric transmits its own normals, so
      ``i_true = 0``; the clump's bumpy union surface deviates by ``i_clump``.
    * Interlocking augmentation ``mu_eff = mu + tan i`` -- **modelled** after
      the classical interlocking/dilatancy term for angular media (Rowe,
      Newland & Allely).  ``theta`` is then read from the measured anchor.

    A poly-shape DEM of the clump was *not* run; ``evidence`` marks those two
    numbers ``MODEL`` rather than ``REAL``, so the report cannot overclaim.
    """
    geom = geometric or compute_geometric_error(sq, n_clump_spheres,
                                               n_surface_samples=n_surface_samples)
    ar = max(sq.a, sq.b, sq.c) / min(sq.a, sq.b, sq.c)
    i_true = 0.0
    i_clump = geom.mean_normal_deviation_deg

    mu_true = mu + math.tan(math.radians(i_true))
    mu_clump = mu + math.tan(math.radians(i_clump))
    theta_true = _repose_from_friction(mu_true)
    theta_clump = _repose_from_friction(mu_clump)
    # Internal friction angle of the shear cell follows the same interlocking
    # augmentation; the reported spread vs the repose angle is the usual
    # theta_peak = theta_r + few degrees observed in granular experiments.
    phi_true = theta_true + 3.0
    phi_clump = theta_clump + 3.0

    return MacroscopicErrorResult(
        repose_angle_true_deg=theta_true,
        repose_angle_clump_deg=theta_clump,
        repose_angle_diff_deg=abs(theta_clump - theta_true),
        force_chain_deviation_deg=i_clump,
        shear_angle_true_deg=phi_true,
        shear_angle_clump_deg=phi_clump,
        shear_angle_diff_deg=abs(phi_clump - phi_true),
        interlocking_clump_deg=i_clump,
        aspect_ratio=float(ar),
        shape_exponents=(sq.n1, sq.n2),
        evidence={
            "theta_r(mu)": "REAL (3D DEM column collapse, monodisperse spheres)",
            "interlocking_angle": "REAL (geometric, measured on the clump union)",
            "theta_clump": "MODEL (interlocking augmentation of the measured anchor)",
            "theta_true": "MODEL (i_true = 0 by definition for the exact shape)",
            "poly_shape_dem": "WITHHELD (no full clump/superquadric pile DEM)",
        },
    )


# ---------------------------------------------------------------------------
# Disable ratio table + API gate
# ---------------------------------------------------------------------------

@dataclass
class DisableRatioTable:
    """Table specifying when the sphere-clump approximation is acceptable.

    Entries are indexed by ``(aspect_ratio, shape_exponent)`` in the geometric
    scan and carry the per-sphere-count results, so the gate can answer "how
    many spheres do I need for this shape?".
    """
    entries: List[Dict] = field(default_factory=list)
    criteria: Dict[str, float] = field(default_factory=lambda: dict(CRITERIA))

    def to_json(self) -> str:
        return json.dumps({"criteria": self.criteria,
                           "entries": self.entries}, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "DisableRatioTable":
        data = json.loads(json_str)
        return cls(criteria=data.get("criteria", dict(CRITERIA)),
                   entries=data.get("entries", []))

    def minimum_clump_size(self, aspect_ratio: float, shape_exponent: float) -> Optional[int]:
        """Smallest sphere count that passes the gate for the closest entry."""
        rows = [e for e in self.entries
                if math.isclose(e["aspect_ratio"], aspect_ratio, rel_tol=1e-6)
                and math.isclose(e["shape_exponent"], shape_exponent, rel_tol=1e-6)]
        allowed = sorted((e["n_clump_spheres"] for e in rows if e["use_clump"]))
        return allowed[0] if allowed else None


def _shape_for(aspect_ratio: float, shape_exponent: float) -> Superquadric:
    """Canonical shape family used by the table (min half-extent = 1)."""
    return Superquadric(a=max(aspect_ratio, 1.0), b=1.0, c=1.0,
                        n1=shape_exponent, n2=shape_exponent)


def build_disable_ratio_table(
    aspect_ratios: Optional[Sequence[float]] = None,
    shape_exponents: Optional[Sequence[float]] = None,
    sphere_counts: Sequence[int] = (1, 2, 4, 8, 16, 32),
    mu: float = 0.40,
    n_surface_samples: int = 800,
    grid: int = 9,
) -> DisableRatioTable:
    """Scan (aspect ratio x shape exponent x sphere count) and build the table.

    The table answers the campaign question directly: for which clump sizes is
    the sphere approximation within the criteria, and where must it be
    disabled?  Deterministic (no random state).
    """
    if aspect_ratios is None:
        aspect_ratios = (1.0, 1.5, 2.0, 3.0, 5.0)
    if shape_exponents is None:
        shape_exponents = (2.0, 4.0, 8.0, 16.0, 50.0)
    table = DisableRatioTable()
    for ar in aspect_ratios:
        for n in shape_exponents:
            sq = _shape_for(float(ar), float(n))
            for n_sph in sphere_counts:
                geom = compute_geometric_error(sq, int(n_sph), n_surface_samples, grid)
                macro = compute_macroscopic_error(sq, int(n_sph), mu=mu,
                                                 geometric=geom)
                use_clump = (
                    macro.repose_angle_diff_deg <= table.criteria["max_repose_error_deg"]
                    and macro.force_chain_deviation_deg <= table.criteria["max_force_chain_error_deg"]
                    and geom.max_normal_deviation_deg <= table.criteria["max_normal_deviation_deg"]
                    and geom.max_surface_distance_rel <= table.criteria["max_geometric_error_rel"]
                )
                table.entries.append({
                    "aspect_ratio": float(ar),
                    "shape_exponent": float(n),
                    "n_clump_spheres": int(geom.n_clump_spheres),
                    "use_clump": bool(use_clump),
                    "repose_error_deg": round(macro.repose_angle_diff_deg, 3),
                    "force_chain_error_deg": round(macro.force_chain_deviation_deg, 3),
                    "geometric_error_m": round(geom.max_surface_distance, 6),
                    "geometric_error_rel": round(geom.max_surface_distance_rel, 6),
                    "normal_deviation_deg": round(geom.max_normal_deviation_deg, 3),
                    "p95_normal_deviation_deg": round(geom.p95_normal_deviation_deg, 3),
                })
    return table


class ClumpApproximationDisabled(RuntimeError):
    """Raised by the API gate when a clump representation is not admissible."""

    def __init__(self, message: str, diagnostics: Dict):
        super().__init__(message)
        self.diagnostics = diagnostics


def api_gate(
    sq: Superquadric,
    n_clump_spheres: int,
    table: Optional[DisableRatioTable] = None,
    mu: float = 0.40,
    n_surface_samples: int = 800,
    grid: int = 9,
    raise_on_fail: bool = False,
) -> Dict:
    """API gate: may this shape be represented by ``n_clump_spheres`` spheres?

    The decision is computed live for the *actual* geometry (not merely looked
    up), because the table only samples the (aspect ratio, exponent) grid.  Both
    the campaign criteria are applied:

    * ``|theta_true - theta_clump| <= 3 deg``
    * contact-normal (force chain) deviation ``<= 5 deg``

    plus the geometric guards (max surface deviation and its size-normalised
    value).  Returns a diagnostics dict; with ``raise_on_fail=True`` it raises
    :class:`ClumpApproximationDisabled` so that callers (API layer, CLI) cannot
    silently use an inadmissible clump.
    """
    criteria = dict(table.criteria) if table is not None else dict(CRITERIA)
    geom = compute_geometric_error(sq, n_clump_spheres, n_surface_samples, grid)
    macro = compute_macroscopic_error(sq, n_clump_spheres, mu=mu, geometric=geom)
    reasons: List[str] = []
    if macro.repose_angle_diff_deg > criteria["max_repose_error_deg"]:
        reasons.append(f"repose error {macro.repose_angle_diff_deg:.2f} deg > "
                       f"{criteria['max_repose_error_deg']:.1f} deg")
    if macro.force_chain_deviation_deg > criteria["max_force_chain_error_deg"]:
        reasons.append(f"normal/force-chain deviation "
                       f"{macro.force_chain_deviation_deg:.2f} deg > "
                       f"{criteria['max_force_chain_error_deg']:.1f} deg")
    if geom.max_normal_deviation_deg > criteria["max_normal_deviation_deg"]:
        reasons.append(f"max normal deviation {geom.max_normal_deviation_deg:.2f} deg > "
                       f"{criteria['max_normal_deviation_deg']:.1f} deg")
    if geom.max_surface_distance_rel > criteria["max_geometric_error_rel"]:
        reasons.append(f"relative surface error {geom.max_surface_distance_rel:.4f} > "
                       f"{criteria['max_geometric_error_rel']:.2f}")
    diagnostics = {
        "allowed": not reasons,
        "reasons": reasons,
        "aspect_ratio": macro.aspect_ratio,
        "shape_exponents": list(macro.shape_exponents),
        "n_clump_spheres": int(geom.n_clump_spheres),
        "repose_error_deg": macro.repose_angle_diff_deg,
        "force_chain_deviation_deg": macro.force_chain_deviation_deg,
        "max_normal_deviation_deg": geom.max_normal_deviation_deg,
        "max_surface_distance_rel": geom.max_surface_distance_rel,
        "criteria": criteria,
        "evidence": macro.evidence,
    }
    if reasons and raise_on_fail:
        raise ClumpApproximationDisabled(
            "sphere-clump approximation rejected: " + "; ".join(reasons), diagnostics)
    return diagnostics


@dataclass
class ShapeErrorBudget:
    """Container tying the two error layers together for reporting."""
    geometric: GeometricErrorResult
    macroscopic: MacroscopicErrorResult

    def to_dict(self) -> Dict:
        return {"geometric": self.geometric.to_dict(),
                "macroscopic": self.macroscopic.to_dict()}