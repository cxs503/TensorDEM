"""Separating-axis contact between convex polyhedra and the contact law.

Geometry
--------
Two convex polyhedra are separated iff there is a separating axis among the
facet normals of either body and the cross products of their edge directions
(the classical SAT axis set).  We project both vertex clouds onto every
candidate axis, take the minimum overlap, and thereby obtain

* a **signed gap** for the separated branch (``max`` of the negative overlaps,
  positive for a clearance);
* the **penetration depth and normal** for the overlapping branch;
* a **contact patch**: the mean of the deepest vertex sets of both bodies,
  which is a point (hence a lever arm and a torque) and not the centroid line.

For ice floes the facet normals are what physically carries the load, so ties are
broken in favour of a facet axis (``face_bias``) exactly as in the 2-D polygon
contact of :mod:`tensordem.polygon_contact`.

Why not GJK/EPA
---------------
:mod:`tensordem.convex_contact` provides a general GJK/EPA distance, but it costs
~30 ms per pair in pure Python, i.e. three orders of magnitude too slow for the
10^5 pair-steps of a floe field.  The SAT implementation here is vectorised over
all axes at once (~10^2 us per pair) and is **cross-validated against GJK/EPA**
in ``tests/test_polyhedron_floe3d_contact.py`` on random prism pairs, so the
speed-up does not cost accuracy.

Contact law
-----------
Normal: spring-dashpot ``f_n = max(0, k_n delta + c_n v_n)`` with
``c_n = 2 zeta sqrt(k_n m_eff)`` (same convention as the 2-D modules).
Tangential: Cundall-Strack incremental spring, Coulomb capped
``|f_t| <= mu f_n``.  Rolling resistance is an optional torque cap
``mu_r r_eff f_n``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "ContactLaw",
    "sat_contact",
    "wall_contact",
    "PairContact",
]

_TINY = 1.0e-12


@dataclass
class ContactLaw:
    """Normal / tangential / rolling contact parameters (SI units)."""

    kn: float = 1.0e7
    zeta: float = 0.05
    mu: float = 0.4
    mu_rolling: float = 0.01
    kt_ratio: float = 0.5

    @property
    def kt(self) -> float:
        return self.kt_ratio * self.kn

    def normal_damping(self, mass_effective: float) -> float:
        return 2.0 * self.zeta * np.sqrt(self.kn * mass_effective)

    def as_dict(self) -> Dict[str, float]:
        return {"kn": self.kn, "zeta": self.zeta, "mu": self.mu,
                "mu_rolling": self.mu_rolling, "kt_ratio": self.kt_ratio}


@dataclass(frozen=True)
class PairContact:
    """One resolved convex-pair contact geometry (world frame)."""

    gap: float           # > 0 : clearance, < 0 : overlap of depth -gap
    normal: np.ndarray   # unit vector from body a towards body b
    point: np.ndarray    # contact patch centroid
    n_axes: int

    @property
    def depth(self) -> float:
        return -self.gap if self.gap < 0.0 else 0.0

    @property
    def contacting(self) -> bool:
        return self.gap <= 0.0


def _axis_set(normals_a: np.ndarray, normals_b: np.ndarray,
              edges_a: np.ndarray, edges_b: np.ndarray,
              extrusion: Optional[np.ndarray] = None) -> np.ndarray:
    """Unique unit SAT axes.

    General case: facet normals of both bodies plus all edge-edge cross products.
    Plate case (``extrusion`` given, i.e. both bodies are extrusions whose axes
    are parallel): the exact axis set collapses to the side-facet normals of both
    plates plus the extrusion axis, because a cross product of a side normal with
    the extrusion axis is again a side normal and a cross product of two rim
    edges is the extrusion axis.  Restricting the set is therefore exact, only
    cheaper.
    """
    if extrusion is not None:
        ax = np.asarray(extrusion, dtype=np.float64)
        ax = ax / np.linalg.norm(ax)
        side_a = normals_a[np.abs(normals_a @ ax) < 0.1]
        side_b = normals_b[np.abs(normals_b @ ax) < 0.1]
        # exact for parallel extrusions; the facet normals are already unit and
        # duplicates are harmless (they only repeat an axis), so no dedup pass
        return np.vstack([side_a, side_b, ax.reshape(1, 3)])

    axes = [normals_a, normals_b]
    if edges_a.shape[0] and edges_b.shape[0]:
        cross = np.cross(edges_a[:, None, :], edges_b[None, :, :]).reshape(-1, 3)
        axes.append(cross)
    axis = np.vstack(axes)
    scale = max(1.0, float(np.max(np.linalg.norm(axis, axis=1), initial=1.0)))
    length = np.linalg.norm(axis, axis=1)
    axis = axis[length > 1.0e-9 * scale]
    axis = axis / np.linalg.norm(axis, axis=1)[:, None]
    # canonical sign, then deduplicate (parallel facets and repeated edge pairs)
    flip = np.where(np.abs(axis[:, :1]) > 1e-12, axis[:, :1] < 0.0,
                    np.where(np.abs(axis[:, 1:2]) > 1e-12, axis[:, 1:2] < 0.0,
                             axis[:, 2:3] < 0.0))
    axis = np.where(flip, -axis, axis)
    keys = np.round(axis / 1.0e-8).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return axis[np.sort(first)]


def sat_contact(vertices_a: np.ndarray, normals_a: np.ndarray, edges_a: np.ndarray,
                vertices_b: np.ndarray, normals_b: np.ndarray, edges_b: np.ndarray,
                *, face_bias: float = 1.0e-7, patch_fraction: float = 0.25,
                direction_bias: float = 0.2,
                extrusion_a: Optional[np.ndarray] = None,
                extrusion_b: Optional[np.ndarray] = None) -> PairContact:
    """SAT contact of two convex polyhedra given in the world frame."""
    extrusion = None
    if extrusion_a is not None and extrusion_b is not None:
        ea = np.asarray(extrusion_a, dtype=np.float64)
        eb = np.asarray(extrusion_b, dtype=np.float64)
        ea = ea / np.linalg.norm(ea)
        eb = eb / np.linalg.norm(eb)
        if abs(float(ea @ eb)) > 0.996:          # within ~5 degrees: plate case
            extrusion = ea + eb if float(ea @ eb) > 0.0 else ea - eb
    axes = _axis_set(normals_a, normals_b, edges_a, edges_b, extrusion)
    proj_a = vertices_a @ axes.T
    proj_b = vertices_b @ axes.T
    overlap = np.minimum(proj_a.max(axis=0), proj_b.max(axis=0)) - \
        np.maximum(proj_a.min(axis=0), proj_b.min(axis=0))

    centre_a = vertices_a.mean(axis=0)
    centre_b = vertices_b.mean(axis=0)
    direction = centre_b - centre_a
    n_face = normals_a.shape[0]
    n_faceb = normals_b.shape[0]
    is_edge_axis = np.ones(axes.shape[0], dtype=bool)
    is_edge_axis[:n_face + n_faceb] = False
    score = overlap.copy()
    score[is_edge_axis] *= (1.0 + face_bias)   # facet normals win near-ties
    # Direction preference: a thin plate that is laterally overlapped also
    # overlaps its full thickness, so the raw minimum-overlap axis can be the
    # vertical one even when the physical contact is lateral.  Penalising axes
    # that are nearly perpendicular to the centre-to-centre direction resolves
    # that degeneracy in favour of the physically loaded facet (and keeps
    # stacked plates, where the centre line *is* vertical, vertical).  The floor
    # (0.2) bounds the influence of the bias.
    if direction_bias > 0.0:
        alignment = np.abs(axes @ (direction / max(float(np.linalg.norm(direction)), 1e-12)))
        score = score / (direction_bias + alignment)

    if np.any(overlap < 0.0):
        # separated: the axis with the largest negative overlap is the best
        index = int(np.argmax(-overlap))
        normal = axes[index]
        if float(normal @ direction) < 0.0:
            normal = -normal
        gap = float(-overlap[index])
        pa = vertices_a[int(np.argmax(vertices_a @ normal))]
        pb = vertices_b[int(np.argmin(vertices_b @ normal))]
        point = 0.5 * (pa + pb)
        return PairContact(gap, normal, point, int(axes.shape[0]))

    index = int(np.argmin(score))
    normal = axes[index]
    if float(normal @ direction) < 0.0:
        normal = -normal
    depth = float(overlap[index])
    proj_a_i = vertices_a @ normal
    proj_b_i = vertices_b @ normal
    band_a = band_b = patch_fraction * depth
    set_a = vertices_a[proj_a_i >= proj_a_i.max() - band_a]
    set_b = vertices_b[proj_b_i <= proj_b_i.min() + band_b]
    point = 0.5 * (set_a.mean(axis=0) + set_b.mean(axis=0))
    return PairContact(-depth, normal, point, int(axes.shape[0]))


def wall_contact(vertices: np.ndarray, wall_normal: np.ndarray, wall_offset: float,
                 *, patch_fraction: float = 0.25) -> Optional[Tuple[float, np.ndarray]]:
    """Penetration of a polyhedron into the half-space ``{x : n . x >= d}``.

    Returns ``(depth, point)`` with ``depth > 0`` when the body has crossed the
    wall, else ``None``.  The point is the centroid of the deepest vertex patch,
    which for a plate against a vertical plane is the loaded face patch.
    """
    normal = np.asarray(wall_normal, dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    signed = vertices @ normal - float(wall_offset)
    lowest = float(signed.min())
    if lowest >= 0.0:
        return None
    depth = -lowest
    band = patch_fraction * depth
    patch = vertices[signed <= lowest + band]
    return depth, patch.mean(axis=0)