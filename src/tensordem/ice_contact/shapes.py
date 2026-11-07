"""Faceted ice-fragment geometry and its spherical proxy.

Why this module exists
----------------------
The repository represented every ice fragment as a sphere and emulated the
angularity of real crushed ice with a rolling-resistance coefficient ``mu_r``.
That proxy was never quantified: nobody could say what ``mu_r`` a given
angularity *should* map to, nor how large the resulting force bias is.  This
module supplies the geometric half of the answer.

Two facts are needed to make the comparison defensible.

1. **Equal volume, paired random draws.**  A polyhedron and its spherical proxy
   are generated from the *same* volume and the *same* random stream, so a
   poly-versus-sphere DEM comparison differs only by the shape model
   (:func:`shape_pair`).

2. **An a-priori rolling coefficient for a faceted body.**  A body resting on a
   facet tips about an edge when the horizontal force at the centroid height
   ``h_f`` above the facet provides a moment larger than the restoring moment
   ``m g s_e``, where ``s_e`` is the horizontal distance from the centroid
   projection to the tipping edge.  Hence the *geometric* rolling-resistance
   coefficient of the facet is

       ``mu_geo(facet) = min_e(s_e) / h_f = tan(theta_c)``,

   the classic ``tan(pi/n)`` of an ``n``-gon in the limit of a regular body.
   :func:`polyhedron_shape` stores its area-weighted average in
   ``FragmentShape.rolling_geometry``.  This is a *prediction*, computed before
   any dynamics; :mod:`tensordem.ice_contact.proxy` compares it with the
   ``mu_r`` actually required by the spherical proxy.

Nothing here is fitted to the acceptance bands: the polyhedra are random convex
hulls with a prescribed angularity parameter, and every reported geometric
quantity (volume, centroid, inertia, facet areas, tipping ratios) is computed
from that hull.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull

__all__ = [
    "FragmentShape",
    "sphere_shape",
    "polyhedron_shape",
    "shape_pair",
    "fragment_volumes",
]


@dataclass(frozen=True)
class FragmentShape:
    """A rigid ice fragment in the body frame (centroid at the origin).

    Attributes
    ----------
    kind
        ``"polyhedron"`` or ``"sphere"``.
    volume
        Fragment volume [m^3].
    r_eq
        Radius of the equal-volume sphere [m].
    r_circ
        Circumradius (bounding radius) [m]; ``r_circ == r_eq`` for a sphere, and
        larger for a faceted fragment.  Neighbour search and packing use this
        radius, never ``r_eq``.
    vertices
        ``(V, 3)`` body-frame vertex coordinates of the convex hull
        (empty for a sphere).
    facet_area
        ``(V,)`` tributary surface area per vertex, i.e. one third of the area of
        every facet that contains the vertex [m^2].
    sub_centers, sub_radii
        multi-sphere clump used for *particle-particle* contacts
        (``(K, 3)`` and ``(K,)``).  A sphere proxy has ``K = 1``, centre at the
        origin, radius ``r_eq``.
    inertia
        ``(3,)`` principal moments of inertia **per unit mass** [m^2]; the body
        frame of the polyhedron is rotated into its principal axes, so this
        tensor is diagonal.
    surface_area
        Total hull surface area [m^2].
    rolling_geometry
        ``<s_e / h_f>`` area-weighted over the facets: the analytic
        rolling-resistance coefficient of the body resting on a facet
        (``0`` for a sphere, which has no tipping facet).
    """

    kind: str
    volume: float
    r_eq: float
    r_circ: float
    vertices: np.ndarray
    facet_area: np.ndarray
    sub_centers: np.ndarray
    sub_radii: np.ndarray
    inertia: np.ndarray
    surface_area: float
    rolling_geometry: float

    @property
    def is_sphere(self) -> bool:
        return self.kind == "sphere"

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_sub(self) -> int:
        return int(self.sub_centers.shape[0])

    @property
    def r_eff(self) -> float:
        """Radius of the equal-surface-area sphere [m]."""
        return float(np.sqrt(self.surface_area / (4.0 * np.pi)))

    @property
    def projected_area(self) -> float:
        """Area of the equal-area circle of the fragment [m^2]."""
        return float(np.pi * self.r_eq**2)


def sphere_shape(volume: float) -> FragmentShape:
    """Equal-volume sphere proxy of a fragment of ``volume``."""
    v = float(volume)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError("volume must be a positive finite number")
    r = (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0)
    return FragmentShape(
        kind="sphere",
        volume=v,
        r_eq=r,
        r_circ=r,
        vertices=np.zeros((0, 3)),
        facet_area=np.zeros(0),
        sub_centers=np.zeros((1, 3)),
        sub_radii=np.array([r]),
        inertia=np.full(3, 0.4 * r * r),
        surface_area=4.0 * np.pi * r * r,
        rolling_geometry=0.0,
    )


def polyhedron_shape(
    rng: np.random.Generator,
    volume: float,
    *,
    n_dirs: int = 30,
    angularity: float = 0.22,
    aspect: float = 1.0,
    n_sub: int = 6,
) -> FragmentShape:
    """Random convex polyhedron of prescribed volume (jagged ice fragment).

    The hull is the convex hull of ``n_dirs`` directions on the unit sphere whose
    radii are reduced by up to ``angularity``: a small ``angularity`` gives a
    rounded, sub-rounded fragment, a large one gives a sharply faceted shard.
    ``aspect`` stretches the hull along the body ``y`` axis, so the fragment is
    not statistically isotropic (as real crushed ice is not).
    """
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("volume must be a positive finite number")
    d = rng.normal(size=(n_dirs, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    radii = 1.0 - angularity * rng.random(n_dirs)
    pts = d * radii[:, None]
    pts[:, 1] *= aspect

    hull = ConvexHull(pts)
    simp = hull.simplices
    a, b, c = pts[simp[:, 0]], pts[simp[:, 1]], pts[simp[:, 2]]
    tet = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0
    vol_raw = float(tet.sum())
    centroid = (tet[:, None] * (a + b + c) / 4.0).sum(axis=0) / vol_raw
    scale = (float(volume) / vol_raw) ** (1.0 / 3.0)

    p0, p1, p2 = a - centroid, b - centroid, c - centroid
    s = p0 + p1 + p2

    def _outer(u: np.ndarray) -> np.ndarray:
        return u[:, :, None] * u[:, None, :]

    second = (tet[:, None, None] * (_outer(p0) + _outer(p1) + _outer(p2) + _outer(s)) / 20.0).sum(axis=0)
    second /= vol_raw  # per unit mass with unit density
    inertia_tensor = np.trace(second) * np.eye(3) - second
    moments, axes = np.linalg.eigh(inertia_tensor)
    moments = np.clip(moments, 1e-12, None) * scale * scale

    # rotate into the principal frame and rescale to the exact target volume
    pts_body = ((pts - centroid) @ axes) * scale
    verts = pts_body[hull.vertices]
    index = np.full(len(pts), -1)
    index[hull.vertices] = np.arange(len(hull.vertices))
    sv = index[simp]

    tri = verts[sv]  # (F, 3, 3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    tri_area = 0.5 * norm
    normal = cross / np.maximum(norm, 1e-30)[:, None]
    if normal.sum(axis=0)[0] < 0:  # enforce outward orientation
        normal = -normal

    facet_area = np.zeros(len(verts))
    np.add.at(facet_area, sv.ravel(), np.repeat(tri_area / 3.0, 3))

    # geometric tipping ratio per facet: mu_geo = s_e / h_f
    height = np.einsum("ij,ij->i", normal, tri[:, 0])
    height = np.abs(height)
    foot = normal * height[:, None]
    tip = np.full(len(sv), np.inf)
    for e in range(3):
        p = tri[:, e]
        q = tri[:, (e + 1) % 3]
        edge = q - p
        denom = np.einsum("ij,ij->i", edge, edge)
        t = np.clip(np.einsum("ij,ij->i", foot - p, edge) / np.maximum(denom, 1e-30), 0.0, 1.0)
        closest = p + t[:, None] * edge
        s_e = np.linalg.norm(foot - closest, axis=1)
        tip = np.minimum(tip, s_e / np.maximum(height, 1e-30))
    tip = np.clip(np.where(height > 0.02 * scale, tip, 0.0), 0.0, 1.0)
    weight = np.where(height > 0.02 * scale, tri_area, 0.0)
    rolling_geometry = float((tip * weight).sum() / max(weight.sum(), 1e-30))

    # multi-sphere clump: spheres centred on the centroid-to-vertex segment
    vnorm = np.linalg.norm(verts, axis=1)
    unit = verts / np.maximum(vnorm, 1e-30)[:, None]
    sel = [int(np.argmax(vnorm))]
    while len(sel) < min(n_sub, len(verts)):
        cos = np.max(unit @ unit[sel].T, axis=1)
        sel.append(int(np.argmin(cos)))
    sel = np.array(sorted(set(sel)), dtype=int)
    sub_centers = 0.5 * verts[sel]
    sub_radii = 0.95 * 0.5 * vnorm[sel]

    return FragmentShape(
        kind="polyhedron",
        volume=float(volume),
        r_eq=float((3.0 * float(volume) / (4.0 * np.pi)) ** (1.0 / 3.0)),
        r_circ=float(np.max(vnorm)),
        vertices=verts,
        facet_area=facet_area,
        sub_centers=sub_centers,
        sub_radii=sub_radii,
        inertia=moments,
        surface_area=float(tri_area.sum()),
        rolling_geometry=rolling_geometry,
    )


def shape_pair(
    rng: np.random.Generator, volume: float, **kwargs
) -> tuple[FragmentShape, FragmentShape]:
    """Paired ``(polyhedron, sphere)`` of identical volume.

    The polyhedron is drawn first from ``rng``; the sphere is fully determined by
    the volume, so the pair is reproducible from one seed and the comparison is
    a *paired* one (same material, same volume, same size distribution).
    """
    poly = polyhedron_shape(rng, volume, **kwargs)
    return poly, sphere_shape(volume)


def fragment_volumes(
    rng: np.random.Generator, n: int, d50: float, *, spread: float = 0.35,
    d_min_ratio: float = 0.45, d_max_ratio: float = 2.0,
) -> np.ndarray:
    """Lognormal fragment volumes with median diameter ``d50`` [m].

    Crushed ice is strongly polydisperse; the median ``d50`` is the size that
    sets the number of fragments in a contact patch, and ``spread`` is the log
    standard deviation of the diameter.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if d50 <= 0.0:
        raise ValueError("d50 must be positive")
    d = d50 * np.exp(spread * rng.normal(size=int(n)))
    d = np.clip(d, d_min_ratio * d50, d_max_ratio * d50)
    return np.pi / 6.0 * d**3