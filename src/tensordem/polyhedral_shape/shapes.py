"""Convex polyhedral particle representation and analytic shape descriptors.

Why this module exists
----------------------
``tensordem.convex_contact`` answers the *contact* question for convex support
bodies, but a DEM shape study also needs the *shape* question: given a convex
polyhedron (or an ellipsoid), what are its integral geometric descriptors and
how do they distribute over a family of particles?

Two descriptors are used throughout the DEM/PIV literature for granular and
crushed-ice particles:

* **sphericity** (Wadell 1935)::

      Psi = pi**(1/3) * (6 V)**(2/3) / A

  the surface area of the volume-equivalent sphere over the actual surface area.
  ``Psi = 1`` for a sphere; the regular tetrahedron gives ``0.6706``.
* **roundness** (Wadell-type, 3-D)::

      R = r_in / r_circ

  with ``r_in`` the radius of the largest inscribed sphere (Chebyshev centre,
  solved here as a linear programme over the facet half-spaces) and ``r_circ``
  the radius of the smallest enclosing sphere about that centre.  Sharp
  polyhedra have ``R << 1``; the sphere has ``R = 1``.

* **aspect ratio**: ``sqrt(l_max / l_min)`` where ``l`` are the eigenvalues of
  the particle's second-moment (covariance) tensor.  The covariance is
  integrated *exactly* over the solid by tetrahedral decomposition, not
  estimated from the vertex cloud, so it is independent of vertex sampling.

All descriptors are closed-form functions of the facet geometry, so the test
suite can pin them against exact values (cube: ``Psi = pi**(1/3) 6**(2/3)/6``;
regular tetrahedron: ``Psi = 6**(2/3) pi**(1/3) / (sqrt(3) * 3**(1/4) * ...)``,
evaluated symbolically in the tests) to machine precision.

The module also supplies

* :class:`Ellipsoid` -- a closed-form support body (``M M^T d / |M^T d|``) so
  that GJK/EPA in :mod:`tensordem.convex_contact` can be exercised on smooth
  non-spherical shapes *without* faceting error;
* :func:`spherical_polyhedron` / :func:`random_polyhedron` -- families of
  convex polyhedra with a controllable vertex count ``N_v``.  ``N_v -> inf``
  with all vertices on a common sphere is the polyhedron -> sphere limit used
  by the negative controls.

Units: SI everywhere; arrays are ``float64`` with shape ``(3,)`` / ``(V, 3)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from ..convex_contact import ConvexShape

__all__ = [
    "PolyhedralShape",
    "Ellipsoid",
    "ShapeDescriptors",
    "describe",
    "describe_ellipsoid",
    "cube_vertices",
    "regular_tetrahedron_vertices",
    "regular_octahedron_vertices",
    "regular_icosahedron_vertices",
    "regular_dodecahedron_vertices",
    "spherical_polyhedron",
    "random_polyhedron",
    "named_shape",
    "volume_equivalent_radius",
    "sphericity",
]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _as_array(value, shape=None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if shape is not None and arr.shape != shape:
        raise ValueError(f"expected shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("array contains non-finite entries")
    return arr


def _orthonormal(value) -> np.ndarray:
    if value is None:
        return np.eye(3)
    rot = _as_array(value).reshape(3, 3)
    u, _, vt = np.linalg.svd(rot)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        out = out * np.array([1.0, 1.0, -1.0])
    return out


def volume_equivalent_radius(volume: float) -> float:
    """Radius of the sphere with the same volume."""
    if not (math.isfinite(volume) and volume > 0.0):
        raise ValueError("volume must be finite and positive")
    return float((3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0))


def sphericity(volume: float, area: float) -> float:
    """Wadell sphericity ``pi**(1/3) (6V)**(2/3) / A``."""
    if not (math.isfinite(volume) and volume > 0.0):
        raise ValueError("volume must be finite and positive")
    if not (math.isfinite(area) and area > 0.0):
        raise ValueError("area must be finite and positive")
    return float(math.pi ** (1.0 / 3.0) * (6.0 * volume) ** (2.0 / 3.0) / area)


# ---------------------------------------------------------------------------
# polyhedral shape
# ---------------------------------------------------------------------------


class PolyhedralShape(ConvexShape):
    """Convex polyhedron given by a vertex cloud in a body frame.

    The convex hull is taken on construction; ``support`` is the vertex-cloud
    maximum, which is exact for a convex body.
    """

    def __init__(self, vertices, orientation=None, center=(0.0, 0.0, 0.0),
                 name: str = "polyhedron"):
        verts = _as_array(vertices).reshape(-1, 3)
        if verts.shape[0] < 4:
            raise ValueError("a polyhedron needs at least 4 vertices")
        rot = _orthonormal(orientation)
        local = verts
        pivot = np.zeros(3)
        hull = ConvexHull(local)
        if hull.volume <= 0.0:
            raise ValueError("degenerate (zero-volume) vertex cloud")
        self.name = str(name)
        self._local = local
        self._orientation = rot
        self._center = _as_array(center, (3,))
        self._hull = hull
        self._pivot = pivot

    # -- support-function interface ----------------------------------------
    def support(self, direction: Sequence[float]) -> np.ndarray:
        d = _as_array(direction, (3,))
        if float(np.linalg.norm(d)) == 0.0:
            raise ValueError("support direction must be non-zero")
        local_dir = self._orientation.T @ d
        idx = int(np.argmax(self._local @ local_dir))
        return self._center + self._orientation @ self._local[idx]

    def centroid(self) -> np.ndarray:
        return self._center + self._orientation @ self._centroid_local

    def bounding_radius(self) -> float:
        return float(np.max(np.linalg.norm(self._local - self._centroid_local, axis=1)))

    # -- geometry ---------------------------------------------------------
    @property
    def local_vertices(self) -> np.ndarray:
        return self._local

    @property
    def vertices(self) -> np.ndarray:
        """World-frame vertices (body frame rotated and translated)."""
        return self._center + self._local @ self._orientation.T

    @property
    def n_vertices(self) -> int:
        return int(self._local.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self._hull.simplices.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.n_vertices + self.n_faces - 2)

    def _tetra_decomposition(self):
        """Interior point and per-facet tetrahedra (absolute volumes)."""
        p0 = self._local.mean(axis=0)
        simplices = self._local[self._hull.simplices]
        a, b, c = simplices[:, 0], simplices[:, 1], simplices[:, 2]
        vol = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), a - p0)) / 6.0
        return p0, a, b, c, vol

    @property
    def volume(self) -> float:
        return float(np.sum(self._tetra_decomposition()[4]))

    @property
    def surface_area(self) -> float:
        p0, a, b, c, _ = self._tetra_decomposition()
        return float(np.sum(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)))

    @property
    def _centroid_local(self) -> np.ndarray:
        p0, a, b, c, vol = self._tetra_decomposition()
        # first moment of a tetrahedron = V (v0 + v1 + v2 + v3) / 4
        first = (vol[:, None] * (p0 + a + b + c) / 4.0).sum(axis=0)
        return first / float(vol.sum())

    @property
    def _second_moment_local(self) -> np.ndarray:
        """Exact ``int x x^T dV`` about the origin of the body frame."""
        p0, a, b, c, vol = self._tetra_decomposition()
        v0 = np.broadcast_to(p0, a.shape)
        stack = np.stack([v0, a, b, c], axis=1)  # (n_tri, 4, 3)
        s = stack.sum(axis=1)                    # (n_tri, 3)
        quad = np.einsum("nki,nkj->nij", stack, stack) + np.einsum("ni,nj->nij", s, s)
        total = np.einsum("n,nij->ij", vol, quad) / 20.0
        return total

    @property
    def covariance(self) -> np.ndarray:
        m1 = self._centroid_local
        v = self.volume
        m2 = self._second_moment_local / v - np.outer(m1, m1)
        return 0.5 * (m2 + m2.T)

    @property
    def aspect_ratio(self) -> float:
        eig = np.linalg.eigvalsh(self.covariance)
        eig = np.clip(eig, 1.0e-300, None)
        return float(math.sqrt(float(eig[-1] / eig[0])))

    @property
    def principal_extents(self) -> np.ndarray:
        """Equivalent-ellipsoid semi-axes ``sqrt(5 * eig)`` (uniform body)."""
        eig = np.linalg.eigvalsh(self.covariance)
        return np.sqrt(np.clip(eig, 1.0e-300, None) * 5.0)

    @property
    def inradius(self) -> float:
        """Chebyshev radius (largest inscribed sphere), solved exactly as an LP."""
        planes = self._hull.equations  # n . x + d <= 0 with |n| = 1
        normals = planes[:, :3]
        offsets = -planes[:, 3]
        # maximise r  <=>  minimise -r,  subject to  n . x + r <= offset
        a_ub = np.hstack([normals, np.ones((normals.shape[0], 1))])
        res = linprog(c=[0.0, 0.0, 0.0, -1.0], A_ub=a_ub, b_ub=offsets,
                      bounds=[(None, None)] * 3 + [(0.0, None)], method="highs")
        if not res.success:
            raise RuntimeError(f"inradius LP failed: {res.message}")
        return float(res.x[3])

    @property
    def incenter(self) -> np.ndarray:
        planes = self._hull.equations
        normals = planes[:, :3]
        offsets = -planes[:, 3]
        a_ub = np.hstack([normals, np.ones((normals.shape[0], 1))])
        res = linprog(c=[0.0, 0.0, 0.0, -1.0], A_ub=a_ub, b_ub=offsets,
                      bounds=[(None, None)] * 3 + [(0.0, None)], method="highs")
        if not res.success:
            raise RuntimeError(f"inradius LP failed: {res.message}")
        return np.asarray(res.x[:3], dtype=float)

    @property
    def circumradius(self) -> float:
        """Radius of the smallest sphere about the incentive enclosing the body."""
        c = self.incenter
        return float(np.max(np.linalg.norm(self._local - c, axis=1)))


class Ellipsoid(ConvexShape):
    """Ellipsoid with semi-axes ``semi_axes`` rotated by ``orientation``.

    Support function (exact, no faceting): with ``M = R diag(axes)``,

        ``support(d) = c + M M^T d / |M^T d|``
    """

    def __init__(self, semi_axes, center=(0.0, 0.0, 0.0), orientation=None,
                 name: str = "ellipsoid"):
        axes = _as_array(semi_axes, (3,))
        if np.any(axes <= 0.0):
            raise ValueError("ellipsoid semi-axes must be positive")
        self.semi_axes = axes
        self.center_vec = _as_array(center, (3,))
        self.orientation = _orthonormal(orientation)
        self.name = str(name)
        self._m = self.orientation @ np.diag(axes)

    def support(self, direction: Sequence[float]) -> np.ndarray:
        d = _as_array(direction, (3,))
        local = self._m.T @ d
        norm = float(np.linalg.norm(local))
        if norm == 0.0:
            raise ValueError("support direction must be non-zero")
        return self.center_vec + self._m @ (local / norm)

    def centroid(self) -> np.ndarray:
        return self.center_vec

    def bounding_radius(self) -> float:
        return float(np.max(self.semi_axes))

    @property
    def volume(self) -> float:
        return float(4.0 / 3.0 * math.pi * np.prod(self.semi_axes))

    @property
    def surface_area(self) -> float:
        """Knud Thomsen's p = 1.6075 approximation (relative error < 1.1%)."""
        a, b, c = (float(x) for x in np.sort(self.semi_axes)[::-1])
        p = 1.6075
        t = ((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3.0
        return float(4.0 * math.pi * t ** (1.0 / p))

    @property
    def exact_surface_area_sphere_limit(self) -> float:
        """Exact area when all three axes are equal (sphere)."""
        a = float(self.semi_axes[0])
        if np.allclose(self.semi_axes, a, rtol=0.0, atol=1e-15):
            return 4.0 * math.pi * a * a
        raise ValueError("not a sphere")


# ---------------------------------------------------------------------------
# descriptors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShapeDescriptors:
    """Integral geometric descriptors of a convex particle."""

    name: str
    volume: float
    surface_area: float
    n_vertices: int
    n_faces: int
    n_edges: int
    sphericity: float
    roundness: float
    aspect_ratio: float
    inradius: float
    circumradius: float
    equivalent_radius: float
    principal_extents: tuple

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "volume": self.volume,
            "surface_area": self.surface_area,
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_edges": self.n_edges,
            "sphericity": self.sphericity,
            "roundness": self.roundness,
            "aspect_ratio": self.aspect_ratio,
            "inradius": self.inradius,
            "circumradius": self.circumradius,
            "equivalent_radius": self.equivalent_radius,
            "principal_extents": list(self.principal_extents),
        }


def describe(shape: PolyhedralShape) -> ShapeDescriptors:
    """Compute all descriptors of a polyhedron in one pass."""
    volume = shape.volume
    area = shape.surface_area
    r_in = shape.inradius
    r_circ = shape.circumradius
    return ShapeDescriptors(
        name=shape.name,
        volume=volume,
        surface_area=area,
        n_vertices=shape.n_vertices,
        n_faces=shape.n_faces,
        n_edges=shape.n_edges,
        sphericity=sphericity(volume, area),
        roundness=float(r_in / r_circ),
        aspect_ratio=shape.aspect_ratio,
        inradius=r_in,
        circumradius=r_circ,
        equivalent_radius=volume_equivalent_radius(volume),
        principal_extents=tuple(float(x) for x in shape.principal_extents[::-1]),
    )


def describe_ellipsoid(shape: Ellipsoid, *, surface_area: Optional[float] = None) -> ShapeDescriptors:
    """Descriptors of an ellipsoid (surface area via the Thomsen approximation)."""
    volume = shape.volume
    area = float(surface_area) if surface_area is not None else shape.surface_area
    axes = np.sort(np.asarray(shape.semi_axes, dtype=float))[::-1]
    return ShapeDescriptors(
        name=shape.name,
        volume=volume,
        surface_area=area,
        n_vertices=0,
        n_faces=0,
        n_edges=0,
        sphericity=sphericity(volume, area),
        roundness=float(axes[-1] / axes[0]),
        aspect_ratio=float(axes[0] / axes[-1]),
        inradius=float(axes[-1]),
        circumradius=float(axes[0]),
        equivalent_radius=volume_equivalent_radius(volume),
        principal_extents=tuple(float(x) for x in axes),
    )


# ---------------------------------------------------------------------------
# canonical and random polyhedra
# ---------------------------------------------------------------------------


def cube_vertices(edge: float = 1.0) -> np.ndarray:
    h = 0.5 * float(edge)
    if h <= 0.0:
        raise ValueError("edge must be positive")
    return np.array([[sx * h, sy * h, sz * h]
                     for sx in (-1.0, 1.0)
                     for sy in (-1.0, 1.0)
                     for sz in (-1.0, 1.0)], dtype=float)


def regular_tetrahedron_vertices(edge: float = 1.0) -> np.ndarray:
    e = float(edge)
    if e <= 0.0:
        raise ValueError("edge must be positive")
    base = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0],
                     [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]]) / math.sqrt(8.0)
    return base * e


def regular_octahedron_vertices(edge: float = 1.0) -> np.ndarray:
    e = float(edge)
    if e <= 0.0:
        raise ValueError("edge must be positive")
    r = e / math.sqrt(2.0)
    return np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0],
                     [0, 0, r], [0, 0, -r]], dtype=float)


def regular_icosahedron_vertices(edge: float = 1.0) -> np.ndarray:
    e = float(edge)
    if e <= 0.0:
        raise ValueError("edge must be positive")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = []
    for a in (-1.0, 1.0):
        for b in (-phi, phi):
            raw += [[0.0, a, b], [a, b, 0.0], [b, 0.0, a]]
    raw = np.array(raw, dtype=float)
    raw *= e / 2.0  # edge length of the golden-ratio coordinates is 2
    return raw


def regular_dodecahedron_vertices(edge: float = 1.0) -> np.ndarray:
    e = float(edge)
    if e <= 0.0:
        raise ValueError("edge must be positive")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                raw.append([sx, sy, sz])
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            raw += [[0.0, s1 / phi, s2 * phi], [s1 / phi, s2 * phi, 0.0],
                    [s2 * phi, 0.0, s1 / phi]]
    raw = np.array(raw, dtype=float)
    raw *= e / (2.0 / phi)  # edge length of these coordinates is 2/phi
    return raw


def named_shape(name: str, size: float = 1.0) -> PolyhedralShape:
    """Canonical polyhedra scaled to unit volume."""
    key = str(name).strip().lower()
    table = {
        "tetrahedron": regular_tetrahedron_vertices,
        "tetra": regular_tetrahedron_vertices,
        "cube": cube_vertices,
        "hexahedron": cube_vertices,
        "octahedron": regular_octahedron_vertices,
        "dodecahedron": regular_dodecahedron_vertices,
        "icosahedron": regular_icosahedron_vertices,
    }
    if key not in table:
        raise ValueError(f"unknown shape '{name}'; known: {sorted(table)}")
    verts = table[key](size)
    scale = (1.0 / (ConvexHull(verts).volume)) ** (1.0 / 3.0)
    return PolyhedralShape(verts * scale, name=key)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """`n` near-uniform unit directions (golden-angle spiral)."""
    i = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * i / n
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    theta = math.pi * (1.0 + math.sqrt(5.0)) * i
    return np.column_stack([r * np.cos(theta), r * np.sin(theta), z])


def spherical_polyhedron(n_vertices: int, radius: float = 1.0,
                         rng: Optional[np.random.Generator] = None,
                         jitter: float = 0.0) -> PolyhedralShape:
    """Convex hull of `n_vertices` points on a sphere (the sphere-limit family).

    With ``jitter = 0`` every sampled point is an extreme point, so the hull has
    exactly ``n_vertices`` vertices and sphericity increases monotonically
    towards 1 as ``n_vertices -> inf`` (the inscribed-hull limit).
    """
    n = int(n_vertices)
    if n < 4:
        raise ValueError("n_vertices must be >= 4")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    dirs = _fibonacci_sphere(n)
    if jitter > 0.0:
        if rng is None:
            rng = np.random.default_rng(0)
        dirs = dirs + jitter * rng.normal(size=dirs.shape)
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return PolyhedralShape(dirs * radius, name=f"spherical-{n}")


def random_polyhedron(n_vertices: int, rng: Optional[np.random.Generator] = None,
                      radius_spread: float = 0.4,
                      radius: float = 1.0) -> PolyhedralShape:
    """Convex hull of `n_vertices` random points on a jittered radial surface.

    The radial jitter breaks the spherical symmetry (lower sphericity, finite
    roundness) while keeping the cloud in convex position on average, so the
    family spans the sphericity range of crushed-ice particles.
    """
    n = int(n_vertices)
    if n < 4:
        raise ValueError("n_vertices must be >= 4")
    if not (0.0 <= radius_spread < 1.0):
        raise ValueError("radius_spread must lie in [0, 1)")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    rng = np.random.default_rng(rng)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    radii = radius * (1.0 - radius_spread * rng.random(n))
    return PolyhedralShape(dirs * radii[:, None],
                           name=f"random-{n}")