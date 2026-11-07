"""Rigid multi-sphere (clump) particles: geometry, mass properties, contacts.

Why a clump module
------------------
The bonded-sphere engines in this repository (``dem.py``, ``dem3d.py``) represent
ice as *spheres that stay bonded*; every particle is an independent rigid body.
Many DEM applications (rock/rubble, ice floes, sintered aggregates, ballast) need
the other classical representation: a **rigid clump** -- a set of overlapping
spheres glued into one non-deformable body with its own mass, centre of mass and
inertia tensor.  Clumps reproduce non-spherical shapes (dumbbells, tetrahedra,
rounded polyhedra) while keeping the cheapest possible contact detection: every
member is still a sphere, so a clump contact is a **sum over member-sphere
contacts**, each producing a force at its own contact point and therefore a
torque about the clump centre of mass.

What this module provides
-------------------------
* :class:`SphereMember` / :class:`Clump` -- immutable clump geometry with
  validation (radius, density, member overlap limit, connectivity, triple
  overlap) and mass properties.
* **Union** mass properties: the overlapping part of two members is counted
  once, as it must be for a rigid body.  Sphere moments are closed-form; the
  intersection ("lens") moments are integrated exactly by Gauss-Legendre
  quadrature over horizontal slices, whose integrands are polynomials of degree
  <= 4 in the slice coordinate, so an 8-node rule is algebraically exact and the
  result is machine-precision exact.
* ``inertia_model="member_sum"`` -- the naive parallel-axis sum that
  double-counts overlap; kept for comparison and for codes that use that
  convention.
* :func:`double_sphere_inertia_analytic` -- independent closed-form reference
  for the two-sphere clump used by the verification suite.
* :func:`mass_properties_by_voxel` -- independent brute-force midpoint voxel
  integration of the *union* geometry, sharing no code path with the exact
  moments; used as a numerical cross-check.
* :class:`PlaneContact`, :class:`SphereObstacle`, :class:`ClumpBody` in
  :mod:`tensordem.clump_dynamics` -- rigid-body dynamics with member-sphere
  summation for clump-wall and clump-sphere contacts.

Roadmap note
------------
True polyhedral (convex-hull) particles and clump-clump contact with a proper
narrow phase are *not* implemented here; see
``docs/clump_multisphere_particles.md`` for the staged route (member-sphere
summation -> Minkowski/hull narrow phase -> polyhedron with SAT/clipping).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "Clump",
    "ClumpGeometryError",
    "ClumpOverlapError",
    "ClumpTopologyError",
    "SphereMember",
    "double_sphere_clump",
    "double_sphere_inertia_analytic",
    "double_sphere_mass_analytic",
    "gauss_legendre_nodes",
    "inertia_tensor_from_second_moments",
    "lens_moments",
    "mass_properties_by_voxel",
    "sphere_moments",
    "union_moments",
]

_FOUR_PI_OVER_3 = 4.0 / 3.0 * math.pi

# Overlap bookkeeping tolerances.  ``_GEOM_TOL`` is a length tolerance used only
# to decide "touching vs gap"; ``_REL_TOL`` is the relative slack on the overlap
# limit so that a clump built exactly at the limit is accepted.
_GEOM_TOL = 1e-12
_REL_TOL = 1e-12


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class ClumpGeometryError(ValueError):
    """Invalid clump geometry (radius, density, connectivity, overlap)."""


class ClumpOverlapError(ClumpGeometryError):
    """Two members overlap more than the configured limit allows."""


class ClumpTopologyError(ClumpGeometryError):
    """Member graph is disconnected, or overlap regions are not exclusive."""


# --------------------------------------------------------------------------
# Quadrature
# --------------------------------------------------------------------------
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def gauss_legendre_nodes(order: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on ``[-1, 1]`` (cached, NumPy only).

    ``order=8`` integrates polynomials of degree <= 15 exactly, which is why the
    slice integrals below are exact rather than merely accurate.
    """
    if order < 1:
        raise ValueError("quadrature order must be >= 1")
    if order not in _GL_CACHE:
        x, w = np.polynomial.legendre.leggauss(int(order))
        _GL_CACHE[order] = (x, w)
    x, w = _GL_CACHE[order]
    return x.copy(), w.copy()


def _gl_interval(a: float, b: float, order: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Return nodes/weights mapped from ``[-1, 1]`` onto ``[a, b]``."""
    x, w = gauss_legendre_nodes(order)
    half = 0.5 * (b - a)
    mid = 0.5 * (a + b)
    return mid + half * x, half * w


# --------------------------------------------------------------------------
# Members
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SphereMember:
    """One spherical member of a clump, given in the clump's body frame."""

    center: tuple[float, float, float]
    radius: float

    def center_array(self) -> np.ndarray:
        return np.asarray(self.center, dtype=np.float64)


def _as_members(members: Iterable[SphereMember]) -> tuple[SphereMember, ...]:
    out = tuple(members)
    if not out:
        raise ClumpGeometryError("a clump needs at least one member sphere")
    for index, member in enumerate(out):
        if not isinstance(member, SphereMember):
            raise ClumpGeometryError(
                f"member {index} is {type(member).__name__}, expected SphereMember")
    return out


# --------------------------------------------------------------------------
# Exact moments of primitive bodies
# --------------------------------------------------------------------------
def sphere_moments(center: Sequence[float], radius: float) -> tuple[float, np.ndarray, np.ndarray]:
    """``(V, S, Q)`` of a ball about the *global* origin.

    ``V = int dV``, ``S = int x dV``, ``Q = int x x^T dV`` with unit density.
    For a ball ``Q = V c c^T + (V r^2 / 5) I`` exactly.
    """
    c = np.asarray(center, dtype=np.float64)
    r = float(radius)
    volume = _FOUR_PI_OVER_3 * r ** 3
    second = volume * r * r / 5.0
    q = volume * np.outer(c, c) + second * np.eye(3)
    return volume, volume * c, q


def inertia_tensor_from_second_moments(q: np.ndarray, volume: float,
                                       first: np.ndarray) -> np.ndarray:
    """Inertia tensor about the centre of mass from raw second moments."""
    q = np.asarray(q, dtype=np.float64)
    first = np.asarray(first, dtype=np.float64)
    if volume <= 0.0:
        raise ClumpGeometryError("volume must be positive")
    com = first / volume
    i_origin = np.trace(q) * np.eye(3) - q
    return i_origin - volume * (float(com @ com) * np.eye(3) - np.outer(com, com))


def _line_of_centers_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal frame whose third axis is ``direction``."""
    ez = direction / np.linalg.norm(direction)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(ez[0]) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    ex = np.cross(ez, seed)
    ex /= np.linalg.norm(ex)
    ey = np.cross(ez, ex)
    return ex, ey, ez


def lens_moments(center_a: Sequence[float], radius_a: float,
                 center_b: Sequence[float], radius_b: float,
                 order: int = 8) -> tuple[float, np.ndarray, np.ndarray]:
    """``(V, S, Q)`` of the intersection of two balls (unit density).

    The lens is swept along the line of centres.  At height ``z`` the
    cross-section is a disc of radius ``a(z)^2 = R^2 - (z - z_c)^2`` (one of the
    two spheres, whichever is smaller there); ``a^2`` and ``a^4`` are
    polynomials in ``z`` of degree <= 4, and the two sphere branches switch at
    the radical plane, so each smooth sub-interval is integrated *exactly* by
    Gauss-Legendre.  Cross moments vanish by axisymmetry.
    """
    ca = np.asarray(center_a, dtype=np.float64)
    cb = np.asarray(center_b, dtype=np.float64)
    ra, rb = float(radius_a), float(radius_b)
    if ra <= 0.0 or rb <= 0.0:
        raise ClumpGeometryError("lens radii must be positive")
    d = cb - ca
    distance = float(np.linalg.norm(d))
    zero = (0.0, np.zeros(3), np.zeros((3, 3)))

    # Containment: the intersection is the smaller ball.
    if distance <= _GEOM_TOL:
        if abs(ra - rb) <= _GEOM_TOL:
            return sphere_moments(ca, ra)
        small, small_r = (ca, ra) if ra < rb else (cb, rb)
        return sphere_moments(small, small_r)
    if distance + min(ra, rb) <= max(ra, rb) + _GEOM_TOL:
        small, small_r = (ca, ra) if ra < rb else (cb, rb)
        return sphere_moments(small, small_r)
    if distance >= ra + rb - _GEOM_TOL:  # disjoint or touching
        return zero

    ex, ey, ez = _line_of_centers_frame(d)
    z_lo = max(-ra, distance - rb)
    z_hi = min(ra, distance + rb)
    z_radical = (ra * ra - rb * rb + distance * distance) / (2.0 * distance)
    z_radical = min(max(z_radical, z_lo), z_hi)

    volume = 0.0
    first_z = 0.0
    q_axial = 0.0   # int (x^2 + y^2)/2 dV per transverse direction => Q_xx = Q_yy
    q_z = 0.0
    for z_a, z_b in ((z_lo, z_radical), (z_radical, z_hi)):
        if z_b - z_a <= 0.0:
            continue
        z_mid = 0.5 * (z_a + z_b)
        a_sq = ra * ra - z_mid * z_mid
        b_sq = rb * rb - (z_mid - distance) ** 2
        radius, z_center = (ra, 0.0) if a_sq <= b_sq else (rb, distance)
        # |z - z_center| <= radius on this sub-interval by construction
        z, w = _gl_interval(z_a, z_b, order)
        section = np.maximum(radius * radius - (z - z_center) ** 2, 0.0)
        volume += float(np.sum(w * math.pi * section))
        first_z += float(np.sum(w * z * math.pi * section))
        q_axial += float(np.sum(w * math.pi * section ** 2 / 4.0))
        q_z += float(np.sum(w * z * z * math.pi * section))

    if volume <= 0.0:
        return zero
    # Local moments are taken about ``ca`` (the origin of the slice frame) in the
    # (ex, ey, ez) basis; transport them to the global origin.  Dropping these
    # transport terms leaves a *systematic* error in the transverse inertia of an
    # overlapping pair (measured 2.5 % here), which no amount of voxel refinement
    # can expose as a discretisation error -- it was caught by exactly that check.
    basis = np.column_stack((ex, ey, ez))
    first_local = np.array([0.0, 0.0, first_z])
    q_local = np.diag([q_axial, q_axial, q_z])
    first_world = basis @ first_local
    q_world = (volume * np.outer(ca, ca)
               + np.outer(ca, first_world) + np.outer(first_world, ca)
               + basis @ q_local @ basis.T)
    return volume, ca * volume + first_world, q_world


def union_moments(centers: np.ndarray, radii: np.ndarray,
                  order: int = 8) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact ``(V, S, Q)`` of a union of balls with *exclusive* pair overlaps.

    Inclusion-exclusion over members and pairwise intersections.  It is exact
    only when no three members share a common point; that precondition is
    checked by :func:`triple_intersections` before this function is used by
    :class:`Clump`.
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    volume = 0.0
    first = np.zeros(3)
    q = np.zeros((3, 3))
    for c, r in zip(centers, radii):
        v_i, s_i, q_i = sphere_moments(c, r)
        volume += v_i
        first += s_i
        q += q_i
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            v_ij, s_ij, q_ij = lens_moments(centers[i], radii[i],
                                            centers[j], radii[j], order=order)
            volume -= v_ij
            first -= s_ij
            q -= q_ij
    return volume, first, q


def _circle_point_distance(point: np.ndarray, center: np.ndarray,
                           axis: np.ndarray, radius: float) -> float:
    """Distance from ``point`` to a circle (centre ``center``, unit ``axis``)."""
    delta = point - center
    along = float(delta @ axis)
    radial = delta - along * axis
    radial_length = float(np.linalg.norm(radial))
    return math.hypot(along, max(0.0, radial_length - radius))


def _inside_ball(point: np.ndarray, center: np.ndarray, radius: float) -> bool:
    return float(np.linalg.norm(point - center)) <= radius + 1e-12


def balls_intersect(center_a: np.ndarray, radius_a: float,
                    center_b: np.ndarray, radius_b: float) -> bool:
    return float(np.linalg.norm(center_b - center_a)) <= radius_a + radius_b + _GEOM_TOL


def balls_intersection_circle(center_a: np.ndarray, radius_a: float,
                              center_b: np.ndarray, radius_b: float
                              ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Centre, unit axis and radius of the intersection circle of two spheres."""
    d = center_b - center_a
    distance = float(np.linalg.norm(d))
    if distance <= _GEOM_TOL or distance > radius_a + radius_b + _GEOM_TOL:
        return None
    axis = d / distance
    t = (distance ** 2 + radius_a ** 2 - radius_b ** 2) / (2.0 * distance)
    rho_sq = radius_a ** 2 - t ** 2
    rho = math.sqrt(max(rho_sq, 0.0))
    return center_a + t * axis, axis, rho


def triple_intersection(centers: np.ndarray, radii: np.ndarray,
                        indices: Sequence[int]) -> bool:
    """True if the three balls have a common point (positive-measure overlap).

    Intersection of three balls is non-empty iff one centre lies in all three
    balls, or the intersection circle of some pair meets the third ball.
    """
    idx = tuple(indices)
    for k in idx:
        if all(_inside_ball(centers[k], centers[m], radii[m]) for m in idx):
            return True
    for i, j in ((idx[0], idx[1]), (idx[0], idx[2]), (idx[1], idx[2])):
        k = [m for m in idx if m not in (i, j)][0]
        circle = balls_intersection_circle(centers[i], radii[i], centers[j], radii[j])
        if circle is None:
            continue
        q, axis, rho = circle
        if _circle_point_distance(centers[k], q, axis, rho) <= radii[k] + 1e-12:
            return True
    return False


def triple_intersections(centers: np.ndarray, radii: np.ndarray) -> list[tuple[int, int, int]]:
    """All member triples whose three balls share a common point."""
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(radii)
    if n < 3:
        return []
    touching = [[j > i and balls_intersect(centers[i], radii[i], centers[j], radii[j])
                 for j in range(n)] for i in range(n)]
    found = []
    for i in range(n):
        for j in range(i + 1, n):
            if not touching[i][j]:
                continue
            for k in range(j + 1, n):
                if not (touching[i][k] and touching[j][k]):
                    continue
                if triple_intersection(centers, radii, (i, j, k)):
                    found.append((i, j, k))
    return found


# --------------------------------------------------------------------------
# Independent numerical cross-check
# --------------------------------------------------------------------------
def mass_properties_by_voxel(centers: np.ndarray, radii: np.ndarray,
                             density: float = 1.0, resolution: int = 96,
                             slab: int = 16) -> dict[str, np.ndarray | float]:
    """Union mass properties by midpoint voxel integration (independent path).

    The membership test counts a voxel once if it lies in *any* member, so the
    overlap is not double-counted.  Midpoint quadrature on a uniform grid has an
    ``O(h^2)`` error for this integrand; ``resolution`` therefore controls the
    achievable tolerance (see ``docs/clump_multisphere_particles.md`` for the
    measured convergence).
    """
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    if resolution < 4:
        raise ValueError("resolution must be >= 4")
    lo = (centers - radii[:, None]).min(axis=0)
    hi = (centers + radii[:, None]).max(axis=0)
    pad = 1e-3 * float(np.max(hi - lo))
    lo -= pad
    hi += pad
    h = (hi - lo) / (resolution - 1)
    xs = lo[0] + h[0] * np.arange(resolution)
    ys = lo[1] + h[1] * np.arange(resolution)
    zs = lo[2] + h[2] * np.arange(resolution)
    cell = float(h[0] * h[1] * h[2])

    volume = 0.0
    first = np.zeros(3)
    q = np.zeros((3, 3))
    for start in range(0, resolution, slab):
        z_block = zs[start:start + slab]
        zz, yy, xx = np.meshgrid(z_block, ys, xs, indexing="ij")
        shape = xx.shape
        mask = np.zeros(shape, dtype=bool)
        for c, r in zip(centers, radii):
            inside = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2) <= r * r
            mask |= inside
        x = xx[mask]
        y = yy[mask]
        z = zz[mask]
        count = x.size
        volume += count * cell
        first += cell * np.array([x.sum(), y.sum(), z.sum()])
        q[0, 0] += cell * float((x * x).sum())
        q[1, 1] += cell * float((y * y).sum())
        q[2, 2] += cell * float((z * z).sum())
        q[0, 1] += cell * float((x * y).sum())
        q[1, 2] += cell * float((y * z).sum())
        q[0, 2] += cell * float((x * z).sum())
    q[1, 0] = q[0, 1]
    q[2, 1] = q[1, 2]
    q[2, 0] = q[0, 2]
    inertia = inertia_tensor_from_second_moments(q, volume, first)
    return {
        "volume": volume,
        "mass": density * volume,
        "center_of_mass": first / volume,
        "inertia": density * inertia,
        "resolution": resolution,
    }


# --------------------------------------------------------------------------
# Analytic references for the two-sphere clump
# --------------------------------------------------------------------------
def double_sphere_mass_analytic(radius: float, separation: float,
                                density: float = 1.0) -> float:
    """Mass of a two-sphere clump; exact only for non-overlapping members."""
    if separation < 2.0 * radius - _GEOM_TOL:
        raise ClumpGeometryError(
            "double_sphere_mass_analytic requires non-overlapping members")
    return 2.0 * density * _FOUR_PI_OVER_3 * radius ** 3


def double_sphere_inertia_analytic(radius: float, separation: float,
                                   density: float = 1.0) -> np.ndarray:
    """Closed-form inertia tensor of two equal spheres at ``+-separation/2`` on x.

    ``I_xx = 2 * (2/5) m r^2`` (line of centres) and
    ``I_yy = I_zz = 2 * (2/5 m r^2 + m (s/2)^2)`` (parallel-axis), which is the
    textbook value for a dumbbell; valid for ``separation >= 2 radius`` so that
    the union equals the sum of the two balls.
    """
    if separation < 2.0 * radius - _GEOM_TOL:
        raise ClumpGeometryError(
            "double_sphere_inertia_analytic requires non-overlapping members")
    m = density * _FOUR_PI_OVER_3 * radius ** 3
    axial = 2.0 * 0.4 * m * radius ** 2
    transverse = 2.0 * (0.4 * m * radius ** 2 + m * (0.5 * separation) ** 2)
    return np.diag([axial, transverse, transverse])


def double_sphere_clump(radius: float, separation: float | None = None,
                        density: float = 900.0, **kwargs) -> "Clump":
    """Dumbbell clump: two equal spheres on the body-frame x axis.

    ``separation`` defaults to ``2 * radius`` (members exactly touching, i.e. the
    closed-form reference applies unchanged).
    """
    s = 2.0 * radius if separation is None else float(separation)
    members = (
        SphereMember((-0.5 * s, 0.0, 0.0), radius),
        SphereMember((0.5 * s, 0.0, 0.0), radius),
    )
    return Clump(members, density=density, **kwargs)


# --------------------------------------------------------------------------
# Clump
# --------------------------------------------------------------------------
class Clump:
    """Immutable rigid aggregate of overlapping spheres in a body frame.

    Parameters
    ----------
    members:
        Member spheres (``SphereMember``) in the body frame.  The body frame need
        not be centred on the centre of mass; :attr:`local_centers` is.
    density:
        Uniform mass density of the clump material [kg/m^3].
    max_overlap_fraction:
        Largest allowed penetration ``r_i + r_j - |c_i - c_j|`` normalised by the
        smaller radius.  ``0`` forbids overlap entirely (touching allowed).
    require_connected:
        Require the "overlap or touch" member graph to be a single component; a
        disconnected aggregate is not a rigid body.
    inertia_model:
        ``"union"`` (default, physical: overlap counted once) or
        ``"member_sum"`` (naive parallel-axis sum, double-counts overlap).
    """

    def __init__(self, members: Iterable[SphereMember], density: float = 900.0, *,
                 max_overlap_fraction: float = 0.25,
                 require_connected: bool = True,
                 inertia_model: str = "union",
                 name: str = "clump") -> None:
        member_tuple = _as_members(members)
        if not math.isfinite(density) or density <= 0.0:
            raise ClumpGeometryError("density must be finite and positive")
        if inertia_model not in ("union", "member_sum"):
            raise ClumpGeometryError(
                f"unknown inertia_model {inertia_model!r}; use 'union' or 'member_sum'")
        if not math.isfinite(max_overlap_fraction) or max_overlap_fraction < 0.0:
            raise ClumpGeometryError("max_overlap_fraction must be finite and >= 0")

        centers = np.array([m.center_array() for m in member_tuple], dtype=np.float64)
        radii = np.array([float(m.radius) for m in member_tuple], dtype=np.float64)
        if not np.all(np.isfinite(centers)):
            raise ClumpGeometryError("member centers must be finite")
        if not np.all(np.isfinite(radii)):
            raise ClumpGeometryError("member radii must be finite")
        if np.any(radii <= 0.0):
            bad = int(np.flatnonzero(radii <= 0.0)[0])
            raise ClumpGeometryError(
                f"member {bad} radius must be positive (got {radii[bad]!r})")

        self.name = str(name)
        self.density = float(density)
        self.max_overlap_fraction = float(max_overlap_fraction)
        self.inertia_model = inertia_model
        self.centers = centers
        self.radii = radii
        self._check_member_overlaps()
        if require_connected:
            self._check_connectivity()

        volume, first, q = union_moments(centers, radii)
        self.volume = float(volume)
        self.mass = self.density * self.volume
        self.center_of_mass = first / self.volume
        self.local_centers = centers - self.center_of_mass
        self._union_inertia = self.density * inertia_tensor_from_second_moments(
            q, self.volume, first)

        v_ms = 0.0
        s_ms = np.zeros(3)
        q_ms = np.zeros((3, 3))
        for c, r in zip(centers, radii):
            v_i, s_i, q_i = sphere_moments(c, r)
            v_ms += v_i
            s_ms += s_i
            q_ms += q_i
        self.member_sum_volume = float(v_ms)
        self._member_sum_com = s_ms / v_ms
        self._member_sum_inertia = self.density * inertia_tensor_from_second_moments(
            q_ms, v_ms, s_ms)

        if inertia_model == "union":
            self._check_exclusive_overlaps()
        else:
            triple = triple_intersections(centers, radii)
            if triple:
                raise ClumpTopologyError(
                    f"members {triple[0]} share a common point; 'member_sum' is not a"
                    " physical inertia for triple overlaps")

        self.inertia_about_com = (
            self._union_inertia if inertia_model == "union" else self._member_sum_inertia)
        eigenvalues, eigenvectors = np.linalg.eigh(self.inertia_about_com)
        order = np.argsort(eigenvalues)
        self.principal_moments = eigenvalues[order]
        self.principal_axes = eigenvectors[:, order]

    # -- validation ------------------------------------------------------
    def _pair_overlap_fraction(self) -> np.ndarray:
        n = len(self.radii)
        fractions = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                distance = float(np.linalg.norm(self.centers[j] - self.centers[i]))
                penetration = self.radii[i] + self.radii[j] - distance
                if penetration <= 0.0:
                    continue
                smaller = min(self.radii[i], self.radii[j])
                fractions[i, j] = fractions[j, i] = penetration / smaller
        return fractions

    def _check_member_overlaps(self) -> None:
        fractions = self._pair_overlap_fraction()
        n = len(self.radii)
        for i in range(n):
            for j in range(i + 1, n):
                if fractions[i, j] > self.max_overlap_fraction + 1e-12:
                    raise ClumpOverlapError(
                        f"members {i} and {j} overlap by {fractions[i, j]:.6g} of the"
                        f" smaller radius, exceeding max_overlap_fraction="
                        f"{self.max_overlap_fraction:.6g}")
        self.overlap_fractions = fractions

    def _check_connectivity(self) -> None:
        n = len(self.radii)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i in range(n):
            for j in range(i + 1, n):
                if balls_intersect(self.centers[i], self.radii[i],
                                   self.centers[j], self.radii[j]):
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        components = {find(i) for i in range(n)}
        if len(components) > 1:
            raise ClumpTopologyError(
                f"clump members form {len(components)} disconnected groups; a rigid"
                " clump must be one connected body (or pass require_connected=False)")

    def _check_exclusive_overlaps(self) -> None:
        triple = triple_intersections(self.centers, self.radii)
        if triple:
            raise ClumpTopologyError(
                f"members {triple[0]} share a common point; union inertia for"
                " non-exclusive overlaps is not implemented (reduce overlap or use"
                " inertia_model='member_sum')")

    # -- derived quantities ----------------------------------------------
    def member_masses(self) -> np.ndarray:
        return self.density * _FOUR_PI_OVER_3 * self.radii ** 3

    @property
    def inertia_member_sum(self) -> np.ndarray:
        """Naive parallel-axis sum, overlap double-counted (kg m^2)."""
        return self._member_sum_inertia.copy()

    @property
    def inertia_union(self) -> np.ndarray:
        """Union inertia about the union centre of mass (kg m^2)."""
        return self._union_inertia.copy()

    @property
    def equivalent_radius(self) -> float:
        """Radius of the sphere with the same union volume."""
        return (3.0 * self.volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    @property
    def gyration_radius(self) -> float:
        """``sqrt(trace(I)/2 / m)`` -- the RMS radius of gyration."""
        return math.sqrt(float(np.trace(self.inertia_about_com)) / (2.0 * self.mass))

    def inertia_about_axis(self, axis: Sequence[float]) -> float:
        """Moment of inertia about a COM axis given in the body frame."""
        unit = np.asarray(axis, dtype=np.float64)
        unit = unit / np.linalg.norm(unit)
        return float(unit @ self.inertia_about_com @ unit)

    def bounding_radius(self) -> float:
        """Largest member-far extent from the centre of mass."""
        return float(np.max(np.linalg.norm(self.local_centers, axis=1) + self.radii))

    def summary(self) -> dict[str, object]:
        """JSON-serialisable digest used by the verification artefacts."""
        return {
            "name": self.name,
            "members": len(self.radii),
            "radii": self.radii.tolist(),
            "centers": self.centers.tolist(),
            "density": self.density,
            "mass": self.mass,
            "volume": self.volume,
            "member_sum_volume": self.member_sum_volume,
            "overlap_volume_removed": self.member_sum_volume - self.volume,
            "overlap_fraction_max": float(self.overlap_fractions.max()) if self.overlap_fractions.size else 0.0,
            "center_of_mass": self.center_of_mass.tolist(),
            "inertia_model": self.inertia_model,
            "inertia_about_com": self.inertia_about_com.tolist(),
            "inertia_member_sum": self._member_sum_inertia.tolist(),
            "principal_moments": self.principal_moments.tolist(),
            "equivalent_radius": self.equivalent_radius,
            "gyration_radius": self.gyration_radius,
            "bounding_radius": self.bounding_radius(),
        }

    def __len__(self) -> int:
        return len(self.radii)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return (f"Clump(name={self.name!r}, members={len(self)}, "
                f"mass={self.mass:.6g} kg, R_eq={self.equivalent_radius:.6g} m)")