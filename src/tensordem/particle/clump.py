"""Rigid multi-sphere ("clump") particles with a controlled aspect ratio.

Why this module
---------------
``tensordem.clump`` already provides two-sphere clumps with *exact* union
(lens-integrated) mass properties, and ``tensordem.clump_geometry`` fits
N-sphere clumps to superquadrics.  Neither gives what the non-spherical DEM
validation face needs:

* an **aspect-ratio parameterised family** (a 2:1 prolate ellipsoid and its
  coarsenings) whose members are *rigidly bonded* and whose union volume is
  matched to the analytic ellipsoid volume, so that a clump and a sphere of the
  same volume can be compared at equal mass;
* a **degenerate branch** that collapses to a single sphere when ``a == b``
  (exactly, bit for bit), which is the acceptance requirement "clump degenerated
  to one sphere reproduces the spherical baseline";
* **union** mass properties for an arbitrary number of overlapping members,
  computed by midpoint voxel integration of the union and cross-checked against
  (i) the analytic single/sphere values and (ii) the naive parallel-axis member
  sum, which double counts the overlap.

Geometry convention
-------------------
Members are stored in the *body frame with the origin at the centre of mass*
(``com`` is removed at build time), so the rigid-body equations of motion use
``r_i`` directly.  The 2:1 ellipsoid has semi-axes ``(a, b, b)`` with
``a = 2 b``; its members are placed along the ``x`` axis with midpoint-rule
radii ``b sqrt(1 - (x_i/a)^2)`` and a common scale factor ``s`` chosen so that
the **union volume equals the ellipsoid volume** (bisection on the voxel
volume).  Consecutive members overlap, which is what makes the assembly rigid
in the clump contact model.

All quantities are SI (m, kg, s).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

__all__ = [
    "SphereMember",
    "ClumpParticle",
    "single_sphere",
    "ellipsoid_clump",
    "ellipsoid_volume",
    "voxel_mass_properties",
    "member_sum_mass_properties",
    "clump_extents",
    "equal_volume_sphere",
    "polyhedron_clump",
    "quat_to_matrix",
    "quat_rotate",
    "quat_mul",
    "quat_from_axis_angle",
    "quat_integrate",
    "random_quaternions",
]


# ---------------------------------------------------------------------------
# Quaternions (w, x, y, z)
# ---------------------------------------------------------------------------
def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (body -> world) from a unit quaternion.

    Accepts a single quaternion ``(4,)`` or a batch ``(n, 4)``.
    """
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3))
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector(s) ``v`` by quaternion ``q``."""
    return quat_to_matrix(q) @ np.asarray(v, dtype=float)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions ``a * b`` (batched over the leading axis)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    w = aw * bw - np.sum(av * bv, axis=-1, keepdims=True)
    v = aw * bv + bw * av + np.cross(av, bv)
    return np.concatenate([w, v], axis=-1)


def quat_from_axis_angle(axis: Sequence[float], angle: float) -> np.ndarray:
    """Unit quaternion for a rotation of ``angle`` about ``axis``."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    h = 0.5 * float(angle)
    return np.concatenate([[math.cos(h)], math.sin(h) * axis])


def quat_integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Advance quaternion(s) by body-frame angular velocity ``omega`` over ``dt``."""
    q = np.asarray(q, dtype=float)
    omega = np.asarray(omega, dtype=float)
    w = q[..., :1]
    v = q[..., 1:]
    dq_w = -0.5 * dt * np.sum(omega * v, axis=-1, keepdims=True)
    dq_v = 0.5 * dt * (w * omega + np.cross(omega, v))
    qn = np.concatenate([w + dq_w, v + dq_v], axis=-1)
    return qn / np.linalg.norm(qn, axis=-1, keepdims=True)


def random_quaternions(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` uniformly distributed random unit quaternions (Shoemake)."""
    u = rng.random((n, 3))
    q = np.stack(
        [
            np.sqrt(1 - u[:, 0]) * np.sin(2 * math.pi * u[:, 1]),
            np.sqrt(1 - u[:, 0]) * np.cos(2 * math.pi * u[:, 1]),
            np.sqrt(u[:, 0]) * np.sin(2 * math.pi * u[:, 2]),
            np.sqrt(u[:, 0]) * np.cos(2 * math.pi * u[:, 2]),
        ],
        axis=1,
    )
    return q / np.linalg.norm(q, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SphereMember:
    """A member sphere of a clump, body frame (origin = clump centre of mass)."""

    center: np.ndarray
    radius: float

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=float).reshape(3)
        object.__setattr__(self, "center", c)
        r = float(self.radius)
        if not (r > 0.0):
            raise ValueError(f"member radius must be positive, got {r}")
        object.__setattr__(self, "radius", r)


def ellipsoid_volume(a: float, b: float) -> float:
    """Volume of the prolate ellipsoid with semi-axes ``(a, b, b)``."""
    return 4.0 / 3.0 * math.pi * float(a) * float(b) ** 2


def voxel_mass_properties(
    members: Sequence[SphereMember],
    density: float,
    grid: int = 72,
) -> dict:
    """Union mass properties by midpoint voxel integration.

    The union of the member spheres is voxelised on a regular grid; each voxel
    centre is counted once if it lies inside *any* member, so the overlap
    between members is counted exactly once (as required for a rigid body).

    Parameters
    ----------
    members : sequence of SphereMember
    density : float
        Uniform density [kg/m^3].
    grid : int
        Number of voxels along the longest bounding-box axis (72 gives ~0.1 %
        volume error for a sphere).

    Returns
    -------
    dict with ``volume``, ``mass``, ``com`` and ``inertia`` (3x3 about ``com``).
    """
    if len(members) == 0:
        raise ValueError("a clump needs at least one member")
    centers = np.array([m.center for m in members], dtype=float)
    radii = np.array([m.radius for m in members], dtype=float)
    lo = (centers - radii[:, None]).min(axis=0)
    hi = (centers + radii[:, None]).max(axis=0)
    extent = hi - lo
    h = float(extent.max()) / float(grid)
    n = np.maximum(2, np.ceil(extent / h).astype(int))
    axes = [lo[k] + (np.arange(n[k]) + 0.5) * h for k in range(3)]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    inside = np.zeros(X.shape, dtype=bool)
    for c, r in zip(centers, radii):
        inside |= (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2 <= r * r
    dv = h ** 3
    n_vox = int(inside.sum())
    volume = n_vox * dv
    if n_vox == 0:
        raise ValueError("voxel grid resolved no member volume; increase grid")
    pts = np.stack([X[inside], Y[inside], Z[inside]], axis=1)
    com = pts.mean(axis=0)
    d = pts - com
    r2 = np.sum(d * d, axis=1)
    inertia = density * dv * (
        np.eye(3) * float(r2.sum()) - d.T @ d
    )
    return {
        "volume": float(volume),
        "mass": float(density * volume),
        "com": com,
        "inertia": inertia,
        "grid_spacing": h,
        "n_voxels": n_vox,
    }


def member_sum_mass_properties(
    members: Sequence[SphereMember], density: float
) -> dict:
    """Naive parallel-axis member sum (overlap double counted) -- comparison only."""
    m_tot = 0.0
    com = np.zeros(3)
    ms = []
    for m in members:
        mi = density * 4.0 / 3.0 * math.pi * m.radius ** 3
        ms.append((mi, m.center, m.radius))
        m_tot += mi
        com += mi * m.center
    com /= m_tot
    inertia = np.zeros((3, 3))
    for mi, c, r in ms:
        d = c - com
        i_own = 0.4 * mi * r * r * np.eye(3)
        inertia += i_own + mi * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return {"volume": m_tot / density, "mass": m_tot, "com": com, "inertia": inertia}


@dataclass(frozen=True)
class ClumpParticle:
    """A rigid multi-sphere particle (union of member spheres).

    The body frame origin is the union centre of mass; ``members`` centres are
    stored relative to it.  ``mass``/``volume``/``inertia`` are the *union*
    values (overlap counted once) unless ``n_members == 1``, in which case they
    are exact analytic sphere values (no voxel error at all).
    """

    members: Tuple[SphereMember, ...]
    density: float = 2650.0
    name: str = "clump"
    volume: float = field(init=False)
    mass: float = field(init=False)
    inertia: np.ndarray = field(init=False)
    inertia_member_sum: np.ndarray = field(init=False)
    aspect_ratio: float = field(init=False)
    bounding_radius: float = field(init=False)

    def __post_init__(self) -> None:
        members = tuple(self.members)
        if not members:
            raise ValueError("a clump needs at least one member")
        if not (self.density > 0.0):
            raise ValueError("density must be positive")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "name", str(self.name))
        if len(members) == 1:
            m = members[0]
            r = m.radius
            vol = 4.0 / 3.0 * math.pi * r ** 3
            mass = self.density * vol
            inertia = 0.4 * mass * r * r * np.eye(3)
            ms = inertia
            com = m.center.copy()
            # shift the single member onto the body origin
            if np.any(np.abs(com) > 0.0):
                object.__setattr__(
                    self, "members", (SphereMember(np.zeros(3), r),)
                )
        else:
            props = voxel_mass_properties(members, self.density, grid=72)
            vol = props["volume"]
            mass = props["mass"]
            com = props["com"]
            inertia = props["inertia"]
            ms = member_sum_mass_properties(members, self.density)["inertia"]
            if np.linalg.norm(com) > 0.0:
                shifted = tuple(SphereMember(m.center - com, m.radius) for m in members)
                object.__setattr__(self, "members", shifted)
        centers = np.array([m.center for m in self.members])
        radii = np.array([m.radius for m in self.members])
        lo = (centers - radii[:, None]).min(axis=0)
        hi = (centers + radii[:, None]).max(axis=0)
        ext = hi - lo
        object.__setattr__(self, "volume", float(vol))
        object.__setattr__(self, "mass", float(mass))
        object.__setattr__(self, "inertia", np.asarray(inertia, dtype=float))
        object.__setattr__(self, "inertia_member_sum", np.asarray(ms, dtype=float))
        object.__setattr__(self, "aspect_ratio", float(ext.max() / ext.min()))
        object.__setattr__(
            self, "bounding_radius", float(np.linalg.norm(centers, axis=1).max() + radii.max())
        )

    # -- convenience -------------------------------------------------------
    @property
    def n_members(self) -> int:
        return len(self.members)

    @property
    def local_centers(self) -> np.ndarray:
        return np.array([m.center for m in self.members], dtype=float)

    @property
    def radii(self) -> np.ndarray:
        return np.array([m.radius for m in self.members], dtype=float)

    def extended(self) -> dict:
        """Plain-dict copy of the geometry (JSON friendly, members only)."""
        return {
            "name": self.name,
            "n_members": self.n_members,
            "density": self.density,
            "volume": self.volume,
            "mass": self.mass,
            "aspect_ratio": self.aspect_ratio,
            "bounding_radius": self.bounding_radius,
            "inertia": self.inertia.tolist(),
            "inertia_member_sum": self.inertia_member_sum.tolist(),
            "members": [
                {"center": m.center.tolist(), "radius": m.radius} for m in self.members
            ],
        }


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def single_sphere(radius: float, density: float = 2650.0, name: str = "sphere") -> ClumpParticle:
    """The degenerate clump: one member sphere (spherical baseline)."""
    return ClumpParticle(
        members=(SphereMember(np.zeros(3), radius),), density=density, name=name
    )


def clump_extents(particle: ClumpParticle) -> np.ndarray:
    """Axis-aligned extents of the union bounding box (body frame)."""
    c = particle.local_centers
    r = particle.radii
    return (c + r[:, None]).max(axis=0) - (c - r[:, None]).min(axis=0)


def ellipsoid_clump(
    a: float,
    b: float,
    n_spheres: int = 8,
    density: float = 2650.0,
    name: str | None = None,
    volume_matched: bool = True,
    grid: int = 48,
) -> ClumpParticle:
    """Multi-sphere clump approximating the prolate ellipsoid with semi-axes ``(a, b, b)``.

    Parameters
    ----------
    a, b : float
        Semi-axis along the body ``x`` axis and the two transverse semi-axes [m].
    n_spheres : int
        Number of bonded member spheres (>= 2).  ``n_spheres == 1`` returns the
        equal-volume sphere (a documented degeneration, used by the clump-count
        convergence study).
    density : float
        Solid density [kg/m^3].
    volume_matched : bool
        If True (default) the member radii carry a common scale factor solved by
        bisection so that the **union** volume matches ``ellipsoid_volume(a, b)``
        to within ~0.5 %; the clump and the baseline sphere then have the same
        mass.  If False, radii are the plain midpoint-rule values (union volume
        slightly low).
    grid : int
        Voxel resolution used by the volume-matching bisection (not by the final
        mass properties, which always use ``grid=72`` in ``ClumpParticle``).

    Notes
    -----
    ``a == b`` (within machine epsilon relative to the larger axis) returns
    :func:`single_sphere` with radius ``b`` **exactly**, so a degenerated clump is
    bit-for-bit a sphere.
    """
    a = float(a)
    b = float(b)
    if not (a > 0.0 and b > 0.0):
        raise ValueError("semi-axes must be positive")
    if a < b:
        raise ValueError("this family is prolate: a >= b required")
    if n_spheres < 1:
        raise ValueError("n_spheres must be >= 1")
    if abs(a - b) <= 1e-12 * max(a, b):
        return single_sphere(b, density, name=name or "ellipsoid-degenerated-sphere")
    if n_spheres < 2:
        r_eq = (a * b * b) ** (1.0 / 3.0)
        return single_sphere(r_eq, density, name=name or "ellipsoid-equal-volume-sphere")

    x = (np.arange(n_spheres) + 0.5) * (2.0 * a / n_spheres) - a
    base = b * np.sqrt(np.clip(1.0 - (x / a) ** 2, 0.0, None))
    # the outermost member sets the tip of the union: its centre is pulled in so
    # that |x| + s*r == a, which fixes the union long axis to exactly 2a (the
    # transverse extent is 2*s*b, hence the achieved aspect ratio is a/(s*b)).
    x_edge = float(abs(x[-1]))
    r_edge = float(base[-1])

    def build(scale: float) -> Tuple[SphereMember, ...]:
        alpha = (a - scale * r_edge) / x_edge if (a - scale * r_edge) > 0 else 1.0
        return tuple(
            SphereMember(np.array([alpha * xi, 0.0, 0.0]), scale * ri)
            for xi, ri in zip(x, base)
        )

    scale = 1.0
    if volume_matched:
        target = ellipsoid_volume(a, b)

        def union_volume(s: float) -> float:
            return voxel_mass_properties(build(s), density, grid=grid)["volume"]

        lo, hi = 0.4, 1.6
        v_lo, v_hi = union_volume(lo), union_volume(hi)
        if v_lo > target:
            hi, v_hi = lo, v_lo
            lo = 0.2
            v_lo = union_volume(lo)
        if v_hi < target:
            lo, v_lo = hi, v_hi
            hi = 2.2
            v_hi = union_volume(hi)
        if v_lo <= target <= v_hi:
            for _ in range(28):
                mid = 0.5 * (lo + hi)
                if union_volume(mid) < target:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 5e-4:
                    break
            scale = 0.5 * (lo + hi)
        else:
            scale = 1.0

    members = build(scale)
    # re-centre onto the union centre of mass so the body origin is the COM
    com = voxel_mass_properties(members, density, grid=grid)["com"]
    members = tuple(SphereMember(m.center - com, m.radius) for m in members)
    return ClumpParticle(
        members=members,
        density=density,
        name=name or f"ellipsoid-{a / b:.2f}-n{n_spheres}",
    )

def equal_volume_sphere(particle: ClumpParticle, density: float | None = None) -> ClumpParticle:
    """Sphere with the same volume (and therefore mass) as ``particle``.

    This is the fair spherical baseline for any clump shape: comparing at equal
    mass removes the "heavier particles fall differently" confound from the
    repose-angle and force-chain comparisons.
    """
    rho = particle.density if density is None else density
    r = (3.0 * particle.volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    return single_sphere(r, rho, name=f"{particle.name}-equal-volume-sphere")


def polyhedron_clump(
    a: float,
    b: float,
    level: int = 1,
    density: float = 2650.0,
    name: str | None = None,
    radius_ratio: float = 0.75,
) -> ClumpParticle:
    """Multi-sphere clump of a **2:1 square prism** (an angular polyhedron).

    Geometry
    --------
    The prism has half sizes ``(a, b, b)`` (aspect ratio ``a/b``, 2:1 here).
    Member spheres of radius ``r = radius_ratio * b`` sit on the four long edges,
    in ``n_rings`` cross-sections spaced ``<= 2r`` apart along the ``x`` axis and
    inset by ``r`` from the end faces, so that

    * the union is **connected** (a valid rigid body: every member overlaps at
      least one neighbour),
    * the union bounding box equals the prism exactly, hence the aspect ratio is
      ``a/b`` to machine precision,
    * the surface keeps the four **flat faces and sharp edges** of the prism --
      a faceted cross-section cannot roll like a wheel about its long axis, which
      is the interlocking mechanism the repose-angle test looks for.

    ``level`` = 1, 2, 3 add the edge mid-points of the long faces and the face
    centres (a refinement family for the clump-count convergence check).

    The union volume is *not* forced to the prism volume; the baseline is the
    equal-volume sphere (:func:`equal_volume_sphere`), so clump and sphere are
    always compared at equal mass.
    """
    a = float(a)
    b = float(b)
    if not (a > 0.0 and b > 0.0):
        raise ValueError("half sizes must be positive")
    if a < b:
        raise ValueError("this family is prolate: a >= b required")
    if level < 1:
        raise ValueError("level must be >= 1")
    r = radius_ratio * b
    if r >= b:
        raise ValueError("radius_ratio must be < 1 so the prism edges stay sharp")
    if 2.0 * r <= 2.0 * (b - r) * 0.75:
        raise ValueError(
            "member radius too small for a connected union; increase radius_ratio"
        )
    ax, ay = a - r, b - r
    span = 2.0 * ax
    n_rings = max(2, int(math.ceil(span / (1.8 * r))) + 1)
    xs = np.linspace(-ax, ax, n_rings)
    pts = set()
    for x in xs:
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.add((float(x), sy * ay, sz * ay))
    # the set above already uses the four edges; add refinements
    if level >= 2:
        for x in xs:
            for s in (-1.0, 1.0):
                pts.add((float(x), 0.0, s * ay))
                pts.add((float(x), s * ay, 0.0))
    if level >= 3:
        for s in (-1.0, 1.0):
            pts.add((s * ax, 0.0, 0.0))
            for x in xs:
                pts.add((float(x), 0.0, 0.0))
    members = tuple(SphereMember(np.array(p, dtype=float), r) for p in sorted(pts))
    return ClumpParticle(
        members=members,
        density=density,
        name=name or f"polyhedron-prism-{a / b:.2f}-L{level}",
    )
