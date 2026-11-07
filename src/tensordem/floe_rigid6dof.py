"""Six-degree-of-freedom rigid-body kinematics and contact physics for ice floes (IN29-N02).

The existing floe model in :mod:`tensordem.floe_6dof` (IN22-N02) is a *disk*
special case: hydrostatics are evaluated with Gauss-Legendre quadrature over the
footprint of a cylinder and the in-plane dynamics come from the H1 channel
engine.  This module is the general rigid-body counterpart needed to drive a
discrete-element (DEM) field of broken ice:

* **Arbitrary closed polyhedra** -- convex *and* concave -- are supported for all
  hydrostatics.  Mass properties come from a signed tetrahedral decomposition
  (the signed contributions cancel outside a concave mesh); submerged volume,
  centre of buoyancy and first moment come from a surface-flux integral
  (divergence theorem) over the part of the mesh below the waterline, which is
  exact for any closed oriented surface.  The waterline cap needs no explicit
  triangulation: in a frame shifted so the free surface is ``zeta = 0`` every
  integrand carries a factor ``zeta`` and the cap contribution vanishes.
* **Quaternion attitude** ``q`` (scalar first) advanced with the exponential map
  ``q <- q (x) exp(omega_b h / 2)`` and renormalised after every stage, so
  ``| |q| - 1 |`` stays at round-off level.
* **Hydrostatics + hydrodynamics**: buoyancy force/moment from the submerged
  centroid, potential energy ``U_b = -rho_w g int_sub z dV`` whose gradient
  reproduces exactly the buoyancy force and moment (locked in by a finite
  difference test), Morison-type added mass folded into an effective inertia
  tensor, quadratic relative-velocity drag and linear heave / rotational
  damping (the latter drives the analytic free-decay benchmark).
* **Contacts**: Hertz normal law with Hunt-Crossley damping and regularised
  Coulomb friction, against static/moving half-spaces (seabed, ice sheet,
  inclined ship hull) and between floes (vertex-in-convex-hull, with
  equal-and-opposite forces applied at a common contact point, so linear *and*
  angular momentum are conserved to round-off).
* **Integrators**: velocity Verlet (symplectic; used for the long energy audits)
  and a Lie-group RK4 with step doubling for adaptive ``dt``.
* **Energy ledger**: kinetic (with the effective inertia) + gravity + buoyancy
  potential, plus accumulated input work, dissipated work and contact work.

Everything is deterministic (no RNG), so every validation number is reproducible.

Modelling limits (also stated in ``docs/in29-n02-floe6dof/REPORT.md``):
contact detection uses the **convex hull** of each floe, so floe-floe contact of
a deeply concave shape is approximated by its hull while its *buoyancy* stays
exact; contact is a vertex-vs-face model (no edge-edge pair), so grazing
edge-edge impacts are resolved only through the vertices of the two meshes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .floe_6dof import (
    integrate_quaternion,
    quat_from_axis_angle,
    quat_from_euler,
    quat_identity,
    quat_multiply,
    quat_normalize,
    quat_norm_error,
    quat_to_matrix,
)

__all__ = [
    "Environment",
    "Hydrodynamics",
    "ContactSettings",
    "PlaneBoundary",
    "MassProperties",
    "SubmergedMoments",
    "FloeGeometry",
    "State",
    "FloeWorld",
    "analytic_heave_period",
    "angular_momentum",
    "box_mesh",
    "build_floe",
    "floe_buoyancy",
    "l_prism_mesh",
    "linear_momentum",
    "mesh_mass_properties",
    "mesh_signed_volume",
    "mesh_topology_report",
    "quat_derivative",
    "regular_prism_mesh",
    "submerged_moments",
]

_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Mesh helpers
# ---------------------------------------------------------------------------

def _as_vertices(vertices) -> np.ndarray:
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if v.shape[0] < 4:
        raise ValueError("a closed polyhedron needs at least four vertices")
    if not np.isfinite(v).all():
        raise ValueError("vertices contain non-finite entries")
    return v


def _as_faces(faces, n_vertices: int) -> np.ndarray:
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must be a triangulation of shape (F, 3)")
    if f.shape[0] < 4:
        raise ValueError("a closed polyhedron needs at least four triangles")
    if f.min() < 0 or f.max() >= n_vertices:
        raise ValueError("face index out of range")
    return f


def mesh_signed_volume(vertices, faces) -> float:
    """Signed volume from the tetrahedral decomposition about the origin."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def mesh_topology_report(vertices, faces) -> Dict[str, object]:
    """Watertightness / orientation audit of a triangulated mesh.

    A consistently oriented closed surface visits every *directed* edge exactly
    once and every *undirected* edge exactly twice; both conditions are checked.
    """
    v = _as_vertices(vertices)
    f = _as_faces(faces, v.shape[0])
    directed = set()
    duplicates = 0
    for tri in f:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(a), int(b))
            if key in directed:
                duplicates += 1
            directed.add(key)
    undirected: Dict[Tuple[int, int], int] = {}
    for a, b in directed:
        key = (a, b) if a < b else (b, a)
        undirected[key] = undirected.get(key, 0) + 1
    boundary = sum(1 for c in undirected.values() if c == 1)
    folded = sum(1 for c in undirected.values() if c > 2)
    return {
        "n_vertices": int(v.shape[0]),
        "n_triangles": int(f.shape[0]),
        "directed_edge_duplicates": int(duplicates),
        "open_edges": int(boundary),
        "non_manifold_edges": int(folded),
        "signed_volume": float(mesh_signed_volume(v, f)),
        "closed": bool(boundary == 0 and duplicates == 0 and folded == 0),
    }


@dataclass(frozen=True)
class MassProperties:
    """Mass properties of a closed polyhedron."""

    volume: float
    centroid: torch.Tensor      # (3,) in the input frame
    mass: float
    inertia: torch.Tensor       # (3,3) about the centroid
    extent: torch.Tensor        # (3,) bounding-box size

    def summary(self) -> Dict[str, object]:
        return {
            "volume": float(self.volume),
            "mass": float(self.mass),
            "centroid": [float(x) for x in self.centroid],
            "inertia_diag": [float(x) for x in torch.diagonal(self.inertia)],
            "extent": [float(x) for x in self.extent],
        }


def mesh_mass_properties(vertices, faces, density: float = 917.0) -> MassProperties:
    """Volume, centroid and inertia tensor of a closed polyhedron.

    Signed tetrahedral decomposition about the origin: for the tetra
    ``(0, a, b, c)`` with signed volume ``Vs = det[a,b,c]/6``,

    ``int x_i x_j dV = Vs/20 * [(sum_p p_i)(sum_p p_j) + sum_p p_i p_j]``

    summed over the four tetra vertices (origin included).  Signed volumes
    cancel outside a concave mesh, so no convexity assumption is made.
    """
    v = _as_vertices(vertices)
    f = _as_faces(faces, v.shape[0])
    if not math.isfinite(float(density)) or float(density) <= 0.0:
        raise ValueError("density must be finite and strictly positive")
    report = mesh_topology_report(v, f)
    if not report["closed"]:
        raise ValueError(
            "mesh is not a closed, consistently oriented surface: "
            f"{report['open_edges']} open edges, "
            f"{report['directed_edge_duplicates']} duplicate directed edges, "
            f"{report['non_manifold_edges']} non-manifold edges")
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = float(vol.sum())
    if total <= 0.0:
        raise ValueError("mesh orientation is inverted (signed volume <= 0); "
                         "reverse the face winding")
    sums = a + b + c
    m1 = (vol[:, None] * sums / 4.0).sum(axis=0)
    centroid = m1 / total
    outer = (np.einsum("fi,fj->fij", a, a) + np.einsum("fi,fj->fij", b, b)
             + np.einsum("fi,fj->fij", c, c))
    m2 = ((vol / 20.0)[:, None, None]
          * (np.einsum("fi,fj->fij", sums, sums) + outer)).sum(axis=0)
    mass = float(density) * total
    inertia_o = float(density) * (float(np.trace(m2)) * np.eye(3) - m2)
    rr = float(centroid @ centroid)
    inertia_c = inertia_o - mass * (rr * np.eye(3) - np.outer(centroid, centroid))
    extent = v.max(axis=0) - v.min(axis=0)
    return MassProperties(volume=float(total),
                          centroid=torch.as_tensor(centroid, dtype=_DTYPE),
                          mass=mass,
                          inertia=torch.as_tensor(inertia_c, dtype=_DTYPE),
                          extent=torch.as_tensor(extent, dtype=_DTYPE))


# ---------------------------------------------------------------------------
# Mesh factories
# ---------------------------------------------------------------------------

def box_mesh(length: float, width: float, thickness: float):
    """Axis-aligned cuboid centred at the origin (12 triangles, outward normals)."""
    for name, value in (("length", length), ("width", width), ("thickness", thickness)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    x, y, z = 0.5 * length, 0.5 * width, 0.5 * thickness
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], dtype=np.float64)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    faces = []
    for q in quads:
        faces.append((q[0], q[1], q[2]))
        faces.append((q[0], q[2], q[3]))
    return v, np.asarray(faces, dtype=np.int64)


def regular_prism_mesh(radius: float, thickness: float, n_segments: int = 32):
    """Regular ``n``-gon prism (a discretised disk floe)."""
    if not math.isfinite(float(radius)) or float(radius) <= 0.0:
        raise ValueError("radius must be finite and strictly positive")
    if not math.isfinite(float(thickness)) or float(thickness) <= 0.0:
        raise ValueError("thickness must be finite and strictly positive")
    if int(n_segments) < 3:
        raise ValueError("n_segments must be at least 3")
    n = int(n_segments)
    ang = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    ring = np.stack([radius * np.cos(ang), radius * np.sin(ang)], axis=1)
    z = 0.5 * thickness
    body = np.concatenate([np.column_stack([ring, np.full(n, -z)]),
                           np.column_stack([ring, np.full(n, z)])], axis=0)
    verts = np.concatenate([body, np.array([[0.0, 0.0, -z], [0.0, 0.0, z]])], axis=0)
    bottom_c, top_c = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
        faces.append((bottom_c, j, i))
        faces.append((top_c, n + i, n + j))
    return verts, np.asarray(faces, dtype=np.int64)


def l_prism_mesh(length: float, width: float, thickness: float,
                 notch_length: float, notch_width: float):
    """Concave L-shaped prism, used for the concavity tests.

    Cross-section: ``(0,0)-(length,0)-(length,notch_width)-(notch_length,
    notch_width)-(notch_length,width)-(0,width)``.
    """
    for name, value in (("length", length), ("width", width), ("thickness", thickness),
                        ("notch_length", notch_length), ("notch_width", notch_width)):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and strictly positive")
    if notch_length >= length or notch_width >= width:
        raise ValueError("notch must be smaller than the outer rectangle")
    poly = np.array([[0.0, 0.0], [length, 0.0], [length, notch_width],
                     [notch_length, notch_width], [notch_length, width],
                     [0.0, width]], dtype=np.float64)
    n = poly.shape[0]
    z = 0.5 * thickness
    verts = np.concatenate([np.column_stack([poly, np.full(n, -z)]),
                            np.column_stack([poly, np.full(n, z)])], axis=0)
    fan = [(0, i, i + 1) for i in range(1, n - 1)]   # the L is star-shaped from 0
    faces = []
    for (i, j, k) in fan:
        faces.append((i, k, j))                      # bottom cap, outward -z
        faces.append((n + i, n + j, n + k))          # top cap, outward +z
    for i in range(n):
        j = (i + 1) % n
        faces.append((i, j, n + j))
        faces.append((i, n + j, n + i))
    return verts, np.asarray(faces, dtype=np.int64)


# ---------------------------------------------------------------------------
# Submerged-volume moments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubmergedMoments:
    """Integrals over ``body & {z < water_level}``."""

    volume: float
    first_moment: torch.Tensor    # (3,) int x dV, world axes
    centroid: torch.Tensor        # (3,) centre of buoyancy, world axes
    zeta_moment: float            # int (z - water_level) dV

    def summary(self) -> Dict[str, object]:
        return {"volume": float(self.volume),
                "centroid": [float(x) for x in self.centroid],
                "zeta_moment": float(self.zeta_moment)}


def _tri_flux(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    """Flux terms of an oriented triangle batch.

    Returns ``(int_T x.n dA, int_T x_i (x.n) dA)`` with explicit component
    arithmetic (``np.cross``/``np.einsum`` overhead dominates at this size).
    """
    ux, uy, uz = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1], b[:, 2] - a[:, 2]
    vx, vy, vz = c[:, 0] - a[:, 0], c[:, 1] - a[:, 1], c[:, 2] - a[:, 2]
    nx = 0.5 * (uy * vz - uz * vy)
    ny = 0.5 * (uz * vx - ux * vz)
    nz = 0.5 * (ux * vy - uy * vx)
    sx = a[:, 0] + b[:, 0] + c[:, 0]
    sy = a[:, 1] + b[:, 1] + c[:, 1]
    sz = a[:, 2] + b[:, 2] + c[:, 2]
    ns = nx * sx + ny * sy + nz * sz
    na = nx * a[:, 0] + ny * a[:, 1] + nz * a[:, 2]
    nb = nx * b[:, 0] + ny * b[:, 1] + nz * b[:, 2]
    nc = nx * c[:, 0] + ny * c[:, 1] + nz * c[:, 2]
    q = np.empty((a.shape[0], 3))
    for i in range(3):
        q[:, i] = (sx * ns if i == 0 else (sy * ns if i == 1 else sz * ns))
        q[:, i] += a[:, i] * na + b[:, i] * nb + c[:, i] * nc
    return ns / 3.0, q / 12.0


def _submerged_integrals(vertices: np.ndarray, faces: np.ndarray):
    """Vectorised clipped-surface integrals below ``z = 0``.

    Every triangle is clipped by the half-space with the exact
    Sutherland-Hodgman case split (0/1/2/3 vertices wet); the clipped polygons
    are fan-triangulated and their divergence-theorem flux accumulated in a
    single batch.  The cost is a fixed number of numpy operations independent of
    the mesh size.
    """
    if faces.size == 0:
        return 0.0, np.zeros(3)
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    ia = a[:, 2] <= 0.0
    ib = b[:, 2] <= 0.0
    ic = c[:, 2] <= 0.0
    n_in = ia.astype(np.int8) + ib.astype(np.int8) + ic.astype(np.int8)

    za, zb, zc = a[:, 2], b[:, 2], c[:, 2]

    def inter(p, q, zp, zq):
        dz = zq - zp
        safe = np.where(np.abs(dz) < 1e-300, 1.0, dz)
        t = (-zp / safe)[:, None]
        return p + t * (q - p)

    iab = inter(a, b, za, zb)
    ibc = inter(b, c, zb, zc)
    ica = inter(c, a, zc, za)

    parts_a: List[np.ndarray] = []
    parts_b: List[np.ndarray] = []
    parts_c: List[np.ndarray] = []

    def push(va, vb, vc, mask):
        if mask.any():
            parts_a.append(va[mask])
            parts_b.append(vb[mask])
            parts_c.append(vc[mask])

    push(a, b, c, n_in == 3)
    push(a, iab, ica, (n_in == 1) & ia)
    push(b, ibc, iab, (n_in == 1) & ib)
    push(c, ica, ibc, (n_in == 1) & ic)
    m = (n_in == 2) & (~ic)
    push(a, b, ibc, m)
    push(a, ibc, ica, m)
    m = (n_in == 2) & (~ia)
    push(b, c, ica, m)
    push(b, ica, iab, m)
    m = (n_in == 2) & (~ib)
    push(c, a, iab, m)
    push(c, iab, ibc, m)
    if not parts_a:
        return 0.0, np.zeros(3)
    vol, q = _tri_flux(np.concatenate(parts_a), np.concatenate(parts_b),
                       np.concatenate(parts_c))
    return float(vol.sum()), q.sum(axis=0)


def _wet_np(vertices_world: np.ndarray, faces: np.ndarray,
            water_level: float):
    """Volume, world-frame first moment and ``int (z - water_level) dV``."""
    shifted = vertices_world
    if water_level != 0.0:
        shifted = vertices_world.copy()
        shifted[:, 2] -= water_level
    vol_term, q = _submerged_integrals(shifted, faces)
    volume = vol_term / 3.0
    q = q / 4.0
    first = np.array([q[0], q[1], q[2] + water_level * volume])
    return volume, first, float(q[2])


def submerged_moments(vertices_world, faces, water_level: float = 0.0) -> SubmergedMoments:
    """Volume, first moment and buoyancy-potential integral of the wet part.

    ``V = (1/3) int x.n dA`` and ``int x dV = (1/4) int x (x.n) dA`` are taken
    over the clipped body surface via the divergence theorem.  In the shifted
    frame ``zeta = z - water_level`` the waterline cap carries ``zeta = 0``, so
    its flux vanishes identically and no cap triangulation is needed.  Exact for
    any closed oriented surface, convex or concave.
    """
    v = np.asarray(vertices_world, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if not np.isfinite(v).all():
        raise ValueError("vertices contain non-finite entries")
    if not math.isfinite(float(water_level)):
        raise ValueError("water_level must be finite")
    f = _as_faces(faces, v.shape[0])
    volume, first, zeta = _wet_np(v, f, float(water_level))
    if abs(volume) < 1e-14:
        volume = 0.0
        centroid = np.array([0.0, 0.0, float(water_level)])
    else:
        centroid = first / volume
    return SubmergedMoments(volume=float(volume),
                            first_moment=torch.as_tensor(first, dtype=_DTYPE),
                            centroid=torch.as_tensor(centroid, dtype=_DTYPE),
                            zeta_moment=float(zeta))


# ---------------------------------------------------------------------------
# Floe geometry
# ---------------------------------------------------------------------------

@dataclass
class FloeGeometry:
    """Rigid floe geometry whose centre of mass sits at the body origin."""

    vertices: torch.Tensor       # (V,3) body frame, COM at origin
    faces: torch.Tensor          # (F,3)
    volume: float
    mass: float
    inertia: torch.Tensor        # (3,3) about the COM
    extent: torch.Tensor         # (3,)
    hull_planes: torch.Tensor    # (P,4): n.x + d <= 0 inside (body frame)
    name: str = "floe"
    vertices_np: np.ndarray = field(default=None, repr=False)
    faces_np: np.ndarray = field(default=None, repr=False)
    hull_np: np.ndarray = field(default=None, repr=False)

    @property
    def n_vertices(self) -> int:
        return int(self.vertices.shape[0])

    def nodes_world(self, position: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """Contact nodes (all mesh vertices) in world axes."""
        quat = torch.as_tensor(quat, dtype=_DTYPE).reshape(1, 4)
        pos = torch.as_tensor(position, dtype=_DTYPE).reshape(1, 3)
        r = quat_to_matrix(quat)[0]
        return self.vertices @ r.T + pos

    def projected_area(self) -> torch.Tensor:
        return torch.stack([self.extent[1] * self.extent[2],
                            self.extent[0] * self.extent[2],
                            self.extent[0] * self.extent[1]])


def build_floe(vertices, faces, density: float = 917.0, name: str = "floe") -> FloeGeometry:
    """Validate a mesh and package it as a rigid floe."""
    props = mesh_mass_properties(vertices, faces, density)
    v = np.asarray(vertices, dtype=np.float64) - props.centroid.numpy()
    f = np.asarray(faces, dtype=np.int64)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(v)
    planes = np.concatenate([hull.equations[:, :3], hull.equations[:, 3:4]], axis=1)
    return FloeGeometry(vertices=torch.as_tensor(v, dtype=_DTYPE),
                        faces=torch.as_tensor(f, dtype=torch.int64),
                        volume=float(props.volume),
                        mass=float(props.mass),
                        inertia=props.inertia.clone(),
                        extent=props.extent.clone(),
                        hull_planes=torch.as_tensor(planes, dtype=_DTYPE),
                        name=str(name),
                        vertices_np=np.ascontiguousarray(v),
                        faces_np=np.ascontiguousarray(f),
                        hull_np=np.ascontiguousarray(planes))


def floe_buoyancy(geometry: FloeGeometry, position, quat, env: "Environment"):
    """Buoyancy force (world), moment about the COM (world) and wet moments."""
    q = quat_normalize(torch.as_tensor(quat, dtype=_DTYPE).reshape(1, 4))[0].tolist()
    p = tuple(float(x) for x in torch.as_tensor(position, dtype=_DTYPE).reshape(3))
    nodes = _nodes_py([tuple(float(c) for c in v) for v in geometry.vertices_np],
                      _rot_py(*q), p)
    volume, first, zeta = _wet_py(nodes, [(int(a), int(b), int(c))
                                          for a, b, c in geometry.faces_np],
                                  float(env.water_level))
    rho_g = env.water_density * env.gravity
    force_np = np.array([0.0, 0.0, rho_g * volume])
    if abs(volume) < 1e-14:
        centroid = np.array(p)
    else:
        centroid = np.asarray(first) / volume
    moment_np = np.cross(centroid - np.asarray(p), force_np)
    wet = SubmergedMoments(volume=float(volume),
                           first_moment=torch.as_tensor(np.asarray(first), dtype=_DTYPE),
                           centroid=torch.as_tensor(centroid, dtype=_DTYPE),
                           zeta_moment=float(zeta))
    return (torch.as_tensor(force_np, dtype=_DTYPE),
            torch.as_tensor(moment_np, dtype=_DTYPE), wet)


# ---------------------------------------------------------------------------
# Environment and coefficients
# ---------------------------------------------------------------------------

@dataclass
class Environment:
    """Ambient water and gravity (free surface at ``water_level``)."""

    water_density: float = 1025.0
    ice_density: float = 917.0
    gravity: float = 9.81
    water_level: float = 0.0
    current_velocity: Sequence[float] = (0.0, 0.0, 0.0)
    enable_buoyancy: bool = True
    enable_gravity: bool = True

    def current(self) -> torch.Tensor:
        return torch.as_tensor(list(self.current_velocity), dtype=_DTYPE)


@dataclass
class Hydrodynamics:
    """Morison-type added mass and drag coefficients.

    ``added_mass_ratio`` multiplies the displaced mass at the equilibrium
    draught, i.e. the body mass (Archimedes).  Keeping it constant lets the
    effective inertia be folded into the equation of motion and keeps the energy
    ledger exact; the attitude dependence of the added mass is neglected
    (documented limitation).
    """

    added_mass_ratio: float = 0.5
    added_inertia_ratio: float = 0.2
    drag_coefficient: float = 1.2
    rotational_damping: float = 0.0
    linear_heave_damping: float = 0.0
    enable_drag: bool = True

    def validate(self) -> None:
        for name in ("added_mass_ratio", "added_inertia_ratio", "drag_coefficient",
                     "rotational_damping", "linear_heave_damping"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass
class ContactSettings:
    """Hertz normal contact with Hunt-Crossley damping and Coulomb friction."""

    normal_stiffness: float = 5.0e6    # N / m^1.5
    normal_damping: float = 0.0        # 1 / (m/s)
    friction: float = 0.0              # Coulomb mu
    velocity_scale: float = 1.0e-3     # m/s, tanh regularisation

    def validate(self) -> None:
        if not math.isfinite(self.normal_stiffness) or self.normal_stiffness <= 0.0:
            raise ValueError("normal_stiffness must be finite and positive")
        for name in ("normal_damping", "friction"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.velocity_scale) or self.velocity_scale <= 0.0:
            raise ValueError("velocity_scale must be finite and positive")


@dataclass(frozen=True)
class PlaneBoundary:
    """Half-space: the solid occupies ``n . x < offset``.

    ``velocity`` is the bulk velocity of the boundary (a moving hull face); it
    enters the contact kinematics and the external-work ledger.
    """

    normal: Tuple[float, float, float]
    offset: float = 0.0
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = "plane"

    def unit_normal(self) -> torch.Tensor:
        n = torch.as_tensor(list(self.normal), dtype=_DTYPE)
        norm = float(torch.linalg.norm(n))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("plane normal must be finite and non-zero")
        return n / norm


# ---------------------------------------------------------------------------
# State / reference solutions
# ---------------------------------------------------------------------------

@dataclass
class State:
    position: torch.Tensor   # (N,3)
    velocity: torch.Tensor   # (N,3)
    quat: torch.Tensor       # (N,4)
    omega: torch.Tensor      # (N,3) body frame

    def clone(self) -> "State":
        return State(self.position.clone(), self.velocity.clone(),
                     self.quat.clone(), self.omega.clone())


def quat_derivative(quat: torch.Tensor, omega_body: torch.Tensor) -> torch.Tensor:
    """``qdot = 1/2 q (x) (0, omega_b)``."""
    zero = torch.zeros(quat.shape[0], 1, dtype=quat.dtype)
    return quat_multiply(quat, torch.cat([zero, omega_body], dim=1))


def analytic_heave_period(waterplane_area: float, mass: float,
                          added_mass_ratio: float, damping: float,
                          water_density: float = 1025.0,
                          gravity: float = 9.81) -> Dict[str, float]:
    """Closed-form damped heave oscillator used as the reference solution.

    ``omega_n = sqrt(k / m_eff)``, ``zeta = c / (2 sqrt(k m_eff))``,
    ``omega_d = omega_n sqrt(1 - zeta^2)`` with ``k = rho_w g A_wp`` and
    ``m_eff = m (1 + C_a)``.
    """
    k = water_density * gravity * waterplane_area
    m_eff = mass * (1.0 + added_mass_ratio)
    omega_n = math.sqrt(k / m_eff)
    zeta = damping / (2.0 * math.sqrt(k * m_eff))
    omega_d = omega_n * math.sqrt(max(1.0 - zeta * zeta, 0.0))
    return {"stiffness": k, "mass_effective": m_eff, "omega_n": omega_n,
            "zeta": zeta, "omega_d": omega_d,
            "period_damped": 2.0 * math.pi / omega_d,
            "period_undamped": 2.0 * math.pi / omega_n,
            "log_decrement": 2.0 * math.pi * zeta / math.sqrt(max(1.0 - zeta * zeta, 1e-30))}


@dataclass
class _Forces:
    force: torch.Tensor           # (N,3) world
    moment: torch.Tensor          # (N,3) world about the COM
    power_diss: torch.Tensor      # (N,) W
    power_input: torch.Tensor     # (N,) W
    power_contact: torch.Tensor   # (N,) W
    buoyancy_potential: torch.Tensor  # (N,) J

# ---------------------------------------------------------------------------
# Fast numpy kernel (the public API stays torch; the hot loop is numpy because
# torch operator overhead on the O(10) elements of a floe dominates the cost)
# ---------------------------------------------------------------------------

def _cross_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az = a[:, 0], a[:, 1], a[:, 2]
    bx, by, bz = b[:, 0], b[:, 1], b[:, 2]
    return np.stack([ay * bz - az * by,
                     az * bx - ax * bz,
                     ax * by - ay * bx], axis=1)


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=1)


def _quat_norm_np(q: np.ndarray) -> np.ndarray:
    norm = np.sqrt((q * q).sum(axis=1, keepdims=True))
    if not np.isfinite(norm).all() or (norm < 1e-14).any():
        raise ValueError("quaternion norm is numerically zero or non-finite")
    return q / norm


def _rot_np(q: np.ndarray) -> np.ndarray:
    """Body -> world rotation matrices of a batch of unit quaternions."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3))
    r[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    r[:, 0, 1] = 2.0 * (x * y - w * z)
    r[:, 0, 2] = 2.0 * (x * z + w * y)
    r[:, 1, 0] = 2.0 * (x * y + w * z)
    r[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    r[:, 1, 2] = 2.0 * (y * z - w * x)
    r[:, 2, 0] = 2.0 * (x * z - w * y)
    r[:, 2, 1] = 2.0 * (y * z + w * x)
    r[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return r


def _quat_integrate_np(q: np.ndarray, omega: np.ndarray, h: float) -> np.ndarray:
    """Exponential-map attitude update ``q (x) exp(omega h / 2)`` + renormalise."""
    speed = np.sqrt((omega * omega).sum(axis=1))
    half = 0.5 * speed * h
    ratio = np.where(speed > 1e-12, np.sin(half) / np.maximum(speed, 1e-300), 0.5 * h)
    delta = np.concatenate([np.cos(half)[:, None], omega * ratio[:, None]], axis=1)
    return _quat_norm_np(_quat_mul_np(q, delta))


# ---------------------------------------------------------------------------
# Fast float kernel for the hot loop
#
# The public API is torch (it interoperates with :mod:`tensordem.floe_6dof`),
# but a floe carries only O(10..100) vertices, so numpy operator overhead
# (~5 us per call in this build) would dominate the run time.  The per-body
# force assembly therefore runs on plain Python floats over lists of tuples; the
# vectorised numpy versions (:func:`_wet_np`, :func:`submerged_moments`) remain
# available and are cross-checked against the float kernel in the test suite.
# ---------------------------------------------------------------------------

def _rot_py(w: float, x: float, y: float, z: float):
    """Body -> world rotation matrix as three row tuples."""
    return ((1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)),
            (2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)),
            (2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)))


def _apply_py(r, v):
    return (r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
            r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
            r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2])


def _apply_t_py(r, v):
    return (r[0][0] * v[0] + r[1][0] * v[1] + r[2][0] * v[2],
            r[0][1] * v[0] + r[1][1] * v[1] + r[2][1] * v[2],
            r[0][2] * v[0] + r[1][2] * v[1] + r[2][2] * v[2])


def _nodes_py(vertices, r, p):
    """World-frame vertices of a body from a python rotation matrix."""
    return [(r[0][0] * x + r[0][1] * y + r[0][2] * z + p[0],
             r[1][0] * x + r[1][1] * y + r[1][2] * z + p[1],
             r[2][0] * x + r[2][1] * y + r[2][2] * z + p[2])
            for (x, y, z) in vertices]


def _clip_tri_py(a, b, c, z_limit: float = 0.0):
    """Sutherland-Hodgman clip of one triangle to ``z <= z_limit``."""
    out = []
    for p, q in ((a, b), (b, c), (c, a)):
        pin = p[2] <= z_limit
        qin = q[2] <= z_limit
        if pin:
            out.append(p)
        if pin != qin:
            t = (z_limit - p[2]) / (q[2] - p[2])
            out.append((p[0] + t * (q[0] - p[0]),
                        p[1] + t * (q[1] - p[1]),
                        p[2] + t * (q[2] - p[2])))
    return out


def _wet_py(vertices, faces, water_level: float = 0.0):
    """Volume, world first moment and ``int (z - water_level) dV`` of the wet part.

    Pure-float counterpart of :func:`_wet_np` for one body; ``vertices`` is a
    list of world-frame ``(x, y, z)`` tuples and ``faces`` a list of index
    triples.
    """
    z0 = float(water_level)
    shifted = [(x, y, z - z0) for (x, y, z) in vertices]
    vol_term = 0.0
    qx = qy = qz = 0.0
    for (i0, i1, i2) in faces:
        poly = _clip_tri_py(shifted[i0], shifted[i1], shifted[i2], 0.0)
        m = len(poly)
        if m < 3:
            continue
        a = poly[0]
        for k in range(1, m - 1):
            b = poly[k]
            c = poly[k + 1]
            ux = b[0] - a[0]
            uy = b[1] - a[1]
            uz = b[2] - a[2]
            vx = c[0] - a[0]
            vy = c[1] - a[1]
            vz = c[2] - a[2]
            nx = 0.5 * (uy * vz - uz * vy)
            ny = 0.5 * (uz * vx - ux * vz)
            nz = 0.5 * (ux * vy - uy * vx)
            sx = a[0] + b[0] + c[0]
            sy = a[1] + b[1] + c[1]
            sz = a[2] + b[2] + c[2]
            ns = nx * sx + ny * sy + nz * sz
            vol_term += ns / 3.0
            na = nx * a[0] + ny * a[1] + nz * a[2]
            nb = nx * b[0] + ny * b[1] + nz * b[2]
            nc = nx * c[0] + ny * c[1] + nz * c[2]
            qx += (sx * ns + a[0] * na + b[0] * nb + c[0] * nc) / 12.0
            qy += (sy * ns + a[1] * na + b[1] * nb + c[1] * nc) / 12.0
            qz += (sz * ns + a[2] * na + b[2] * nb + c[2] * nc) / 12.0
    volume = vol_term / 3.0
    qx *= 0.25
    qy *= 0.25
    qz *= 0.25
    return volume, (qx, qy, qz + z0 * volume), qz


class FloeWorld:
    """Rigid-body scene of floating ice floes with contacts and a fluid.

    The state is held as torch tensors (:class:`State`), so it interoperates
    with the quaternion utilities of :mod:`tensordem.floe_6dof`; the force
    assembly and the integrators run on a private numpy float64 view because the
    torch operator overhead on O(10)-element floe tensors would dominate the
    run time of a DEM field.  Both views carry the same float64 numbers.
    """

    def __init__(self, geometries: Sequence[FloeGeometry], positions,
                 quats=None, velocities=None, omegas=None,
                 env: Optional[Environment] = None,
                 hydro: Optional[Hydrodynamics] = None,
                 contact: Optional[ContactSettings] = None,
                 planes: Sequence[PlaneBoundary] = (),
                 integrator: str = "verlet") -> None:
        if len(geometries) == 0:
            raise ValueError("at least one floe geometry is required")
        self.geometries: List[FloeGeometry] = list(geometries)
        n = len(self.geometries)
        self.env = env or Environment()
        self.hydro = hydro or Hydrodynamics()
        self.contact = contact or ContactSettings()
        self.hydro.validate()
        self.contact.validate()
        if not math.isfinite(float(self.env.water_density)) or self.env.water_density < 0.0:
            raise ValueError("water_density must be finite and non-negative")
        if not math.isfinite(float(self.env.gravity)) or self.env.gravity < 0.0:
            raise ValueError("gravity must be finite and non-negative")
        self.planes: List[PlaneBoundary] = list(planes)
        self._planes = []
        for plane in self.planes:
            nrm = plane.unit_normal().numpy()
            vel = np.asarray(list(plane.velocity), dtype=np.float64)
            if vel.shape != (3,) or not np.isfinite(vel).all():
                raise ValueError("plane velocity must be three finite components")
            self._planes.append((nrm, float(plane.offset), vel))
        self.integrator = str(integrator)
        if self.integrator not in ("verlet", "rk4"):
            raise ValueError("integrator must be 'verlet' or 'rk4'")

        pos = torch.as_tensor(positions, dtype=_DTYPE).reshape(n, 3)
        if not torch.isfinite(pos).all():
            raise ValueError("positions must be finite")
        quat = quat_identity(n) if quats is None else torch.as_tensor(
            quats, dtype=_DTYPE).reshape(n, 4)
        quat = quat_normalize(quat)
        vel = (torch.zeros(n, 3, dtype=_DTYPE) if velocities is None
               else torch.as_tensor(velocities, dtype=_DTYPE).reshape(n, 3))
        omega = (torch.zeros(n, 3, dtype=_DTYPE) if omegas is None
                 else torch.as_tensor(omegas, dtype=_DTYPE).reshape(n, 3))
        for tensor, label in ((vel, "velocities"), (omega, "omegas")):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{label} must be finite")
        self.state = State(pos, vel, quat, omega)

        self.mass = torch.tensor([g.mass for g in self.geometries], dtype=_DTYPE)
        inertia = torch.stack([g.inertia for g in self.geometries], dim=0)
        extent = torch.stack([g.extent for g in self.geometries], dim=0)
        self.mass_eff = self.mass * (1.0 + self.hydro.added_mass_ratio)
        self.inertia_eff = inertia * (1.0 + self.hydro.added_inertia_ratio)
        self.inv_inertia_eff = torch.linalg.inv(self.inertia_eff)
        self.ref_area = torch.stack([extent[:, 1] * extent[:, 2],
                                     extent[:, 0] * extent[:, 2],
                                     extent[:, 0] * extent[:, 1]], dim=1)
        self.length_scale = torch.linalg.norm(extent, dim=1)

        self._mass = self.mass.numpy()
        self._mass_eff = self.mass_eff.numpy()
        self._inertia_eff = self.inertia_eff.numpy()
        self._inv_inertia_eff = self.inv_inertia_eff.numpy()
        self._ref_area = self.ref_area.numpy()
        self._length_scale = self.length_scale.numpy()
        self._vertices_py = [[tuple(float(c) for c in v) for v in g.vertices_np]
                             for g in self.geometries]
        self._faces_py = [[(int(a), int(b), int(c)) for a, b, c in g.faces_np]
                          for g in self.geometries]
        self._hull_py = [[tuple(float(v) for v in row) for row in g.hull_np]
                         for g in self.geometries]
        self._current = tuple(float(x) for x in
                              np.asarray(list(self.env.current_velocity), dtype=float))

        self.time = 0.0
        self.steps = 0
        self.dissipated = 0.0
        self.input_work = 0.0
        self.contact_work = 0.0
        self.max_penetration = 0.0
        self.initial_energy = self.energy()["total"]

    # -- numpy state view --------------------------------------------------
    def _np_state(self):
        s = self.state
        return (s.position.numpy(), s.velocity.numpy(),
                s.quat.numpy(), s.omega.numpy())

    def _set_np_state(self, st) -> None:
        self.state = State(*(torch.as_tensor(np.ascontiguousarray(x), dtype=_DTYPE)
                             for x in st))

    # -- force assembly ----------------------------------------------------
    def _forces(self, st):
        """Per-body force assembly (plain floats; see the kernel note above)."""
        pos, vel, quat, omega = st
        n = len(self.geometries)
        pos_l = pos.tolist()
        vel_l = vel.tolist()
        rots = [_rot_py(*q) for q in quat.tolist()]
        omega_world = np.einsum("nij,nj->ni", _rot_np(quat), omega)
        force = [[0.0, 0.0, 0.0] for _ in range(n)]
        moment = [[0.0, 0.0, 0.0] for _ in range(n)]
        p_diss = [0.0] * n
        p_input = [0.0] * n
        p_contact = [0.0] * n
        potential = [0.0] * n
        elastic = [0.0] * n
        rho_w = float(self.env.water_density)
        grav = float(self.env.gravity)
        current = self._current
        drag_on = bool(self.hydro.enable_drag and self.hydro.drag_coefficient > 0.0)
        water_level = float(self.env.water_level)

        for b in range(n):
            r = rots[b]
            p = pos_l[b]
            v = vel_l[b]
            nodes = _nodes_py(self._vertices_py[b], r, p)
            fx = fy = fz = mx = my = mz = 0.0
            if self.env.enable_gravity:
                fz -= self._mass[b] * grav
            if self.env.enable_buoyancy:
                volume, first, zeta = _wet_py(nodes, self._faces_py[b], water_level)
                f_bz = rho_w * grav * volume
                if volume > 1e-14:
                    cx = first[0] / volume - p[0]
                    cy = first[1] / volume - p[1]
                else:
                    cx = cy = 0.0
                fz += f_bz
                mx += cy * f_bz
                my += -cx * f_bz
                potential[b] = -rho_w * grav * zeta
            vrx = v[0] - current[0]
            vry = v[1] - current[1]
            vrz = v[2] - current[2]
            if drag_on:
                vb = _apply_t_py(r, (vrx, vry, vrz))
                area = self._ref_area[b]
                coef = -0.5 * rho_w * self.hydro.drag_coefficient
                fd = _apply_py(r, (coef * area[0] * abs(vb[0]) * vb[0],
                                   coef * area[1] * abs(vb[1]) * vb[1],
                                   coef * area[2] * abs(vb[2]) * vb[2]))
                fx += fd[0]
                fy += fd[1]
                fz += fd[2]
                p_diss[b] -= fd[0] * vrx + fd[1] * vry + fd[2] * vrz
                p_input[b] += fd[0] * current[0] + fd[1] * current[1] + fd[2] * current[2]
            if self.hydro.linear_heave_damping > 0.0:
                flz = -self.hydro.linear_heave_damping * vrz
                fz += flz
                p_diss[b] -= flz * vrz
                p_input[b] += flz * current[2]
            if self.hydro.rotational_damping > 0.0:
                w = omega.tolist()[b]
                mw = _apply_py(r, (-self.hydro.rotational_damping * w[0],
                                   -self.hydro.rotational_damping * w[1],
                                   -self.hydro.rotational_damping * w[2]))
                mx += mw[0]
                my += mw[1]
                mz += mw[2]
                p_diss[b] += self.hydro.rotational_damping * (
                    w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
            if self._planes:
                cfx, cfy, cfz, cmx, cmy, cmz, pcon, pin, econs = self._plane_contacts(
                    nodes, p, v, tuple(omega_world[b]))
                fx += cfx
                fy += cfy
                fz += cfz
                mx += cmx
                my += cmy
                mz += cmz
                p_contact[b] += pcon
                p_input[b] += pin
                elastic[b] += econs
            force[b][0] = fx
            force[b][1] = fy
            force[b][2] = fz
            moment[b][0] = mx
            moment[b][1] = my
            moment[b][2] = mz

        for i in range(n):
            for j in range(i + 1, n):
                self._pair_contact(pos_l, vel_l, rots, omega_world, i, j,
                                   force, moment, p_contact, elastic)
        return (np.asarray(force), np.asarray(moment), np.asarray(p_diss),
                np.asarray(p_input), np.asarray(p_contact), np.asarray(potential),
                np.asarray(elastic))

    def _contact_force(self, depth, nrm, v_rel):
        """Hertz normal force (Hunt-Crossley damping) + regularised Coulomb."""
        vn = v_rel[0] * nrm[0] + v_rel[1] * nrm[1] + v_rel[2] * nrm[2]
        f_n = self.contact.normal_stiffness * depth ** 1.5
        if self.contact.normal_damping > 0.0 and vn < 0.0:
            f_n *= (1.0 - self.contact.normal_damping * vn)
        if f_n <= 0.0:
            return (0.0, 0.0, 0.0)
        fx = nrm[0] * f_n
        fy = nrm[1] * f_n
        fz = nrm[2] * f_n
        mu = self.contact.friction
        if mu > 0.0:
            tx = v_rel[0] - vn * nrm[0]
            ty = v_rel[1] - vn * nrm[1]
            tz = v_rel[2] - vn * nrm[2]
            speed = math.sqrt(tx * tx + ty * ty + tz * tz)
            if speed > 0.0:
                scale = mu * f_n * math.tanh(speed / self.contact.velocity_scale) / speed
                fx -= scale * tx
                fy -= scale * ty
                fz -= scale * tz
        return (fx, fy, fz)

    def _plane_contacts(self, nodes, p, v, omega_world):
        fx = fy = fz = mx = my = mz = 0.0
        pcon = pin = elastic = 0.0
        for nrm, offset, vel_plane in self._planes:
            for node in nodes:
                gap = node[0] * nrm[0] + node[1] * nrm[1] + node[2] * nrm[2] - offset
                if gap >= 0.0:
                    continue
                depth = -gap
                rx = node[0] - p[0]
                ry = node[1] - p[1]
                rz = node[2] - p[2]
                vpx = v[0] + omega_world[1] * rz - omega_world[2] * ry
                vpy = v[1] + omega_world[2] * rx - omega_world[0] * rz
                vpz = v[2] + omega_world[0] * ry - omega_world[1] * rx
                f = self._contact_force(depth, nrm, (vpx - vel_plane[0],
                                                    vpy - vel_plane[1],
                                                    vpz - vel_plane[2]))
                fx += f[0]
                fy += f[1]
                fz += f[2]
                mx += ry * f[2] - rz * f[1]
                my += rz * f[0] - rx * f[2]
                mz += rx * f[1] - ry * f[0]
                pcon += f[0] * vpx + f[1] * vpy + f[2] * vpz
                pin -= f[0] * vel_plane[0] + f[1] * vel_plane[1] + f[2] * vel_plane[2]
                if self.contact.normal_damping <= 0.0:
                    elastic += 0.4 * self.contact.normal_stiffness * depth ** 2.5
                if depth > self.max_penetration:
                    self.max_penetration = depth
        return fx, fy, fz, mx, my, mz, pcon, pin, elastic

    def _sat_axis(self, nodes_a, nodes_b, rots, i, j):
        """Minimum-overlap separating axis over the face normals of both floes.

        Returns the unit contact normal pointing from floe ``j`` towards floe
        ``i`` (the direction that pushes ``i`` out of ``j``), or ``None`` when
        the convex hulls are disjoint.  Only face-normal axes are tested (no
        edge-edge axes), so an edge-on-edge contact of two prisms may be missed
        -- see the module docstring.
        """
        axes = []
        for body, other in ((i, j), (j, i)):
            r = rots[body]
            for plane in self._hull_py[body]:
                axes.append(_apply_py(r, (plane[0], plane[1], plane[2])))
        best = None
        for n in axes:
            pa = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v in nodes_a]
            pb = [v[0] * n[0] + v[1] * n[1] + v[2] * n[2] for v in nodes_b]
            overlap = min(max(pa), max(pb)) - max(min(pa), min(pb))
            if overlap <= 0.0:
                return None
            if best is None or overlap < best[0]:
                ca = sum(pa) / len(pa)
                cb = sum(pb) / len(pb)
                direction = n if ca >= cb else (-n[0], -n[1], -n[2])
                best = (overlap, direction)
        return None if best is None else best[1]

    def _pair_contact(self, pos_l, vel_l, rots, omega_world, i, j,
                      force, moment, p_contact, elastic):
        """Floe-floe contact on the minimum-overlap (SAT) axis.

        The normal is the pair's minimum-overlap axis over the face normals of
        both convex hulls; every vertex of one floe that lies inside the other's
        hull then receives a Hertz force along that normal, with a depth equal to
        its distance behind the parallel face.  Both bodies share the same
        contact point, so the force pair conserves linear and angular momentum to
        round-off.
        """
        nodes_a = _nodes_py(self._vertices_py[i], rots[i], pos_l[i])
        nodes_b = _nodes_py(self._vertices_py[j], rots[j], pos_l[j])
        axis = self._sat_axis(nodes_a, nodes_b, rots, i, j)
        if axis is None:
            return
        wa = omega_world[i]
        wb = omega_world[j]
        for a, b, nodes, push in ((i, j, nodes_a, axis),
                                  (j, i, nodes_b, (-axis[0], -axis[1], -axis[2]))):
            r_b = rots[b]
            pb = pos_l[b]
            pa = pos_l[a]
            wa_ = omega_world[a]
            wb_ = omega_world[b]
            va = vel_l[a]
            vb = vel_l[b]
            for node in nodes:
                local = _apply_t_py(r_b, (node[0] - pb[0], node[1] - pb[1],
                                          node[2] - pb[2]))
                gaps = [local[0] * pl[0] + local[1] * pl[1] + local[2] * pl[2] + pl[3]
                        for pl in self._hull_py[b]]
                worst = max(gaps)
                if worst > 0.0:
                    continue
                # depth behind the face most parallel to the contact normal
                best_dot = None
                depth = 0.0
                for idx, pl in enumerate(self._hull_py[b]):
                    nw = _apply_py(r_b, (pl[0], pl[1], pl[2]))
                    dot = nw[0] * push[0] + nw[1] * push[1] + nw[2] * push[2]
                    if best_dot is None or dot > best_dot:
                        best_dot = dot
                        depth = -gaps[idx]
                if depth <= 0.0:
                    continue
                rax = node[0] - pa[0]
                ray = node[1] - pa[1]
                raz = node[2] - pa[2]
                rbx = node[0] - pb[0]
                rby = node[1] - pb[1]
                rbz = node[2] - pb[2]
                vax = va[0] + wa_[1] * raz - wa_[2] * ray
                vay = va[1] + wa_[2] * rax - wa_[0] * raz
                vaz = va[2] + wa_[0] * ray - wa_[1] * rax
                vbx = vb[0] + wb_[1] * rbz - wb_[2] * rby
                vby = vb[1] + wb_[2] * rbx - wb_[0] * rbz
                vbz = vb[2] + wb_[0] * rby - wb_[1] * rbx
                f = self._contact_force(depth, push,
                                        (vax - vbx, vay - vby, vaz - vbz))
                force[a][0] += f[0]
                force[a][1] += f[1]
                force[a][2] += f[2]
                force[b][0] -= f[0]
                force[b][1] -= f[1]
                force[b][2] -= f[2]
                moment[a][0] += ray * f[2] - raz * f[1]
                moment[a][1] += raz * f[0] - rax * f[2]
                moment[a][2] += rax * f[1] - ray * f[0]
                moment[b][0] -= rby * f[2] - rbz * f[1]
                moment[b][1] -= rbz * f[0] - rbx * f[2]
                moment[b][2] -= rbx * f[1] - rby * f[0]
                p_contact[a] += f[0] * vax + f[1] * vay + f[2] * vaz
                p_contact[b] -= f[0] * vbx + f[1] * vby + f[2] * vbz
                if self.contact.normal_damping <= 0.0:
                    stored = 0.4 * self.contact.normal_stiffness * depth ** 2.5
                    elastic[a] += 0.5 * stored
                    elastic[b] += 0.5 * stored
                if depth > self.max_penetration:
                    self.max_penetration = depth


    # -- equations of motion ----------------------------------------------
    def _accelerations(self, st, forces):
        pos, vel, quat, omega = st
        force, moment = forces[0], forces[1]
        rot = _rot_np(quat)
        acc = force / self._mass_eff[:, None]
        torque_body = np.einsum("nij,ni->nj", rot, moment)
        gyro = _cross_np(omega, np.einsum("nij,nj->ni", self._inertia_eff, omega))
        alpha = np.einsum("nij,nj->ni", self._inv_inertia_eff, torque_body - gyro)
        return acc, alpha

    def _derivative(self, st):
        forces = self._forces(st)
        acc, alpha = self._accelerations(st, forces)
        pos, vel, quat, omega = st
        zero = np.zeros((quat.shape[0], 1))
        dquat = 0.5 * _quat_mul_np(quat, np.concatenate([zero, omega], axis=1))
        return dquat, acc, alpha, forces

    @staticmethod
    def _stage(st, acc, alpha, h: float):
        pos, vel, quat, omega = st
        return (pos + h * vel,
                vel + h * acc,
                _quat_integrate_np(quat, omega, h),
                omega + h * alpha)

    def _accumulate(self, f0, f1, dt: float) -> None:
        """Trapezoidal integration of the ledger power channels."""
        self.dissipated += 0.5 * float(f0[2].sum() + f1[2].sum()) * dt
        self.input_work += 0.5 * float(f0[3].sum() + f1[3].sum()) * dt
        self.contact_work += 0.5 * float(f0[4].sum() + f1[4].sum()) * dt

    # -- integrators -------------------------------------------------------
    def _step_verlet(self, dt: float) -> None:
        """Velocity Verlet (kick-drift-kick) with the exponential attitude map."""
        st = self._np_state()
        pos, vel, quat, omega = st
        f0 = self._forces(st)
        acc0, alpha0 = self._accelerations(st, f0)
        v_half = vel + 0.5 * dt * acc0
        w_half = omega + 0.5 * dt * alpha0
        pos_new = pos + dt * v_half
        quat_new = _quat_integrate_np(quat, w_half, dt)
        mid = (pos_new, v_half, quat_new, w_half)
        f1 = self._forces(mid)
        acc1, alpha1 = self._accelerations(mid, f1)
        self._set_np_state((pos_new, v_half + 0.5 * dt * acc1, quat_new,
                            w_half + 0.5 * dt * alpha1))
        self._accumulate(f0, f1, dt)
        self.time += dt
        self.steps += 1

    def _step_rk4(self, dt: float) -> None:
        """Lie-group RK4: classical stages with exact exponential attitude maps."""
        st = self._np_state()
        d1 = self._derivative(st)
        s2 = self._stage(st, d1[1], d1[2], 0.5 * dt)
        d2 = self._derivative(s2)
        s3 = self._stage(st, d2[1], d2[2], 0.5 * dt)
        d3 = self._derivative(s3)
        s4 = self._stage(st, d3[1], d3[2], dt)
        d4 = self._derivative(s4)
        pos, vel, quat, omega = st
        pos_new = pos + dt / 6.0 * (vel + 2.0 * s2[1] + 2.0 * s3[1] + s4[1])
        vel_new = vel + dt / 6.0 * (d1[1] + 2.0 * d2[1] + 2.0 * d3[1] + d4[1])
        w_new = omega + dt / 6.0 * (d1[2] + 2.0 * d2[2] + 2.0 * d3[2] + d4[2])
        quat_new = _quat_norm_np(quat + dt / 6.0 * (d1[0] + 2.0 * d2[0]
                                                    + 2.0 * d3[0] + d4[0]))
        self._set_np_state((pos_new, vel_new, quat_new, w_new))
        f1 = self._forces((pos_new, vel_new, quat_new, w_new))
        self._accumulate(d1[3], f1, dt)
        self.time += dt
        self.steps += 1

    def step(self, dt: float) -> None:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        if self.integrator == "verlet":
            self._step_verlet(dt)
        else:
            self._step_rk4(dt)

    def _adaptive_ratio(self, h: float, tol: float) -> float:
        """Step-doubling error estimate of the RK4 step, normalised by ``tol``."""
        st = self._np_state()
        counters = (self.state, self.dissipated, self.input_work, self.contact_work,
                    self.time, self.steps, self.max_penetration)
        self._step_rk4(h)
        full = self._np_state()
        self._set_np_state(st)
        self._step_rk4(0.5 * h)
        self._step_rk4(0.5 * h)
        half = self._np_state()
        (self.state, self.dissipated, self.input_work, self.contact_work,
         self.time, self.steps, self.max_penetration) = counters
        v_scale = max(float(np.abs(full[1]).max()), 1e-3)
        w_scale = max(float(np.abs(full[3]).max()), 1e-3)
        err = max(float(np.abs(half[1] - full[1]).max()) / v_scale,
                  float(np.abs(half[3] - full[3]).max()) / w_scale)
        return err / max(tol, 1e-300)

    def run(self, t_end: float, dt: float, adaptive: bool = False,
            tol: float = 1e-6, dt_bounds: Tuple[float, float] = (1e-6, 5e-2),
            callback=None) -> Dict[str, float]:
        """March to ``t_end``; ``adaptive`` switches on step-doubling control."""
        t_end = float(t_end)
        if not math.isfinite(t_end) or t_end < 0.0:
            raise ValueError("t_end must be finite and non-negative")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and strictly positive")
        lo, hi = float(dt_bounds[0]), float(dt_bounds[1])
        dt_now = min(max(dt, lo), hi)
        n_steps = 0
        dt_sum = 0.0
        dt_last = dt_now
        while self.time < t_end - 1e-15:
            h = min(dt_now, t_end - self.time)
            if adaptive:
                ratio = self._adaptive_ratio(h, tol)
                if ratio > 1.0 and h > lo * 1.0000001:
                    dt_now = max(lo, 0.5 * h)
                    continue
                self._step_rk4(h)
                factor = 5.0 if ratio <= 0.0 else min(5.0, max(0.2, 0.9 * ratio ** -0.2))
                dt_now = min(hi, max(lo, h * factor))
            else:
                self.step(h)
            n_steps += 1
            dt_sum += h
            dt_last = h
            if callback is not None:
                callback(self)
            if n_steps > 2_000_000:
                raise RuntimeError("run() exceeded the step budget")
        return {"steps": float(n_steps), "time": float(self.time),
                "dt_mean": dt_sum / max(n_steps, 1), "dt_last": float(dt_last)}

    # -- diagnostics -------------------------------------------------------
    def _buoyancy_potential(self) -> float:
        if not self.env.enable_buoyancy:
            return 0.0
        pos, vel, quat, omega = self._np_state()
        pos_l = pos.tolist()
        water_level = float(self.env.water_level)
        total = 0.0
        for b, q in enumerate(quat.tolist()):
            nodes = _nodes_py(self._vertices_py[b], _rot_py(*q), pos_l[b])
            _, _, zeta = _wet_py(nodes, self._faces_py[b], water_level)
            total += -self.env.water_density * self.env.gravity * zeta
        return total

    def energy(self) -> Dict[str, float]:
        pos, vel, quat, omega = self._np_state()
        ke_t = 0.5 * float((self._mass_eff[:, None] * vel * vel).sum())
        ke_r = 0.5 * float((omega * np.einsum("nij,nj->ni",
                                              self._inertia_eff, omega)).sum())
        pe_g = (float((self._mass * self.env.gravity * pos[:, 2]).sum())
                if self.env.enable_gravity else 0.0)
        pe_b = self._buoyancy_potential()
        return {"kinetic_translation": ke_t,
                "kinetic_rotation": ke_r,
                "potential_gravity": pe_g,
                "potential_buoyancy": pe_b,
                "total": ke_t + ke_r + pe_g + pe_b}

    def contact_elastic_energy(self) -> float:
        """Elastic energy currently stored in the Hertz springs (undamped ties)."""
        if self.contact.normal_damping > 0.0:
            return 0.0
        st = self._np_state()
        forces = self._forces(st)
        return float(forces[6].sum())

    def energy_ledger(self) -> Dict[str, float]:
        """Kinetic + potential energy audit against the accumulated work terms.

        ``residual = (E - E0) - (input - dissipated + contact_work)``.  The
        contact term is the elastic energy temporarily stored in the Hertz
        springs (zero once all contacts separate); the input term is the work of
        moving boundaries and currents.
        """
        now = self.energy()
        delta = now["total"] - self.initial_energy
        stored = self.contact_elastic_energy()
        budget = self.input_work - self.dissipated + self.contact_work
        residual = delta - budget
        residual_state = delta + stored - (self.input_work - self.dissipated)
        scale = max(abs(self.initial_energy), abs(now["total"]), 1e-30)
        return {**now,
                "contact_elastic": float(stored),
                "residual_state_relative": float(abs(residual_state) / scale),
                "initial": float(self.initial_energy),
                "delta": float(delta),
                "input_work": float(self.input_work),
                "dissipated": float(self.dissipated),
                "contact_work": float(self.contact_work),
                "residual": float(residual),
                "residual_relative": float(abs(residual) / scale),
                "quaternion_norm_error": self.quat_norm_error()}

    def residual_loads(self) -> Dict[str, object]:
        """Net force / moment per floe, normalised by weight and weight*length."""
        st = self._np_state()
        forces = self._forces(st)
        weight = np.maximum(self._mass * self.env.gravity, 1e-30)
        f_norm = np.sqrt((forces[0] ** 2).sum(axis=1)) / weight
        m_norm = np.sqrt((forces[1] ** 2).sum(axis=1)) / (weight * self._length_scale)
        return {"force_ratio": [float(x) for x in f_norm],
                "moment_ratio": [float(x) for x in m_norm],
                "force_ratio_max": float(f_norm.max()),
                "moment_ratio_max": float(m_norm.max()),
                "velocity_max": float(np.sqrt((st[1] ** 2).sum(axis=1)).max()),
                "omega_max": float(np.sqrt((st[3] ** 2).sum(axis=1)).max())}

    def momentum(self):
        return linear_momentum(self), angular_momentum(self)

    def quat_norm_error(self) -> float:
        return float(quat_norm_error(self.state.quat))

    def nodes_world(self, body: int) -> torch.Tensor:
        """Current world-space contact nodes of ``body``."""
        pos, vel, quat, omega = self._np_state()
        nodes = _nodes_py(self._vertices_py[body], _rot_py(*quat[body].tolist()),
                          pos[body].tolist())
        return torch.as_tensor(nodes, dtype=_DTYPE)


def linear_momentum(world: FloeWorld) -> torch.Tensor:
    """Total linear momentum ``sum m_i v_i`` (world)."""
    pos, vel, quat, omega = world._np_state()
    return torch.as_tensor((world._mass[:, None] * vel).sum(axis=0), dtype=_DTYPE)


def angular_momentum(world: FloeWorld, origin=(0.0, 0.0, 0.0)) -> torch.Tensor:
    """Total angular momentum about ``origin`` (spin + orbital, world axes)."""
    o = np.asarray(list(origin), dtype=np.float64)
    pos, vel, quat, omega = world._np_state()
    rot = _rot_np(quat)
    total = np.zeros(3)
    for b in range(len(world.geometries)):
        r = rot[b]
        inertia_world = r @ world._inertia_eff[b] @ r.T
        total += inertia_world @ (r @ omega[b])
        total += np.cross(pos[b] - o, world._mass[b] * vel[b])
    return torch.as_tensor(total, dtype=_DTYPE)
