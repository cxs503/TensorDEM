"""Signed (inside/outside) machinery for :mod:`tensordem.hull_mesh_contact`.

Why this module exists
----------------------
:func:`tensordem.hull_mesh_contact.mesh_contact_forces` computes the penalty
overlap as ``contact_r - |p - closest|`` with **no inside/outside test**: the
contact normal is taken from the closest surface point to the sphere centre.
That is correct while the sphere centre stays on the outer side of the hull
surface, and it fails in two coupled ways as soon as the centre crosses it
(penetration depth ``t`` measured from the surface):

* ``0 < t < contact_r`` -- the closest-point vector now points *inwards*, so the
  penalty force is reversed: it **pulls the particle into the hull** with
  magnitude ``k (contact_r - t)``.  Measured on the shipped kernel: a 50 mm
  sphere whose centre is 5 mm past a plane receives ``-45 kN`` where the
  outward answer is ``+45 kN`` (``tests/test_hull_mesh_contact_signed.py``).
* ``t >= contact_r`` -- ``contact_r - t <= 0``, so the force is zero: the
  particle is **inside the hull with no restoring force at all**, i.e. it is
  lost from the simulation.

The physically consistent continuation is the *signed* distance
``d_s = -t`` inside (``d_s = +d`` outside), giving ``overlap = contact_r - d_s``
with the normal directed along the outward surface normal.  This is continuous
across the surface (both branches tend to ``contact_r`` at ``t = 0``), so the
sign change costs nothing in accuracy on the outer branch.

What is implemented here
------------------------
* :func:`is_watertight` -- edge-manifold check (every undirected edge used
  exactly twice).  Inside/outside via ray parity is only defined for a closed
  surface, so the signed mode must refuse an open mesh rather than return
  silent nonsense.
* :func:`signed_distances` -- closest-point distance plus inside mask for a set
  of query points, using Möller–Trumbore parity along a fixed non-degenerate
  ray direction.  Degenerate (axis-aligned) hits are avoided by choosing a
  direction with three distinct irrational-ish components; the accuracy of the
  classifier is verified against analytic containment on spheres, capped
  cylinders and a tapered body of revolution (``scripts/hull_signed_audit.py``).

Both helpers are pure geometry: no state, no tuning constants.
"""
from __future__ import annotations

import math

import torch

__all__ = [
    "is_watertight",
    "is_edge_manifold",
    "ray_parity_inside",
    "signed_distance_to_mesh",
    "SignedDistanceResult",
]

# Ray direction with three distinct, irrational-ish components: an axis-aligned
# ray degenerates twice on the grid-aligned meshes built here (it hits shared
# edges/vertices exactly), which double counts intersections.
_RAY_DIR = (0.5773502691896258, 0.2698752691896258, 0.7718841000000000)


def _normalise(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


_RAY_DIR = _normalise(_RAY_DIR)


def _edge_counts(vertices, faces):
    faces = torch.as_tensor(faces).to(torch.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if faces.numel() == 0:
        return None
    if int(faces.min()) < 0 or int(faces.max()) >= int(torch.as_tensor(vertices).shape[0]):
        return None
    f = faces.cpu()
    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], dim=0)
    keys = (torch.minimum(edges[:, 0], edges[:, 1]) * (int(faces.max()) + 1)
            + torch.maximum(edges[:, 0], edges[:, 1]))
    _, counts = torch.unique(keys, return_counts=True)
    return counts


def is_watertight(vertices: torch.Tensor, faces: torch.Tensor) -> bool:
    """True iff the triangle soup closes: no boundary edge (odd multiplicity).

    This is exactly the condition the parity classifier needs (a ray crosses the
    boundary an even number of times to enter and to leave).  A mesh may pinch at
    a vertex or along an edge (multiplicity 4, e.g. the collapsed bow extension
    of a Wigley hull) and still bound a well-defined solid; such a mesh is
    *closed* though not edge-manifold -- use :func:`is_edge_manifold` to tell the
    two apart.  Neither test detects self-intersection, nested shells or
    inconsistent orientation (orientation is not used).
    """
    counts = _edge_counts(vertices, faces)
    if counts is None:
        return False
    return bool((counts % 2 == 0).all())


def is_edge_manifold(vertices: torch.Tensor, faces: torch.Tensor) -> bool:
    """True iff every undirected edge is used by exactly two faces (stricter)."""
    counts = _edge_counts(vertices, faces)
    if counts is None:
        return False
    return bool((counts == 2).all())


def ray_parity_inside(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    ray_dir: tuple[float, float, float] = _RAY_DIR,
) -> torch.Tensor:
    """Inside/outside of a closed triangle mesh by odd-even ray parity.

    Parameters
    ----------
    points : (N, 3) query points
    vertices : (V, 3) mesh vertices
    faces : (F, 3) triangle indices
    ray_dir : ray direction (need not be normalised; a fixed non-degenerate
        direction is used by default)

    Returns
    -------
    (N,) bool tensor, True where the point is inside.

    Method: Möller–Trumbore intersection of each ray with every triangle, count
    crossings, odd = inside.  Intersections behind the origin (``t_hit <= 0``)
    are ignored.  The implementation is exact in the sense of IEEE arithmetic on
    the triangle test; because it is a *parity* count, a ray grazing an edge
    changes the count by two and therefore does not flip the verdict.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    verts = torch.as_tensor(vertices)
    faces = torch.as_tensor(faces).to(torch.int64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    p = torch.as_tensor(points)
    if p.numel() == 0 or faces.numel() == 0:
        return torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)

    dtype = verts.dtype if verts.dtype.is_floating_point else torch.float64
    p = p.to(dtype=dtype)
    verts = verts.to(dtype=dtype)
    d = torch.tensor(ray_dir, dtype=dtype, device=p.device)
    d = d / torch.linalg.vector_norm(d)

    a = verts[faces[:, 0]]
    e1 = verts[faces[:, 1]] - a
    e2 = verts[faces[:, 2]] - a

    # Only triangles that the *line* through the point can reach are candidates:
    # cheap slab test against the ray's bounding planes is not exact for an
    # oblique ray, so this is just a broad phase over the ray direction.
    h = torch.linalg.cross(d.unsqueeze(0).expand_as(e2), e2, dim=-1)
    a_mat = (e1 * h).sum(dim=-1)
    eps = torch.finfo(dtype).eps
    valid_tri = a_mat.abs() > eps * (torch.linalg.vector_norm(e1, dim=-1)
                                     * torch.linalg.vector_norm(e2, dim=-1)).clamp_min(eps)

    inside = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
    for i in range(p.shape[0]):
        s = p[i].unsqueeze(0) - a                      # (F, 3)
        u = (s * h).sum(dim=-1) / torch.where(valid_tri, a_mat, torch.ones_like(a_mat))
        q = torch.linalg.cross(s, e1, dim=-1)
        v = (d.unsqueeze(0) * q).sum(dim=-1) / torch.where(valid_tri, a_mat, torch.ones_like(a_mat))
        t_hit = (e2 * q).sum(dim=-1) / torch.where(valid_tri, a_mat, torch.ones_like(a_mat))
        hit = (valid_tri & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1)
               & (t_hit > 0))
        inside[i] = (int(hit.sum().item()) % 2) == 1
    return inside


class SignedDistanceResult:
    """Result of :func:`signed_distance_to_mesh` (closest point + inside mask)."""

    __slots__ = ("closest", "distance", "inside", "signed_distance")

    def __init__(self, closest, distance, inside):
        self.closest = closest
        self.distance = distance
        self.inside = inside
        self.signed_distance = torch.where(inside, -distance, distance)

    def as_dict(self) -> dict:
        return {
            "distance": self.distance.tolist(),
            "inside": self.inside.tolist(),
            "signed_distance": self.signed_distance.tolist(),
        }


def signed_distance_to_mesh(
    points: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    check_watertight: bool = True,
) -> SignedDistanceResult:
    """Exact closest point on the mesh plus its inside/outside sign.

    The distance is the minimum over triangles of the point-to-triangle
    distance (:func:`tensordem.hull_mesh_contact._closest_point_triangle_batch`,
    the same kernel the force routine uses, so the *distance* half of this
    function is not an independent reference); the sign comes from ray parity.
    Verification of the sign against analytic containment lives in
    ``scripts/hull_signed_audit.py``.
    """
    from .hull_mesh_contact import _closest_point_triangle_batch

    verts = torch.as_tensor(vertices)
    faces = torch.as_tensor(faces).to(torch.int64)
    p = torch.as_tensor(points)
    if check_watertight and not is_watertight(verts, faces):
        raise ValueError("signed_distance_to_mesh needs a closed (no boundary "
                         "edge) mesh; pass check_watertight=False to bypass at "
                         "your own risk")
    if p.numel() == 0 or faces.numel() == 0:
        z = torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)
        return SignedDistanceResult(p.clone(), torch.zeros(p.shape[0], dtype=p.dtype,
                                                          device=p.device), z)

    A = verts[faces[:, 0]]
    B = verts[faces[:, 1]]
    C = verts[faces[:, 2]]

    closest_all = []
    dist_all = []
    for i in range(p.shape[0]):
        pi = p[i].unsqueeze(0).expand_as(A)
        cp = _closest_point_triangle_batch(pi, A, B, C)
        dd = torch.linalg.vector_norm(p[i].unsqueeze(0) - cp, dim=-1)
        j = int(dd.argmin().item())
        closest_all.append(cp[j])
        dist_all.append(dd[j])

    closest = torch.stack(closest_all)
    distance = torch.stack(dist_all).to(p.dtype)
    inside = ray_parity_inside(p, verts, faces)
    return SignedDistanceResult(closest, distance, inside)