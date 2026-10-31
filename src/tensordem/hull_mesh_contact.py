"""Penalty contact between DEM spheres and a 3-D triangle-mesh hull (STL/OBJ).

This module extends the existing 2-D polyline hull contact to full 3-D
triangle-mesh surfaces, enabling realistic ship-hull / ice contact in the
TensorDEM ice-breaking simulation framework.

Algorithm
---------
For each sphere the closest point on every triangle is computed via the
signed-Voronoi-region method (Real-Time Collision Detection, Ericson 2005,
§5.1.5).  A uniform spatial hash (cell width = max sphere diameter) limits
the per-sphere triangle candidates so that total cost is O(N_spheres +
N_triangles) amortised.  Contact force uses a linear penalty spring with
optional viscous damping, consistent with the 2-D polyline model.

Evidence level: REAL – the closest-point and force formulas are standard
computational-contact mechanics; accuracy is verified against the analytical
sphere-on-plane Hertz/penalty solution in the test suite.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Mesh I/O
# ---------------------------------------------------------------------------

def load_stl(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read an ASCII or binary STL file.

    Returns
    -------
    vertices : (3*F, 3) float64 tensor – three vertices per face (duplicated).
    faces    : (F, 3) int64 tensor – indices into *vertices* (0,1,2), (3,4,5)…
    """
    p = Path(path)
    raw = p.read_bytes()
    # Detect binary STL: starts with 80-byte header, then uint32 triangle count.
    if raw[:5].lower() == b"solid" and b"facet" in raw[:500]:
        return _parse_ascii_stl(raw.decode(errors="replace"))
    return _parse_binary_stl(raw)


def _parse_ascii_stl(text: str) -> tuple[torch.Tensor, torch.Tensor]:
    verts: list[list[float]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip().lower()
        if line.startswith("vertex"):
            parts = lines[i].strip().split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        i += 1
    if len(verts) % 3 != 0:
        raise ValueError(f"STL vertex count {len(verts)} not divisible by 3")
    n_faces = len(verts) // 3
    vertices = torch.tensor(verts, dtype=torch.float64).reshape(-1, 3)
    faces = torch.arange(n_faces * 3, dtype=torch.int64).reshape(-1, 3)
    return vertices, faces


def _parse_binary_stl(raw: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    n_tri = struct.unpack_from("<I", raw, 80)[0]
    expected = 84 + n_tri * 50
    if len(raw) < expected:
        raise ValueError(f"Binary STL truncated: expected {expected} bytes, got {len(raw)}")
    verts = torch.empty((n_tri * 3, 3), dtype=torch.float64)
    offset = 84
    for f in range(n_tri):
        # skip normal (12 bytes), read 3 vertices (36 bytes), skip attr (2 bytes)
        offset += 12
        for v in range(3):
            x, y, z = struct.unpack_from("<fff", raw, offset)
            verts[f * 3 + v] = torch.tensor([x, y, z], dtype=torch.float64)
            offset += 12
        offset += 2
    faces = torch.arange(n_tri * 3, dtype=torch.int64).reshape(-1, 3)
    return verts, faces


def load_obj(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Read a Wavefront OBJ file (vertices + triangular faces).

    Returns
    -------
    vertices : (V, 3) float64 tensor
    faces    : (F, 3) int64 tensor – 0-based indices
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            parts = line.split()[1:]
            # OBJ faces can be v, v/vt, v/vt/vn, v//vn
            idx = [int(p.split("/")[0]) - 1 for p in parts]
            if len(idx) == 3:
                faces.append(idx)
            elif len(idx) > 3:
                # fan triangulation
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    if not verts or not faces:
        raise ValueError("OBJ file has no vertices or faces")
    return (torch.tensor(verts, dtype=torch.float64),
            torch.tensor(faces, dtype=torch.int64))


# ---------------------------------------------------------------------------
# Closest point on triangle (Ericson 2005, §5.1.5)
# ---------------------------------------------------------------------------

def _closest_point_triangle_batch(
    P: torch.Tensor,   # (N, 3) query points
    A: torch.Tensor,   # (N, 3) triangle vertex 0
    B: torch.Tensor,   # (N, 3) triangle vertex 1
    C: torch.Tensor,   # (N, 3) triangle vertex 2
) -> torch.Tensor:
    """Vectorised closest point on triangle for each (point, triangle) pair.

    All inputs have shape (N, 3).  Returns (N, 3) closest points.
    """
    eps = torch.finfo(P.dtype).eps
    AB = B - A
    AC = C - A
    AP = P - A

    d1 = (AB * AP).sum(dim=-1)
    d2 = (AC * AP).sum(dim=-1)

    # Vertex region A
    cond_a = (d1 <= 0) & (d2 <= 0)

    # Vertex region B
    BP = P - B
    d3 = (AB * BP).sum(dim=-1)
    d4 = (AC * BP).sum(dim=-1)
    cond_b = (d3 >= 0) & (d4 <= d3)

    # Edge region AB
    vc = d1 * d4 - d3 * d2
    cond_ab = (d1 >= 0) & (d3 <= 0) & (vc <= 0)
    v_ab = d1.clamp_min(eps) / (d1 - d3).clamp_min(eps)
    result_ab = A + v_ab.unsqueeze(-1) * AB

    # Vertex region C
    CP = P - C
    d5 = (AB * CP).sum(dim=-1)
    d6 = (AC * CP).sum(dim=-1)
    cond_c = (d6 >= 0) & (d5 <= d6)

    # Edge region AC
    vb = d5 * d2 - d1 * d6
    cond_ac = (d2 >= 0) & (d6 <= 0) & (vb <= 0)
    w_ac = d2.clamp_min(eps) / (d2 - d6).clamp_min(eps)
    result_ac = A + w_ac.unsqueeze(-1) * AC

    # Edge region BC
    va = d3 * d6 - d5 * d4
    cond_bc = ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & (va <= 0)
    w_bc = ((d4 - d3).clamp_min(eps)) / ((d4 - d3) + (d5 - d6)).clamp_min(eps)
    result_bc = B + w_bc.unsqueeze(-1) * (C - B)

    # Interior region
    denom = (va + vb + vc).clamp_min(eps)
    v_int = vb / denom
    w_int = vc / denom
    result_int = A + v_int.unsqueeze(-1) * AB + w_int.unsqueeze(-1) * AC

    # Combine regions using a priority scheme (first match wins)
    result = result_int.clone()
    result = torch.where(cond_bc.unsqueeze(-1), result_bc, result)
    result = torch.where(cond_ac.unsqueeze(-1), result_ac, result)
    result = torch.where(cond_c.unsqueeze(-1), C, result)
    result = torch.where(cond_ab.unsqueeze(-1), result_ab, result)
    result = torch.where(cond_b.unsqueeze(-1), B, result)
    result = torch.where(cond_a.unsqueeze(-1), A, result)

    return result


# ---------------------------------------------------------------------------
# Spatial hash for O(N) broad-phase
# ---------------------------------------------------------------------------

class _SpatialHash:
    """Uniform-grid spatial hash for triangle AABBs.

    Each triangle is inserted into every cell its AABB overlaps.  Query
    returns candidate triangle indices for a sphere AABB.  Build + query
    are O(N_tri + N_spheres) amortised when cell width ≈ sphere diameter.
    """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor, cell_width: float):
        self.cell_width = cell_width
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        tri_min, _ = torch.stack([v0, v1, v2], dim=0).min(dim=0)
        tri_max, _ = torch.stack([v0, v1, v2], dim=0).max(dim=0)
        self.tri_min = tri_min  # (F, 3)
        self.tri_max = tri_max  # (F, 3)
        inv_w = 1.0 / cell_width
        cell_min = torch.floor(tri_min * inv_w).to(torch.int64)
        cell_max = torch.floor(tri_max * inv_w).to(torch.int64)
        self.cells: dict[tuple[int, int, int], list[int]] = {}
        for fi in range(len(faces)):
            cmin = cell_min[fi].tolist()
            cmax = cell_max[fi].tolist()
            for cx in range(cmin[0], cmax[0] + 1):
                for cy in range(cmin[1], cmax[1] + 1):
                    for cz in range(cmin[2], cmax[2] + 1):
                        self.cells.setdefault((cx, cy, cz), []).append(fi)

    def query(self, centers: torch.Tensor, radius: float) -> list[list[int]]:
        """Return candidate triangle indices for each sphere center."""
        inv_w = 1.0 / self.cell_width
        cmin_all = torch.floor((centers - radius) * inv_w).to(torch.int64)
        cmax_all = torch.floor((centers + radius) * inv_w).to(torch.int64)
        results: list[list[int]] = []
        for i in range(len(centers)):
            cands: set[int] = set()
            cmin = cmin_all[i].tolist()
            cmax = cmax_all[i].tolist()
            for cx in range(cmin[0], cmax[0] + 1):
                for cy in range(cmin[1], cmax[1] + 1):
                    for cz in range(cmin[2], cmax[2] + 1):
                        for fi in self.cells.get((cx, cy, cz), []):
                            cands.add(fi)
            results.append(sorted(cands))
        return results


# ---------------------------------------------------------------------------
# Main contact force computation
# ---------------------------------------------------------------------------

def mesh_contact_forces(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    hull_velocity: torch.Tensor,
    *,
    particle_radius: float | torch.Tensor,
    hull_radius: float = 0.0,
    stiffness: float = 1e4,
    damping: float = 0.0,
    cell_width: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute penalty contact forces between spheres and a triangle mesh.

    Parameters
    ----------
    positions   : (N, 3) sphere centres
    velocities  : (N, 3) sphere velocities
    vertices    : (V, 3) mesh vertices
    faces       : (F, 3) triangle face indices (0-based)
    hull_velocity : (3,) prescribed rigid-body velocity of the mesh
    particle_radius : scalar or (N,) sphere radii
    hull_radius : additional radius offset (default 0)
    stiffness   : penalty spring constant (N/m)
    damping     : viscous damping coefficient (N·s/m)
    cell_width  : spatial hash cell width; defaults to 4 × max(particle_radius)

    Returns
    -------
    forces   : (N, 3) contact forces on each sphere
    reaction : (3,) equal-and-opposite total force on the hull
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if velocities.shape != positions.shape:
        raise ValueError("velocities must match positions shape")
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if hull_velocity.numel() != 3:
        raise ValueError("hull_velocity must have 3 components")
    if stiffness <= 0 or not math.isfinite(stiffness):
        raise ValueError("stiffness must be finite and positive")
    if damping < 0 or not math.isfinite(damping):
        raise ValueError("damping must be finite and non-negative")

    device, dtype = positions.device, positions.dtype
    N = positions.shape[0]
    F = faces.shape[0]

    if N == 0 or F == 0:
        return (torch.zeros_like(positions),
                torch.zeros(3, dtype=dtype, device=device))

    p = positions.to(device=device, dtype=dtype)
    v = velocities.to(device=device, dtype=dtype)
    verts = vertices.to(device=device, dtype=dtype)
    hv = hull_velocity.to(device=device, dtype=dtype).reshape(3)

    if isinstance(particle_radius, torch.Tensor):
        radii = particle_radius.to(device=device, dtype=dtype).reshape(N)
    else:
        radii = torch.full((N,), float(particle_radius), dtype=dtype, device=device)

    max_r = float(radii.max().item()) + float(hull_radius)

    # Broad phase: spatial hash
    if cell_width is None:
        cell_width = max(4.0 * max_r, 1e-6)
    sh = _SpatialHash(verts, faces, cell_width)
    candidates = sh.query(p, max_r + max_r)  # generous envelope

    # Narrow phase: closest point on triangle
    forces = torch.zeros_like(p)
    for i in range(N):
        r_i = float(radii[i].item())
        contact_r = r_i + float(hull_radius)
        cands = candidates[i]
        if not cands:
            continue
        cand_faces = faces[cands]  # (C, 3)
        A = verts[cand_faces[:, 0]]
        B = verts[cand_faces[:, 1]]
        C = verts[cand_faces[:, 2]]
        pi = p[i].unsqueeze(0).expand_as(A)
        closest = _closest_point_triangle_batch(pi, A, B, C)
        delta = p[i] - closest  # (C, 3)
        dist_sq = (delta * delta).sum(dim=-1)
        dist = torch.sqrt(dist_sq)
        # Find closest triangle
        min_idx = int(dist.argmin().item())
        d_min = float(dist[min_idx].item())
        overlap = contact_r - d_min
        if overlap <= 0:
            continue
        # Contact normal (from closest point to sphere centre)
        delta_i = delta[min_idx]
        if d_min > 1e-15:
            normal = delta_i / d_min
        else:
            # Degenerate: sphere centre on triangle – use triangle normal
            e1 = B[min_idx] - A[min_idx]
            e2 = C[min_idx] - A[min_idx]
            n = torch.linalg.cross(e1, e2)
            n_len = torch.linalg.vector_norm(n)
            if n_len > 1e-15:
                normal = n / n_len
            else:
                normal = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
        # Relative velocity
        rel_v = v[i] - hv
        vn = (rel_v * normal).sum()
        # Penalty force
        fn = stiffness * overlap - damping * float(vn.item())
        if fn < 0:
            fn = 0.0
        forces[i] = fn * normal

    reaction = -forces.sum(dim=0)
    return forces, reaction


# ---------------------------------------------------------------------------
# Convenience: build a simple hull mesh for testing
# ---------------------------------------------------------------------------

def make_flat_mesh(
    size: float = 2.0,
    normal: str = "z",
    offset: float = 0.0,
    nx: int = 1,
    ny: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a flat triangulated plane for testing.

    Returns (vertices, faces) suitable for mesh_contact_forces.
    """
    xs = torch.linspace(-size / 2, size / 2, nx + 1, dtype=torch.float64)
    ys = torch.linspace(-size / 2, size / 2, ny + 1, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    grid_z = torch.full_like(grid_x, offset)
    if normal == "z":
        verts = torch.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], dim=-1)
    elif normal == "y":
        verts = torch.stack([grid_x.ravel(), grid_z.ravel(), grid_y.ravel()], dim=-1)
    elif normal == "x":
        verts = torch.stack([grid_z.ravel(), grid_x.ravel(), grid_y.ravel()], dim=-1)
    else:
        raise ValueError(f"Unknown normal direction: {normal}")

    faces_list = []
    for i in range(nx):
        for j in range(ny):
            v00 = i * (ny + 1) + j
            v10 = (i + 1) * (ny + 1) + j
            v01 = i * (ny + 1) + (j + 1)
            v11 = (i + 1) * (ny + 1) + (j + 1)
            faces_list.append([v00, v10, v11])
            faces_list.append([v00, v11, v01])
    faces = torch.tensor(faces_list, dtype=torch.int64)
    return verts, faces
