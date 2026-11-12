"""One-off patch: extract the mesh narrow phase into ``narrow_phase_contacts``.

Kept in the tree only as a record of how the refactor was applied; not imported
by the package.  Run once from the repo root.
"""
from pathlib import Path

p = Path("src/tensordem/hull_mesh_contact.py")
lines = p.read_text().splitlines(keepends=True)

start = next(i for i, l in enumerate(lines) if l.startswith("    max_r = float(radii.max()"))
end = next(i for i, l in enumerate(lines) if l.startswith("    reaction = -forces.sum(dim=0)"))
body = "".join(lines[start:end])

new_call = '''    max_r = float(radii.max().item()) + float(hull_radius)

    contacts = narrow_phase_contacts(
        p, verts, faces,
        radii=radii,
        hull_radius=float(hull_radius),
        cell_width=cell_width,
        closed_surface=closed_surface,
    )

    forces = torch.zeros_like(p)
    if contacts["particle"].numel():
        rel_v = v[contacts["particle"]] - hv
        vn = (rel_v * contacts["normal"]).sum(dim=-1)
        fn = stiffness * contacts["overlap"] - damping * vn
        fn = fn.clamp_min(0.0)
        forces.index_add_(0, contacts["particle"], fn.unsqueeze(-1) * contacts["normal"])

'''

# ---------------------------------------------------------------- narrow phase
narrow = '''# ---------------------------------------------------------------------------
# Narrow phase: the geometry contract shared by every contact law on the mesh
# ---------------------------------------------------------------------------

def narrow_phase_contacts(
    positions: torch.Tensor,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    *,
    radii: torch.Tensor,
    hull_radius: float = 0.0,
    cell_width: Optional[float] = None,
    closed_surface: bool = False,
) -> dict[str, torch.Tensor]:
    """All sphere-triangle contacts, as flat arrays (one row per contact).

    The normal-force law (:func:`mesh_contact_forces`) and the friction layer
    (:mod:`tensordem.hull_mesh_friction`) consume the *same* rows, so a
    tangential law can never disagree with the normal law about which triangle
    a sphere is touching, where the contact point is, or how deep it is.

    Parameters
    ----------
    positions : (N, 3) sphere centres
    vertices  : (V, 3) mesh vertices
    faces     : (F, 3) triangle indices
    radii     : (N,) sphere radii (already validated by the caller)
    hull_radius : additional contact radius offset (a moulded hull skin, m)
    cell_width  : spatial-hash cell width; defaults to 4 x max contact radius
    closed_surface : signed-distance mode for a watertight hull (a centre that
        has crossed the surface keeps a restoring force at any depth)

    Returns
    -------
    dict with, for K active contacts (overlap > 0):

    ``particle`` (K,) int64  sphere index
    ``triangle`` (K,) int64  face index (the *nearest* triangle of that sphere)
    ``closest``  (K, 3)      contact point on the triangle
    ``normal``   (K, 3)      unit normal, pointing from the surface to the
                             sphere centre (out of the hull)
    ``overlap``  (K,)        penetration along the normal (m, > 0)
    ``inside``   (K,) bool   centre behind the surface (closed_surface only)

    Every sphere contributes at most one contact -- the nearest triangle --
    which is the historical kernel behaviour (``mesh_contact_forces``) and is
    what the verified hull audits score.
    """
    N = positions.shape[0]
    F = faces.shape[0]
    dtype = positions.dtype
    device = positions.device

    max_r = float(radii.max().item()) + float(hull_radius)
    if cell_width is None:
        cell_width = max(4.0 * max_r, 1e-6)
    sh = _SpatialHash(vertices, faces, cell_width)
    candidates = sh.query(positions, max_r + max_r)  # generous envelope

    all_faces = torch.arange(F, device=device)
    out_particle: list[int] = []
    out_triangle: list[int] = []
    out_closest: list[torch.Tensor] = []
    out_normal: list[torch.Tensor] = []
    out_overlap: list[float] = []
    out_inside: list[bool] = []

    def _nearest_on(face_idx: torch.Tensor, centre: torch.Tensor):
        """Closest point over ``face_idx``: (A, B, C, dist, closest_point)."""
        f = faces[face_idx]
        A_ = vertices[f[:, 0]]
        B_ = vertices[f[:, 1]]
        C_ = vertices[f[:, 2]]
        c_ = _closest_point_triangle_batch(centre.unsqueeze(0).expand_as(A_), A_, B_, C_)
        d_ = torch.linalg.vector_norm(centre.unsqueeze(0) - c_, dim=-1)
        return A_, B_, C_, d_, c_

    for i in range(N):
        r_i = float(radii[i].item())
        contact_r = r_i + float(hull_radius)
        cands = candidates[i]
        if not cands and not closed_surface:
            continue
        if cands:
            A, B, C, dist, closest = _nearest_on(
                torch.as_tensor(cands, dtype=torch.int64, device=device), positions[i])
            min_idx = int(dist.argmin().item())
            d_min = float(dist[min_idx].item())
            tri_i = int(cands[min_idx])
            delta_i = positions[i] - closest[min_idx]
        else:
            A = B = C = closest = None
            min_idx = 0
            tri_i = -1
            d_min = float("inf")
            delta_i = torch.zeros(3, dtype=dtype, device=device)
        overlap = contact_r - d_min
        inside = False
        if closed_surface:
            inside = bool(ray_parity_inside(positions[i:i + 1], vertices, faces)[0].item())
            if inside and (not math.isfinite(d_min) or overlap <= 0.0):
                # A closed hull puts no bound on the proximity of a centre that
                # has crossed the surface: the broad phase only reaches
                # ``2 * (radius + hull_radius)``, so a deeper (or fully escaped)
                # centre has to fall back to a nearest-triangle search over the
                # whole mesh.  Without this the "restoring force at any
                # penetration depth" contract is silently violated (m16872).
                A, B, C, dist, closest = _nearest_on(all_faces, positions[i])
                min_idx = int(dist.argmin().item())
                d_min = float(dist[min_idx].item())
                tri_i = int(min_idx)
                delta_i = positions[i] - closest[min_idx]
                overlap = contact_r - d_min
            if inside:
                # Signed distance: the centre is behind the surface, so the
                # outward penetration is contact_r + d_min and the outward
                # normal is (closest - centre) / d_min.
                overlap = contact_r + d_min
        if overlap <= 0.0:
            continue
        if inside and d_min > 1e-15:
            normal = -delta_i / d_min
        elif d_min > 1e-15:
            normal = delta_i / d_min
        else:
            # Degenerate: sphere centre on triangle - use triangle normal
            e1 = B[min_idx] - A[min_idx]
            e2 = C[min_idx] - A[min_idx]
            n = torch.linalg.cross(e1, e2)
            n_len = torch.linalg.vector_norm(n)
            if n_len > 1e-15:
                normal = n / n_len
            else:
                normal = torch.tensor([0., 0., 1.], dtype=dtype, device=device)
        out_particle.append(i)
        out_triangle.append(tri_i)
        out_closest.append(closest[min_idx].clone())
        out_normal.append(normal.clone())
        out_overlap.append(float(overlap))
        out_inside.append(bool(inside))

    return {
        "particle": torch.as_tensor(out_particle, dtype=torch.int64, device=device),
        "triangle": torch.as_tensor(out_triangle, dtype=torch.int64, device=device),
        "closest": (torch.stack(out_closest) if out_closest
                    else torch.zeros((0, 3), dtype=dtype, device=device)),
        "normal": (torch.stack(out_normal) if out_normal
                   else torch.zeros((0, 3), dtype=dtype, device=device)),
        "overlap": torch.as_tensor(out_overlap, dtype=dtype, device=device),
        "inside": torch.as_tensor(out_inside, dtype=torch.bool, device=device),
    }


'''
# ------------------------------------------------------- public API aliasing
lines[start:end] = [new_call]
text = "".join(lines)
marker = "# ---------------------------------------------------------------------------\n# Main contact force computation\n"
idx = text.index(marker)
text = text[:idx] + narrow + text[idx:]
p.write_text(text)
print("patched", len(body.splitlines()), "lines replaced")