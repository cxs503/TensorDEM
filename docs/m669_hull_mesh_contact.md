# 3D Hull Surface Contact (STL/OBJ Triangle Mesh) — m669

## Summary

This module implements penalty contact detection between DEM spheres and 3D triangle-mesh hulls (STL/OBJ format), replacing the previous 2D polyline-only limitation. The implementation uses:

- **Ericson's closest-point-on-triangle algorithm** (Real-Time Collision Detection, §5.1.5) for exact narrow-phase detection
- **Uniform spatial hashing** for O(N) broad-phase culling
- **Linear penalty spring + viscous damping** consistent with the existing 2D model

## Acceptance Criteria

| Criterion | Required | Achieved | Status |
|-----------|----------|----------|--------|
| `sphere_contact_force_error` | < 0.1 | 0.0 (machine precision) | ✅ PASS |
| `contact_detection_complexity` | ≤ O(N) | O(N) confirmed by scaling test | ✅ PASS |
| `pytest_test_hull_mesh_contact` | PASS | 17/17 tests pass | ✅ PASS |

## Files

- `src/tensordem/hull_mesh_contact.py` — Core module
- `tests/test_hull_mesh_contact.py` — 17 pytest tests
- `artifacts/fig1_force_error.png` — Force vs analytical
- `artifacts/fig2_complexity.png` — Scaling benchmark
- `artifacts/fig3_3d_contact.png` — 3D visualization

## Usage

```python
from tensordem.hull_mesh_contact import mesh_contact_forces, load_stl, load_obj

# Load a hull mesh
vertices, faces = load_stl("hull.stl")  # or load_obj("hull.obj")

# Compute contact forces
forces, reaction = mesh_contact_forces(
    sphere_positions,    # (N, 3)
    sphere_velocities,   # (N, 3)
    vertices,            # (V, 3)
    faces,               # (F, 3)
    hull_velocity,       # (3,)
    particle_radius=0.05,
    stiffness=10000.0,
    damping=10.0,
)
```

## Algorithm Details

### Broad Phase (Spatial Hash)
- Uniform grid with cell width = 4 × max(sphere_radius)
- Each triangle inserted into cells overlapping its AABB
- Query returns candidate triangles per sphere in O(1) amortised
- Total build + query: O(N_tri + N_spheres)

### Narrow Phase (Closest Point on Triangle)
- Vectorised Ericson algorithm for batch (sphere, triangle) pairs
- Handles all 7 Voronoi regions: 3 vertices, 3 edges, 1 interior
- Returns exact closest point and distance

### Force Model
- Linear penalty: F = k · δ (δ = overlap)
- Viscous damping: F_d = -c · v_n (v_n = normal relative velocity)
- Force clamped to repulsive only (no attraction on separation)
- Action-reaction guaranteed by construction

## Limitations

1. **CPU-only**: The per-sphere loop is sequential; GPU acceleration would require restructuring the narrow phase
2. **No friction**: Current implementation only computes normal forces; tangential friction is not yet implemented
3. **Single-contact per sphere**: Each sphere interacts with only its closest triangle; multi-contact scenarios (e.g., sphere in a corner) may under-report forces
4. **No mesh deformation**: The hull is treated as a rigid body; coupled FEM-DEM would require additional infrastructure

## References

- Ericson, C. (2005). *Real-Time Collision Detection*. Morgan Kaufmann. §5.1.5
- Teschner, M. et al. (2003). "Optimized Spatial Hashing for Collision Detection of Deformable Objects". VMV 2003.
