"""Probe: does ``closed_surface=True`` restore contact for deep penetration?

Builds a watertight axis-aligned box hull (1 m cube, 12 triangles), pushes a
sphere centre to depth ``t`` along -z from the top face, and reports the penalty
force magnitude for both the historical one-sided path and the signed path.
"""
from __future__ import annotations

import json
import sys

import torch

from tensordem.hull_mesh_contact import mesh_contact_forces
from tensordem.hull_mesh_contact_signed import is_watertight, ray_parity_inside

F64 = torch.float64
HALF = 0.5


def box_mesh(half: float = HALF):
    c = [(-half, -half, -half), (half, -half, -half), (half, half, -half),
         (-half, half, -half), (-half, -half, half), (half, -half, half),
         (half, half, half), (-half, half, half)]
    v = torch.tensor(c, dtype=F64)
    f = torch.tensor([
        [0, 2, 1], [0, 3, 2],      # bottom (z=-h)
        [4, 5, 6], [4, 6, 7],      # top    (z=+h)
        [0, 1, 5], [0, 5, 4],      # y=-h
        [1, 2, 6], [1, 6, 5],      # x=+h
        [2, 3, 7], [2, 7, 6],      # y=+h
        [3, 0, 4], [3, 4, 7],      # x=-h
    ], dtype=torch.int64)
    return v, f


def main() -> int:
    verts, faces = box_mesh()
    print("watertight:", is_watertight(verts, faces))
    radius, stiffness = 0.05, 1.0e6
    rows = []
    # t > 0 means the centre is *inside* the hull, measured from the top face
    for t in (-0.04, -0.01, -1e-6, 1e-6, 0.01, 0.04, 0.049, 0.051, 0.1, 0.3):
        z = HALF - t
        centre = torch.tensor([[0.0, 0.0, z]], dtype=F64)
        inside = bool(ray_parity_inside(centre, verts, faces)[0].item())
        f_open, _ = mesh_contact_forces(
            centre, torch.zeros(1, 3, dtype=F64), verts, faces,
            torch.zeros(3, dtype=F64), particle_radius=radius,
            stiffness=stiffness, closed_surface=False)
        f_closed, _ = mesh_contact_forces(
            centre, torch.zeros(1, 3, dtype=F64), verts, faces,
            torch.zeros(3, dtype=F64), particle_radius=radius,
            stiffness=stiffness, closed_surface=True)
        rows.append({
            "t_m": t,
            "inside": inside,
            "discrete_fz_N": float(f_open[0, 2]),
            "signed_fz_N": float(f_closed[0, 2]),
        })
        print(f"t={t:+.4f} m inside={inside!s:5s} "
              f"one-sided Fz={float(f_open[0, 2]):+10.1f} N "
              f"signed Fz={float(f_closed[0, 2]):+10.1f} N")
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/probe.json", "w"),
              indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())