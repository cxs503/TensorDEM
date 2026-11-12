"""m16872 audit: does ``closed_surface=True`` hold to *any* penetration depth?

The shipped ``docs/hull_mesh_contact_signed_distance.md`` §1 promises the signed
continuation gives ``overlap = contact_r - d_s`` and therefore "keeps a restoring
force at any penetration depth", but its own §3 table reports 100 % force error
for the cylinder at depths 0.25-3 R.  This script isolates why by sweeping the
signed depth ``t`` across ``contact_r`` on analytic geometry (box: exact
faceting-free mesh; capped cylinder: faceted) and either ships the sweep of the
*working tree* or, with ``PYTHONPATH`` pointed at a checkout of the pre-fix
tree, the same sweep before the fix, so before/after differ only in the module.

Usage::

    PYTHONPATH=src python scripts/hull_signed_deep_penetration_audit.py \\
        --label patched --out runs/m16872/sweep_after.json \\
        --figure runs/m16872/fig_deep_penetration_after.png
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordem.hull_mesh_contact import mesh_contact_forces
from tensordem.hull_mesh_contact_signed import is_watertight, ray_parity_inside

F64 = torch.float64
R, K = 0.05, 1.0e6
HALF = 0.5


# ---------------------------------------------------------------------------
# geometry (mirrors scripts/hull_signed_audit.py)
# ---------------------------------------------------------------------------
def box_mesh(half: float = HALF):
    c = [(-half, -half, -half), (half, -half, -half), (half, half, -half),
         (-half, half, -half), (-half, -half, half), (half, -half, half),
         (half, half, half), (-half, half, half)]
    v = torch.tensor(c, dtype=F64)
    f = torch.tensor([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], dtype=torch.int64)
    return v, f


def capped_cylinder_mesh(radius: float, length: float, n_circ: int = 64):
    ang = torch.linspace(0.0, 2.0 * math.pi, n_circ + 1, dtype=F64)[:-1]
    ring = torch.stack([radius * torch.cos(ang), radius * torch.sin(ang)], dim=-1)
    z0, z1 = -length / 2.0, length / 2.0
    verts = []
    for z in (z0, z1):
        for (x, y) in ring.tolist():
            verts.append([x, y, z])
    verts.append([0.0, 0.0, z0])
    verts.append([0.0, 0.0, z1])
    v = torch.tensor(verts, dtype=F64)
    c0, c1, b0, b1 = 0, n_circ, 2 * n_circ, 2 * n_circ + 1
    faces = []
    for i in range(n_circ):
        j = (i + 1) % n_circ
        faces.append([c0 + i, c0 + j, c1 + j])
        faces.append([c0 + i, c1 + j, c1 + i])
        faces.append([b0, c0 + j, c0 + i])
        faces.append([b1, c1 + i, c1 + j])
    return v, torch.tensor(faces, dtype=torch.int64)


# ---------------------------------------------------------------------------
# force probe
# ---------------------------------------------------------------------------
def force_mag(centre, verts, faces, closed):
    f, reaction = mesh_contact_forces(
        centre.reshape(1, 3), torch.zeros(1, 3, dtype=F64), verts, faces,
        torch.zeros(3, dtype=F64), particle_radius=R, stiffness=K,
        closed_surface=closed)
    return f[0], reaction


def sweep_cylinder(depths, az_deg=(0.0, 47.0, 133.0, 311.0)):
    verts, faces = capped_cylinder_mesh(0.5, 2.0, 64)
    rows = []
    for az in az_deg:
        a = math.radians(az)
        n_out = torch.tensor([math.cos(a), math.sin(a), 0.0], dtype=F64)
        for frac in depths:
            rho = 0.5 - frac * R
            c = n_out * rho
            inside = bool(ray_parity_inside(c.reshape(1, 3), verts, faces)[0].item())
            f, _ = force_mag(c, verts, faces, closed=True)
            dropped = f.abs().max().item() == 0.0 and inside
            mag = float(torch.linalg.vector_norm(f))
            cos = float(torch.dot(f / mag, n_out)) if mag > 0 else 0.0
            exact = K * (R + frac * R) if frac > 0 else K * (R + frac * R)
            rows.append({
                "geometry": "capped_cylinder",
                "az_deg": az, "depth_frac_R": frac, "depth_m": frac * R,
                "inside": inside, "force_N": mag,
                "expected_N": exact, "rel_err": (abs(mag - exact) / exact),
                "outward_cos": cos, "silent_drop": dropped,
            })
    return rows


def sweep_box(depths):
    """Box hull: the mesh is exact (no faceting), so any error is code, not geometry."""
    verts, faces = box_mesh()
    rows = []
    for frac in depths:
        z = HALF - frac * R
        c = torch.tensor([0.0, 0.0, z], dtype=F64)
        inside = bool(ray_parity_inside(c.reshape(1, 3), verts, faces)[0].item())
        f, reaction = force_mag(c, verts, faces, closed=True)
        mag = float(f[2])
        exact = K * (R + frac * R)
        rows.append({
            "geometry": "box", "depth_frac_R": frac, "depth_m": frac * R,
            "inside": inside, "force_N": mag, "expected_N": exact,
            "rel_err": abs(mag - exact) / exact,
            "reaction_ok": bool(abs(float(reaction[2]) + mag) < 1e-6 * max(1.0, abs(mag))),
        })
    return rows


def deep_recovery():
    """Integrate a 50 mm sphere released 0.2 m inside a 1 m box (k = 1 MN/m)."""
    verts, faces = box_mesh()
    mass = 4.0 / 3.0 * math.pi * R ** 3 * 2500.0
    x = torch.tensor([0.0, 0.0, HALF - 0.2], dtype=F64)
    v = torch.zeros(3, dtype=F64)
    dt = 2.0e-5
    nstep = int(3.0e-3 / dt)
    out = []
    for _ in range(nstep):
        f, _ = force_mag(x, verts, faces, closed=True)
        v = v + (f / mass) * dt
        x = x + v * dt
        out.append((float(x[2]), float(v[2] * mass)))
    return {"t_end": nstep * dt, "z_end": out[-1][0], "vz_end": out[-1][1] / mass,
            "z_series": [o[0] for o in out], "t": nstep * dt / nstep}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="patched",
                    help="tag written into metrics.json (e.g. pre-fix / patched)")
    ap.add_argument("--out", default="runs/m16872/sweep.json")
    ap.add_argument("--figure", default=None)
    args = ap.parse_args()

    depths = (-0.5, -0.2, 0.0, 0.25, 0.5, 0.9, 1.0, 1.01, 1.5, 2.0, 3.0, 4.0)
    cyl = sweep_cylinder(depths)
    box = sweep_box((-0.5, 0.0, 0.25, 0.5, 0.9, 1.0, 1.02, 1.5, 2.0, 3.0, 6.0))
    manif = {"cylinder_watertight": bool(is_watertight(*capped_cylinder_mesh(0.5, 2.0, 64))),
             "box_watertight": bool(is_watertight(*box_mesh()))}

    drops = [r for r in cyl if r["silent_drop"]]
    deep = [r for r in cyl if r["depth_frac_R"] >= 1.0]
    metrics = {
        "mode": args.label,
        "R": R, "K": K, "mesh": manif,
        "cylinder_sweep": cyl,
        "box_sweep": box,
        "verdict": {
            "silent_drop_points": len(drops),
            "deep_points": len(deep),
            "deep_max_rel_err": max((r["rel_err"] for r in deep), default=0.0),
            "deep_max_rel_err_box": max(r["rel_err"] for r in box if r["depth_frac_R"] >= 1.0),
            "outside_max_rel_err": max(r["rel_err"] for r in cyl if r["depth_frac_R"] <= 0.0),
            "shallow_max_rel_err": max(r["rel_err"] for r in cyl
                                       if 0.0 < r["depth_frac_R"] < 1.0),
            "min_outward_cos_deep": min((r["outward_cos"] for r in deep), default=0.0),
        },
        "deep_recovery": deep_recovery(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    v = metrics["verdict"]
    print(f"mode={metrics['mode']}  silent drops={v['silent_drop_points']}/{len(cyl)}  "
          f"deep max err={v['deep_max_rel_err']*100:.2f}%  "
          f"(box {v['deep_max_rel_err_box']*100:.3f}%)  "
          f"outside max err={v['outside_max_rel_err']*100:.3f}%  "
          f"min outward cos={v['min_outward_cos_deep']:.4f}  "
          f"recovery z_end={metrics['deep_recovery']['z_end']:.6f}")

    if args.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
        t = [r["depth_m"] * 1e3 for r in cyl if r["az_deg"] == 0.0]
        fm = [r["force_N"] / 1e3 for r in cyl if r["az_deg"] == 0.0]
        ex = [r["expected_N"] / 1e3 for r in cyl if r["az_deg"] == 0.0]
        ax[0].plot(t, ex, "k-", label=r"$k\,(r_c+t)$ / $k\,(r_c-t)$")
        ax[0].plot(t, fm, "o", ms=4,
                   label=f"{args.label} signed")
        ax[0].axvline(R * 1e3, color="0.6", ls="--", lw=0.9, label=r"$t=r_c$")
        ax[0].set_xlabel("signed depth $t$ [mm]  (t>0 inside)")
        ax[0].set_ylabel("penalty force magnitude [kN]")
        ax[0].set_title("capped cylinder, $R_{hull}$=0.5 m")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)
        ax[1].plot(t, [r["outward_cos"] for r in cyl if r["az_deg"] == 0.0], "s-", ms=3)
        ax[1].set_xlabel("signed depth $t$ [mm]")
        ax[1].set_ylabel(r"$\cos(\hat F,\ \hat n_{out})$")
        ax[1].set_ylim(-1.2, 1.2)
        ax[1].set_title("force direction (must be +1)")
        ax[1].grid(alpha=0.3)
        png = Path(args.figure)
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        print("figure ->", png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())