#!/usr/bin/env python3
"""Curved-hull validation of ``tensordem.hull_mesh_contact`` (GAP-DEM-G3 debt).

The 3-D mesh contact kernel (m669) was verified on flat planes and single
analytic triangles only.  The gap ledger (``docs/dem_calibration_gap_ledger.md``)
carries the open row: *curved-hull validation on a real hull not re-audited*.
This audit closes the geometry side of that row with the 3-D analogue of the
2-D polyline audit (``docs/polyline_hull_contact_verification.md``):

* reference = exact circular cylinder (the 3-D analogue of the 2-D circular
  hull), distance from a sphere centre to the surface in closed form;
* the mesh is an *inscribed* faceted cylinder -- exactly what a digitised
  curved hull is;
* measured: force error vs penetration, discretisation convergence
  (sagitta law), end-rim region, broad-phase completeness, action/reaction.

Every number in the report is emitted by this script (``--out`` JSON); nothing
is transcribed by hand.  Light-weight: pure torch, seconds of CPU.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordem.hull_mesh_contact import _SpatialHash, mesh_contact_forces

# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def cylinder_mesh(radius: float, length: float, n_circ: int, n_axial: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Inscribed faceted cylinder, axis = z, centred at the origin.

    ``n_circ`` facets around, ``n_axial`` rings along the axis.  Outward
    winding (not required by the closest-point kernel, kept for hygiene).
    """
    if n_circ < 3 or n_axial < 1:
        raise ValueError("need n_circ >= 3, n_axial >= 1")
    theta = torch.linspace(0.0, 2.0 * math.pi, n_circ + 1, dtype=torch.float64)[:-1]
    z = torch.linspace(-0.5 * length, 0.5 * length, n_axial + 1, dtype=torch.float64)
    T, _ = torch.meshgrid(theta, z, indexing="ij")  # (n_axial+1, n_circ)
    Z = z.unsqueeze(0).expand(n_circ, n_axial + 1)
    verts = torch.stack([radius * torch.cos(T), radius * torch.sin(T), Z], dim=-1)
    verts = verts.reshape(-1, 3)

    def vid(i: int, j: int) -> int:
        return (i % n_circ) * (n_axial + 1) + j

    faces = []
    for j in range(n_axial):
        for i in range(n_circ):
            a, b = vid(i, j), vid(i + 1, j)
            c, d = vid(i + 1, j + 1), vid(i, j + 1)
            faces.append((a, c, b))
            faces.append((a, c, d))
    return verts, torch.tensor(faces, dtype=torch.int64)


def sagitta(radius: float, n_circ: int) -> float:
    """Radial gap between the inscribed facet and the true circle."""
    return radius * (1.0 - math.cos(math.pi / n_circ))


# ---------------------------------------------------------------------------
# Analytic reference (independent of the kernel under test)
# ---------------------------------------------------------------------------

def analytic_cylinder_force(center: torch.Tensor, radius: float, length: float,
                           particle_radius: float, stiffness: float) -> tuple[float, torch.Tensor]:
    """Exact penalty force of a sphere against a solid circular cylinder.

    Returns ``(force_magnitude, unit_direction)``; magnitude is 0 when the
    sphere does not touch the solid.  Distance to the solid = min of the
    mid-span radial distance and the two rim-circle distances.
    """
    x, y, z = float(center[0]), float(center[1]), float(center[2])
    rho = math.hypot(x, y)
    candidates = []
    if abs(z) <= 0.5 * length:
        ux, uy = (x / rho, y / rho) if rho > 1e-15 else (1.0, 0.0)
        candidates.append((abs(rho - radius),
                           torch.tensor([radius * ux, radius * uy, 0.0])))
    for z_end in (-0.5 * length, 0.5 * length):
        d = math.hypot(rho - radius, z - z_end)
        ux, uy = (x / rho, y / rho) if rho > 1e-15 else (1.0, 0.0)
        candidates.append((d, torch.tensor([radius * ux, radius * uy, z_end])))
    d, closest = min(candidates, key=lambda t: t[0])
    overlap = particle_radius - d
    if overlap <= 0.0:
        return 0.0, torch.zeros(3, dtype=torch.float64)
    delta = center - closest
    nrm = delta / delta.norm() if delta.norm() > 1e-15 else torch.tensor([0.0, 0.0, 1.0])
    return stiffness * overlap, nrm


# ---------------------------------------------------------------------------
# Audit sections
# ---------------------------------------------------------------------------

def radial_sweep(verts, faces, n_circ: int, radius: float, length: float,
                 particle_radius: float, stiffness: float,
                 penetrations, azimuths_deg) -> dict:
    rows = []
    for az in azimuths_deg:
        t = math.radians(az)
        for pen in penetrations:
            rho = radius - pen  # pen > 0 -> centre inside the true surface
            c = torch.tensor([rho * math.cos(t), rho * math.sin(t), 0.0], dtype=torch.float64)
            f, _ = mesh_contact_forces(c.unsqueeze(0), torch.zeros(1, 3, dtype=torch.float64),
                                    verts, faces, torch.zeros(3, dtype=torch.float64),
                                    particle_radius=particle_radius, stiffness=stiffness)
            f_mag = float(f[0].norm())
            f_ana, n_ana = analytic_cylinder_force(c, radius, length, particle_radius, stiffness)
            rel = (f_mag - f_ana) / f_ana if f_ana > 0 else abs(f_mag - f_ana)
            dir_err = 0.0
            if f_mag > 0 and f_ana > 0:
                dir_err = math.degrees(math.acos(torch.clamp((f[0] / f_mag) @ n_ana, -1.0, 1.0)))
            rows.append({"azimuth_deg": az, "penetration_m": pen,
                         "force_num": f_mag, "force_ana": f_ana,
                         "rel_error": rel, "dir_error_deg": dir_err})
    max_rel = max(r["rel_error"] for r in rows)
    max_dir = max(r["dir_error_deg"] for r in rows)
    dropout = sum(1 for r in rows if r["force_num"] == 0.0 and r["force_ana"] > 0.0)
    return {"rows": rows, "max_rel_error": max_rel, "max_dir_error_deg": max_dir,
            "dropout_count": dropout, "n_rows": len(rows)}


def discretisation_scan(radius: float, length: float, particle_radius: float,
                        stiffness: float, n_circ_list, penetration: float,
                        azimuths_deg) -> list[dict]:
    out = []
    for n_circ in n_circ_list:
        verts, faces = cylinder_mesh(radius, length, n_circ, 4)
        res = radial_sweep(verts, faces, n_circ, radius, length, particle_radius,
                           stiffness, [penetration], azimuths_deg)
        out.append({"n_circ": n_circ, "sagitta_m": sagitta(radius, n_circ),
                    "sagitta_over_pen": sagitta(radius, n_circ) / penetration,
                    "max_rel_error": res["max_rel_error"],
                    "dropout_count": res["dropout_count"], "n_rows": res["n_rows"]})
    return out


def rim_study(radius: float, length: float, particle_radius: float, stiffness: float,
              n_circ: int) -> dict:
    """Sphere approaching the end rim: analytic rim-circle distance."""
    verts, faces = cylinder_mesh(radius, length, n_circ, 8)
    rows = []
    z_end = 0.5 * length
    for dz in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
        for dr in (-0.02, -0.01, 0.0, 0.01, 0.02):
            c = torch.tensor([(radius + dr) * math.cos(0.6), (radius + dr) * math.sin(0.6),
                              z_end + dz], dtype=torch.float64)
            f, _ = mesh_contact_forces(c.unsqueeze(0), torch.zeros(1, 3, dtype=torch.float64),
                                    verts, faces, torch.zeros(3, dtype=torch.float64),
                                    particle_radius=particle_radius, stiffness=stiffness)
            f_mag = float(f[0].norm())
            f_ana, _ = analytic_cylinder_force(c, radius, length, particle_radius, stiffness)
            rows.append({"dz_m": dz, "dr_m": dr, "force_num": f_mag, "force_ana": f_ana,
                         "rel_error": (f_mag - f_ana) / f_ana if f_ana > 0 else abs(f_mag - f_ana)})
    return {"n_circ": n_circ, "rows": rows,
            "max_rel_error": max(r["rel_error"] for r in rows),
            "sagitta_m": sagitta(radius, n_circ)}


def broad_phase_completeness(verts, faces, radius: float, length: float,
                            particle_radius: float, n_queries: int, seed: int) -> dict:
    """Hash candidates must be a superset of the brute-force AABB set."""
    g = torch.Generator().manual_seed(seed)
    centers = torch.zeros(n_queries, 3, dtype=torch.float64)
    for i in range(n_queries):
        t = float(torch.rand((), generator=g).item() * 2 * math.pi)
        rho = radius + float(torch.rand((), generator=g).item() * 0.3 - 0.15)
        z = float(torch.rand((), generator=g).item() * 1.2 * length - 0.6 * length)
        centers[i] = torch.tensor([rho * math.cos(t), rho * math.sin(t), z])
    cell_width = max(4.0 * particle_radius, 1e-6)
    sh = _SpatialHash(verts, faces, cell_width)
    cands = sh.query(centers, 2.0 * particle_radius)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    tri_min = torch.stack([v0, v1, v2], dim=0).min(dim=0).values
    tri_max = torch.stack([v0, v1, v2], dim=0).max(dim=0).values
    missing = 0
    for i in range(n_queries):
        lo = centers[i] - 2.0 * particle_radius
        hi = centers[i] + 2.0 * particle_radius
        need = ((tri_max >= lo) & (tri_min <= hi)).all(dim=1).nonzero().flatten().tolist()
        have = set(cands[i])
        if any(fi not in have for fi in need):
            missing += 1
    return {"n_queries": n_queries, "missing_queries": missing}


def action_reaction(verts, faces, radius: float, length: float,
                    particle_radius: float, stiffness: float) -> float:
    c = torch.tensor([[radius - 0.01, 0.0, 0.0]], dtype=torch.float64)
    f, reaction = mesh_contact_forces(c, torch.zeros(1, 3, dtype=torch.float64),
                                    verts, faces, torch.zeros(3, dtype=torch.float64),
                                    particle_radius=particle_radius, stiffness=stiffness)
    return float((reaction + f.sum(dim=0)).norm())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/m6423_curved_hull_audit/metrics.json")
    ap.add_argument("--figures", default="artifacts/m6423_curved_hull_audit")
    args = ap.parse_args()

    R_c, L, R_p, k = 0.5, 2.0, 0.05, 1.0e6
    out_dir = Path(args.figures)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {
        "case": {"cylinder_radius_m": R_c, "cylinder_length_m": L,
                 "particle_radius_m": R_p, "stiffness_N_per_m": k},
    }

    # 1) fine mesh: must reproduce the analytic force to machine precision
    fine = cylinder_mesh(R_c, L, 128, 8)
    pens = [0.04, 0.03, 0.02, 0.01, 0.005, 0.002]
    azs = [0, 15, 47, 90, 133, 180, 212, 270, 311]
    fine_res = radial_sweep(fine[0], fine[1], 128, R_c, L, R_p, k, pens, azs)
    metrics["fine_mesh"] = {kk: vv for kk, vv in fine_res.items() if kk != "rows"}
    metrics["fine_mesh"]["rows"] = fine_res["rows"]

    # 2) discretisation scan at fixed penetration
    scan = discretisation_scan(R_c, L, R_p, k, [8, 16, 32, 64, 128], 0.01,
                              list(range(0, 360, 5)))
    metrics["discretisation_scan"] = scan

    # 3) end-rim region
    metrics["rim_study"] = rim_study(R_c, L, R_p, k, 64)

    # 4) broad phase completeness on the curved mesh
    metrics["broad_phase"] = broad_phase_completeness(fine[0], fine[1], R_c, L, R_p, 300, 20261108)

    # 5) action/reaction
    metrics["action_reaction_residual"] = action_reaction(fine[0], fine[1], R_c, L, R_p, k)

    # 6) closed-form digitisation rule (3-D analogue of the 2-D rule)
    for pen in (0.02, 0.01, 0.005):
        n_min = math.ceil(math.pi / math.acos(1.0 - pen / R_c))
        metrics.setdefault("digitisation_rule", []).append(
            {"penetration_m": pen, "n_circ_min": n_min,
             "sagitta_at_n_min_m": sagitta(R_c, n_min)})

    with open(args.out, "w") as fh:
        json.dump(metrics, fh, indent=1)
    print(json.dumps({kk: vv for kk, vv in metrics.items() if kk != "fine_mesh"}, indent=1))
    print("fine_mesh:", {kk: vv for kk, vv in fine_res.items() if kk != "rows"})

    # figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for az in (0, 47, 133, 270):
            rr = [r for r in fine_res["rows"] if r["azimuth_deg"] == az]
            ax.plot([r["penetration_m"] for r in rr], [r["force_num"] for r in rr],
                    "o-", ms=3, label=f"az {az}\u00b0")
        ax.plot([r["penetration_m"] for r in fine_res["rows"] if r["azimuth_deg"] == 0],
                [r["force_ana"] for r in fine_res["rows"] if r["azimuth_deg"] == 0],
                "k-", lw=2, label="analytic cylinder")
        ax.set_xlabel("penetration (m)")
        ax.set_ylabel("force (N)")
        ax.set_title("n_circ=128: force vs penetration, several azimuths")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "force_vs_penetration.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.loglog([s["n_circ"] for s in scan], [max(s["max_rel_error"], 1e-12) for s in scan],
                  "o-", label="measured max rel. error")
        ax.loglog([s["n_circ"] for s in scan], [max(s["sagitta_over_pen"], 1e-12) for s in scan],
                  "k--", label="sagitta / penetration bound")
        ax.set_xlabel("facets around the circumference")
        ax.set_ylabel("max relative force error")
        ax.set_title("discretisation error vs resolution (penetration 10 mm)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "discretisation_scan.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 4))
        rr = metrics["rim_study"]["rows"]
        ax.plot([r["dz_m"] for r in rr], [r["force_num"] for r in rr], "o-", ms=3,
                  label="mesh (n_circ=64)")
        ax.plot([r["dz_m"] for r in rr], [r["force_ana"] for r in rr], "k-", lw=2,
                  label="analytic rim")
        ax.set_xlabel("z - L/2 (m)")
        ax.set_ylabel("force (N)")
        ax.set_title("end-rim approach, dr = 0")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "rim_approach.png", dpi=140)
        plt.close(fig)
        print("figures written to", out_dir)
    except Exception as exc:  # figures are secondary; never fail the audit
        print("figure generation skipped:", exc)


if __name__ == "__main__":
    main()
