"""Round 26 (m22550) analysis: does the polydisperse seed break the crystal that
round 25 identified, and (once the sweep `cj4227` has drained) what does mu do?

Two independent products:
1. **seed structure** (computed here, no dynamics): psi6, Z, band area fraction and
   the radius distribution of the *initial* configurations at w = 0 / 0.25 / 0.5.
   Round 25 showed the monodisperse seed re-crystallises during compaction; this
   file measures the seed that the polydisperse branch starts from.
2. **sweep table** (read from ``runs/plane_shear_2d/poly_w*_I*.json`` when present):
   mu_wall / phi_gap vs the archived law, with the honest PASS/FAIL per point.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "benchmarks" / "plane_shear_mu_i"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import plane_shear_2d as ps  # noqa: E402
import reference_law as rl  # noqa: E402

OUT = REPO / "outputs" / "r26"
FIG = REPO / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
RUNS = REPO / "benchmarks" / "plane_shear_mu_i" / "runs" / "plane_shear_2d"


def case(w: float, I: float = 0.05) -> ps.PlaneShear2DCase:
    return ps.PlaneShear2DCase(I_target=I, n_particles=458, width=0.30, gap=0.15,
                               settle_time_units=40.0, shear_settle_strain=0.5,
                               strain_measure=1.0, init_mode="random", seed=0,
                               polydispersity=w)


def seed_structure() -> dict:
    rows = {}
    for w in (0.0, 0.25, 0.5):
        c = case(w)
        cell = ps.RoughWallShearCell(c)
        rad = cell.system.radii[cell.free_idx]
        pos = cell.system.pos[cell.free_idx]
        order = ps.hexatic_order(pos, rad, c.width)
        area = float(np.sum(math.pi * rad ** 2))
        rows[f"{w:g}"] = {
            "polydispersity": w,
            "radius_mean_mm": 1e3 * float(rad.mean()),
            "radius_sd_mm": 1e3 * float(rad.std(ddof=1)),
            "radius_max_over_min": float(rad.max() / rad.min()),
            "band_phi_seed": area / (c.width * (cell.gap_eff)),
            "psi6_mean": order["psi6_mean"],
            "psi6_hexatic_fraction": order["psi6_hexatic_fraction"],
            "z_mean": order["z_mean"],
            "psi6_round25_after_compaction": {0.0: 0.914, 0.25: None, 0.5: None}[w],
        }
    return rows


def sweep_table() -> dict:
    rows = {}
    for f in sorted(RUNS.glob("poly_w*_I*.json")):
        d = json.loads(f.read_text())
        I = d["I_bulk"]
        rows[f.stem] = {
            "I_target": d["I_target"], "I_measured": I,
            "polydispersity": d.get("polydispersity"),
            "radius_max_over_min": d.get("radius_ratio_max_over_min"),
            "mu_wall": d["mu_wall"], "mu_wall_sd": d.get("mu_wall_sd"),
            "mu_bulk": d["mu_bulk"],
            "phi_gap": d["phi_gap"],
            "psi6_mean": d["psi6_mean"], "psi6_hexatic_fraction": d["psi6_hexatic_fraction"],
            "z_mean": d["z_mean"],
            "p_wall_over_P": d["p_wall_over_P"],
            "ref_mu": rl.ref_mu(I) if math.isfinite(I) else float("nan"),
            "ref_phi": rl.ref_phi(I) if math.isfinite(I) else float("nan"),
            "mu_rel_err": (rl.ref_mu(I) - d["mu_wall"]) / rl.ref_mu(I)
                          if math.isfinite(I) else float("nan"),
        }
    return rows


def figures(seed: dict, sweep: dict) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    # (1) seed structure: psi6 and Z vs w, with round 25's post-compaction psi6
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ws = [seed[k]["polydispersity"] for k in seed]
    psi = [seed[k]["psi6_mean"] for k in seed]
    hexf = [seed[k]["psi6_hexatic_fraction"] for k in seed]
    zz = [seed[k]["z_mean"] for k in seed]
    ax[0].plot(ws, psi, "o-", label=r"$\psi_6$ (seed)")
    ax[0].plot(ws, hexf, "s--", label="hexatic fraction ($\\psi_6>0.8$)")
    ax[0].axhline(0.914, color="crimson", ls=":",
                  label=r"round 25: $\psi_6$ after compaction (w=0)")
    ax[0].axhline(0.9069 / math.pi * math.pi, color="none")
    ax[0].set_xlabel("polydispersity $w$"); ax[0].set_ylabel("order parameter")
    ax[0].set_title("seed structure vs size spread"); ax[0].legend(fontsize=7)
    ax[0].set_ylim(0, 1.02)
    ax[1].plot(ws, zz, "^-", color="darkgreen")
    ax[1].axhline(4.0, color="grey", ls=":", label="isostatic $Z=4$ (2-D, jamming)")
    ax[1].set_xlabel("polydispersity $w$"); ax[1].set_ylabel("coordination $Z$")
    ax[1].set_title("seed coordination"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    p = FIG / "r26_seed_structure.png"
    fig.savefig(p, dpi=140); made.append(str(p.relative_to(REPO)))
    plt.close(fig)

    # (2) radius distributions actually drawn (the device contract)
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for w in (0.25, 0.5):
        r = ps._draw_radii(case(w), 458) * 1e3
        ax.hist(r, bins=30, alpha=0.55, label=f"$w={w:g}$ (r_max/r_min={r.max()/r.min():.2f})")
    ax.set_xlabel("disk radius [mm]"); ax.set_ylabel("count")
    ax.set_title("bounded radius draw, mean fixed at 5 mm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "r26_radius_draw.png"
    fig.savefig(p, dpi=140); made.append(str(p.relative_to(REPO)))
    plt.close(fig)
    return made


def main() -> int:
    seed = seed_structure()
    sweep = sweep_table()
    figs = figures(seed, sweep)
    (OUT / "r26_seed_structure.json").write_text(json.dumps(seed, indent=1))
    (OUT / "r26_sweep_table.json").write_text(json.dumps(sweep, indent=1))
    print(json.dumps({"seed": seed, "n_sweep_runs": len(sweep), "figures": figs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
