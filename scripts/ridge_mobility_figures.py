#!/usr/bin/env python
"""Figures for the grounded ice-ridge mobility report (reads metrics.json)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "ridge_mobility"
sys.path.insert(0, str(ROOT / "src"))

from tensordem.ridge_mobility import (  # noqa: E402
    RHO_ICE, RHO_WATER, DemKeelSpec, RidgeMobilityCase, dem_grounding,
    free_float_draft_ratio,
)

M = json.loads((OUT / "metrics.json").read_text())


def fig_dem_state() -> Path:
    """DEM keel section: grains, bonds, seabed contact forces, waterline."""
    spec = DemKeelSpec(keel_depth=1.0, porosity=0.30, radius=0.03, flank_angle_deg=30.0)
    case = RidgeMobilityCase(keel_depth=1.0, water_depth=0.7, seabed_friction=0.35,
                            ship_thrust=1.5e6, ship_contact_length=30.0)
    led, cf = dem_grounding(case, spec, max_steps=9000, tol=1e-7)
    # rebuild the final state for plotting (deterministic seed and protocol)
    from tensordem.ridge_mobility import DemKeelRig  # local import: plotting only
    rig = DemKeelRig(spec)
    ground = -case.water_depth
    rig.positions = rig.positions - rig.positions[:, 1].min() + spec.radius * 0.5 + ground
    rig.relax(waterline=0.0, ground_depth=ground, seabed_friction=case.seabed_friction,
              drive_force_per_length=case.driving_force, max_steps=9000, tol=1e-7)
    p = rig.positions
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6),
                                  gridspec_kw={"width_ratios": [1.35, 1.0]})
    bi, bj = rig.bonds
    for a, b in zip(bi.tolist(), bj.tolist()):
        ax.plot([p[a, 0], p[b, 0]], [p[a, 1], p[b, 1]], color="0.85", lw=0.4, zorder=1)
    sc = ax.scatter(p[:, 0], p[:, 1], s=18, c=p[:, 1], cmap="Blues_r", vmin=-1.0, vmax=0.0,
                    edgecolors="0.3", linewidths=0.3, zorder=2)
    ax.axhline(0.0, color="tab:blue", lw=1.2, label="waterline")
    ax.axhline(ground, color="saddlebrown", lw=2.2, label="seabed")
    bed = (ground - (p[:, 1] - spec.radius)) > 0
    for k in range(int(bed.sum())):
        pass
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"DEM keel pack (N={rig.n}, r={spec.radius*100:.0f} cm, "
                 f"n_meas={led['measured_porosity']:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    plt.colorbar(sc, ax=ax, label="z [m]")

    # draft / Archimedes view
    t = np.linspace(0, 1.15, 200)
    ax2.plot(t, t ** 2, "k-", label=r"self-similar prism $\phi=(t/h_k)^2$")
    ax2.axhline(RHO_ICE / RHO_WATER, color="tab:red", ls="--",
                label=r"Archimedes $\rho_i/\rho_w$")
    ratio = free_float_draft_ratio()
    ax2.axvline(ratio, color="tab:green", ls=":",
                label=fr"closed form $t_{{ff}}/h_k$={ratio:.4f}")
    c2 = M["C2_free_float"]
    ax2.plot(c2["draft_ratio_dem_area"], c2["archimedes_ratio_dem"], "o", ms=9,
             color="tab:purple", label="DEM free float (area)")
    ax2.plot(c2["draft_ratio_dem_geometric"], c2["archimedes_ratio_dem"], "s", ms=8,
             color="tab:orange", label="DEM free float (geometric)")
    ax2.set_xlabel(r"draft ratio $t/h_k$  ($h_k$ from the same body)")
    ax2.set_ylabel(r"submerged solid fraction $\phi$")
    ax2.set_title("Free-float equilibrium (C2)")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7.5)
    fig.tight_layout()
    path = OUT / "fig1_dem_keel_state.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_buoyancy_vs_depth() -> Path:
    """Grounding force vs water depth: closed form + DEM points + free-float limit."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    hk = 5.0
    s = 1.0 / math.tan(math.radians(30.0))
    hh = np.linspace(0.45, 1.05, 200)
    N = np.array([max(0.0, (1 - 0.30) * 9.81 * s * (RHO_ICE * hk ** 2 - RHO_WATER * (r * hk) ** 2))
                  for r in hh])
    ax.plot(hh, N / 1e3, "k-", label="closed form (n=0.30)")
    ratio = free_float_draft_ratio()
    ax.axvline(ratio, color="tab:red", ls="--", label=r"$t_{ff}/h_k$")
    rows = M["C3_mu_star_boundary"]["rows"]
    xs, ys = [], []
    for row in rows:
        xs.append(row["closure"]["water_depth_over_keel"])
        ys.append(row["ground_normal_N_per_m"] / 1e3)
    ax.plot(xs, ys, "o", ms=9, color="tab:blue", label="DEM (r/h_k=1.5 %)"
            if not M["mode"] == "quick" else "DEM (r/h_k=3 %)")
    base = M["grounded_base"]
    ax.plot(base["closure"]["water_depth_over_keel"],
            base["closure"]["normal_force_dem_static_N_per_m"] / 1e3, "D", ms=8,
            color="tab:purple", label="DEM base case")
    ax.set_xlabel(r"water depth / keel depth  $H/h_k$")
    ax.set_ylabel(r"seabed normal force $N$ [kN/m]")
    ax.set_title("Grounded-ridge normal force: closed form vs DEM")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "fig2_normal_force_vs_depth.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_mu_star_boundary() -> Path:
    """Critical seabed friction vs keel-depth/draft ratio."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    curve = M["boundary_curve"]
    x = [r["keel_over_draft"] for r in curve]
    y = [r["mu_star"] for r in curve]
    ax.plot(x, y, "k-", label=r"closed form $\mu^*(h_k/d)=(q-c_a b)/N$")
    demo = M["C3_mu_star_boundary"]["rows"]
    ax.plot([r["keel_over_draft"] for r in demo], [r["mu_star_dem"] for r in demo],
            "o", ms=10, color="tab:blue", label="DEM boundary")
    ax.plot([r["keel_over_draft"] for r in demo], [r["mu_star_cf"] for r in demo],
            "x", ms=9, color="tab:red", label="closed form at DEM cases")
    for r in demo:
        ax.annotate(f"{r['error_vs_cf_pct']:.1f}%",
                    (r["keel_over_draft"], r["mu_star_dem"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axvspan(*[4.0, 6.0], color="tab:green", alpha=0.08,
               label="literature band $h_k/d\\in[4,6]$")
    ax.axhline(0.35, color="0.4", ls=":", label=r"$\mu_b=0.35$ seabed")
    ax.set_xlabel(r"$h_k/d$  (keel depth / ambient level-ice draft)")
    ax.set_ylabel(r"critical friction $\mu^*$")
    ax.set_title("Mobility boundary in $(h_k/d,\\,\\mu_b)$")
    ax.set_xlim(2, 8.5)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    path = OUT / "fig3_mu_star_boundary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_mobility_map() -> Path:
    """Mobile / immobile regions in the (h_k/d, mu_b) plane with DEM sweep."""
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    curve = M["boundary_curve"]
    x = np.array([r["keel_over_draft"] for r in curve])
    y = np.array([r["mu_star"] for r in curve])
    ax.fill_between(x, 0, y, color="tab:green", alpha=0.18, label="mobile (slides)")
    ax.fill_between(x, y, 1.2, color="tab:red", alpha=0.15, label="immobile (held)")
    ax.plot(x, y, "k-", lw=2, label="mobility boundary")
    sweep = M["mu_sweep"]
    for row in sweep:
        held = not row["mobile"]
        ax.plot(5.0, row["seabed_friction"], "o", ms=10,
                color="tab:red" if held else "tab:green",
                markeredgecolor="k", zorder=5)
    ax.plot([], [], "o", color="tab:red", markeredgecolor="k", label="DEM held")
    ax.plot([], [], "o", color="tab:green", markeredgecolor="k", label="DEM slides")
    ax.axhline(0.35, color="0.3", ls=":", lw=1)
    ax.text(2.15, 0.38, r"$\mu_b=0.35$", fontsize=8, color="0.3")
    ax.set_xlim(2, 8.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"$h_k/d$")
    ax.set_ylabel(r"seabed friction $\mu_b$")
    ax.set_title("Grounded-ridge mobility map ($q=50$ kN/m)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    path = OUT / "fig4_mobility_map.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    paths = [fig_dem_state(), fig_buoyancy_vs_depth(), fig_mu_star_boundary(),
             fig_mobility_map()]
    print("\n".join(str(p) for p in paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())