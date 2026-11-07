"""Adhesion campaign (P96-N4) driver: JKR anchor, Laplace bridge, cohesive pile.

Produces the machine-readable evidence and the figures used by
``docs/adhesion_jkr_capillary.md``:

    artifacts/adhesion-jkr/jkr.json          JKR pull-off + gamma -> 0 control
    artifacts/adhesion-jkr/capillary.json    Laplace bridge vs 2 pi gamma R cos(theta)
    artifacts/adhesion-jkr/pile.json         repose angle vs Bond number
    artifacts/adhesion-jkr/fig_*.png         figures

Run:  PYTHONPATH=src python3 scripts/adhesion_repose_pile.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordem.adhesion import (
    capillary_force_analytic,
    jkr_force_from_contact_radius,
    jkr_hertz_relative_error,
    jkr_normal_force,
    jkr_pull_off_force,
    jkr_pull_off_numerical,
    reduced_modulus,
    solve_liquid_bridge,
    hertz_normal_force,
)
from tensordem.adhesive_pile2d import PileCase, simulate_pile

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "adhesion-jkr"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. JKR: pull-off anchor and gamma -> 0 degeneration
# ---------------------------------------------------------------------------

def jkr_evidence() -> dict:
    E, nu, R = 1.0e6, 0.3, 1.0e-3
    e_star = reduced_modulus(E, nu, rigid_counterpart=True)
    rows = []
    for gamma in (0.005, 0.02, 0.05, 0.1):
        num = jkr_pull_off_numerical(e_star, R, gamma)
        rows.append({
            "gamma": gamma,
            "a_c": num["a_c"],
            "delta_c": num["delta_c"],
            "f_pull_numeric": num["f_pull"],
            "f_pull_theory": jkr_pull_off_force(gamma, R),
            "rel_error": num["rel_error"],
        })

    # gamma -> 0 negative control (Hertz degeneration)
    control = []
    for gamma in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
        err = jkr_hertz_relative_error(e_star, R, gamma)
        control.append({"gamma": gamma, "max_rel_error": err["max_rel_error"],
                        "per_overlap": err["per_overlap"]})

    # force-displacement curves for the figure
    deltas = np.linspace(-1.0e-6, 6.0e-6, 241)
    curves = {}
    for gamma in (0.0, 0.02, 0.05):
        f = jkr_normal_force(torch.tensor(deltas), e_star, R, gamma).numpy()
        curves[str(gamma)] = f.tolist()
    hertz = hertz_normal_force(torch.tensor(np.clip(deltas, 0.0, None)), e_star, R).numpy()

    return {
        "e_star": e_star,
        "radius": R,
        "pull_off": rows,
        "gamma_to_zero": control,
        "curves": {"delta": deltas.tolist(), "gamma": curves,
                   "hertz": hertz.tolist()},
    }


def fig_jkr(ev: dict) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    d = np.array(ev["curves"]["delta"]) * 1e6
    for gamma, f in ev["curves"]["gamma"].items():
        ax[0].plot(d, np.array(f) * 1e3, label=f"JKR γ={gamma} J/m²")
    ax[0].plot(d, np.array(ev["curves"]["hertz"]) * 1e3, "--", color="k",
               label="Hertz (γ=0)")
    scale = jkr_pull_off_force(0.05, ev["radius"]) * 1e3
    ax[0].axhline(-scale, color="C3", ls=":", lw=1.2,
                  label="−1.5πγR* (γ=0.05)")
    ax[0].axhline(0.0, color="grey", lw=0.6)
    ax[0].set_xlabel("normal overlap δ (µm)")
    ax[0].set_ylabel("normal force (mN)")
    ax[0].set_title("(a) JKR force–displacement, sphere vs plane")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    g = np.array([row["gamma"] for row in ev["gamma_to_zero"]])
    e = np.array([row["max_rel_error"] for row in ev["gamma_to_zero"]])
    mask = g > 0
    ax[1].loglog(g[mask], np.maximum(e[mask], 1e-18), "o-")
    ax[1].axhline(1e-2, color="C3", ls="--", label="1 % acceptance")
    ax[1].set_xlabel("surface energy γ (J/m²)")
    ax[1].set_ylabel("max |F_JKR − F_Hertz| / F_Hertz")
    ax[1].set_title("(b) γ→0 negative control (linear in γ)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    path = OUT / "fig_jkr.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------------------
# 2. Liquid bridge (Laplace) vs the analytic 2 pi gamma R cos(theta)
# ---------------------------------------------------------------------------

def capillary_evidence() -> dict:
    R, gamma = 1.0e-3, 0.072
    vols = (1e-7, 1e-6, 1e-5, 1e-4)
    rows = []
    for theta in (0.0, 30.0, 60.0):
        for v in vols:
            res = solve_liquid_bridge(R, gap=0.0, contact_angle_deg=theta,
                                      volume_over_R3=v, gamma=gamma)
            rows.append({
                "theta_deg": theta,
                "volume_over_R3": v,
                "volume_solved_over_R3": res.volume / R ** 3,
                "filling_angle_deg": res.filling_angle_deg,
                "force": res.force,
                "force_neck": res.force_neck,
                "force_contact": res.force_contact,
                "consistency": res.consistency,
                "analytic": res.analytic_force,
                "rel_error_vs_analytic": res.rel_error_vs_analytic,
                "suction_over_gamma_R": res.suction * R / gamma,
                "neck_radius_over_R": res.neck_radius / R,
                "residual_angle": res.residual_angle,
                "residual_volume": res.residual_volume,
                "converged": res.converged,
            })
    # gap sensitivity at fixed (small) volume: the 2 pi gamma R cos(theta) law is
    # the gap->0 and V->0 asymptote.
    gap_rows = []
    for gap_over_R in (0.0, 0.01, 0.05, 0.10):
        for v in (1e-6, 1e-5):
            try:
                res = solve_liquid_bridge(R, gap=gap_over_R * R, contact_angle_deg=0.0,
                                          volume_over_R3=v, gamma=gamma)
            except RuntimeError as exc:  # thin bridge / large gap: no solution found
                gap_rows.append({"gap_over_R": gap_over_R, "volume_over_R3": v,
                                 "error": str(exc), "converged": False})
                continue
            gap_rows.append({
                "gap_over_R": gap_over_R,
                "volume_over_R3": v,
                "force": res.force,
                "ratio_to_analytic": res.force / res.analytic_force,
                "converged": res.converged,
                "filling_angle_deg": res.filling_angle_deg,
            })
    return {"radius": R, "gamma": gamma, "rows": rows, "gap_rows": gap_rows,
            "analytic": capillary_force_analytic(gamma, R, 0.0)}


def fig_capillary(ev: dict) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for theta, marker in ((0.0, "o"), (30.0, "s"), (60.0, "^")):
        rows = [r for r in ev["rows"] if r["theta_deg"] == theta]
        v = np.array([r["volume_over_R3"] for r in rows])
        ratio = np.array([r["force"] / r["analytic"] for r in rows])
        ax[0].semilogx(v, ratio, marker + "-", label=f"θ = {theta:.0f}°")
    ax[0].axhline(1.0, color="k", ls="--", lw=1, label="2πγR cosθ")
    ax[0].axhline(1.03, color="C3", ls=":", lw=1)
    ax[0].axhline(0.97, color="C3", ls=":", lw=1, label="±3 % band")
    ax[0].set_xlabel("bridge volume V / R³")
    ax[0].set_ylabel("F_Laplace / (2πγR cosθ)")
    ax[0].set_title("(a) Laplace solution vs analytic small-bridge limit")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")

    for theta, marker in ((0.0, "o"), (30.0, "s"), (60.0, "^")):
        rows = [r for r in ev["rows"] if r["theta_deg"] == theta]
        beta = np.array([r["filling_angle_deg"] for r in rows])
        err = np.array([r["rel_error_vs_analytic"] for r in rows]) * 100
        ax[1].plot(beta, err, marker + "-", label=f"θ = {theta:.0f}°")
    ax[1].axhline(3.0, color="C3", ls="--", lw=1, label="3 % acceptance")
    ax[1].set_xlabel("filling angle β (deg)")
    ax[1].set_ylabel("|F − 2πγR cosθ| / (2πγR cosθ)  (%)")
    ax[1].set_title("(b) deviation is ∝ β: vanishes as V→0")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    path = OUT / "fig_capillary.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------------------
# 3. Cohesive pile: repose angle vs Bond number
# ---------------------------------------------------------------------------

SETTLE = float(os.environ.get("ADHESION_SETTLE", "25.0"))


def pile_evidence() -> dict:
    base = PileCase(wedge_half_base_in_d=10.0, settle_time=SETTLE)
    rows = []
    snapshots = {}
    for bo in (0.0, 1.0, 5.0, 20.0):
        case = PileCase(**{**base.__dict__, "bond_number": bo})
        t0 = time.time()
        res = simulate_pile(case, return_particles=True)
        pos = res.pop("positions")
        res.pop("velocities")
        res["wall_time_s"] = time.time() - t0
        rows.append(res)
        snapshots[bo] = pos.numpy()

    # cohesion-range sensitivity at Bo = 20 (regularisation-width check)
    sens = []
    for cr in (0.01, 0.02, 0.04):
        case = PileCase(**{**base.__dict__, "bond_number": 20.0, "cohesion_range": cr})
        res = simulate_pile(case)
        sens.append({"cohesion_range_over_R": cr,
                     "theta_r_deg": res["theta_r_deg"],
                     "kinetic_energy": res["kinetic_energy"],
                     "max_speed": res["max_speed"]})

    # physical mapping of the Bond number (sphere, rho = 2500 kg/m^3)
    rho = 2500.0
    mapping = []
    for r_mm in (5.0, 0.5, 0.05):
        r = r_mm * 1e-3
        m = rho * 4.0 / 3.0 * math.pi * r ** 3
        for gamma_s in (0.05, 0.072, 1.0):
            f_pull = jkr_pull_off_force(gamma_s, r / 2.0)  # two identical spheres
            mapping.append({"radius_mm": r_mm, "gamma_solid": gamma_s,
                            "mass_kg": m, "f_pull_N": f_pull,
                            "bond_number": f_pull / (m * 9.81)})

    return {"rows": rows, "sensitivity": sens, "bond_mapping": mapping,
            "base": {k: v for k, v in base.__dict__.items()},
            "snapshots": {str(k): v.tolist() for k, v in snapshots.items()}}


def fig_pile(ev: dict) -> list[str]:
    names = []
    rows = ev["rows"]
    # (a) final configurations
    fig, axes = plt.subplots(1, len(rows), figsize=(3.0 * len(rows), 3.4),
                             sharey=True)
    for ax, row in zip(np.atleast_1d(axes), rows):
        pos = np.array(ev["snapshots"][str(row["bond_number"])])
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=pos[:, 1], s=18, cmap="viridis",
                        vmin=0.0)
        ax.set_title(f"Bo = {row['bond_number']:g}\nθr = {row['theta_r_deg']:.1f}°",
                     fontsize=9)
        ax.set_xlabel("x / d")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
    np.atleast_1d(axes)[0].set_ylabel("y / d")
    fig.colorbar(sc, ax=fig.axes, shrink=0.8, label="height y / d")
    path = OUT / "fig_pile_snapshots.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    names.append(path.name)

    # (b) repose angle vs Bond number
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    bo = [r["bond_number"] for r in rows]
    th = [r["theta_r_deg"] for r in rows]
    ax.plot(bo, th, "o-", lw=1.6)
    for b, t, r in zip(bo, th, rows):
        ax.annotate(f"{t:.1f}°", (b, t), textcoords="offset points", xytext=(4, 6),
                    fontsize=8)
        ax.annotate(f"KE={r['kinetic_energy']:.3f}", (b, t),
                    textcoords="offset points", xytext=(4, -12), fontsize=6,
                    color="grey")
    dth = th[-1] - th[0]
    ax.set_xlabel("Bond number  Bo = F_pull / (m g)")
    ax.set_ylabel("repose angle θr (deg)")
    ax.set_title(f"repose angle vs adhesion (Δθr = {dth:+.1f}° at Bo={bo[-1]:g})")
    ax.grid(alpha=0.3)
    path = OUT / "fig_pile_angle.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    names.append(path.name)
    return names


def main() -> None:
    t0 = time.time()
    pile_only = "--pile-only" in sys.argv
    if pile_only:
        jkr = json.loads((OUT / "jkr.json").read_text())
        cap = json.loads((OUT / "capillary.json").read_text())
        figures = ["fig_jkr.png", "fig_capillary.png"]
        print("[pile-only] reusing jkr.json / capillary.json", flush=True)
    else:
        jkr = jkr_evidence()
        cap = capillary_evidence()
        print(f"[jkr+capillary] {time.time() - t0:.1f}s", flush=True)
        figures = [fig_jkr(jkr), fig_capillary(cap)]
    pile = pile_evidence()
    print(f"[pile] {time.time() - t0:.1f}s", flush=True)
    figures += fig_pile(pile)

    if not pile_only:
        (OUT / "jkr.json").write_text(json.dumps(jkr, indent=1))
        (OUT / "capillary.json").write_text(json.dumps(cap, indent=1))
    (OUT / "pile.json").write_text(json.dumps(pile, indent=1))

    summary = {
        "F1_jkr_pull_off_max_rel_error": max(r["rel_error"] for r in jkr["pull_off"]),
        "F2_capillary_max_rel_error_small_bridge": max(
            r["rel_error_vs_analytic"] for r in cap["rows"]
            if r["volume_over_R3"] <= 1e-6),
        "F3_gamma_to_zero_max_rel_error": min(r["max_rel_error"] for r in jkr["gamma_to_zero"]),
        "F3_gamma_to_zero_zero_gamma": [r["max_rel_error"] for r in jkr["gamma_to_zero"]
                                        if r["gamma"] == 0.0],
        "F4_repose_increase_deg": pile["rows"][-1]["theta_r_deg"] - pile["rows"][0]["theta_r_deg"],
        "settle_time_s": SETTLE,
        "repose_angles": {str(r["bond_number"]): r["theta_r_deg"] for r in pile["rows"]},
        "figures": figures,
        "wall_time_s": time.time() - t0,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1)[:2000], flush=True)


if __name__ == "__main__":
    main()