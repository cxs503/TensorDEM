#!/usr/bin/env python3
"""W28 figures for papers/artifacts/w28/ice-ridge-keel-grounding/dual_convergence.json.

Reads the artifact (no numbers are retyped) and writes four PNGs next to it:

  fig1_keel_law.png          keel pressure-settlement laws of both bond-law arms
  fig2_particle_axis.png     R, A_c, q_max vs particle count, both arms + Richardson band
  fig3_mesh_axis.png         R, A_c, q_max vs beam mesh size, both arms (observed order)
  fig4_contact_and_spearman.png  contact pressure profiles (three meshes) + R(w) ramp
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = os.path.join("papers", "artifacts", "w28", "ice-ridge-keel-grounding",
                   "dual_convergence.json")
OUT = os.path.dirname(ART)


def main() -> int:
    with open(ART) as fh:
        a = json.load(fh)
    os.makedirs(OUT, exist_ok=True)
    arms = list(a["configuration"]["arms"])
    counts = a["configuration"]["particle_counts"]
    meshes = a["configuration"]["mesh_elements"]
    colors = {arms[0]: "tab:red", arms[1]: "tab:blue"}
    labels = {arms[0]: "legacy bond law (W27 basis)",
              arms[1]: "spacing-consistent bond law"}

    # ---------------- fig 1: keel pressure-settlement laws ----------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for res in a["dem_resolutions"]:
        w = np.array(res["confined"]["settlements_m"])
        p = np.array(res["confined"]["pressures_Pa"])
        ax.plot(w * 1e3, p / 1e6, "o-", ms=3, lw=1.2, color=colors[res["arm"]],
                alpha=0.45 + 0.15 * (counts.index(res["n_particles_requested"])))
        ax.annotate("%s / %dk" % (res["arm"].split("_")[0], res["n_particles_requested"] // 1000),
                    (w[-1] * 1e3, p[-1] / 1e6), fontsize=6,
                    xytext=(3, 2), textcoords="offset points", color=colors[res["arm"]])
    ax.set_xlabel("keel settlement $w$ [mm]")
    ax.set_ylabel("keel pressure $p$ [MPa]")
    ax.set_title("W28 keel pressure–settlement law (confined DEM test)\n"
                 "legacy: $k\\propto 1/a$ (unbounded with $N$); consistent: $k\\approx E/\\Lambda$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_keel_law.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig 2: particle axis ----------------
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    keys = [("R_N_per_m", "$R$ [N/m]", 1.0), ("A_c_m2_per_m", "$A_c$ [m$^2$/m]", 1.0),
            ("q_max_Pa", "$q_{max}$ [Pa]", 1.0)]
    for ax, (key, ylab, scale) in zip(axes, keys):
        for arm in arms:
            vals = [c[key] for c in a["cases"]
                    if c["arm"] == arm and c["n_elements"] == meshes[-1]]
            ax.plot(counts, np.array(vals) * scale, "o-", color=colors[arm], label=labels[arm])
            lim = a["convergence"][arm]["particle_axis"][key]["richardson_limit"]
            band = a["convergence"][arm]["particle_axis"][key]["uncertainty_band_rel"]
            if lim is not None:
                ax.axhline(lim * scale, color=colors[arm], ls="--", lw=1)
                if band:
                    ax.axhspan(lim * (1 - band) * scale, lim * (1 + band) * scale,
                               color=colors[arm], alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel("particle count $N$ (mesh fixed at $h/4$)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
        ax.set_title(key.split("_")[0] + " vs particles")
    axes[0].legend(fontsize=7)
    fig.suptitle("W28 particle axis: dashed = Richardson limit, band = 1.25 safety band", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_particle_axis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- fig 3: mesh axis ----------------
    h = [a["configuration"]["beam_length_m"] / ne for ne in meshes]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (key, ylab, scale) in zip(axes, keys):
        for arm in arms:
            vals = [c[key] for c in a["cases"]
                    if c["arm"] == arm and c["n_particles_requested"] == counts[-1]]
            ax.plot(h, np.array(vals) * scale, "s-", color=colors[arm], label=labels[arm])
            tri = a["convergence"][arm]["mesh_axis"][key]["triads"].get("finest", {})
            lim = tri.get("richardson_limit")
            p = tri.get("observed_order_in_spacing")
            if lim is not None:
                ax.axhline(lim * scale, color=colors[arm], ls="--", lw=1)
                ax.annotate("$p_a$=%.2f" % p, (h[0], lim * scale), fontsize=7,
                            color=colors[arm], xytext=(2, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("beam element size $h$ [m] (particles fixed at 200 k)")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3, which="both")
        ax.set_title(key.split("_")[0] + " vs mesh")
    axes[0].legend(fontsize=7)
    fig.suptitle("W28 mesh axis (uniform family, $h$, $h/2$, $h/4$)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_mesh_axis.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------------- fig 4: contact pressure + penetration ramp ----------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    ax = axes[0]
    grad = a["extra_evidence"]["graded_mesh_diagnostic"]
    ax.set_title("contact pressure across the structure\n(legacy arm, 200 k particles, $w_0$=8 mm)")
    ax.set_xlabel("$x$ along the structure [m]")
    ax.set_ylabel("contact pressure $p$ [MPa]")
    ax.grid(alpha=0.3)
    for c in a["cases"]:
        if c["arm"] == arms[0] and c["n_particles_requested"] == counts[-1]:
            prof = c.get("contact_pressure_profile")
            if prof is None:
                continue
            ax.plot(prof["x_m"], np.array(prof["p_Pa"]) / 1e6, lw=1.0,
                    label="h = %.3f m" % c["element_length_m"])
    ax.legend(fontsize=7)
    ax = axes[1]
    for c in a["cases"]:
        w = [s["settlement_m"] * 1e3 for s in c["settlement_ramp"]]
        f = [s["resistance_N_per_m"] / 1e6 for s in c["settlement_ramp"]]
        ax.plot(w, f, "-", lw=0.9, color=colors[c["arm"]],
                alpha=0.35 + 0.1 * meshes.index(c["n_elements"]))
    sp = a["spearman"]
    ax.set_title("penetration vs contact force (all %d cases)\nSpearman $\\rho_{min}$=%.4f"
                 % (len(a["cases"]), sp["min_rho_per_case"]))
    ax.set_xlabel("platen settlement $w$ [mm]")
    ax.set_ylabel("keel resistance $R$ [MN/m]")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_contact_and_spearman.png"), dpi=150)
    plt.close(fig)
    print("figures written to", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
