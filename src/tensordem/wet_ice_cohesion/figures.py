"""Figures for the P35-N2 wet-ice cohesion campaign.

Reads ``artifacts/wet_ice_cohesion/campaign.json`` and writes PNGs next to it.
Nothing here computes physics: every plotted number comes from the campaign JSON
that :func:`tensordem.wet_ice_cohesion.study.run_all` produced.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import capillary  # noqa: E402


def _capillary_figure(camp, outdir, radius, gamma, f0):
    val = camp["capillary"]["splits"]["validate"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    gaps_over_r = np.linspace(0.0, 0.25, 60)
    for v, style in ((0.02, "-"), (0.05, "--"), (0.1, ":")):
        f = [capillary.capillary_force(radius, gamma, v * radius ** 3, g * radius) / f0
             for g in gaps_over_r]
        ax.plot(gaps_over_r, f, style, label=f"model, V/R$^3$={v}")
    for row in val:
        if row["theta_deg"] == 0.0 and row["gap_over_R"] > 0:
            ax.plot(row["gap_over_R"], row["force_exact_N"] / f0, "o", ms=5, mfc="none",
                    mec="k", alpha=0.8)
    ax.plot([], [], "o", mfc="none", mec="k", label="exact Young-Laplace")
    u = 0.05 ** (1 / 3)
    h = u * radius
    g = np.linspace(0.0, 0.4, 60) * radius
    ax.plot(g / radius,
            [capillary.seeded_campaign_force(radius, gamma, gg, h) / f0 for gg in g],
            "-.", color="crimson", label="seeded P35-N2 form (< 0: no attraction)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("separation s / R")
    ax.set_ylabel(r"$F / (2\pi\gamma R)$")
    ax.set_title("(a) bridge force: model vs exact reference vs seeded form")
    ax.set_ylim(-0.15, 1.0)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    peak = [r for r in val if r["gap_over_R"] == 0.0]
    xv = np.array([r["force_exact_N"] for r in peak])
    yv = np.array([r["force_model_N"] for r in peak])
    ax.plot(xv / f0, yv / f0, "s", color="tab:blue", label="held-out peak forces")
    lim = [0.7, 1.0]
    ax.plot(lim, lim, "k-", lw=0.8, label="1:1")
    err = float(np.max(np.abs(yv - xv) / xv) * 100)
    ax.set_xlabel(r"exact $F_{peak}/(2\pi\gamma R)$")
    ax.set_ylabel(r"model $F_{peak}/(2\pi\gamma R)$")
    ax.set_title(f"(b) peak-force parity, held-out max error {err:.2f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_capillary_bridge.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _aggregation_figure(camp, outdir):
    agg = camp["aggregation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    t = np.asarray(agg["simulation"]["times"])
    ax.plot(t, agg["simulation"]["mean_diameter"], "o-", label="DEM shear agglomeration")
    ax.plot(agg["smoluchowski"]["times"], agg["smoluchowski"]["mean_diameter"], "s--",
            label="Smoluchowski, shear kernel")
    for name, style in (("dt_half", ":"), ("n_double", "-.")):
        ax.plot(t, agg["convergence"][name]["mean_diameter"], style,
                label=f"convergence: {name}")
    ax.set_xlabel("time (units of 1/shear rate)")
    ax.set_ylabel(r"$\langle D\rangle / D_0$")
    ax.set_title(f"(a) aggregate growth, max deviation {100*agg['criterion_b_max_rel_error']:.1f} %")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ["start", "middle", "end"]
    for k, dist in enumerate(agg["size_distributions"]):
        ax.plot(dist["diameter_centres"], dist["number_fraction"], "o-",
                label=f"n(D), {labels[k]}: <D>={dist['mean_diameter']:.2f}")
    ax.set_xlabel(r"equivalent diameter $D/D_0$")
    ax.set_ylabel("number fraction")
    ax.set_title("(b) cluster size distribution n(D, t)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_agglomeration_smoluchowski.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _cohesion_figure(camp, outdir):
    coh = camp["cohesion"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    s = np.asarray(coh["wet"]["separation_history"])
    ax.plot(s, coh["wet"]["traction_history"], "-", label="wet (capillary bridges)")
    # the dry control is run with the same pull velocity, so the separation
    # grid is identical
    ax.plot(s, coh["dry"]["traction_history"], "--", label="dry ($\\gamma=0$)")
    ax.set_xlabel("wall separation / R")
    ax.set_ylabel("wall traction (DEM units)")
    ax.set_title("(a) tension cell: ${%.3f}$ wet vs ${%.3f}$ dry"
                 % (coh["wet"]["cohesion"], coh["dry"]["cohesion"]))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = ["DEM (measured)", "plane-cut ref.", "virial/Rumpf bound"]
    vals = [coh["wet"]["cohesion"], coh["reference"]["cohesion_plane_cut"],
            coh["reference"]["cohesion_virial_bound"]]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    ax.bar(names, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("cohesion (DEM units)")
    ax.set_title("(b) cohesion vs micro-mechanical reference\n"
                 f"F_enh = {coh['enhancement']['F_enh_dem']:.3f} (DEM) vs "
                 f"{coh['enhancement']['F_enh_reference']:.3f} (ref)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_cohesion_cell.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _enhancement_figure(camp, outdir):
    """Enhancement factor vs particle size: where capillary cohesion can matter."""
    from . import ice_resistance

    fig, ax = plt.subplots(figsize=(7, 4.4))
    radii = np.geomspace(5e-5, 2e-2, 60)
    gamma = ice_resistance.GAMMA_WATER_0C
    coordination = camp["cohesion"]["wet"]["coordination"]
    sigma_n = 1000.0   # Pa, representative confining stress of a rubble layer
    tan_phi = math.tan(math.radians(30.0))
    f_enh = []
    for r in radii:
        c = ice_resistance.capillary_cohesion(r, gamma, coordination, 0.02, 0.0)
        f_enh.append(ice_resistance.enhancement_factor(sigma_n * tan_phi + c, sigma_n * tan_phi))
    ax.semilogx(radii * 1e3, f_enh, "-", label=r"$F_{enh}=1+c_{cap}/(\sigma_n\tan\phi)$")
    r_star = math.sqrt(gamma / (917.0 * 9.81))
    ax.axvline(r_star * 1e3, color="k", ls="--", lw=0.8,
               label=f"Bond number = 1 (R = {r_star*1e3:.2f} mm)")
    ax.axhline(camp["cohesion"]["enhancement"]["F_enh_dem"], color="tab:red", ls=":",
               label=f"DEM 2D cell index {camp['cohesion']['enhancement']['F_enh_dem']:.2f}")
    ax.axhline(camp["cohesion"]["enhancement"]["timco_weeks"]["enhancement_factor"],
               color="tab:purple", ls="-.", label="Timco-Weeks relation branch")
    ax.set_xlabel("particle radius (mm)")
    ax.set_ylabel(r"$F_{wet}/F_{dry}$")
    ax.set_title(r"ice-resistance enhancement vs particle size ($\sigma_n$ = 1 kPa)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_enhancement_vs_size.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all(campaign_path: str = "artifacts/wet_ice_cohesion/campaign.json") -> list:
    with open(campaign_path) as fh:
        camp = json.load(fh)
    outdir = os.path.dirname(os.path.abspath(campaign_path))
    meta = camp["capillary"]["reference_meta"]
    radius = meta["particle_radius_m"]
    gamma = meta["surface_tension_N_per_m"]
    f0 = camp["capillary"]["f0_N"]
    return [
        _capillary_figure(camp, outdir, radius, gamma, f0),
        _aggregation_figure(camp, outdir),
        _cohesion_figure(camp, outdir),
        _enhancement_figure(camp, outdir),
    ]