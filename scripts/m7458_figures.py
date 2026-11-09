"""m7458 campaign figures: wet-ice capillary bridge cohesion.

Generates PNG figures for the campaign report from the test results.
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordem.wet_ice_cohesion import aggregation, capillary
from tensordem.wet_ice_cohesion.lian_reference import (
    lian_bridge_force,
    lian_rupture_distance,
)

# Constants
R_ICE = 1.0e-3
GAMMA_WATER = 0.072
THETA_ICE = 0.0
V_L_FRACTIONS = [0.01, 0.05, 0.10]
D_F_TARGET = 2.5
N_PARTICLES_TEST = 200
SHEAR_RATE = 1.0
TOTAL_STRAIN_TEST = 1.0


def _bridge_volume(vl_frac, radius=R_ICE):
    return vl_frac * (4.0 / 3.0) * math.pi * radius ** 3


def make_figures(outdir: str = "artifacts/m7458") -> list:
    """Generate all campaign figures."""
    os.makedirs(outdir, exist_ok=True)
    paths = []

    # Figure 1: Capillary bridge force vs gap (Lian model)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    gaps = np.linspace(0, 0.5e-3, 100)
    for vl_frac, style in zip(V_L_FRACTIONS, ["-", "--", ":"]):
        v_bridge = _bridge_volume(vl_frac)
        f_vals = [lian_bridge_force(R_ICE, GAMMA_WATER, v_bridge, g, THETA_ICE) for g in gaps]
        ax.plot(gaps * 1e3, f_vals, style, label=f"V_l/V = {vl_frac}")
    ax.set_xlabel("Gap s (mm)")
    ax.set_ylabel("Capillary force F (N)")
    ax.set_title("Lian et al. (1998) liquid-bridge force vs separation")
    ax.legend()
    ax.grid(alpha=0.3)
    path1 = os.path.join(outdir, "fig1_lian_bridge_force.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    paths.append(path1)

    # Figure 2: Aggregation <D> vs time (DEM vs Smoluchowski)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for vl_frac, style in zip(V_L_FRACTIONS, ["o-", "s--", "^:"]):
        res = aggregation.simulate_shear_aggregation(
            n_particles=N_PARTICLES_TEST, shear_rate=SHEAR_RATE,
            total_strain=TOTAL_STRAIN_TEST, dt_fraction=0.01, seed=42,
            dimension_fractal=D_F_TARGET, sticking_probability=1.0 if vl_frac > 0 else 0.0,
        )
        volume = res.params["box"] ** 3
        sm = aggregation.smoluchowski_shear(
            n_particles=N_PARTICLES_TEST, volume=volume, shear_rate=SHEAR_RATE,
            times=res.times, radius_primary=1.0, n_classes=40,
            dimension_fractal=D_F_TARGET,
        )
        ax.plot(res.times, res.mean_diameter, style, label=f"DEM V_l={vl_frac}")
        ax.plot(sm["times"], sm["mean_diameter"], style.replace("o", "").replace("s", "").replace("^", "") + "k",
                alpha=0.5, label=f"Smoluchowski V_l={vl_frac}")
    ax.set_xlabel("Time (1/shear_rate)")
    ax.set_ylabel(r"$\langle D\rangle / D_0$")
    ax.set_title("Aggregate growth: DEM vs Smoluchowski mean-field")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    path2 = os.path.join(outdir, "fig2_aggregation_growth.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    paths.append(path2)

    # Figure 3: Enhancement factor vs V_l
    fig, ax = plt.subplots(figsize=(6, 4))
    f_vals = []
    for vl_frac in V_L_FRACTIONS:
        F_contact = 2.0 * math.pi * GAMMA_WATER * R_ICE
        Z = 6.0
        area = math.pi * R_ICE ** 2
        c_cap = Z * vl_frac * F_contact / area
        sigma_n = 100.0
        tan_phi = math.tan(math.radians(30.0))
        tau_dry = sigma_n * tan_phi
        tau_wet = tau_dry + c_cap
        f_vals.append(tau_wet / tau_dry)
    ax.plot(V_L_FRACTIONS, f_vals, "o-", color="tab:blue")
    ax.set_xlabel("Liquid volume fraction V_l/V_total")
    ax.set_ylabel("Enhancement factor f(V_l)")
    ax.set_title("Ice-resistance enhancement factor vs liquid content")
    ax.grid(alpha=0.3)
    path3 = os.path.join(outdir, "fig3_enhancement_factor.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    paths.append(path3)

    # Figure 4: Cluster size distribution at steady state
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for vl_frac, style in zip(V_L_FRACTIONS, ["-", "--", ":"]):
        res = aggregation.simulate_shear_aggregation(
            n_particles=N_PARTICLES_TEST, shear_rate=SHEAR_RATE,
            total_strain=TOTAL_STRAIN_TEST, dt_fraction=0.01, seed=42,
            dimension_fractal=D_F_TARGET, sticking_probability=1.0 if vl_frac > 0 else 0.0,
        )
        masses = res.snapshots[-1]
        d_equiv = 2.0 * masses ** (1.0 / D_F_TARGET)
        counts, edges = np.histogram(d_equiv, bins=15)
        centres = 0.5 * (edges[1:] + edges[:-1])
        ax.plot(centres, counts / max(1, len(masses)), style,
                label=f"V_l={vl_frac}: <D>={np.mean(d_equiv):.2f}")
    ax.set_xlabel(r"Equivalent diameter $D/D_0$")
    ax.set_ylabel("Number fraction")
    ax.set_title("Cluster size distribution n(D) at steady state")
    ax.legend()
    ax.grid(alpha=0.3)
    path4 = os.path.join(outdir, "fig4_size_distribution.png")
    fig.savefig(path4, dpi=150)
    plt.close(fig)
    paths.append(path4)

    return paths


if __name__ == "__main__":
    paths = make_figures()
    for p in paths:
        print(f"Generated: {p}")
