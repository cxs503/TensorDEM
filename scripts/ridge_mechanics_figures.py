"""Figures for the IN100-N07 ridge mechanics report (from acceptance.json)."""
from __future__ import annotations

import base64
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join("artifacts", "ridge_mechanics")


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    with open(os.path.join(OUT, "acceptance.json")) as handle:
        data = json.load(handle)
    os.makedirs(OUT, exist_ok=True)
    paths = []

    # 1. ridge section before / after the crossing
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)
    for ax, key, title in ((axes[0], "ridge_particles_before", "ridge before crossing"),
                           (axes[1], "ridge_particles_after", "after crossing")):
        pts = np.array(data[key])
        ax.scatter(pts[:, 0], pts[:, 1], s=(pts[:, 2] * 140) ** 2, c="#8ab6d6",
                   edgecolors="#2b5d7d", linewidths=0.4)
        ax.axhline(0.0, color="#1f6f8b", lw=1.2, ls="--")
        ax.set_aspect("equal")
        ax.set_ylabel("z (m)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
    axes[1].set_xlabel("x (m)   [ship moves +x]")
    paths.append(_save(fig, "fig1_ridge_section.png"))

    # 2. hull force history with mode onsets
    hist = data["crossing"]["force_history"]
    t = [row["time"] for row in hist]
    f = [row["force_x"] / 1e6 for row in hist]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(t, f, color="#c0392b", lw=1.1)
    for mode, colour in (("crushing", "#34495e"), ("bending", "#8e44ad"),
                         ("ride_up", "#16a085"), ("clearing", "#d68910")):
        if mode in data["F6"]["phase_onsets"]:
            ax.axvline(data["F6"]["phase_onsets"][mode], color=colour, ls=":",
                       lw=1.0, label=f"{mode} onset")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("hull resistance F_x (MN/m)")
    ax.set_title("ship crossing: resistance and failure-mode onsets", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    paths.append(_save(fig, "fig2_force_history.png"))

    # 3. mode indicators (normalised)
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    for mode in ("crushing", "bending", "ride_up", "clearing"):
        series = np.array([row[mode] for row in data["crossing"]["mode_trace"]])
        if series.max() > 0:
            ax.plot(t[: len(series)], series / series.max(), lw=1.2, label=mode)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("indicator / peak (-)")
    ax.set_title("failure-mode indicators", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    paths.append(_save(fig, "fig3_mode_indicators.png"))

    # 4. sail height scaling versus Hopky + porosity
    sweep = data["scaling_sweep"]
    d = [row["squeeze"] for row in sweep]
    hs = [row["sail_height"] for row in sweep]
    hk = [0.4 * np.sqrt(1.6 * dd) for dd in d]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    axes[0].plot(d, hk, "s--", label="Hopky 0.4*sqrt(h d)", lw=1.2)
    axes[0].plot(d, hs, "o-", label="DEM (measured)", lw=1.2)
    axes[0].set_xlabel("convergence d (m)")
    axes[0].set_ylabel("sail height H_sail (m)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)
    axes[1].plot(d, [row["ratio"] for row in sweep], "o-", lw=1.2)
    ax2 = axes[1].twinx()
    ax2.plot(d, [row["porosity"] for row in sweep], "s--", color="#8e44ad", lw=1.0)
    axes[1].axhspan(3.0, 5.0, color="#2ecc71", alpha=0.15)
    axes[1].set_xlabel("convergence d (m)")
    axes[1].set_ylabel("D_keel / H_sail (-)")
    ax2.set_ylabel("porosity n (-)", color="#8e44ad")
    axes[1].set_title("accepted D/H band shaded", fontsize=8)
    axes[1].grid(alpha=0.25)
    paths.append(_save(fig, "fig4_scaling.png"))
    print("\n".join(paths))


if __name__ == "__main__":
    main()
