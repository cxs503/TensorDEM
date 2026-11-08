"""Figures for the non-spherical representation verdict (campaign P27W2-N07).

Every curve is read from the JSON fragments written by the campaign script, so
the figures cannot drift away from the reported numbers.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Dict, List

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(path: str):
    with open(path) as fh:
        return json.load(fh)


def make_figures(art_dir: str, fig_dir: str) -> List[str]:
    os.makedirs(fig_dir, exist_ok=True)
    made: List[str] = []

    hertz = _load(os.path.join(art_dir, "hertz_anchors.json"))
    geom = _load(os.path.join(art_dir, "geom_convergence.json"))

    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "src"))
    from tensordem.shape_representation.hertz_ellipsoid import ellipsoid_plane_contact

    # ---- fig 1: Hertz stiffness and its verification residual --------------
    ratios = np.linspace(0.1, 1.0, 60)
    k = []
    for r in ratios:
        c = ellipsoid_plane_contact(1.0 / r, 1.0, 1.0e9, load=1.0)
        k.append(c.stiffness / 1.0e6)     # MN/m
    rows = hertz["rows"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=170)
    ax[0].plot(ratios, k, "-", lw=2, color="#1f4e9c",
               label="ellipsoid/plane Hertz (closed form)")
    ax[0].plot([r["curvature_ratio_B_over_A"] for r in rows],
               [r["stiffness_closed_form_n_per_m"] / 1e6 for r in rows],
               "o", ms=5, mfc="none", color="#c0392b", label="anchor points (C1 table)")
    ax[0].set_xlabel(r"curvature ratio $B/A = R_{sharp}/R_{flat}$")
    ax[0].set_ylabel(r"$k_n$  [MN/m]   ($E^*=1$ GPa, $P=1$ N)")
    ax[0].set_title("(a) ellipsoid-on-plane normal stiffness\nsphere limit $k_n=2E^*a$ reproduced exactly",
                    fontsize=9)
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    ax[1].semilogy([r["curvature_ratio_B_over_A"] for r in rows],
                   [max(r["stiffness_rel_error"], 1e-16) for r in rows], "o-",
                   color="#1e8449", label="closed form vs integral equation")
    ax[1].semilogy([r["curvature_ratio_B_over_A"] for r in rows],
                   [max(r["flatness_spread_rel"], 1e-16) for r in rows], "s--",
                   color="#7d3c98", label=r"residual spread of $w+g$  (flatness)")
    ax[1].axhline(2e-2, color="#c0392b", ls=":", lw=1.4, label="C1 tolerance 2%")
    ax[1].set_xlabel(r"curvature ratio $B/A$")
    ax[1].set_ylabel("relative error")
    ax[1].set_title("(b) verification: independent quadrature of Love's equation", fontsize=9)
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(fig_dir, "nonspherical_hertz_anchor.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    # ---- fig 2: superellipsoid pole power law ------------------------------
    laws = hertz["power_law_indenter"]
    ms = [l["power_m"] for l in laws]
    alpha = [l["numerical_force_exponent"] for l in laws]
    fig, ax = plt.subplots(figsize=(5.6, 3.9), dpi=170)
    mm = np.linspace(1.0, 8.5, 200)
    ax.plot(mm, 1.0 + 1.0 / mm, "-", color="#c0392b", lw=1.6,
            label=r"analytic $1+1/m$")
    ax.plot(ms, alpha, "o", ms=7, color="#1f4e9c", label="Love-equation solve (numerical)")
    for m_, a_ in zip(ms, alpha):
        ax.annotate(f"m={m_:g}", (m_, a_), textcoords="offset points", xytext=(4, -10),
                    fontsize=7)
    ax.set_xlabel("superellipsoid / indenter exponent $m$")
    ax.set_ylabel(r"force exponent $\alpha$  ($P \sim \delta^{\alpha}$)")
    ax.set_title("superellipsoid pole contact law\nclump = Hertz ($\\alpha$=1.5) for every shape",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(fig_dir, "nonspherical_pole_exponent.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    # ---- fig 3: clump convergence ----------------------------------------
    fig, ax = plt.subplots(figsize=(5.8, 4.0), dpi=170)
    colors = {"m4": "#1f4e9c", "m8": "#c0392b"}
    for key, block in geom["shapes"].items():
        rows = block["rows"]
        ns = [r["n_spheres"] for r in rows]
        ax.loglog(ns, [r["covering"]["max_distance_ratio"] for r in rows], "o-",
                  color=colors.get(key, "k"), label=f"covering fit, {key}")
        ax.loglog(ns, [r["kmeans"]["max_distance_ratio"] for r in rows], "s--",
                  color=colors.get(key, "k"), alpha=0.55, label=f"k-means fit, {key}")
    ns = np.array([8, 256])
    ax.loglog(ns, 2.3 * ns ** -0.5, ":", color="#555555", lw=1.2,
              label=r"$N^{-1/2}$ reference (surface covering)")
    ax.axhline(0.05, color="#1e8449", ls=":", lw=1.2, label="5% bulk tolerance")
    ax.axhline(0.01, color="#7d3c98", ls=":", lw=1.2, label="1% contact-critical tolerance")
    ax.set_xlabel("number of spheres in the clump $N$")
    ax.set_ylabel("max two-sided surface deviation\n/ smallest half-extent")
    ax.set_title("clump representation error vs N (criterion C2, geometry half)", fontsize=9)
    ax.legend(fontsize=6.4, ncol=1)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(fig_dir, "nonspherical_clump_convergence.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    # ---- fig 4: verdict map ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=170)
    for key, block in geom["shapes"].items():
        rows = block["rows"]
        ns = np.array([r["n_spheres"] for r in rows], dtype=float)
        es = np.array([r["covering"]["max_distance_ratio"] for r in rows], dtype=float)
        p_exp = block["convergence_exponent_covering"]
        n_fine = np.geomspace(ns.min(), 4096, 200)
        e_fine = es[0] * (n_fine / ns[0]) ** p_exp
        ax.loglog(n_fine, e_fine, "-", color=colors.get(key, "k"), lw=1.4,
                  label=f"covering fit, {key} (exp {p_exp:.2f})")
        ax.loglog(ns, es, "o", color=colors.get(key, "k"), ms=6)
    ax.axhspan(0, 0.01, color="#1e8449", alpha=0.12, label="contact-critical zone (<=1%)")
    ax.axhspan(0.01, 0.05, color="#f1c40f", alpha=0.15, label="bulk admissible zone (<=5%)")
    for nn in (32, 128, 1024):
        ax.axvline(nn, color="#999999", ls=":", lw=0.9)
    ax.set_xlabel("N (spheres per particle)")
    ax.set_ylabel("surface deviation ratio")
    ax.set_title("clump admissibility map (measured trend extended by the fitted exponent)", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = os.path.join(fig_dir, "nonspherical_verdict_map.png")
    fig.savefig(p)
    plt.close(fig)
    made.append(p)

    return made


if __name__ == "__main__":
    import sys
    art = sys.argv[1] if len(sys.argv) > 1 else "artifacts/shape_representation"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs/figures"
    print(make_figures(art, out))