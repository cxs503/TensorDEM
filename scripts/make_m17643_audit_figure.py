#!/usr/bin/env python3
"""Board-audit figure for m17643: recomputed points, not the mission's own plots.

Reads the mission commit's results.json through git (zero hand-copy) and draws
(a) the spin-down sweep against the analytic (5/7) mu_r g reference with the
pre-declared 15% band, and (b) the pair-lever arms that carry the conversion factor.
"""
from __future__ import annotations

import json
import os
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMIT = "e1cdcab1"
RES = "artifacts/xval_dem_round2_method_selection/results.json"
OUT = os.path.join(REPO, "artifacts/xval_m17643_board_audit/fig1_audit_recompute.png")


def load() -> dict:
    blob = subprocess.run(["git", "-C", REPO, "cat-file", "blob", f"{COMMIT}:{RES}"],
                          check=True, capture_output=True).stdout
    return json.loads(blob)


def main() -> None:
    res = load()
    rows = res["spin_down"]["rows"]
    mu = [r["mu_r"] for r in rows]
    th = [r["t_stop_theory_s"] * 1e3 for r in rows]
    kern = [r["t_stop"]["kernel"] * 1e3 for r in rows]
    oss = [r["t_stop"]["oss_epsd2"] * 1e3 for r in rows]
    cdt = [r["t_stop"]["oss_cdt"] * 1e3 for r in rows]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.fill_between(mu, [t * 0.85 for t in th], [t * 1.15 for t in th],
                   color="0.85", label="analytic ±15% (pre-declared gate)")
    a.plot(mu, th, "k--", lw=1.2, label="analytic v0/((5/7) mu_r g)")
    a.plot(mu, kern, "o-", label="TensorDEM kernel")
    a.plot(mu, oss, "s-", ms=4, label="LIGGGHTS EPSD2 transcription")
    a.plot(mu, cdt, "^--", ms=4, lw=0.9, label="LIGGGHTS CDT transcription")
    a.set_yscale("log")
    a.set_xlabel(r"$\mu_r$ (TensorDEM lever = R)")
    a.set_ylabel("spin-down stop time [ms]")
    a.set_title("E2: kernel vs analytic vs OSS — audit recomputation")
    a.annotate("mu_r=0.8: +17.63% vs analytic\n(mu_r>mu_s sliding saturation,\nall three laws agree)",
               xy=(0.8, 63.0), xytext=(0.24, 30), fontsize=7.5,
               arrowprops=dict(arrowstyle="->", lw=0.8))
    a.legend(fontsize=6.5, loc="upper right")
    a.grid(alpha=0.3)

    b = ax[1]
    pair = res["pair_lever"]
    arms = ["kernel_reff_r", "oss_epsd2_reff_r2", "oss_cdt_reff_r2",
            "kernel_mapped_mu_half"]
    labels = ["kernel\n$r_{eff}$=R, $\\mu_r$=0.5", "OSS EPSD2\n$r_{eff}$=R/2",
              "OSS CDT\n$r_{eff}$=R/2", "kernel mapped\n$r_{eff}$=R, $\\mu_r$/2"]
    vals = [pair["t_half"][k] * 1e3 for k in arms]
    th_pair = [pair["t_half_theory"][k] * 1e3 for k in arms]
    x = range(len(arms))
    b.bar([i - 0.2 for i in x], vals, width=0.4, label="measured")
    b.bar([i + 0.2 for i in x], th_pair, width=0.4, label="closed form 0.25 w0 I/(mu_r reff Fn)")
    b.set_xticks(list(x))
    b.set_xticklabels(labels, fontsize=6.5)
    b.set_ylabel("pair spin half-life [ms]")
    b.set_title("E3: lever arm 2x ⇒ $\\mu_{r,OSS}$ = 2 $\\mu_{r,TD}$ (audit: ratio 2.000)")
    b.legend(fontsize=7)
    b.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print(OUT)


if __name__ == "__main__":
    main()