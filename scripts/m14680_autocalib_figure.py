"""Figure of the GAP-DEM-AUTOCALIB round-1 diagnosis (reads the artefact).

    PYTHONPATH=src python scripts/m14680_autocalib_figure.py

Writes ``artifacts/gap_dem_autocalib/fig1_repose_diagnosis.png`` (θ_r vs μ with
the replica noise band and the literature band of dry sand, plus the residual
kinetic-energy state of the two termination rules).
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACT = os.path.join("artifacts", "gap_dem_autocalib", "diagnosis.json")
PNG = os.path.join("artifacts", "gap_dem_autocalib", "fig1_repose_diagnosis.png")


def main() -> int:
    d = json.load(open(ARTIFACT, encoding="utf-8"))
    p1, p2, p3 = d["p1_friction_response"], d["p2_replica_noise"], d["p3_rest_criterion"]
    s = d["summary"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.3))

    mu = [r["mu"] for r in p1]
    th = [r["theta_r_deg"] for r in p1]
    ax.plot(mu, th, "o-", color="#1f77b4", lw=1.6, ms=6,
            label=r"DEM pluviation heap, $\theta_r(\mu)$")
    rep = [r["theta_r_deg"] for r in p2]
    ax.axhspan(min(rep), max(rep), color="#d62728", alpha=0.18,
               label=f"replica spread at fixed params ({min(rep):.1f}-{max(rep):.1f}$^\\circ$)")
    ax.axhline(34.0, color="k", ls=":", lw=1.2, label="literature dry sand $34^\\circ$")
    lo, hi = s["literature_theta_band_deg"]
    ax.axhspan(lo, hi, color="0.75", alpha=0.35,
               label=f"literature band {lo:.0f}-{hi:.0f}$^\\circ$")
    ax.set_xlabel(r"sliding friction $\mu$ [-]")
    ax.set_ylabel(r"angle of repose $\theta_r$ [deg]")
    ax.set_title(f"forward map is non-monotone and cannot reach literature\n"
                 f"Spearman={s['spearman_mu_theta']:.2f}, "
                 f"signal {s['signal_span_deg']:.1f}$^\\circ$ vs noise "
                 f"{s['replica_noise_span_deg']:.1f}$^\\circ$ (N={s['n_particles']})")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    labels, vals = [], []
    for r in p3["rows"]:
        labels += [f"$\\mu$={r['mu']:.2f}\nfixed window",
                   f"$\\mu$={r['mu']:.2f}\n$k_e$ stop"]
        vals += [r["fixed_window"]["ke_ratio"], r["ke_stop_window"]["ke_ratio"]]
    cols = ["#7f7f7f", "#2ca02c"] * len(p3["rows"])
    bx.bar(range(len(vals)), vals, color=cols, width=0.7)
    bx.axhline(0.02, color="k", ls="--", lw=1.0, label="rest threshold 2%")
    bx.set_xticks(range(len(labels)))
    bx.set_xticklabels(labels, fontsize=7)
    bx.set_ylabel(r"residual $K$ / released PE [-]")
    bx.set_title("$\\theta_r$ of the settled column collapse: 0.8-4.5$^\\circ$\n"
                 "(deposit is a monolayer, not a heap)")
    bx.legend(fontsize=8)
    bx.grid(alpha=0.3, axis="y")

    fig.suptitle("GAP-DEM-AUTOCALIB round 1 — why the literature-anchored calibration "
                 "cannot be closed with the shipped forward model", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(PNG), exist_ok=True)
    fig.savefig(PNG, dpi=140)
    print(f"wrote {PNG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())