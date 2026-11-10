"""Figures for the weakest-link bond campaign.

Reads ``artifacts/ice_bond_weakest_link/verification.json`` (the only source of
numbers) and writes three PNGs next to it:

* ``chain_strength_ecdf.png``   -- empirical CDF of the chain strength vs the
  analytic Weibull weakest-link CDF, for N = 1 ... 64;
* ``size_effect.png``           -- log-log median chain strength vs N with the
  analytic ``N**(-1/m)`` line and the dynamic-ensemble medians;
* ``quasi_static.png``          -- ramp-duration study: load error and bond-force
  spread against the ramp duration T.

Usage
-----
    PYTHONPATH=src python scripts/make_weakest_link_figures.py
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensordem.ice_bond_viscoelastic import weibull_bond_strengths

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "ice_bond_weakest_link"


def main() -> int:
    payload = json.loads((ART / "verification.json").read_text())
    material = payload["material"]
    modulus = material["weibull_modulus"]
    scale = payload["analytic_references"]["bond_strength_scale_N"]
    median_force = material["bond_stiffness_N_m"] * material["median_breaking_strain"] * material["rest_length_m"]

    # --- 1. ECDF of the chain strength -----------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    samples = payload["arm_ecdf"]["samples"]
    generator = torch.Generator().manual_seed(20260101)
    colours = plt.cm.viridis([i / 6 for i in range(7)])
    for index, row in enumerate(payload["arm_ecdf"]["rows"]):
        n_bonds = row["n_bonds"]
        draws = weibull_bond_strengths(samples * n_bonds, 1.0, modulus, generator=generator)
        strength = draws.reshape(samples, n_bonds).min(dim=1).values * median_force
        ordered, _ = torch.sort(strength)
        empirical = torch.arange(1, samples + 1, dtype=torch.float64) / samples
        ax.plot(ordered.numpy(), empirical.numpy(), lw=1.0, color=colours[index],
                label=f"N={n_bonds} (KS={row['ks_statistic']:.3f})")
        grid = torch.linspace(float(ordered[0]), float(ordered[-1]), 256, dtype=torch.float64)
        analytic = 1.0 - torch.exp(-n_bonds * (grid / scale).pow(modulus))
        ax.plot(grid.numpy(), analytic.numpy(), "--", lw=1.0, color=colours[index])
    ax.set_xlabel("chain strength [N]")
    ax.set_ylabel("empirical CDF  (dashed = analytic weakest link)")
    ax.set_title("Chain strength vs Weibull weakest-link CDF (m = %g)" % modulus)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ART / "chain_strength_ecdf.png", dpi=150)
    plt.close(fig)

    # --- 2. size effect ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    rows = payload["arm_ecdf"]["rows"]
    sizes = [row["n_bonds"] for row in rows]
    analytic = [row["median_analytic_N"] for row in rows]
    measured = [row["median_measured_N"] for row in rows]
    ax.loglog(sizes, analytic, "k--", lw=1.2, label=r"analytic $F_0(\ln 2/N)^{1/m}$")
    ax.loglog(sizes, measured, "o", ms=5, label="static (min of N draws, R=%d)" % samples)
    per_size = payload["arm_dynamic"]["per_size"]
    dynamic_sizes = sorted(int(k) for k in per_size)
    ax.loglog(dynamic_sizes, [per_size[str(n)]["median_load_N"] for n in dynamic_sizes],
              "s", ms=6, color="tab:red",
              label="dynamic engine (load at first failure)")
    fit = payload["arm_dynamic"]["size_effect_fit_all_points"]
    ax.set_xlabel("chain length N [bonds]")
    ax.set_ylabel("median chain strength [N]")
    ax.set_title("Weakest-link size effect: fitted slope %.3f (analytic %.3f)"
                 % (fit["fitted_slope"], fit["analytic_slope"]))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "size_effect.png", dpi=150)
    plt.close(fig)

    # --- 3. quasi-static convergence --------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    rate_rows = payload["arm_rate"]["rows"]
    durations = [row["ramp_duration_s"] for row in rate_rows]
    ax.plot(durations, [abs(row["rel_error"]) * 100 for row in rate_rows], "o-",
            label="|load error| [%]")
    ax.plot(durations, [row["force_spread"] * 100 for row in rate_rows], "s-",
            label="bond-force spread [%]")
    ax.set_xscale("log")
    ax.set_xlabel("ramp duration T [s]  (N = 4 chain)")
    ax.set_ylabel("percent")
    ax.set_title("Quasi-static convergence of the dynamic engine")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "quasi_static.png", dpi=150)
    plt.close(fig)

    print("wrote", *(p.name for p in sorted(ART.glob("*.png"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())