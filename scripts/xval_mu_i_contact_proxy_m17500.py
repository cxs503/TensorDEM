"""XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE -- cross-validation run (mission m17500).

Compares, on one and the same steady-shear state of the repository's
constant-stress servo shear cell:

  REF  bulk route   : I = gdot_measured*d/sqrt(P/rho) (Love-Weber slab P,
                      measured shear-rate profile), mu = tau/P
  GOT  contact proxy: I_c = |v_t|/sqrt(f_n/(rho d^2)) per live contact,
                      mu(I_c) with the Jop (2006) constants, f_n-weighted

Outputs (all inside the worktree):
  artifacts/xval_mu_i_contact_proxy_m17500/results.json
  artifacts/xval_mu_i_contact_proxy_m17500/fig_*.png

Run:  PYTHONPATH=src python3 scripts/xval_mu_i_contact_proxy_m17500.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from tensordem.mu_i_rheology import ServoShearCase, simulate_servo_shear  # noqa: E402
from tensordem.mu_i_contact_proxy import (  # noqa: E402
    aggregate_proxy, contact_proxy_stats, xval_row, xval_summary, DEFAULT_TOL,
)

OUT = os.path.join(HERE, "artifacts", "xval_mu_i_contact_proxy_m17500")
os.makedirs(OUT, exist_ok=True)


def run_case(label: str, *, inertial_number: float, normal_stress: float = 2.0e3,
             kn: float = 1.0e3, seed: int = 5279) -> dict:
    case = ServoShearCase(inertial_number=inertial_number,
                          normal_stress=normal_stress, kn=kn, seed=seed,
                          contact_proxy=True, bulk_stress=True)
    out = simulate_servo_shear(case)
    proxy = out["contact_proxy"]
    tail = aggregate_proxy(out.get("contact_proxy_series", []), frac=0.6)
    row = xval_row(label=label,
                   inertial_number_bulk=out["inertial_number_bulk"],
                   mu_bulk=out["bulk_mu"],
                   proxy=proxy,
                   mu_contact=case.mu)
    row.update({
        "kn": kn, "normal_stress_pa": normal_stress,
        "i_target": inertial_number,
        "i_wall_measured": out["inertial_number"],
        "mu_wall_measured": out["mu_effective"],
        "i_proxy_tail60": tail["i_fn_weighted"],
        "mu_proxy_tail60": tail["mu_fn_weighted"],
        "shear_rate_measured": out["shear_rate_measured"],
        "coordination": out["coordination_number"],
        "phi": out["solid_fraction"],
        "servo_valid": out["servo_valid"],
        "pressure_control_error_final": out["pressure_control_error_final"],
        "kappa": out.get("kappa"),
        "dt": out["dt"],
        "shear_steps": out["shear_steps_executed"],
        "wall_clock_s": out["wall_clock_s"],
        "n_proxy_samples": len(out.get("contact_proxy_series", [])),
    })
    print(f"[{label}] I_bulk={row['i_bulk_measured']:.4f} mu_bulk={row['mu_bulk_measured']:.4f} "
          f"I_c={row['i_proxy_fn_weighted']:.4f} mu_proxy={row['mu_proxy_fn_weighted']:.4f} "
          f"i_err={row['i_mapping_error']:+.1%} mu_err={row['mu_error']:+.1%} "
          f"shortfall={row['shortfall']:+.1%} ({row['wall_clock_s']:.0f}s)", flush=True)
    return row


def main() -> int:
    t0 = time.time()
    rows = []
    # ---- base sweep: dense/slow band -> collisional band
    for i_t in (0.01, 0.1, 0.6):
        rows.append(run_case(f"base_I{i_t:g}", inertial_number=i_t))
    # ---- sensitivity S1: confining pressure (2 kPa -> 8 kPa) at I = 0.1
    rows.append(run_case("sens_P8k_I0.1", inertial_number=0.1,
                         normal_stress=8.0e3))
    # ---- sensitivity S2: contact stiffness (kn x4 -> kappa x4) at I = 0.1
    rows.append(run_case("sens_kn4e3_I0.1", inertial_number=0.1, kn=4.0e3))

    base = [r for r in rows if r["label"].startswith("base_")]
    summary = xval_summary(base, tol=DEFAULT_TOL)
    summary_all = xval_summary(rows, tol=DEFAULT_TOL)
    # sensitivity: how much do the two cross-validation metrics drift?
    sens = {}
    ref = next(r for r in base if "0.1" in r["label"])
    for r in rows:
        if r["label"].startswith("sens_"):
            sens[r["label"]] = {
                "i_mapping_error_delta": r["i_mapping_error"] - ref["i_mapping_error"],
                "mu_error_delta": r["mu_error"] - ref["mu_error"],
                "i_bulk_ratio": r["i_bulk_measured"] / ref["i_bulk_measured"],
                "i_proxy_ratio": r["i_proxy_fn_weighted"] / ref["i_proxy_fn_weighted"],
                "shortfall": r["shortfall"],
            }
    # window convergence (numerical sensitivity, same runs)
    window = [{"label": r["label"],
               "i_proxy_tail60_vs_final20": (r["i_proxy_tail60"]
                                             / r["i_proxy_fn_weighted"] - 1.0),
               "mu_proxy_tail60_vs_final20": (r["mu_proxy_tail60"]
                                              / r["mu_proxy_fn_weighted"] - 1.0)}
              for r in rows]

    payload = {
        "mission": 17500,
        "anchor": "XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE",
        "ref_source": ("oss_solutions id 421 (tag mu-i-contact-proxy-negative, "
                       "repo TensorDEM, feature_slug CASE-COLUMN-COLLAPSE); "
                       "law: Jop/Forterre/Pouliquen 2006 Nature 441:727-730; "
                       "contact gauge: src/tensordem/granular_column/inertial_friction.py"),
        "tolerance": DEFAULT_TOL,
        "rows": rows,
        "summary_base_sweep": summary,
        "summary_all_runs": summary_all,
        "sensitivity": sens,
        "window_convergence": window,
        "wall_clock_total_s": time.time() - t0,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    # ---------------- figures (matplotlib, Agg) ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("matplotlib unavailable:", exc)
        return 0

    b = sorted(base, key=lambda r: r["i_bulk_measured"])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ii = np.logspace(-2.2, 0.2, 60)
    ax.loglog(ii, [0.36 + 0.48 * i / (0.30 + i) for i in ii], "k--",
              label="Jop 2006 mu(I) (0.36/0.84/0.30)")
    ax.loglog([r["i_bulk_measured"] for r in b], [r["mu_bulk_measured"] for r in b],
              "o-", color="#1f77b4", label="REF bulk route (tau/P, Love-Weber)")
    ax.loglog([r["i_proxy_fn_weighted"] for r in b],
              [r["mu_proxy_fn_weighted"] for r in b], "s--", color="#d62728",
              label="GOT contact proxy mu(I_c)")
    ax.set_xlabel("inertial number I  (or contact proxy I_c)")
    ax.set_ylabel("effective friction mu")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_mu_vs_I.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    x = np.arange(len(b))
    ax.bar(x - 0.2, [r["i_mapping_error"] * 100 for r in b], 0.4,
           label="|I_c/I_bulk - 1|")
    ax.bar(x + 0.2, [r["mu_error"] * 100 for r in b], 0.4,
           label="|mu_proxy/mu_bulk - 1|")
    ax.axhline(15.0, color="k", ls="--", lw=1, label="15 % owner threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].replace("base_", "") for r in b])
    ax.set_ylabel("relative error [%]")
    ax.set_title("contact-proxy xval errors (servo shear cell)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_errors.png"), dpi=150)
    plt.close(fig)
    print("figures written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())