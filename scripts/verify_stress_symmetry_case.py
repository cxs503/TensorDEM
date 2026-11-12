#!/usr/bin/env python3
"""Quantify the non-central (lever-arm) part of the reported shear stress.

Context
-------
``elastic_limit_shear`` (``CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT``) reported a
single scalar shear stress

    tau = -[sum_p m v_x v_y + sum_c F_{i,x} (r_i - r_j)_y] / A   (= sigma_yx),

i.e. one off-diagonal component of the *raw* Virial tensor, added to a
symmetric kinetic term.  For central (frictionless) contact forces the raw
Virial is symmetric and the choice is invisible; with Coulomb tangential forces
it is not, and the two off-diagonal components differ by the per-contact
lever-arm torque

    sigma_xy - sigma_yx = (1/A) sum_c |l| f_t .

The certified instrument (:mod:`tensordem.granular3d.stress`, reused for 2-D
through :mod:`tensordem.granular2d.stress_arrays`) returns both components, so
the size of that bookkeeping choice is now measurable instead of assumed.

What this script does
---------------------
1. sweeps the shear rate at the case's default friction (mu = 0.5) and reports
   ``tau`` (historical, sigma_yx), ``tau_symmetric``, their relative gap and the
   raw asymmetry;
2. re-runs the same sweep at two other friction coefficients (mu = 0.0 control,
   mu = 0.3) to show that the gap is carried by the tangential force and that it
   vanishes for central forces;
3. fits the log-log rate exponent of both ordinates, i.e. answers whether the
   gated conclusion (elastic limit: ``n = d ln tau / d ln gamma_dot -> 0``)
   survives the correction;
4. writes ``artifacts/stress_symmetry_case/verification.json`` + a figure.

Run:  PYTHONPATH=src python3 scripts/verify_stress_symmetry_case.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from tensordem.elastic_limit_shear import (
    ElasticLimitShearCase,
    shear_rate_exponent,
    simulate_plane_shear,
)

OUT = os.path.join("artifacts", "stress_symmetry_case")

SHEAR_RATES = (0.1, 0.3, 1.0)
FRICTIONS = (0.5, 0.0, 0.3)
N_PARTICLES = 400


def run_point(rate: float, friction: float) -> dict:
    case = ElasticLimitShearCase(n_particles=N_PARTICLES, shear_rate=rate,
                                 friction=friction)
    t0 = time.time()
    res = simulate_plane_shear(case)
    return {
        "shear_rate": rate,
        "friction": friction,
        "case": res["case"],
        "kappa": res["kappa"],
        "tau_star": res["tau_star"],
        "I": res["I"],
        "gates": res["gates"],
        "tau": res["tau"],
        "tau_symmetric": res["tau_symmetric"],
        "tau_minus_symmetric": res["tau_minus_symmetric"],
        "symmetry_defect_relative": res["tau_symmetry_defect"],
        "stress_asymmetry": res["stress_asymmetry"],
        "stress_asymmetry_relative": res["stress_asymmetry_relative"],
        "pressure": res["pressure"],
        "mu_eff": res["mu_eff"],
        "mu_eff_symmetric": (res["tau_symmetric"] / res["pressure"]
                             if res["pressure"] > 0 else float("nan")),
        "coordination": res["coordination"],
        "n_samples": res["samples"],
        "gates_all": res["gates"]["all"],
        "runtime_s": round(time.time() - t0, 2),
    }


def loglog_slope(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    ok = (x > 0) & (y > 0)
    if np.count_nonzero(ok) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)[0])


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for mu in FRICTIONS:
        for rate in SHEAR_RATES:
            row = run_point(rate, mu)
            rows.append(row)
            print(f"mu={mu:.2f} gdot={rate:.2f}  tau={row['tau']:.4f} "
                  f"tau_sym={row['tau_symmetric']:.4f} "
                  f"defect={row['symmetry_defect_relative']*100:6.2f}%  "
                  f"asym={row['stress_asymmetry_relative']*100:6.2f}%  "
                  f"gates={row['gates_all']}  ({row['runtime_s']}s)", flush=True)

    slopes = {}
    for mu in FRICTIONS:
        sub = [r for r in rows if r["friction"] == mu]
        rates = [r["shear_rate"] for r in sub]
        slopes[str(mu)] = {
            "n_historical": loglog_slope(rates, [r["tau"] for r in sub]),
            "n_symmetric": loglog_slope(rates, [r["tau_symmetric"] for r in sub]),
        }
        d = abs(slopes[str(mu)]["n_symmetric"] - slopes[str(mu)]["n_historical"])
        slopes[str(mu)]["abs_slope_change"] = d
        print(f"mu={mu}: n(historical)={slopes[str(mu)]['n_historical']:.5f} "
              f"n(symmetric)={slopes[str(mu)]['n_symmetric']:.5f} "
              f"|dn|={d:.5f}", flush=True)

    # also exercise the library's own exponent helper on the two ordinates
    lib = {}
    sub = [r for r in rows if r["friction"] == 0.5]
    pts_hist = [dict(r) for r in sub]
    pts_sym = [dict(r, tau=r["tau_symmetric"]) for r in sub]
    lib["n_historical"] = shear_rate_exponent(pts_hist)
    lib["n_symmetric"] = shear_rate_exponent(pts_sym)
    print("library shear_rate_exponent:", lib, flush=True)

    payload = {
        "case": "CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT",
        "module": "tensordem.elastic_limit_shear",
        "instrument": "tensordem.granular2d.stress_arrays -> "
                      "tensordem.granular3d.stress.contact_outer",
        "setup": {"n_particles": N_PARTICLES, "shear_rates": SHEAR_RATES,
                  "frictions": FRICTIONS,
                  "case_defaults": ElasticLimitShearCase().describe()},
        "points": rows,
        "loglog_slopes": slopes,
        "library_exponent": lib,
        "defect_summary": {
            "max_relative_tau_defect": max(r["symmetry_defect_relative"]
                                           for r in rows),
            "max_relative_asymmetry_mu0p5": max(
                r["stress_asymmetry_relative"] for r in rows
                if r["friction"] == 0.5),
            "frictionless_asymmetry": [r["stress_asymmetry_relative"]
                                       for r in rows if r["friction"] == 0.0],
        },
    }
    with open(os.path.join(OUT, "verification.json"), "w") as fh:
        json.dump(payload, fh, indent=2)

    try:
        make_figure(payload)
    except Exception as exc:                       # pragma: no cover - optional
        print(f"figure skipped: {exc}", file=sys.stderr)
    return 0


def make_figure(payload: dict) -> None:            # pragma: no cover - plotting
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    for mu, style in ((0.5, "o-"), (0.3, "s--"), (0.0, "^:")):
        sub = [r for r in payload["points"] if r["friction"] == mu]
        g = [r["shear_rate"] for r in sub]
        ax.loglog(g, [r["tau"] for r in sub], style, color="C0" if mu == 0.5 else
                  ("C2" if mu == 0.3 else "C3"), label=f"sigma_yx (mu={mu})")
        ax.loglog(g, [r["tau_symmetric"] for r in sub], style,
                  color="C1" if mu == 0.5 else ("C4" if mu == 0.3 else "C5"),
                  mfc="none", label=f"sy (mu={mu})")
    ax.set_xlabel(r"shear rate $\dot\gamma$")
    ax.set_ylabel(r"shear stress $\tau$ (DEM units)")
    ax.set_title("raw vs symmetric shear stress")
    ax.legend(fontsize=7)

    ax = axes[1]
    widths = 0.35
    idx = np.arange(len(SHEAR_RATES))
    for k, mu in enumerate((0.5, 0.3, 0.0)):
        sub = [r for r in payload["points"] if r["friction"] == mu]
        ax.bar(idx + (k - 1) * widths, [100 * r["symmetry_defect_relative"]
                                        for r in sub], widths,
               label=f"mu={mu}")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{g:g}" for g in SHEAR_RATES])
    ax.set_xlabel(r"shear rate $\dot\gamma$")
    ax.set_ylabel(r"$|\tau-\sigma_{xy}|/|\sigma_{xy}|$  [%]")
    ax.set_title("bookkeeping defect of the historical ordinate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_stress_symmetry.png"), dpi=160)
    print("figure ->", os.path.join(OUT, "fig_stress_symmetry.png"))


if __name__ == "__main__":
    sys.exit(main())