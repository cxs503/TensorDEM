#!/usr/bin/env python3
"""Finite-size convergence study for CASE-DEM-RCP-PACKING-FRACTION.

Why this exists
---------------
The anchor's acceptance criterion is a *steady-state porosity of a dropped
packing* (10 000 spheres, frictionless / low-friction).  The production wave
(`benchmarks/case_rcp_deposition/plan.json`) ran N = 10000 for both tiers and
the debt audit (`debt_audit.py`) confirmed the production size and seed
coverage.  What was never *quantified* is the finite-size systematic: how far
the N = 10000 value sits from the infinite-N limit, and whether the RCP
number is converged at the production size or still drifting.

This script closes that gap.  It reads the *existing* run JSONs (no new
simulation) at N = 1000 / 2000 / 4000 / 10000 for the frictionless tier and
N = 2000 / 10000 for the low-friction tier, fits the asymptotic model

    phi(N) = phi_inf + A N^{-p}

to the finite-size ladder, and reports:

* the observed order ``p`` of the finite-size systematic (whether the drift
  is a clean ``1/N`` or ``1/N^{1/2}`` power law, or absent);
* the extrapolated ``phi_inf`` (infinite-N RCP value) and its distance from
  the production N = 10000 value -- the error bar the finite-size
  systematic removes;
* a leave-one-size-out check: fit the coarser sizes, *predict* the largest,
  compare with the measurement (the same falsifiability discipline as
  :func:`tensordem.deposition_packing.extrapolation_selfconsistency`).

Every number in the generated document comes from the run JSONs -- no hand
copying.  The reference is the machine-readable RCP anchor
``phi_rcp = 0.6366`` (Scott 1960 / Bernal & Mason 1960) and the ledger band
``[0.62, 0.66]`` with the < 3 % accuracy standard.

Usage::

    PYTHONPATH=src python benchmarks/case_rcp_deposition/finite_size_convergence.py
    PYTHONPATH=src python benchmarks/case_rcp_deposition/finite_size_convergence.py --json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Reference (machine-readable, same home as the rest of the anchor)
# ---------------------------------------------------------------------------
RCP_REF_PHI = 0.6366   # Bernal & Mason 1960, Nature 188:910
RCP_BAND = (0.62, 0.66)
RCP_TOL_ABS = 0.02     # task tolerance: 0.62-0.66 -> +-0.02 around 0.64

RUNS_DIR = _REPO / "benchmarks" / "case_rcp_deposition" / "runs"


def _load_runs() -> dict:
    """Read all production (kn=4000, no dt-probe) run JSONs, keyed by (N, mu)."""
    rows: dict[tuple[int, float], list[float]] = {}
    for f in glob.glob(str(RUNS_DIR / "deposition_n*.json")):
        if "_dtf" in f:            # timestep probe, not a production run
            continue
        m = re.search(r"n(\d+)_mu([\d.]+)_kn(\d+)_", f)
        if not m:
            continue
        N, mu, kn = int(m.group(1)), float(m.group(2)), int(m.group(3))
        if kn != 4000:             # finite-size ladder is at production stiffness
            continue
        d = json.load(open(f))
        phi = d["metrics"]["phi_bulk"]
        rows.setdefault((N, mu), []).append(phi)
    return rows


def _fit_asymptotic(Ns: list[int], phis: list[float],
                    order_bounds: tuple[float, float] = (0.2, 4.0)) -> dict:
    """Fit phi(N) = phi_inf + A N^{-p} to the finite-size ladder.

    With >= 3 sizes the observed order ``p`` is measurable from the ratio of
    two consecutive refinements ``Delta_1/Delta_2 = r^p`` (``r`` the size
    ratio); with 2 sizes it falls back to ``default_order``.  Returns the
    extrapolated ``phi_inf`` and ``delta_from_last`` -- the distance between
    the largest measured size and the limit, which is the error bar the
    finite-size systematic removes.
    """
    N = np.array([float(n) for n in Ns])
    val = np.array([float(v) for v in phis])
    if len(N) < 2:
        raise ValueError("need at least two sizes")
    if np.any(np.diff(N) <= 0.0):
        raise ValueError("sizes must be strictly increasing")

    deltas = np.diff(val)
    ratios = N[1:] / N[:-1]

    order: float | None = None
    if len(deltas) >= 2 and deltas[-2] != 0.0 and deltas[-1] != 0.0 \
            and np.sign(deltas[-2]) == np.sign(deltas[-1]):
        order = float(math.log(abs(deltas[-2]) / abs(deltas[-1]))
                      / math.log(ratios[-1]))
        if not (order_bounds[0] < order < order_bounds[1]):
            order = None
    p_used = 1.0 if order is None else order
    ratio = float(ratios[-1])
    limit = float(val[-1] + deltas[-1] / (ratio ** p_used - 1.0))
    return {
        "N": N.tolist(),
        "phi": val.tolist(),
        "delta": deltas.tolist(),
        "ratio": ratios.tolist(),
        "order": order,
        "order_used": float(p_used),
        "phi_inf": limit,
        "delta_from_last": float(abs(limit - val[-1])),
        "n_sizes": int(len(N)),
    }


def _leave_one_out(Ns: list[int], phis: list[float]) -> dict:
    """Fit the coarser sizes, predict the largest, compare with measurement."""
    if len(Ns) < 3:
        return {"prediction_error": None, "n_sizes": len(Ns)}
    fit = _fit_asymptotic(Ns[:-1], phis[:-1])
    N_last, val_last = float(Ns[-1]), float(phis[-1])
    # phi(N) = phi_inf + A N^{-p} with A fixed by the last fitted size
    N_fit, val_fit = float(Ns[-2]), float(phis[-2])
    predicted = float(fit["phi_inf"]
                      + (val_fit - fit["phi_inf"]) * (N_last / N_fit) ** (-fit["order_used"]))
    return {
        "prediction_error": float(abs(predicted - val_last)),
        "predicted": predicted,
        "measured": val_last,
        "n_sizes": len(Ns),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="print numbers only")
    ap.add_argument("--out", default="artifacts/case_rcp_deposition",
                    help="output directory")
    args = ap.parse_args()

    rows = _load_runs()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-tier finite-size ladder (mean over seeds at each size) ----
    tiers = {}
    for mu in (0.0, 0.1):
        keyed = {N: np.mean(v) for (N, m), v in rows.items() if m == mu}
        Ns = sorted(keyed)
        phis = [keyed[n] for n in Ns]
        if len(Ns) < 2:
            print(f"[FS] mu={mu}: only {len(Ns)} size(s), skipping", flush=True)
            continue
        fit = _fit_asymptotic(Ns, phis)
        loo = _leave_one_out(Ns, phis)
        tiers[mu] = {
            "sizes": Ns,
            "phi_by_size": {n: float(round(keyed[n], 5)) for n in Ns},
            "fit": fit,
            "leave_one_out": loo,
            "phi_production": float(round(keyed[Ns[-1]], 5)),
            "phi_inf": float(round(fit["phi_inf"], 5)),
            "finite_size_systematic": float(round(abs(fit["phi_inf"] - keyed[Ns[-1]]), 5)),
            "in_band": bool(RCP_BAND[0] <= keyed[Ns[-1]] <= RCP_BAND[1]),
            "lt_3pct": bool(abs(keyed[Ns[-1]] - RCP_REF_PHI) / RCP_REF_PHI < 0.03),
        }
        print(f"[FS] mu={mu}: sizes={Ns} phi={[round(p,4) for p in phis]} "
              f"phi_inf={fit['phi_inf']:.4f} order={fit['order']} "
              f"fs_systematic={abs(fit['phi_inf']-keyed[Ns[-1]]):.4f}", flush=True)

    # ---- figure: finite-size convergence ----
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0.0: "#2196F3", 0.1: "#FF9800"}
    for mu, t in tiers.items():
        Ns = t["sizes"]
        phis = [t["phi_by_size"][n] for n in Ns]
        ax.plot(Ns, phis, "o-", lw=2, color=colors[mu],
                label=f"mu={mu} (production φ={t['phi_production']:.4f})")
        ax.axhline(t["phi_inf"], color=colors[mu], ls="--", lw=1,
                   alpha=0.6, label=f"mu={mu} φ_inf={t['phi_inf']:.4f}")
    ax.axhline(RCP_REF_PHI, color="r", ls=":", lw=1.5,
               label=f"RCP reference φ={RCP_REF_PHI:.4f}")
    ax.axhspan(*RCP_BAND, alpha=0.1, color="green", label="Tolerance band [0.62, 0.66]")
    ax.set_xscale("log")
    ax.set_xlabel("Number of spheres N (log)")
    ax.set_ylabel("Bulk solid fraction φ")
    ax.set_title("RCP finite-size convergence: φ(N) → φ_inf")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.60, 0.66)
    fig.tight_layout()
    fig_path = out_dir / "fig_rcp_finite_size.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    # ---- assemble result ----
    result = {
        "case": "CASE-DEM-RCP-PACKING-FRACTION",
        "study": "finite-size-convergence",
        "reference": {"phi_rcp": RCP_REF_PHI, "band": list(RCP_BAND),
                      "source": "Bernal & Mason (1960) Nature 188:910; Scott (1960) Nature 188:908"},
        "tiers": tiers,
        "figure": "fig_rcp_finite_size.png",
        "all_in_band": all(t["in_band"] for t in tiers.values()),
        "all_lt_3pct": all(t["lt_3pct"] for t in tiers.values()),
    }
    result_path = out_dir / "finite_size_convergence.json"
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[FS] result written to {result_path}", flush=True)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # ---- document ----
    lines = []
    lines.append("# CASE-DEM-RCP-PACKING-FRACTION — finite-size convergence of the RCP value\n")
    lines.append("Anchor: `CASE-DEM-RCP-PACKING-FRACTION` (feature_ledger). This document is\n"
                 "generated by `benchmarks/case_rcp_deposition/finite_size_convergence.py`;\n"
                 "every number below is read from the run JSONs in\n"
                 "`benchmarks/case_rcp_deposition/runs/` — no hand copying.\n")
    lines.append("\n## 1 Why this study\n")
    lines.append("The anchor's acceptance criterion is a *steady-state porosity of a dropped\n"
                 "packing* (10 000 spheres, frictionless / low-friction).  The production wave\n"
                 "ran N = 10000 for both tiers; what was never *quantified* is the finite-size\n"
                 "systematic — how far the N = 10000 value sits from the infinite-N limit.\n"
                 "This study reads the existing run JSONs at N = 1000 / 2000 / 4000 / 10000\n"
                 "(frictionless) and N = 2000 / 10000 (low-friction) and fits\n"
                 "`phi(N) = phi_inf + A N^{-p}` to the ladder.\n")
    lines.append("\n## 2 Finite-size ladder (production stiffness k_n = 4000)\n")
    lines.append("| mu | N | phi_bulk (mean) |\n|---|---|---|")
    for mu, t in tiers.items():
        for n in t["sizes"]:
            lines.append(f"| {mu} | {n} | {t['phi_by_size'][n]:.5f} |")
    lines.append("\n## 3 Asymptotic fit and finite-size systematic\n")
    for mu, t in tiers.items():
        f = t["fit"]
        lines.append(f"- **mu={mu}**: observed order `p={f['order']}` "
                     f"(used {f['order_used']:.2f}); φ_inf = {t['phi_inf']:.5f}; "
                     f"finite-size systematic = {t['finite_size_systematic']:.5f} "
                     f"(production N={t['sizes'][-1]} value {t['phi_production']:.5f}).")
    lines.append("\n## 4 Leave-one-size-out check\n")
    for mu, t in tiers.items():
        loo = t["leave_one_out"]
        if loo["prediction_error"] is not None:
            lines.append(f"- **mu={mu}**: fit the coarser sizes, predict the largest: "
                         f"predicted {loo['predicted']:.5f} vs measured {loo['measured']:.5f}, "
                         f"prediction error {loo['prediction_error']:.5f}.")
        else:
            lines.append(f"- **mu={mu}**: only {loo['n_sizes']} sizes — "
                         f"leave-one-out not possible (needs >= 3).")
    lines.append("\n## 5 Verdict\n")
    for mu, t in tiers.items():
        band = "IN BAND" if t["in_band"] else "OUT OF BAND"
        pct = "OK (<3%)" if t["lt_3pct"] else "FAIL (>=3%)"
        lines.append(f"- **mu={mu}**: production φ={t['phi_production']:.5f} — {band}, "
                     f"vs reference {RCP_REF_PHI:.4f} {pct}; "
                     f"finite-size systematic {t['finite_size_systematic']:.5f}.")
    lines.append(f"\nOverall: {'ALL IN BAND' if result['all_in_band'] else 'SOME OUT'} · "
                 f"{'ALL <3%' if result['all_lt_3pct'] else 'SOME >=3%'}.\n")

    doc_path = _REPO / "docs" / "CASE_RCP_FINITE_SIZE_CONVERGENCE.md"
    with open(doc_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[FS] doc written to {doc_path}", flush=True)


if __name__ == "__main__":
    main()
