"""Assemble the ``phi_0`` axis of CASE-COLUMN-COLLAPSE (m21723).

Reads the sweep results (``runs/case_column_collapse_packing_axis_m21723``),
maps ``phi_0`` onto the m21656 resolution ladder from pure geometry, fits the
runout sensitivity ``d(rho_inf)/d(phi_0)`` per arm, and apportions the ladder's
observed runout spread between the packing drift and everything else.

Every number below is computed here; nothing is transcribed by hand except the
ladder's *measured* runout values, which are parsed (with a sha256 of the source
recorded) from the committed audit report of m21656 -- the run artifacts of that
round are not in the repo, only its tables.

Outputs (``docs/artifacts/column_packing_axis_m21723``): ``summary.json``,
``fig_phi_axis.png``, ``fig_confound.png``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import case_column_collapse_packing_axis_m21723 as run  # noqa: E402
from tensordem.granular_column.packing_axis import (  # noqa: E402
    confound_decomposition, difference_quotient, ols_slope, slope_is_resolved)

LADDER_DOC = "docs/AUDIT_COLUMN_COLLAPSE_GRID_CONVERGENCE_m21656.md"
OUT = os.path.join("docs", "artifacts", "column_packing_axis_m21723")
TOLERANCE = 0.15


MISSING: list[str] = []


def load_cases() -> list[dict]:
    rows = []
    for spec in run.CASES:
        path = os.path.join(run.CASE_DIR, f"{spec['case_id']}.json")
        if not os.path.exists(path):
            MISSING.append(spec["case_id"])
            continue
        with open(path) as fh:
            case = json.load(fh)
        case.setdefault("packing_gap", case["protocol"].get("packing_gap"))
        case["arm"] = spec["arm"]
        case["grains_per_radius"] = float(spec["grains_per_radius"])
        rows.append(case)
    return rows


def ladder_runouts() -> list[dict]:
    """Parity of the m21656 ``a = 3`` ladder: measured rho and its reference."""
    with open(LADDER_DOC) as fh:
        text = fh.read()
    sha = hashlib.sha256(text.encode()).hexdigest()
    rows = []
    for m in re.finditer(r"^\|\s*(\d+)\s*\|\s*[\d.]+\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|"
                         r"\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", text,
                         flags=re.MULTILINE):
        gpr, n, a_meas, phi0, rho, ref = m.groups()
        rows.append({"grains_per_radius": float(gpr), "n_particles": int(n),
                     "aspect_measured": float(a_meas), "phi0": float(phi0),
                     "runout_ratio_p995": float(rho), "reference": float(ref)})
    if len(rows) < 4:
        raise SystemExit(f"{LADDER_DOC}: could not parse the a=3 ladder table "
                         f"({len(rows)} rows)")
    return rows, sha


def arm_fit(rows: list[dict], arm: str) -> dict:
    sel = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["packing_gap"])
    phi = [r["phi0"] for r in sel]
    rho = [r["runout_ratio_p995"] for r in sel]
    if len(sel) < 2:
        fit = {"status": "insufficient", "slope": float("nan"),
               "intercept": float("nan"), "stderr": float("nan"),
               "r2": float("nan"), "n": len(sel), "n_distinct": len(sel),
               "x_span": float("nan"),
               "reason": "arm has fewer than two members in this run"}
    elif len(sel) == 2:
        fit = difference_quotient(phi[0], rho[0], phi[1], rho[1])
    else:
        fit = ols_slope(phi, rho)
    resolved = slope_is_resolved(fit)
    return {"arm": arm, "members": sel, "fit": fit, "resolved": resolved,
            "phi_span": max(phi) - min(phi)}


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    rows = load_cases()
    phi_map = run.phi_map()
    ladder, ladder_sha = ladder_runouts()
    fits = {arm: arm_fit(rows, arm) for arm, *_ in run.ARMS}

    # --- confound: a=3 ladder, phi_0 span from the m21656 ladder geometry.
    phi_span_a3 = float(np.subtract(*reversed(phi_map["phi0_span"]["3"])))
    rho_a3 = [r["runout_ratio_p995"] for r in ladder]
    spread_a3 = max(rho_a3) - min(rho_a3)
    ref_a3 = float(np.mean([r["reference"] for r in ladder]))
    primary = fits["A3g8"]
    conf = confound_decomposition(slope=primary["fit"]["slope"], dphi=phi_span_a3,
                                  observed_spread=spread_a3, reference=ref_a3,
                                  resolved=primary["resolved"]["resolved"],
                                  tolerance=TOLERANCE)
    # Cross-resolution check: does the a=3 slope survive 12 grains per r_i?
    s8, s12 = primary["fit"]["slope"], fits["A3g12"]["fit"]["slope"]
    cross = {"slope_g8": s8, "slope_g12": s12,
             "relative_gap": (abs(s12 - s8) / abs(s8)
                              if s8 and np.isfinite(s12) else float("nan")),
             "note": ("a=3 at 12 grains/r_i carries only one member in this run: "
                      "no cross-resolution slope (queue job)")}
    # Independent witness: the ladder's own rho vs phi_0 (5 rungs, one axis each).
    ladder_fit = ols_slope([r["phi0"] for r in ladder], rho_a3)

    summary = {
        "round": "m21723", "case": "CASE-COLUMN-COLLAPSE", "axis": "PHI",
        "reference": "Lube et al. 2004 JFM 508:175-199", "tolerance": TOLERANCE,
        "phi_map": phi_map,
        "missing_cases": MISSING,
        "arms": {k: {"phi_span": v["phi_span"], "fit": v["fit"],
                     "resolved": v["resolved"],
                     "members": [{kk: m[kk] for kk in
                                  ("case_id", "packing_gap", "phi0",
                                   "runout_ratio_p995", "runout_error_measured",
                                   "aspect_measured", "n_particles",
                                   "stop_reason")} for m in v["members"]]}
                 for k, v in fits.items()},
        "confound": conf, "cross_resolution": cross,
        "ladder": {"source": LADDER_DOC, "source_sha256": ladder_sha,
                   "rows": ladder, "phi0_span": phi_span_a3,
                   "rho_spread": spread_a3, "reference_mean": ref_a3,
                   "fit_rho_vs_phi0": ladder_fit},
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---------------- figures ----------------
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for arm, colour in (("A3g8", "C0"), ("A3g12", "C3"), ("A1g8", "C2")):
        mem = fits[arm]["members"]
        if len(mem) < 2:
            continue
        ax[0].plot([m["phi0"] for m in mem], [m["runout_ratio_p995"] for m in mem],
                   "o-", color=colour, label=f"{arm}")
        if fits[arm]["fit"].get("status") in ("fit", "difference_quotient"):
            x = np.array([min(m["phi0"] for m in mem), max(m["phi0"] for m in mem)])
            f = fits[arm]["fit"]
            icpt = (f["intercept"] if np.isfinite(f["intercept"])
                    else mem[0]["runout_ratio_p995"] - f["slope"] * mem[0]["phi0"])
            ax[0].plot(x, icpt + f["slope"] * x, "--", color=colour, lw=1)
    ax[0].set_xlabel(r"initial solid fraction $\phi_0$")
    ax[0].set_ylabel(r"runout $r_\infty/r_i$  (-)")
    ax[0].set_title("runout vs packing fraction at fixed resolution")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot([r["phi0"] for r in ladder], rho_a3, "s-", color="C4",
               label=r"$a=3$ resolution ladder (m21656)")
    for r in ladder:
        ax[1].annotate(f'{r["grains_per_radius"]:g}', (r["phi0"], r["runout_ratio_p995"]),
                       textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax[1].axhline(ref_a3, color="k", ls=":", lw=1, label="Lube 2004 at $a_{meas}$")
    ax[1].axhspan(ref_a3 * (1 - TOLERANCE), ref_a3 * (1 + TOLERANCE), color="0.85",
                  zorder=0, label="15 % band")
    ax[1].set_xlabel(r"$\phi_0$ of each rung of the resolution ladder")
    ax[1].set_ylabel(r"$r_\infty/r_i$  (-)")
    ax[1].set_title(f"the ladder sweeps $\\phi_0$ too: {conf['verdict']}")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_phi_axis.png"), dpi=160)
    plt.close(fig)

    print(json.dumps({"phi_span_a3": phi_span_a3, "spread_a3": spread_a3,
                      "confound": conf, "cross": cross,
                      "ladder_fit": ladder_fit,
                      "slopes": {k: v["fit"]["slope"] for k, v in fits.items()},
                      "resolved": {k: v["resolved"] for k, v in fits.items()}},
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())