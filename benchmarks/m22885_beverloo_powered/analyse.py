"""Frozen estimator for the m22885 powered Beverloo ensemble (see prereg.md).

Everything here is pure arithmetic on per-run JSON, so the verdicts can be
re-derived from the archived artefacts with one command:

    python3 benchmarks/m22885_beverloo_powered/analyse.py --dir runs/m22885_beverloo_powered

`tests/test_m22885_beverloo_powered.py` pins every branch on synthetic data.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

BEVERLOO_C = 0.58
BEVERLOO_K = 1.4
G = 9.81
TOL = 0.10          # ledger acceptance: |rel| < 10 %
RHO_SPREAD_TOL = 0.03
CV_TOL = 0.10
DRIFT_TOL = 0.10
NOISE_TOL = 0.05

# Student-t 0.975 quantile, two-sided 95 %, by degrees of freedom
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
    22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
    29: 2.045, 30: 2.042,
}


def t975(dof: int) -> float:
    """Two-sided 95 % Student-t quantile (table for small n, normal limit after)."""
    if dof < 1:
        raise ValueError("dof must be >= 1")
    if dof in _T975:
        return _T975[dof]
    if dof <= 60:
        return 2.000 + 0.042 * (30.0 / dof)      # smooth continuation of the table
    if dof <= 120:
        return 1.980
    return 1.960


def rel_error(w_meas: float, w_ref: float) -> float:
    if not (w_ref > 0.0) or not math.isfinite(w_ref):
        raise ValueError("reference rate must be finite and positive")
    return w_meas / w_ref - 1.0


def beverloo_ref(orifice: float, particle: float, rho_b: float) -> float:
    """Literature-coefficient reference rate (never refitted)."""
    eff = orifice - BEVERLOO_K * particle
    if eff <= 0.0:
        return 0.0
    return BEVERLOO_C * rho_b * math.sqrt(G) * eff ** 2.5


def counting_noise(n_exit: int) -> float:
    """Relative 1-sigma floor of a count-limited rate measurement."""
    if n_exit <= 0:
        return float("nan")
    return 1.0 / math.sqrt(n_exit)


def drift_fraction(rates, heights) -> float:
    """Relative rate change across the measured span from an OLS fit W(h).

    ``|slope| * (h_first - h_last) / mean(rates)`` — dimensionless, sign kept so a
    falling bed that slows the discharge shows up as a positive loss.
    """
    r = np.asarray(rates, dtype=float)
    h = np.asarray(heights, dtype=float)
    if r.size < 3 or r.size != h.size:
        raise ValueError("need >= 3 matched windows")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(h)):
        raise ValueError("non-finite window data")
    slope, _ = np.polyfit(h, r, 1)
    span = float(h[0] - h[-1])
    return float(abs(slope) * span / float(np.mean(r)))


def per_run(run: dict, *, rho_key: str = "rho_bulk_measured") -> dict:
    """Reduce one discharge JSON to the frozen per-run quantities."""
    case = run["case"]
    rates = [float(x) for x in run["flow_rate_windows_kg_s"]]
    n_steady = max(1, len(rates) // 2)
    w_s = float(np.mean(rates[-n_steady:]))
    n_exit = int(sum(int(x) for x in run["window_particles"]))
    rho = float(run[rho_key])
    w_ref = beverloo_ref(float(case["orifice_diameter"]), float(case["particle_diameter"]), rho)
    heights = [float(x) for x in run["window_surface_height_m"]]
    return {
        "seed": int(case["seed"]),
        "fill_height": float(case["fill_height"]),
        "D_over_d": float(case["orifice_over_particle"]),
        "w_s": w_s,
        "w_ref": w_ref,
        "rel": rel_error(w_s, w_ref),
        "rho_bulk": float(run["rho_bulk"]),
        "rho_bulk_measured": float(run["rho_bulk_measured"]),
        "surface_height_m": float(run["surface_height_m"]),
        "n_exit": n_exit,
        "noise": counting_noise(n_exit),
        "drift": drift_fraction(rates, heights),
        "c_recovered": w_s / (rho * math.sqrt(G) * (float(case["orifice_diameter"])
                                                    - BEVERLOO_K * float(case["particle_diameter"])) ** 2.5),
        "n_bed": int(run["n_bed"]),
        "steps": int(run["steps"]),
    }


def mean_ci(values) -> dict:
    """Mean with two-sided 95 % Student-t interval."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "se": None, "ci95": None, "t": None}
    mean = float(np.mean(v))
    if n == 1:
        return {"n": 1, "mean": mean, "sd": None, "se": None, "ci95": None, "t": None}
    sd = float(np.std(v, ddof=1))
    se = sd / math.sqrt(n)
    t = t975(n - 1)
    return {"n": n, "mean": mean, "sd": sd, "se": se, "ci95": t * se, "t": t}


def verdict_band(stats: dict, tol: float = TOL) -> str:
    """PASS / FAIL / NOT_DECIDABLE for an interval test against ±tol."""
    if stats.get("mean") is None or stats.get("ci95") is None:
        return "NOT_RUN"
    lo = stats["mean"] - stats["ci95"]
    hi = stats["mean"] + stats["ci95"]
    if -tol < lo and hi < tol:
        return "PASS"
    if hi < -tol or lo > tol:
        return "FAIL"
    return "NOT_DECIDABLE"


def ensemble(runs: list[dict], *, rho_key: str = "rho_bulk_measured") -> dict:
    """Ensemble block for one fill height."""
    rows = [per_run(r, rho_key=rho_key) for r in runs]
    stats = mean_ci([r["rel"] for r in rows])
    w_stats = mean_ci([r["w_s"] for r in rows])
    cv = None
    if w_stats["sd"] is not None and w_stats["mean"]:
        cv = w_stats["sd"] / abs(w_stats["mean"])
    noise = mean_ci([r["noise"] for r in rows])
    drift = mean_ci([r["drift"] for r in rows])
    rho_meas = mean_ci([r["rho_bulk_measured"] for r in rows])
    rho_nom = mean_ci([r["rho_bulk"] for r in rows])
    conv = None
    if rho_nom["mean"]:
        conv = rho_meas["mean"] / rho_nom["mean"] - 1.0
    return {
        "rows": rows,
        "rel": stats,
        "w_s": w_stats,
        "cv": cv,
        "noise": noise,
        "drift": drift,
        "rho_meas": rho_meas,
        "rho_nom": rho_nom,
        "rho_convention_shift": conv,
        "rho_spread": (rho_meas["sd"] / rho_meas["mean"]) if (rho_meas["sd"] and rho_meas["mean"]) else None,
        "V1_ledger": verdict_band(stats),
        "V2_reproducibility": (None if cv is None else ("PASS" if cv < CV_TOL else "FAIL")),
        "V3_steady_state": (None if drift["mean"] is None else
                            ("PASS" if drift["mean"] < DRIFT_TOL else "FAIL")),
        "V4_rho_control": (None if (rho_meas["sd"] is None) else
                           ("PASS" if (rho_meas["sd"] / rho_meas["mean"] < RHO_SPREAD_TOL
                                       and abs(conv or 0.0) > 0.01) else "FAIL")),
        "V6_noise_floor": (None if noise["mean"] is None else
                           ("PASS" if noise["mean"] <= NOISE_TOL else "FAIL")),
    }


def h_independence(base: dict, tall: dict) -> dict:
    """Ratio W_fill(H/D=3.2)/W_fill(H/D=2.4) with a Welch-style interval."""
    a = base["w_s"]
    b = tall["w_s"]
    if a["mean"] is None or b["mean"] is None or not a["mean"] or not b["mean"]:
        return {"ratio": None, "rel": None, "V5_H_independence": "NOT_RUN"}
    ratio = b["mean"] / a["mean"]
    rel = ratio - 1.0
    var = 0.0
    if a["se"] is not None and b["se"] is not None:
        var = (b["se"] / a["mean"]) ** 2 + (b["mean"] * a["se"] / a["mean"] ** 2) ** 2
    half = t975(min(a["n"], b["n"]) - 1) * math.sqrt(var)
    lo, hi = rel - half, rel + half
    if -TOL < lo and hi < TOL:
        verdict = "PASS"
    elif hi < -TOL or lo > TOL:
        verdict = "FAIL"
    else:
        verdict = "NOT_DECIDABLE"
    return {"ratio": ratio, "rel": rel, "ci95": half, "lo": lo, "hi": hi,
            "V5_H_independence": verdict}


def analyse_dir(directory: Path) -> dict:
    files = sorted(glob.glob(str(Path(directory) / "*.json")))
    runs = [json.loads(Path(f).read_text()) for f in files]
    runs = [r for r in runs if "case" in r and "flow_rate_windows_kg_s" in r]
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(f"fill_{r['case']['fill_height'] * 1000:.1f}mm", []).append(r)
    out = {"n_files": len(files), "n_runs": len(runs), "groups": {}}
    for name, grp in sorted(groups.items()):
        out["groups"][name] = {"n": len(grp),
                               "measured": ensemble(grp, rho_key="rho_bulk_measured"),
                               "nominal": ensemble(grp, rho_key="rho_bulk")}
    fills = sorted(groups)
    if len(fills) >= 2:
        base = out["groups"][fills[0]]["measured"]
        tall = out["groups"][fills[-1]]["measured"]
        out["V5_H_independence"] = h_independence(base, tall)
    else:
        out["V5_H_independence"] = {"ratio": None, "rel": None,
                                    "V5_H_independence": "NOT_RUN"}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(ROOT / "runs/m22885_beverloo_powered"))
    ap.add_argument("--out", default=str(ROOT / "runs/m22885_beverloo_powered/summary.json"))
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    res = analyse_dir(Path(args.dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}  (n_runs={res['n_runs']})")
    for name, g in res["groups"].items():
        m = g["measured"]
        print(f"  {name} n={g['n']} rel={m['rel']['mean']} ±{m['rel']['ci95']} "
              f"V1={m['V1_ledger']} V2={m['V2_reproducibility']} V3={m['V3_steady_state']} "
              f"V4={m['V4_rho_control']} V6={m['V6_noise_floor']}")
    print(f"  V5={res['V5_H_independence']['V5_H_independence']} "
          f"ratio={res['V5_H_independence']['ratio']}")
    if args.figures:
        from render_figures import render_all
        render_all(res, ROOT / "docs/figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())