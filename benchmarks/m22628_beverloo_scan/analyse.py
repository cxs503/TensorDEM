#!/usr/bin/env python3
"""CASE-DEM-SILO-BEVERLOO (m22628): invert the Beverloo law from the DEM scan.

The correlation

    W = C · rho_b · sqrt(g) · (D - k d)**p ,        p_law = 5/2

carries three *separately falsifiable* statements.  This driver fits four
nested models to the measured ``(D, d, rho_b, W)`` set and reports which of
them the DEM data actually supports:

| model | free parameters | what it tests |
|-------|-----------------|---------------|
| A | none (C = 0.58, k = 1.4, p = 5/2) | point-by-point amplitude error |
| B | C (k = 1.4, p = 5/2) | can one amplitude rescale every orifice? |
| C | C, k (p = 5/2) | is the finite-size offset ``k d`` recovered in [1.2, 1.6]? |
| D | C, k, p | shape: is the exponent 5/2 within +-0.2? |

All fits are least squares in **log** space, because discharge-rate errors are
multiplicative.  ``C`` is eliminated analytically (``ln C = mean(y - b)``), so
models C and D are minimisations over two (or one) parameters and the reported
standard errors come from the profiled objective.  Numbers are read from
``runs/m22628_beverloo_scan/*.json`` -- nothing is hand-typed.

Output: ``runs/m22628_beverloo_scan/summary.json``.

Usage
-----
    PYTHONPATH=src python benchmarks/m22628_beverloo_scan/analyse.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "runs" / "m22628_beverloo_scan"
SUMMARY = RUN_DIR / "summary.json"

G = 9.81
P_LAW = 2.5
C_REF, K_REF = 0.58, 1.4
C_RANGE = (0.55, 0.60)
K_RANGE = (1.2, 1.6)
P_TOL = 0.2
PRIMARY_TOL = 0.10          # the ledger acceptance: < 10 %

D_SCAN = ["D40", "D45", "D50", "D55", "D60"]
H_SCAN = {"D50": 3.0, "D50_H25": 2.5, "D50_H20": 2.0}   # H/D per case


# ---------------------------------------------------------------------------
# model machinery
# ---------------------------------------------------------------------------
def _design(D: np.ndarray, rho: np.ndarray, d: float, k: float, p: float) -> np.ndarray:
    """``b_i = ln(rho_b sqrt g) + p ln(D_i - k d)`` -- the log-prediction without C."""
    eff = D - k * d
    if np.any(eff <= 0.0):
        return np.full(D.shape, -np.inf)
    return np.log(rho * math.sqrt(G)) + p * np.log(eff)


def _sse(theta: np.ndarray, D, rho, d, y, *, free_p: bool) -> float:
    k = float(theta[0])
    p = float(theta[1]) if free_p else P_LAW
    b = _design(D, rho, d, k, p)
    if not np.all(np.isfinite(b)):
        return 1e30
    resid = y - b
    resid = resid - resid.mean()          # C profiled out
    return float(np.sum(resid ** 2))


def fit(D, rho, d, y, *, k_free: bool, p_free: bool) -> dict:
    """Log-space least squares with ``C`` profiled out."""
    D = np.asarray(D, float)
    rho = np.asarray(rho, float)
    y = np.asarray(y, float)
    n = D.size
    k_hat = K_REF
    p_hat = P_LAW
    sse = _sse(np.array([K_REF, P_LAW]), D, rho, d, y, free_p=False)
    if k_free or p_free:
        from scipy.optimize import minimize

        x0 = np.array([K_REF, P_LAW])
        bounds = [(0.0, 3.0), (1.0, 4.0)]
        objective = lambda t: _sse(t, D, rho, d, y, free_p=p_free)  # noqa: E731
        res = minimize(
            objective, x0, method="L-BFGS-B", bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-12, "maxiter": 2000},
        )
        k_hat = float(res.x[0]) if k_free else K_REF
        p_hat = float(res.x[1]) if p_free else P_LAW
        sse = float(res.fun)
    b = _design(D, rho, d, k_hat, p_hat)
    ln_C = float(np.mean(y - b))
    C_hat = math.exp(ln_C)
    resid = y - b - ln_C
    pred = np.exp(b + ln_C)
    dof = max(n - 1 - int(k_free) - int(p_free), 1)
    sigma2 = sse / dof
    rms_log = math.sqrt(sse / n)
    out = {
        "k": k_hat,
        "p": p_hat,
        "C": C_hat,
        "sse_log": sse,
        "rms_log_residual": rms_log,
        "dof": dof,
        "n_points": int(n),
        "predicted_kg_s": pred.tolist(),
        "measured_kg_s": y.tolist(),
        "rel_err": np.expm1(resid).tolist(),          # (pred - meas)/meas
        "max_abs_rel_err": float(np.max(np.abs(np.expm1(resid)))),
    }
    # profiled standard error of the free parameters from the numeric Hessian
    if (k_free or p_free) and n > 2:
        theta0 = np.array([k_hat, p_hat])
        steps = np.array([1e-4, 1e-4])
        H = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                def f(shift):
                    t = theta0.copy()
                    t[i] += shift[0]
                    t[j] += shift[1]
                    return _sse(t, D, rho, d, y, free_p=p_free)
                H[i, j] = (
                    f((steps[i], steps[j])) - f((steps[i], -steps[j]))
                    - f((-steps[i], steps[j])) + f((-steps[i], -steps[j]))
                ) / (4.0 * steps[i] * steps[j])
        try:
            cov = np.linalg.inv(H) * 2.0 * sigma2
            se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
            out["se_k"] = float(se[0])
            out["se_p"] = float(se[1])
        except np.linalg.LinAlgError:
            out["se_k"] = out["se_p"] = float("nan")
    return out


def k_profile(D, rho, d, y, *, p: float = P_LAW, ks=None) -> list[dict]:
    """SSE as a function of a *fixed* ``k`` (for the confidence figure)."""
    D = np.asarray(D, float)
    rho = np.asarray(rho, float)
    y = np.asarray(y, float)
    if ks is None:
        ks = np.linspace(0.0, 3.0, 61)
    out = []
    for k in ks:
        sse = _sse(np.array([k, p]), D, rho, d, y, free_p=False)
        out.append({"k": float(k), "sse_log": float(sse)})
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def load_runs(run_dir: Path = RUN_DIR) -> dict:
    runs = {}
    for p in sorted(run_dir.glob("*.json")):
        if p.name == "summary.json":
            continue
        data = json.loads(p.read_text())
        if isinstance(data, dict) and "case_id" in data:
            runs[data["case_id"]] = data
    return runs


def beverloo_ref(D: float, d: float, rho: float, *, c=C_REF, k=K_REF, p=P_LAW) -> float:
    eff = D - k * d
    if eff <= 0.0:
        return 0.0
    return c * rho * math.sqrt(G) * eff ** p


def _case_get(runs: dict, ids: list, key: str, default: float) -> float:
    """Read a case field from the first run that carries it (a partial run set,
    or a synthetic run used in the tests, may not carry every field)."""
    for cid in ids:
        value = runs[cid].get("case", {}).get(key)
        if value is not None:
            return float(value)
    return float(default)


def analyse(runs: dict) -> dict:
    available = [cid for cid in D_SCAN if cid in runs]
    if len(available) < 3:
        raise SystemExit(
            f"need >= 3 orifice cases to fit, found {available}; run scan.py first"
        )
    D = np.array([runs[c]["case"]["orifice_diameter"] for c in available])
    rho = np.array([runs[c]["result"]["rho_bulk"] for c in available])
    W = np.array([runs[c]["result"]["flow_rate_kg_s"] for c in available])
    Wsd = np.array([runs[c]["result"]["flow_rate_std_kg_s"] for c in available])
    d = float(runs[available[0]]["case"]["particle_diameter"])
    ratios = [runs[c]["domain"]["ratios"]["D_over_d"] for c in available]
    rho_spread = float((rho.max() - rho.min()) / rho.mean())

    # --- model A: the pure literature prediction, no free parameter ---------
    modelA = []
    for cid, Di, ri, Wi, si in zip(available, D, rho, W, Wsd):
        pred = beverloo_ref(float(Di), d, float(ri))
        modelA.append({
            "case_id": cid,
            "D_over_d": float(Di / d),
            "rho_bulk": float(ri),
            "n_bed": int(runs[cid]["result"].get("n_bed", 0)),
            "mass_fraction_discharged": runs[cid]["result"].get("mass_fraction_discharged"),
            "W_measured_kg_s": float(Wi),
            "W_pred_kg_s": pred,
            "rel_err": float(Wi / pred - 1.0) if pred > 0 else float("nan"),
            "window_rel_std": float(si / Wi) if Wi > 0 else float("nan"),
            "pass_10pct": bool(abs(Wi / pred - 1.0) < PRIMARY_TOL) if pred > 0 else False,
        })

    y = np.log(W)
    fits = {
        "B_free_C": fit(D, rho, d, y, k_free=False, p_free=False),
        "C_free_Ck": fit(D, rho, d, y, k_free=True, p_free=False),
        "D_free_Ckp": fit(D, rho, d, y, k_free=True, p_free=True),
    }
    for tag, f in fits.items():
        f["max_abs_rel_err"] = float(np.max(np.abs(np.expm1(np.array(f["rel_err"])))))

    # --- the three falsifiable statements ----------------------------------
    fitC = fits["C_free_Ck"]
    fitD = fits["D_free_Ckp"]
    p_se = fitD.get("se_p", float("nan"))
    shape_ok = bool(abs(fitD["p"] - P_LAW) <= P_TOL)
    offset_ok = bool(K_RANGE[0] <= fitC["k"] <= K_RANGE[1])
    amplitude_ok = bool(C_RANGE[0] <= fitC["C"] <= C_RANGE[1])
    collapse_ok = bool(fitC["max_abs_rel_err"] < PRIMARY_TOL)
    # is a wrong k the *reason* a single amplitude failed?  compare the spread
    spread_B = fits["B_free_C"]["max_abs_rel_err"]
    spread_C = fitC["max_abs_rel_err"]

    prim = primary_verdict(modelA, fits, available)

    summary = {
        "campaign": "m22628",
        "active_cases": available,
        "particle_diameter_m": d,
        "n_points": len(available),
        "D_over_d_scan": [float(x) for x in ratios],
        "geometry": {
            "silo_radius_m": _case_get(runs, available, "silo_radius", 0.0),
            "fill_height_m": _case_get(runs, available, "fill_height", 0.0),
            "kn": _case_get(runs, available, "kn", 0.0),
            "mu": _case_get(runs, available, "mu", 0.0),
            "mu_rolling": _case_get(runs, available, "mu_rolling", 0.0),
            "wall_mu": _case_get(runs, available, "wall_mu", 0.0),
            "n_bed_range": [int(min(r["n_bed"] for r in modelA)),
                            int(max(r["n_bed"] for r in modelA))],
        },
        "rho_bulk_spread_frac": rho_spread,
        "reference": {
            "C": C_REF, "k": K_REF, "p": P_LAW,
            "C_range": list(C_RANGE), "k_range": list(K_RANGE), "p_tol": P_TOL,
            "tolerance": PRIMARY_TOL,
        },
        "model_A_literature": modelA,
        "fits": fits,
        "statements": {
            "shape_p_equals_2p5": {
                "fitted_p": fitD["p"], "se_p": p_se, "target": P_LAW,
                "tolerance": P_TOL, "pass": shape_ok,
            },
            "offset_k_in_1p2_1p6": {
                "fitted_k": fitC["k"], "se_k": fitC.get("se_k", float("nan")),
                "range": list(K_RANGE), "pass": offset_ok,
            },
            "amplitude_C_in_0p55_0p60": {
                "fitted_C": fitC["C"], "range": list(C_RANGE), "pass": amplitude_ok,
            },
            "residual_collapse_below_10pct": {
                "max_abs_rel_err_modelC": spread_C,
                "max_abs_rel_err_modelB": spread_B,
                "pass": collapse_ok,
            },
        },
        "k_profile": k_profile(D, rho, d, y),
        "primary": prim,
    }

    # --- filling-height independence --------------------------------------
    summary["fill_height_test"] = fill_height_test(runs)

    # --- run-to-run scatter ------------------------------------------------
    if "D50" in runs and "D50_s2" in runs:
        w1 = runs["D50"]["result"]["flow_rate_kg_s"]
        w2 = runs["D50_s2"]["result"]["flow_rate_kg_s"]
        summary["seed_repeat"] = {
            "W_seed_22628": w1, "W_seed_22629": w2,
            "rel_diff": (w2 - w1) / w1,
            "pass": bool(abs((w2 - w1) / w1) < PRIMARY_TOL),
        }
    return summary


def primary_verdict(modelA: list, fits: dict, available: list) -> dict:
    """Ledger acceptance: |W_dem - W_Beverloo| / W_Beverloo < 10 % at the primary
    case, with the primary being ``D/d = 5`` under the *literature* coefficients
    (the anchor compares against the law, not against a re-fitted C)."""
    prim = next((r for r in modelA if abs(r["D_over_d"] - 5.0) < 1e-9), None)
    if prim is None:
        return {"available": False, "reason": "primary case D/d = 5 not in the run set"}
    return {
        "available": True,
        "case_id": prim["case_id"],
        "W_measured_kg_s": prim["W_measured_kg_s"],
        "W_literature_kg_s": prim["W_pred_kg_s"],
        "rel_err_literature": prim["rel_err"],
        "pass_10pct_literature": prim["pass_10pct"],
        "tolerance": PRIMARY_TOL,
    }


def fill_height_test(runs: dict) -> dict:
    rows = []
    for cid, hd in H_SCAN.items():
        run = runs.get(cid)
        if run is None:
            continue
        rows.append({
            "case_id": cid,
            "H_over_D": hd,
            "fill_height_m": run["case"]["fill_height"],
            "n_bed": run["result"]["n_bed"],
            "W_kg_s": run["result"]["flow_rate_kg_s"],
            "W_std_kg_s": run["result"]["flow_rate_std_kg_s"],
        })
    if len(rows) < 2:
        return {"available": False, "rows": rows}
    rows.sort(key=lambda r: r["H_over_D"])
    W = np.array([r["W_kg_s"] for r in rows])
    base = float(W[-1])                      # tallest bed = reference
    for r in rows:
        r["rel_to_tallest"] = float(r["W_kg_s"] / base - 1.0)
    spread = float((W.max() - W.min()) / W.mean())
    return {
        "available": True,
        "rows": rows,
        "max_abs_rel_dev": float(np.max(np.abs(W / base - 1.0))),
        "relative_spread": spread,
        "pass_10pct": bool(spread < PRIMARY_TOL),
        "statement": "Beverloo: W is independent of the filling height "
                     "(for bed_height/D >= 1.5)",
    }


def main() -> int:
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no run JSON in {RUN_DIR}")
    summary = analyse(runs)
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"wrote {SUMMARY.relative_to(ROOT)}")
    for tag, f in summary["fits"].items():
        print(f"  {tag:12s} C={f['C']:.4f} k={f['k']:.3f} p={f['p']:.3f} "
              f"max|err|={f['max_abs_rel_err'] * 100:.2f}%")
    for name, s in summary["statements"].items():
        print(f"  {name:32s} pass={s['pass']}")
    fh = summary["fill_height_test"]
    if fh["available"]:
        print(f"  fill_height: spread={fh['relative_spread'] * 100:.2f}% pass={fh['pass_10pct']}")
    prim = summary["primary"]
    if prim.get("available"):
        print(f"  primary {prim['case_id']}: err={prim['rel_err_literature'] * 100:+.2f}% "
              f"pass={prim['pass_10pct_literature']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())