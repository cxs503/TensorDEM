#!/usr/bin/env python3
"""Seed-ensemble uncertainty budget for the Beverloo discharge rate (m22657).

Frozen estimator (see ``prereg.md`` §4): the ledger quantity is the ensemble mean
of the per-seed discharge rate with a Student-t 95 % interval, compared against
the *literature* coefficients ``C = 0.58``, ``k = 1.4`` — no refitted amplitude
is allowed to rescue the tolerance.

Everything is computed from the per-case JSON files that
``python -m tensordem.silo_flow.study beverloo`` wrote; nothing is hand copied.

Arrest handling (m23184 rule, in main since #1748): a record whose
``discharge_status`` is ``ARRESTED`` describes a bed that arched over the
orifice; its "flow rate" is an arch, not a measurement of the law.  Such
records are reported as-is in ``per_seed`` (prereg.md §7: never resampled)
but are excluded from the ensemble mean, the CI and the V1/V2/V3 verdicts.
With fewer than two flowing seeds the verdict is ``NOT_DECIDABLE`` with the
arching diagnosis - never a point FAIL that would blame the law.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REF_PATH = ROOT / "benchmarks" / "m20055_beverloo" / "ref_beverloo_1961.json"

TOLERANCE = 0.10          # feature_ledger CASE-DEM-SILO-BEVERLOO
STEADY_DRIFT_TOL = 0.10   # V3
RHO_SPREAD_TOL = 0.03     # V4
REPRO_TOL = 0.10          # V2


# --------------------------------------------------------------- statistics
def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-16) -> float:
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)``."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def student_cdf(t: float, df: int) -> float:
    """CDF of Student's t (df > 0)."""
    if df <= 0:
        raise ValueError("df must be positive")
    tail = 0.5 * betainc(0.5 * df, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0.0 else tail


def student_quantile(p: float, df: int) -> float:
    """Quantile of Student's t by bisection on :func:`student_cdf`."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must lie in (0, 1)")
    if df <= 0:
        raise ValueError("df must be positive")
    lo, hi = -1.0e3, 1.0e3
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if student_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def window_trend(rates: np.ndarray) -> dict:
    """Least-squares drift of the window rates over the counted windows.

    A draining bed is not strictly steady: the measured rate drifts while the
    free surface falls.  The normalised drift is ``slope * (n-1) / mean``, i.e.
    the change predicted across the whole counted window set, relative to the
    mean rate — comparable with the 10 % tolerance.
    """
    r = np.asarray(rates, dtype=np.float64)
    n = r.size
    if n < 2:
        return {"n": int(n), "slope": float("nan"), "rel_drift": float("nan"), "t_stat": float("nan")}
    x = np.arange(n, dtype=np.float64)
    xc = x - x.mean()
    sxx = float(np.sum(xc * xc))
    slope = float(np.sum(xc * (r - r.mean())) / sxx)
    resid = r - (r.mean() + slope * xc)
    dof = n - 2
    s2 = float(np.sum(resid ** 2) / dof) if dof > 0 else float("nan")
    se_slope = math.sqrt(s2 / sxx) if s2 > 0.0 else float("nan")
    t_stat = slope / se_slope if se_slope and math.isfinite(se_slope) else float("nan")
    mean = float(r.mean())
    return {
        "n": int(n),
        "slope_kg_s_per_window": slope,
        "rel_drift": slope * (n - 1) / mean if mean > 0 else float("nan"),
        "t_stat": t_stat,
        "mean_kg_s": mean,
    }


def poisson_noise_budget(particles_per_window: list[float], mass_per_particle: float,
                         window_time: float) -> dict:
    """Expected relative noise of the mean rate from counting statistics alone.

    ``W`` is a count of particles per unit time, so ``sd(W_window)/W_window =
    1/sqrt(N_window)`` and the mean of ``k`` independent windows has
    ``1/sqrt(sum N_window)`` — the intrinsic measurement floor of the case.
    """
    n = np.asarray(particles_per_window, dtype=np.float64)
    total = float(np.sum(n))
    if total <= 0.0:
        return {"n_exit_counted": 0.0, "expected_rel_noise": float("nan")}
    return {
        "n_exit_counted": total,
        "mass_counted_kg": total * mass_per_particle,
        "mean_rate_kg_s": total * mass_per_particle / (window_time * n.size),
        "expected_rel_noise": 1.0 / math.sqrt(total),
    }


def ensemble_stats(values: np.ndarray) -> dict:
    """Mean, sd, se, Student-t 95 % CI and dispersion of the per-seed rates."""
    v = np.asarray(values, dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return {"n": 0}
    mean = float(v.mean())
    if n < 2:
        return {"n": n, "mean_kg_s": mean, "sd_kg_s": float("nan"),
                "se_kg_s": float("nan"), "ci95_kg_s": [float("nan"), float("nan")],
                "cv": float("nan"), "t_crit": float("nan")}
    sd = float(v.std(ddof=1))
    se = sd / math.sqrt(n)
    tcrit = student_quantile(0.975, n - 1)
    return {
        "n": n,
        "mean_kg_s": mean,
        "sd_kg_s": sd,
        "se_kg_s": se,
        "t_crit": tcrit,
        "ci95_kg_s": [mean - tcrit * se, mean + tcrit * se],
        "cv": sd / mean if mean > 0 else float("nan"),
    }


def beverloo_reference(orifice: float, particle: float, rho_bulk: float,
                       c: float = 0.58, k: float = 1.4, gravity: float = 9.81) -> float:
    """``W = C rho_b sqrt(g) (D − k d)^(5/2)`` with literature coefficients."""
    eff = orifice - k * particle
    if eff <= 0.0:
        return 0.0
    return c * rho_bulk * math.sqrt(gravity) * eff ** 2.5


# -------------------------------------------------------------------- driver
def load_cases(run_dir: Path) -> list[dict]:
    files = sorted(Path(run_dir).glob("seed*.json"))
    out = []
    for p in files:
        d = json.loads(p.read_text())
        case = d["case"]
        d["_path"] = str(p)
        d["_seed"] = int(case.get("seed", -1))
        out.append(d)
    return out


def analyse(run_dir: Path | str | None = None, ref_path: Path | str | None = None) -> dict:
    run_dir = Path(run_dir or ROOT / "runs" / "m22657_beverloo_ensemble")
    ref = json.loads(Path(ref_path or REF_PATH).read_text())
    c_lit = float(ref["coefficients"]["C"])
    k_lit = float(ref["coefficients"]["k"])

    cases = load_cases(run_dir)
    summary: dict = {
        "run_dir": str(run_dir),
        "reference": {"C": c_lit, "k": k_lit, "source": str(ref_path or REF_PATH),
                      "citation": ref.get("citation", "")},
        "tolerance": TOLERANCE,
        "n_cases": len(cases),
        "per_seed": [],
    }
    if not cases:
        summary["verdicts"] = {"V1_ledger": "NOT_RUN", "V2_reproducibility": "NOT_RUN",
                               "V3_steady_state": "NOT_RUN", "V4_rho_control": "NOT_RUN"}
        summary["reason"] = "no seed*.json artefacts in the run directory"
        return summary

    rates, rhos = [], []
    for c in cases:
        case = c["case"]
        D, d = float(case["orifice_diameter"]), float(case["particle_diameter"])
        rho_b = float(c["rho_bulk"])
        rates_w = np.asarray(c["flow_rate_windows_kg_s"], dtype=np.float64)
        n_steady = max(1, rates_w.size // 2)
        used = rates_w[-n_steady:]
        W = float(np.mean(used))
        W_ref = beverloo_reference(D, d, rho_b, c=c_lit, k=k_lit)
        n_bed = int(c["n_bed"])
        m_particle = float(c["bed_mass_kg"]) / n_bed if n_bed else float("nan")
        window_time = float(c["window_time_s"]) if "window_time_s" in c else float("nan")
        # particles that left during the counted windows = mass / single-particle mass
        n_counted = float(np.sum(used)) * window_time / m_particle if m_particle else float("nan")
        trend = window_trend(used)
        noise = poisson_noise_budget([n_counted / n_steady] * n_steady, m_particle, window_time)
        # m23184 rule: an arrested record (bed arched over the orifice) measures
        # an arch, not a flow rate - it is reported as-is (prereg.md §7) but must
        # never be averaged into the Beverloo verdict.
        status = c.get("discharge_status") or {}
        arrested = bool(status.get("arrested", False))
        if not arrested:
            rates.append(W)
            rhos.append(rho_b)
        summary["per_seed"].append({
            "seed": c["_seed"],
            "file": Path(c["_path"]).name,
            "discharge_status": status.get("status", "UNKNOWN"),
            "arrested": arrested,
            "arrest_reasons": list(status.get("reasons", [])),
            "orifice_diameter_m": D,
            "particle_diameter_m": d,
            "D_over_d": D / d,
            "silo_diameter_over_D": 2.0 * float(case["silo_radius"]) / D,
            "fill_over_D": float(case["fill_height"]) / D,
            "n_bed": n_bed,
            "rho_bulk_kg_m3": rho_b,
            "flow_rate_windows_kg_s": [float(x) for x in rates_w],
            "windows_used": n_steady,
            "W_dem_kg_s": W,
            "W_ref_kg_s": W_ref,
            "rel_err": (W / W_ref - 1.0) if W_ref > 0 else float("nan"),
            "C_recovered": (W / (rho_b * math.sqrt(9.81) * (D - k_lit * d) ** 2.5)
                            if D > k_lit * d else float("nan")),
            "window_trend": trend,
            "poisson_budget": noise,
            "particles_exited": int(c.get("particles_exited", -1)),
            "n_final": int(c.get("n_final", -1)),
            "dt": float(c.get("dt", float("nan"))),
        })

    n_arrested = len(cases) - len(rates)
    summary["n_arrested"] = n_arrested
    summary["arrested_seeds"] = [p["seed"] for p in summary["per_seed"] if p["arrested"]]
    rho_all = [p["rho_bulk_kg_m3"] for p in summary["per_seed"]]
    rho_spread = (max(rho_all) - min(rho_all)) / float(np.mean(rho_all))
    if len(rates) < 2:
        # fewer than two flowing seeds: no interval can be formed, and an
        # arrested rate must not be averaged in (m23184) - the honest verdict
        # is NOT_DECIDABLE with the arching diagnosis, not a point FAIL.
        summary["ensemble"] = {}
        summary["verdicts"] = {
            "V1_ledger": "NOT_DECIDABLE",
            "V2_reproducibility": "NOT_DECIDABLE",
            "V3_steady_state": "NOT_DECIDABLE",
            "V4_rho_control": "PASS" if rho_spread < RHO_SPREAD_TOL else "FAIL",
            "supporting": {
                "n_flow_seeds": len(rates),
                "n_arrested_seeds": n_arrested,
                "V4_relative_rho_spread": rho_spread,
            },
        }
        summary["note"] = (
            f"{n_arrested} of {len(cases)} seeds are ARRESTED (arching at the orifice); "
            "arrested records are excluded from the ensemble (m23184 rule) and "
            f"only {len(rates)} flowing seed(s) remain - no interval verdict is possible")
        return summary

    stats = ensemble_stats(np.asarray(rates))
    # reference at the ensemble-mean bulk density, so the verdict is one number
    rho_mean = float(np.mean(rhos))
    c0 = summary["per_seed"][0]
    W_ref_ens = beverloo_reference(c0["orifice_diameter_m"], c0["particle_diameter_m"],
                                   rho_mean, c=c_lit, k=k_lit)
    rel_ens = stats["mean_kg_s"] / W_ref_ens - 1.0
    lo, hi = stats["ci95_kg_s"]
    rel_lo = lo / W_ref_ens - 1.0
    rel_hi = hi / W_ref_ens - 1.0
    summary["ensemble"] = {
        "rho_bulk_mean_kg_m3": rho_mean,
        "W_ref_kg_s": W_ref_ens,
        "W_ens_kg_s": stats["mean_kg_s"],
        "W_ens_sd_kg_s": stats["sd_kg_s"],
        "W_ens_se_kg_s": stats["se_kg_s"],
        "W_ens_ci95_kg_s": stats["ci95_kg_s"],
        "t_crit": stats["t_crit"],
        "rel_err_ensemble": rel_ens,
        "rel_err_ci95": [rel_lo, rel_hi],
        "cv": stats["cv"],
        "ci_half_width_rel": 0.5 * (rel_hi - rel_lo),
        "C_recovered_ensemble": stats["mean_kg_s"] / (
            rho_mean * math.sqrt(9.81) * (c0["orifice_diameter_m"] - k_lit * c0["particle_diameter_m"]) ** 2.5),
        "single_seed_rel_err": [p["rel_err"] for p in summary["per_seed"] if not p["arrested"]],
        "single_seed_worst_rel_err": max(abs(p["rel_err"])
                                        for p in summary["per_seed"] if not p["arrested"]),
        "expected_rel_noise_mean": float(np.mean([p["poisson_budget"]["expected_rel_noise"]
                                                  for p in summary["per_seed"] if not p["arrested"]])),
        "n_flow_seeds": len(rates),
        "n_arrested_seeds": n_arrested,
    }

    # ---- verdicts (frozen rules, prereg.md §6)
    inside_point = abs(rel_ens) < TOLERANCE
    inside_ci = (abs(rel_lo) < TOLERANCE) and (abs(rel_hi) < TOLERANCE)
    if not inside_point:
        v1 = "FAIL"
    elif inside_ci:
        v1 = "PASS"
    else:
        v1 = "NOT_DECIDABLE"
    v2 = "PASS" if stats["cv"] < REPRO_TOL else "FAIL"
    drifts = [abs(p["window_trend"]["rel_drift"])
              for p in summary["per_seed"] if not p["arrested"]]
    v3 = "PASS" if max(drifts) < STEADY_DRIFT_TOL else "FAIL"
    v4 = "PASS" if rho_spread < RHO_SPREAD_TOL else "FAIL"
    summary["verdicts"] = {
        "V1_ledger": v1,
        "V2_reproducibility": v2,
        "V3_steady_state": v3,
        "V4_rho_control": v4,
        "supporting": {
            "V1_point_inside": bool(inside_point),
            "V1_ci_inside": bool(inside_ci),
            "V2_cv": stats["cv"],
            "V3_max_abs_rel_drift": max(drifts),
            "V4_relative_rho_spread": rho_spread,
            "n_flow_seeds": len(rates),
            "n_arrested_seeds": n_arrested,
        },
    }
    return summary


def format_table(summary: dict) -> str:
    lines = []
    hdr = (f"{'seed':>4} {'n_bed':>6} {'rho_b':>8} {'W_dem':>10} {'W_ref':>10} "
           f"{'rel%':>7} {'C_rec':>6} {'drift%':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for p in summary.get("per_seed", []):
        lines.append(
            f"{p['seed']:>4} {p['n_bed']:>6} {p['rho_bulk_kg_m3']:>8.1f} "
            f"{p['W_dem_kg_s']:>10.6f} {p['W_ref_kg_s']:>10.6f} "
            f"{100 * p['rel_err']:>+7.2f} {p['C_recovered']:>6.3f} "
            f"{100 * p['window_trend']['rel_drift']:>+7.2f}")
    e = summary.get("ensemble")
    if e:
        lines.append("-" * len(hdr))
        lines.append(
            f"{'ens':>4} {'':>6} {e['rho_bulk_mean_kg_m3']:>8.1f} "
            f"{e['W_ens_kg_s']:>10.6f} {e['W_ref_kg_s']:>10.6f} "
            f"{100 * e['rel_err_ensemble']:>+7.2f} {e['C_recovered_ensemble']:>6.3f}")
        lines.append(f"  95% CI on rel err: [{100 * e['rel_err_ci95'][0]:+.2f}%, "
                     f"{100 * e['rel_err_ci95'][1]:+.2f}%]  cv={100 * e['cv']:.2f}%")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Beverloo seed-ensemble analysis")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.dir) if args.dir else ROOT / "runs" / "m22657_beverloo_ensemble"
    summary = analyse(run_dir)
    print(format_table(summary))
    print("\nVerdicts:", json.dumps(summary["verdicts"], indent=1))
    out = Path(args.out) if args.out else run_dir / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {out}")
    if args.figures:
        render_figures(summary, Path(args.figures))
    return 0


def render_figures(summary: dict, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    per = summary.get("per_seed", [])
    if not per:
        return made
    e = summary.get("ensemble", {})

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    seeds = [p["seed"] for p in per]
    rel = [100 * p["rel_err"] for p in per]
    ax.axhspan(-100 * TOLERANCE, 100 * TOLERANCE, color="0.85", label="ledger ±10 %")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.plot(seeds, rel, "o", ms=6, label="single seed")
    if e:
        centre = 100 * e["rel_err_ensemble"]
        yerr = [[centre - 100 * e["rel_err_ci95"][0]], [100 * e["rel_err_ci95"][1] - centre]]
        ax.errorbar([max(seeds) + 1], [centre], yerr=yerr, fmt="s", color="tab:red",
                    capsize=4, label="ensemble mean, 95 % CI")
    ax.set_xlabel("packing seed")
    ax.set_ylabel(r"$(W_{\rm DEM}/W_{\rm Beverloo} - 1)$  [%]")
    ax.set_title("Beverloo D/d = 5: single-seed scatter vs ensemble interval")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p1 = out_dir / "m22657_ensemble_errors.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(str(p1))

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for p in per:
        w = p["flow_rate_windows_kg_s"]
        n_used = p["windows_used"]
        idx = np.arange(1, len(w) + 1)
        ax.plot(idx, w, "o-", ms=3, lw=1, alpha=0.75,
                label=f"seed {p['seed']}" if len(per) <= 6 else None)
        if n_used < len(w):
            ax.axvspan(len(w) - n_used + 0.5, len(w) + 0.5, color="0.9", zorder=0)
    if e:
        ax.axhline(e["W_ref_kg_s"], color="k", ls="--", lw=1.2, label="Beverloo (C=0.58, k=1.4)")
        ax.axhspan(e["W_ens_ci95_kg_s"][0], e["W_ens_ci95_kg_s"][1], color="tab:red", alpha=0.15,
                   label="ensemble 95 % CI")
    ax.set_xlabel("measurement window index (0.05 s each)")
    ax.set_ylabel(r"window rate $W$ [kg/s]")
    ax.set_title("window-rate history (shaded = counted windows)")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    p2 = out_dir / "m22657_window_history.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(str(p2))
    return made


if __name__ == "__main__":
    raise SystemExit(main())