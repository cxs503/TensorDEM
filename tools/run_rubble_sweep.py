#!/usr/bin/env python3
"""IN69-N06 sweep driver: rubble accumulation around a fixed cylinder.

The pre-registered design lives in
``prereg/IN69_N06_rubble_accumulation_prereg.json``.  Three modes:

``pilot``    a couple of cases, printed to stdout (numerical health only);
``sweep``    the one-factor-at-a-time design plus the negative controls, the
             overburden ablation, the seed repetitions and the pile-clearing
             diagnostics; one JSON per case, so the work can be sharded;
``analyse``  merges every case JSON, fits the growth laws, extracts the scaling
             exponents, computes the shielding ratio, writes the summary JSON,
             the figures and the markdown tables.

Every number in the produced artefacts is computed here.

Deviation from the pre-registration (declared, not hidden)
---------------------------------------------------------
The pre-registered baseline for the shielding ratio was the load measured in a
*pile-clearing* control run (the pile is teleported out of an exclusion disc).
That control was run and found unusable: the clearing removes a fragment before
it can press on the cylinder for more than one time step, so the measured
``p_0`` is one to two orders of magnitude below the momentum flux that a bare
cylinder must absorb (numbers in ``summary.json``, section ``control_diagnostics``).
The shielding ratio therefore uses the *secondary* baseline declared in the
pre-registration: the impingement pressure of the approaching field,
``p_0 = rho_i C_i v_i^2``, which is the pressure a bare cylinder of the same
diameter absorbs in the same flux (``R_shield ~ 1`` for a bare cylinder).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensordem import rubble_accumulation as ra  # noqa: E402

BASE = {"v_i": 0.3, "h_i": 0.3, "C_i": 0.3, "D": 3.0}


# --------------------------------------------------------------------------- #
# case definitions
# --------------------------------------------------------------------------- #
def _row_tag(row: dict) -> str:
    return (f"v{row['v_i']:.2f}_h{row['h_i']:.2f}_c{row['C_i']:.2f}_D{row['D']:.1f}"
            .replace(".", "p"))


def sweep_cases() -> list[dict]:
    cases = []
    for idx, row in enumerate(ra.sweep_rows()):
        cases.append({"id": f"{idx:02d}_{_row_tag(row)}", "row": row,
                      "kind": "accumulation", "seed": 0, "clearing": False})
    for seed in (1, 2):
        cases.append({"id": f"rep_seed{seed}", "row": dict(BASE), "kind": "reproducibility",
                      "seed": seed, "clearing": False})
    cases.append({"id": "neg_C0", "row": {**BASE, "C_i": 0.0},
                  "kind": "negative_control_C_i_0", "seed": 0, "clearing": False})
    cases.append({"id": "neg_v0", "row": {**BASE, "v_i": 0.0},
                  "kind": "negative_control_v_i_0", "seed": 0, "clearing": False})
    cases.append({"id": "abl_no_overburden", "row": dict(BASE),
                  "kind": "ablation_overburden_off", "seed": 0, "clearing": False,
                  "overburden_scale": 0.0})
    for tag, row in (("diag_clear_base", dict(BASE)),
                     ("diag_clear_lowv", {**BASE, "v_i": 0.1}),
                     ("diag_clear_highC", {**BASE, "C_i": 0.5})):
        cases.append({"id": tag, "row": row, "kind": "clearing_diagnostic",
                      "seed": 0, "clearing": True})
    return cases


KN = {"value": 3.0e4}


def _config(case: dict, duration: float, sample_dt: float) -> ra.AccumulationConfig:
    domain = ra.DomainSpec()
    row = case["row"]
    return ra.AccumulationConfig(
        material=ra.RubbleMaterial(overburden_scale=case.get("overburden_scale", 1.0),
                                   kn=KN["value"]),
        domain=domain,
        structure=ra.StructureSpec(diameter=float(row["D"]),
                                   x_center=0.62 * domain.length,
                                   y_center=0.5 * domain.width),
        conditions=ra.IceConditions(velocity=float(row["v_i"]),
                                    thickness=float(row["h_i"]),
                                    concentration=float(row["C_i"])),
        duration=duration, sample_dt=sample_dt, seed=int(case.get("seed", 0)),
        clearing=bool(case["clearing"]))


def _run_one_paths(case: dict, duration: float, sample_dt: float, out_dir: str) -> dict:
    return run_one(case, duration, sample_dt, Path(out_dir))


def run_one(case: dict, duration: float, sample_dt: float,
            out_dir: Path | None = None) -> dict:
    cfg = _config(case, duration, sample_dt)
    started = time.perf_counter()
    run = ra.run_accumulation(cfg)
    payload = {
        "id": case["id"], "kind": case["kind"], "row": case["row"],
        "seed": int(case.get("seed", 0)), "clearing": bool(case["clearing"]),
        "overburden_scale": float(case.get("overburden_scale", 1.0)),
        "duration": duration, "sample_dt": sample_dt,
        "wall_seconds": time.perf_counter() - started,
        "dt": run.dt, "drag_time": run.drag_time, "field": run.field,
        "summary": run.summary(), "samples": run.samples,
        "final_state": run.final_state,
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{case['id']}.json").write_text(json.dumps(payload))
    return payload


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
GEOMETRY_KEYS = ("B_p", "H_p", "L_p")
HEAD_KEYS = ("v_i", "h_i", "C_i", "D")


def analyse_case(payload: dict) -> dict:
    samples = payload["samples"]
    time = np.array([s["time"] for s in samples], dtype=float)
    tail = time >= 0.75 * time.max()
    out = {"id": payload["id"], "kind": payload["kind"], "clearing": payload["clearing"],
           "row": payload["row"], "seed": payload["seed"],
           "overburden_scale": payload["overburden_scale"],
           "duration": payload["duration"], "wall_seconds": payload["wall_seconds"],
           "dt": payload["dt"], "drag_time": payload["drag_time"],
           "field": payload["field"], "Fr": payload["summary"]["Fr"],
           "time_series": time.tolist(),
           "max_penetration_ratio": float(np.max(
               [s["max_penetration_ratio"] for s in samples])),
           "final_state": payload.get("final_state")}
    for key in GEOMETRY_KEYS + ("pile_mass", "p_front", "pile_cluster_share",
                                "mass_closure_relative", "mass_trapped"):
        values = np.array([s.get(key, 0.0) for s in samples], dtype=float)
        tail_values = values[tail]
        out[f"{key}_plateau"] = (float(np.nanmean(tail_values))
                                 if np.isfinite(tail_values).any() else float("nan"))
        out[f"{key}_peak"] = (float(np.nanmax(values))
                              if np.isfinite(values).any() else float("nan"))
        out[f"{key}_series"] = values.tolist()
        if key in GEOMETRY_KEYS:
            fit = ra.exponential_growth_fit(time, values)
            out[f"fit_{key}"] = fit
            tau, inf = fit["tau"], fit["X_inf"]
            saturated = bool(np.isfinite(tau) and np.isfinite(inf) and inf > 0.0
                             and abs(out[f"{key}_plateau"] - inf) / inf <= 0.10
                             and tau <= 0.7 * payload["duration"])
            out[f"{key}_saturated"] = saturated
    # smoothing *only* as a descriptive companion to the raw fit (declared)
    smoothed = {}
    for key in GEOMETRY_KEYS:
        values = np.array(out[f"{key}_series"], dtype=float)
        if len(values) >= 8:
            window = max(3, len(values) // 12)
            kernel = np.ones(window) / window
            smoothed[key] = ra.exponential_growth_fit(time, np.convolve(values, kernel,
                                                                       mode="same"))
        else:
            smoothed[key] = ra.exponential_growth_fit(time, values)
    out["smoothed_fit"] = smoothed
    out["pile_cluster_share_peak"] = float(np.max(
        [s.get("pile_cluster_share", 0.0) for s in samples]))
    return out


def _momentum_baseline(row: dict) -> float:
    cond = ra.IceConditions(float(row["v_i"]), float(row["h_i"]), float(row["C_i"]))
    return cond.driving_stress


def analyse(out_dir: Path, *, make_figures: bool = True) -> dict:
    payloads = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))
                if not p.name.startswith("summary")]
    cases = [analyse_case(p) for p in payloads]
    accum = [c for c in cases if c["kind"] in ("accumulation", "reproducibility",
                                               "ablation_overburden_off")]
    result: dict = {
        "n_cases": len(cases),
        "design": {"base": BASE, "duration": accum[0]["duration"] if accum else None,
                   "dt": accum[0]["dt"] if accum else None,
                   "drag_time_base": accum[0]["drag_time"] if accum else None,
                   "domain": {"length": ra.DomainSpec().length,
                              "width": ra.DomainSpec().width}},
        "cases": [{k: v for k, v in c.items()
                   if not k.endswith("_series") and k not in ("final_state", "smoothed_fit")}
                  for c in cases],
    }
    crit = result["criteria"] = {}

    # ---------------------------------------------------------------- (a) growth
    fit_rows = []
    for c in accum:
        row = {"id": c["id"], "kind": c["kind"], "Fr": c["Fr"],
               **{k: c["row"][k] for k in HEAD_KEYS}}
        for key in GEOMETRY_KEYS:
            fit = c[f"fit_{key}"]
            row[f"{key}_X_inf"] = fit["X_inf"]
            row[f"{key}_tau"] = fit["tau"]
            row[f"{key}_r2"] = fit["r2"]
            row[f"{key}_r2_smoothed"] = c["smoothed_fit"][key]["r2"]
            row[f"{key}_plateau"] = c[f"{key}_plateau"]
            row[f"{key}_saturated"] = c[f"{key}_saturated"]
        fit_rows.append(row)
    result["growth_fits"] = fit_rows
    r2_raw = [r[f"{k}_r2"] for r in fit_rows for k in GEOMETRY_KEYS
              if np.isfinite(r[f"{k}_r2"])]
    r2_smooth = [r[f"{k}_r2_smoothed"] for r in fit_rows for k in GEOMETRY_KEYS
                 if np.isfinite(r[f"{k}_r2_smoothed"])]
    nonneg = [r for r in fit_rows if r["id"].startswith(("0", "1"))]
    r2_nonneg = [r[f"{k}_r2"] for r in nonneg for k in GEOMETRY_KEYS
                 if np.isfinite(r[f"{k}_r2"])]
    crit["a_growth_r2"] = {
        "criterion": "B_p(t), H_p(t), L_p(t) fit an exponential growth law with R^2>0.95",
        "threshold": 0.95,
        "n_series": len(r2_raw),
        "median_r2_raw": float(np.median(r2_raw)) if r2_raw else float("nan"),
        "min_r2_raw": float(np.min(r2_raw)) if r2_raw else float("nan"),
        "n_pass_raw": int(sum(v > 0.95 for v in r2_raw)),
        "median_r2_smoothed": float(np.median(r2_smooth)) if r2_smooth else float("nan"),
        "n_pass_smoothed": int(sum(v > 0.95 for v in r2_smooth)),
        "median_r2_raw_sweep_rows_only": float(np.median(r2_nonneg)) if r2_nonneg else None,
        "pass": bool(r2_raw and np.median(r2_raw) > 0.95),
        "grade": "REAL (DEM series) / curve fit",
    }

    # ------------------------------------------------------------- (b) scaling
    union = [c for c in accum if all(c[f"{k}_saturated"] for k in GEOMETRY_KEYS)
             and c["row"]["C_i"] > 0.0 and c["kind"] == "accumulation"]
    partially = [c for c in accum if c["L_p_saturated"] and c["kind"] == "accumulation"]
    scaling = {}
    exponents = {}
    for label, key, ratio in (("H_over_h", "H_p", "h_i"), ("L_over_D", "L_p", "D"),
                              ("B_over_D", "B_p", "D"), ("B_over_L", "B_p", None)):
        values, ids, fr, ci, dia = [], [], [], [], []
        for c in union + []:
            num = c[f"{key}_plateau"]
            den = 1.0 if ratio is None else float(c["row"][ratio])
            if ratio is None:
                den = max(c["L_p_plateau"], 1e-9)
            if den > 0:
                values.append(num / den)
                ids.append(c["id"])
                fr.append(c["Fr"])
                ci.append(float(c["row"]["C_i"]))
                dia.append(float(c["row"]["D"]))
        scaling[label] = {"values": values, "ids": ids, "Fr": fr, "C_i": ci, "D": dia}
    if len(union) >= 3:
        fr = np.array([c["Fr"] for c in union], dtype=float)
        ci = np.array([float(c["row"]["C_i"]) for c in union], dtype=float)
        dia = np.array([float(c["row"]["D"]) for c in union], dtype=float)
        for label, key, ratio in (("H_over_h", "H_p", "h_i"), ("L_over_D", "L_p", "D"),
                                  ("B_over_D", "B_p", "D")):
            values = np.array([c[f"{key}_plateau"] for c in union], dtype=float) / \
                np.array([float(c["row"][ratio]) for c in union], dtype=float)
            exponents[label] = {"vs_Fr": ra.loglog_exponent(fr, values),
                                "vs_C_i": ra.loglog_exponent(ci, values),
                                "vs_D": ra.loglog_exponent(dia, values)}
    result["scaling"] = scaling
    result["exponents"] = exponents
    result["analytic_reference"] = ra.momentum_balance_reference(
        ra.IceConditions(BASE["v_i"], BASE["h_i"], BASE["C_i"]))
    reference = {"H_over_h.vs_Fr": 2.0, "H_over_h.vs_C_i": 1.0,
                 "L_over_D.vs_Fr": 2.0, "B_over_D.vs_Fr": 2.0}
    deviations = {}
    for path, expected in reference.items():
        group, which = path.split(".")
        if group in exponents:
            got = exponents[group][which]["exponent"]
            if np.isfinite(got):
                deviations[path] = {"measured": got, "reference": expected,
                                    "relative_deviation": abs(got - expected) / expected}
    crit["b_scaling"] = {
        "criterion": "steady-state geometry exponents within 20% of the reference",
        "threshold": 0.20, "n_saturated_cases": len(union),
        "n_L_p_saturated": len(partially),
        "deviations": deviations,
        "pass": bool(deviations and all(d["relative_deviation"] <= 0.20
                                        for d in deviations.values())),
        "grade": "ANALYTIC momentum-balance reference; digitised model-test data WITHHELD",
    }

    # ------------------------------------------------------------ (c) shielding
    shield_rows = []
    for c in accum:
        if c["kind"] != "accumulation":
            continue
        p0 = _momentum_baseline(c["row"])
        plateau = c["p_front_plateau"]
        shield_rows.append({"id": c["id"], "Fr": c["Fr"], "C_i": float(c["row"]["C_i"]),
                            "D": float(c["row"]["D"]), "v_i": float(c["row"]["v_i"]),
                            "p_front_plateau": plateau, "p0_momentum": p0,
                            "R_shield": plateau / p0 if p0 > 0 else float("nan")})
    ordered = sorted([r for r in shield_rows if np.isfinite(r["R_shield"])],
                     key=lambda r: r["Fr"])
    crossover = None
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev["R_shield"] < 1.0 <= nxt["R_shield"]:
            crossover = {"Fr": 0.5 * (prev["Fr"] + nxt["Fr"]),
                         "between": [prev["Fr"], nxt["Fr"]]}
            break
    low = [r["R_shield"] for r in ordered if r["Fr"] <= 0.2]
    high = [r["R_shield"] for r in ordered if r["Fr"] >= 0.5]
    result["shielding"] = shield_rows
    crit["c_shielding"] = {
        "criterion": "R_shield<1 at low Fr (shielding), R_shield>1 at high Fr (amplification)",
        "baseline": "p0 = rho_i C_i v_i^2 (impingement pressure of the approaching field)",
        "n_low_Fr": len(low), "n_high_Fr": len(high),
        "mean_R_low_Fr": float(np.mean(low)) if low else float("nan"),
        "mean_R_high_Fr": float(np.mean(high)) if high else float("nan"),
        "min_R": float(np.min([r["R_shield"] for r in ordered])) if ordered else float("nan"),
        "max_R": float(np.max([r["R_shield"] for r in ordered])) if ordered else float("nan"),
        "crossover_Fr": crossover,
        "shielded_at_low_Fr": bool(low and np.mean(low) < 1.0),
        "amplified_at_high_Fr": bool(high and np.mean(high) > 1.0),
        "pass": bool(low and high and np.mean(low) < 1.0 and np.mean(high) > 1.0),
        "grade": "REAL ratio; baseline ANALYTIC",
    }

    # --------------------------------------------------------- (d) mass ledger
    closures = [c["mass_closure_relative_peak"] for c in accum
                if np.isfinite(c["mass_closure_relative_peak"])]
    trapped = [c["mass_trapped_peak"] for c in accum]
    crit["d_mass_conservation"] = {
        "criterion": "pile mass = influx - bypass with closure error < 2%",
        "threshold": 0.02,
        "max_closure_relative": float(np.max(closures)) if closures else float("nan"),
        "median_closure_relative": float(np.median(closures)) if closures else float("nan"),
        "max_trapped_mass": float(np.max(trapped)) if trapped else 0.0,
        "pass": bool(closures and np.max(closures) < 0.02),
        "grade": "BOOKKEEPING (exact if no crossing is missed and no ice is trapped)",
        "note": ("every fragment is followed individually, so this identity can only be "
                 "violated by a missed boundary crossing or by ice trapped inside the "
                 "structure; it is a numerical regression guard, not a physics validation"),
    }

    # --------------------------------------------------- (e) negative controls
    neg = {}
    for c in cases:
        if c["kind"].startswith("negative_control"):
            neg[c["kind"]] = {"id": c["id"], "pile_mass_peak": c["pile_mass_peak"],
                              "pile_discs_peak": c["pile_mass_peak"] / max(
                                  c["field"]["n_pieces"], 1) if c["field"]["n_pieces"] else 0.0,
                              "p_front_plateau": c["p_front_plateau"],
                              "L_p_peak": c["L_p_peak"]}
    crit["e_negative_controls"] = {
        "criterion": "C_i=0 -> no pile; v_i=0 -> no pile",
        "cases": neg,
        "pass": bool(all(abs(v["L_p_peak"]) < 1e-9 for v in neg.values()) and len(neg) == 2),
        "grade": "REAL",
    }

    # ------------------------------------------------------- (f) numerical guard
    pens = [c["max_penetration_ratio"] for c in cases
            if c["kind"] != "clearing_diagnostic"]
    pens_diag = [c["max_penetration_ratio"] for c in cases
                 if c["kind"] == "clearing_diagnostic"]
    critical = [c for c in cases if c["kind"] == "accumulation"]
    crit["numerical_guard"] = {
        "max_penetration_ratio_all_production_cases": float(np.max(pens)) if pens else float("nan"),
        "max_penetration_ratio_sweep_rows": float(np.max(
            [c["max_penetration_ratio"] for c in critical])) if critical else float("nan"),
        "max_penetration_ratio_clearing_diagnostics": float(np.max(pens_diag))
        if pens_diag else None,
        "threshold": 0.05,
        "pass": bool(pens and np.max(pens) <= 0.05),
        "grade": "REAL",
    }

    # -------------------------------------------- reproducibility + control diag
    reps = {c["id"]: {"L_p_plateau": c["L_p_plateau"], "H_p_plateau": c["H_p_plateau"],
                      "B_p_plateau": c["B_p_plateau"], "p_front_plateau": c["p_front_plateau"]}
            for c in accum if c["kind"] == "reproducibility"}
    seed0 = next((c for c in accum if c["id"].startswith("00_")), None)
    if seed0 is not None:
        reps["seed0"] = {"L_p_plateau": seed0["L_p_plateau"],
                         "H_p_plateau": seed0["H_p_plateau"],
                         "B_p_plateau": seed0["B_p_plateau"],
                         "p_front_plateau": seed0["p_front_plateau"]}
    spread = {}
    for key in ("L_p_plateau", "H_p_plateau", "B_p_plateau", "p_front_plateau"):
        values = [v[key] for v in reps.values() if key in v]
        mean = float(np.mean(values)) if values else float("nan")
        spread[key] = {"values": values, "mean": mean,
                       "relative_spread": float(np.std(values) / mean) if mean else float("nan")}
    result["reproducibility"] = spread
    clear = [c for c in cases if c["kind"] == "clearing_diagnostic"]
    result["control_diagnostics"] = [
        {"id": c["id"], "Fr": c["Fr"], "p_front_plateau": c["p_front_plateau"],
         "p0_momentum": _momentum_baseline(c["row"]),
         "ratio_to_momentum": (c["p_front_plateau"] / _momentum_baseline(c["row"])
                               if _momentum_baseline(c["row"]) > 0 else float("nan"))}
        for c in clear]
    result["ablation"] = {
        c["kind"]: {"id": c["id"], "L_p_plateau": c["L_p_plateau"],
                    "H_p_plateau": c["H_p_plateau"], "p_front_plateau": c["p_front_plateau"]}
        for c in cases if c["kind"] == "ablation_overburden_off"}
    if seed0 is not None:
        result["ablation"]["overburden_on_base"] = {
            "id": seed0["id"], "L_p_plateau": seed0["L_p_plateau"],
            "H_p_plateau": seed0["H_p_plateau"], "p_front_plateau": seed0["p_front_plateau"]}

    result["verdict"] = {k: bool(v.get("pass")) for k, v in crit.items()}
    if make_figures:
        result["figures"] = make_figures_from(result, cases, out_dir)
    return result


def make_figures_from(result: dict, cases: list[dict], out_dir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    made = []

    base = next((c for c in cases if c["id"].startswith("00_")), None)
    if base is not None:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        t = np.array(base["time_series"])
        for key, label in (("L_p", "upstream length $L_p$"), ("B_p", "cross-stream width $B_p$"),
                           ("H_p", "pile thickness $H_p$")):
            y = np.array(base[f"{key}_series"])
            ax.plot(t, y, ".", ms=3, alpha=0.5)
            fit = base[f"fit_{key}"]
            tt = np.linspace(0, t.max(), 200)
            ax.plot(tt, fit["X_inf"] * (1 - np.exp(-tt / fit["tau"])),
                    label=f"{label}: $X_\\infty$={fit['X_inf']:.2f} m, "
                          f"$\\tau$={fit['tau']:.1f} s, $R^2$={fit['r2']:.2f}")
        ax.set_xlabel("time t [s]")
        ax.set_ylabel("pile geometry [m]")
        ax.set_title(f"Rubble growth, base case v={BASE['v_i']} m/s, h={BASE['h_i']} m, "
                     f"C={BASE['C_i']}, D={BASE['D']} m")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = fig_dir / "fig1_growth_base.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append({"name": "fig1_growth_base.png", "path": str(path),
                     "caption": "Pile geometry growth and exponential fits (base case)"})

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    sc = result["scaling"]
    for label, marker, name in (("L_over_D", "o", "$L_\\infty/D$"),
                                ("B_over_D", "s", "$B_\\infty/D$"),
                                ("H_over_h", "^", "$H_\\infty/h_i$")):
        if sc[label]["values"]:
            ax.loglog(sc[label]["Fr"], sc[label]["values"], marker, label=name)
    ex = result["exponents"]
    for group, style in (("L_over_D", "--"), ("H_over_h", ":")):
        if group in ex:
            slope = ex[group]["vs_Fr"]["exponent"]
            if np.isfinite(slope):
                fr = np.array(sc[group]["Fr"])
                ref = np.median(np.array(sc[group]["values"]) / (fr ** slope))
                ax.plot(sorted(fr), ref * np.array(sorted(fr)) ** slope, style,
                        label=f"{group}: fitted slope {slope:.2f}")
                ax.plot(sorted(fr), ref * np.array(sorted(fr)) ** 2.0, "-.", alpha=0.4,
                        label=f"{group}: reference slope 2")
    ax.set_xlabel("Froude number Fr = v_i/sqrt(g h_i)")
    ax.set_ylabel("normalised pile geometry")
    ax.set_title("Steady pile geometry vs Froude number")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    path = fig_dir / "fig2_scaling_froude.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append({"name": "fig2_scaling_froude.png", "path": str(path),
                 "caption": "Steady pile geometry vs Fr with fitted and reference slopes"})

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    rows = sorted([r for r in result["shielding"] if np.isfinite(r["R_shield"])],
                  key=lambda r: r["Fr"])
    if rows:
        ax.plot([r["Fr"] for r in rows], [r["R_shield"] for r in rows], "o-",
                label="$R_{shield} = p_{front}/(\\rho_i C_i v_i^2)$")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="no pile (R=1)")
        cross = result["criteria"]["c_shielding"]["crossover_Fr"]
        if cross:
            ax.axvline(cross["Fr"], color="tab:red", ls=":", label=f"crossover Fr={cross['Fr']:.2f}")
    ax.set_xlabel("Froude number Fr")
    ax.set_ylabel("$R_{shield}$")
    ax.set_title("Pile shielding / amplification of the cylinder load")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = fig_dir / "fig3_shielding.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append({"name": "fig3_shielding.png", "path": str(path),
                 "caption": "Shielding ratio vs Froude number (momentum-flux baseline)"})

    if base is not None and base.get("final_state"):
        state = base["final_state"]
        pos = np.array(state["positions"])
        radii = np.array(state["radii"])
        thick = np.array(state["thickness"])
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        im = axes[0].imshow(thick.T, origin="lower", aspect="auto", cmap="viridis",
                            extent=[0, ra.DomainSpec().length, 0, ra.DomainSpec().width])
        axes[0].add_patch(plt.Circle(tuple(state["axis"]), 0.5 * state["diameter"],
                                     color="white", ec="k"))
        axes[0].set_xlabel("x [m]")
        axes[0].set_ylabel("y [m]")
        axes[0].set_title("Pile thickness field $H_p(x,y)$ [m], end of base run")
        fig.colorbar(im, ax=axes[0], label="$H_p$ [m]")
        axes[1].scatter(pos[:, 0], pos[:, 1], s=radii * 60, c="tab:blue", alpha=0.6)
        axes[1].add_patch(plt.Circle(tuple(state["axis"]), 0.5 * state["diameter"],
                                     color="tab:red"))
        dx = pos[:, 0] - state["axis"][0]
        axes[1].annotate("", xy=(state["axis"][0], state["axis"][1] + 2),
                         xytext=(state["axis"][0] - 3, state["axis"][1] + 2),
                         arrowprops={"arrowstyle": "->", "color": "k"})
        axes[1].text(state["axis"][0] - 3.2, state["axis"][1] + 2.4, "$v_i$", fontsize=10)
        axes[1].set_xlabel("x [m]")
        axes[1].set_ylabel("y [m]")
        axes[1].set_title("Fragment field (plan view), end of base run")
        axes[1].set_aspect("equal")
        path = fig_dir / "fig4_field_snapshot.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append({"name": "fig4_field_snapshot.png", "path": str(path),
                     "caption": "Pile thickness field and plan-view fragment field at t_end"})
    return made


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "sweep", "analyse"), default="pilot")
    parser.add_argument("--out", default=str(REPO_ROOT / "runs" / "IN69_N06"))
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--sample-dt", type=float, default=0.5)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--kn", type=float, default=3.0e4,
                        help="contact stiffness [N/m]; the 10x refinement run uses 3e5")
    parser.add_argument("--skip-diagnostics", action="store_true",
                        help="drop the pile-clearing diagnostic cases")
    parser.add_argument("--jobs", type=int, default=1,
                        help="parallel worker processes for --mode sweep")
    parser.add_argument("--ids", nargs="*", default=None)
    args = parser.parse_args(argv)
    KN["value"] = float(args.kn)
    out_dir = Path(args.out)

    if args.mode == "pilot":
        cases = sweep_cases()[:2]
        for case in cases:
            payload = run_one(case, args.duration, args.sample_dt, out_dir)
            final = payload["samples"][-1]
            print(f"{case['id']}: wall={payload['wall_seconds']:.1f}s "
                  f"n={payload['field']['n_pieces']:.0f} dt={payload['dt']:.2e} "
                  f"pile={final['pile_discs']:.0f} L_p={final['L_p']:.2f} "
                  f"pen={final['max_penetration_ratio']:.3f}", flush=True)
        return 0

    if args.mode == "sweep":
        cases = sweep_cases()
        if args.skip_diagnostics:
            cases = [c for c in cases if c["kind"] != "clearing_diagnostic"]
        if args.ids:
            cases = [c for c in cases if c["id"] in args.ids]
        todo = [c for k, c in enumerate(cases) if k % args.shards == args.shard]
        if args.jobs > 1:
            import multiprocessing as mp
            with mp.Pool(args.jobs) as pool:
                args_list = [(c, args.duration, args.sample_dt, str(out_dir)) for c in todo]
                for payload in pool.starmap(_run_one_paths, args_list):
                    final = payload["samples"][-1]
                    print(f"{payload['id']}: wall={payload['wall_seconds']:.0f}s "
                          f"pile={final['pile_discs']:.0f} L_p={final['L_p']:.2f} "
                          f"pen={final['max_penetration_ratio']:.3f}", flush=True)
            return 0
        for case in todo:
            payload = run_one(case, args.duration, args.sample_dt, out_dir)
            final = payload["samples"][-1]
            print(f"[shard {args.shard}] {case['id']}: wall={payload['wall_seconds']:.0f}s "
                  f"pile={final['pile_discs']:.0f} L_p={final['L_p']:.2f} "
                  f"pen={final['max_penetration_ratio']:.3f}", flush=True)
        return 0

    result = analyse(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())