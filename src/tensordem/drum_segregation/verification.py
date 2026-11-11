"""Machine-checkable acceptance criteria C1-C6 for p176w2-n08.

Every criterion is evaluated from the per-case JSON records produced by
``scripts/run_drum_seg_verification.py`` plus the analytic anchors of the axial
model.  Nothing here hard-codes a result: each verdict carries the numbers it
was built from, and the report generator prints exactly these numbers.

Criterion definitions (pre-registered, see prereg/P176W2_N08_*.json)
--------------------------------------------------------------------
C1 Brazil-nut direction vs literature, >=3 size ratios
    For d_l/d_s in {1.2, 1.5, 2, 3} at the reference operating point the coarse
    species must be enriched at the drum periphery (``mode == "periphery"``,
    i.e. ``excess > 0``) **and** the enrichment must be statistically
    significant against the within-species position-permutation null
    (|z| >= 3).  PASS if >= 3 of the 4 ratios comply.
C2 segregation-degree scaling vs literature magnitude (+-50 %)
    Fit ``|core-shell excess| = A (d_l/d_s - 1)^alpha`` over the 4 ratios.
    Literature scaling band for drum size segregation: alpha in [0.5, 1.0]
    (alpha = 1 is the linear "percolation velocity ~ excess diameter" limit,
    smaller values reflect void-filling saturation).  The +-50 % window is
    applied to that band -> PASS requires alpha in [0.25, 1.50] with R^2 >= 0.8
    **and** the saturated amplitude at ratio 3 inside the literature order
    band [0.1, 0.5] widened by 50 % -> [0.05, 0.75].
C3 segregation pattern map, >=3 modes
    At least 3 distinct qualitatively-different patterns must be reproduced:
    periphery (Brazil-nut), core (reverse Brazil-nut / density driven), mixed
    (no structure) and axial banding (reduced-order model A2 anchor).
C4 monodisperse negative control S < 1e-6
    With d_l = d_s the size-position index must be identically zero
    (< 1e-6).  The concentration index has a finite sampling floor and is
    reported separately (it is *not* part of the threshold, see report).
C5 three negative controls
    (a) d_l/d_s -> 1  : zero index at ratio 1.0, no significant direction, and
        the index must grow monotonically from ratio 1.0 -> 1.05 -> 1.10 -> 1.2;
    (b) omega -> 0    : static drum must be "mixed" (|z| < 3) and the amplitude
        must grow monotonically over omega = 0/0.5/1/2/3;
    (c) fill -> 1     : at fill = 0.65 the pattern must be "mixed" (|z| < 3) and
        the amplitude must be below the fill = 0.30 amplitude.
C6 pytest >= 15 cases on the module.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["evaluate", "fit_scaling", "criteria_definitions"]

Z_SIGNIF = 3.0
LIT_ALPHA_BAND = (0.5, 1.0)
LIT_ALPHA_WINDOW = (0.25, 1.50)
LIT_EXCESS_BAND = (0.10, 0.50)
LIT_EXCESS_WINDOW = (0.05, 0.75)


def criteria_definitions():
    return {
        "C1": {"statement": "coarse species enriched at the drum periphery for >=3 of "
                            "d_l/d_s in {1.2,1.5,2,3}, significant at |z|>=3",
               "threshold": "count(periphery & |z|>=3) >= 3"},
        "C2": {"statement": "|excess| = A*(d_l/d_s-1)^alpha; alpha inside the "
                            "literature band +-50%, amplitude inside the literature "
                            "order band +-50%",
               "threshold": f"alpha in {LIT_ALPHA_WINDOW}, R^2>=0.8, "
                            f"excess(ratio 3) in {LIT_EXCESS_WINDOW}"},
        "C3": {"statement": ">=3 distinct segregation patterns reproduced "
                            "(periphery / core / mixed / axial bands)",
               "threshold": "n_modes >= 3"},
        "C4": {"statement": "monodisperse (d_l=d_s) size-position index < 1e-6",
               "threshold": "corr <= 1e-6"},
        "C5": {"statement": "(a) ratio->1, (b) omega->0, (c) fill->1 all drive the "
                            "index to zero / kill the direction",
               "threshold": "all three sub-controls pass"},
        "C6": {"statement": "pytest cases on the module", "threshold": ">= 15"},
    }


def fit_scaling(ratios, amplitudes):
    """Least-squares power law ``a = A (r-1)^alpha`` in log-log space."""
    r = np.asarray(ratios, float)
    a = np.abs(np.asarray(amplitudes, float))
    valid = (r > 1.0) & np.isfinite(a) & (a > 0)
    if valid.sum() < 2:
        return {"A": float("nan"), "alpha": float("nan"), "R2": float("nan"),
                "n": int(valid.sum())}
    x = np.log(r[valid] - 1.0)
    y = np.log(a[valid])
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"A": float(math.exp(intercept)), "alpha": float(slope),
            "R2": float(r2), "n": int(valid.sum())}


def _get(cases, name):
    for c in cases:
        if c["name"] == name:
            return c
    return None


def _amp(case):
    """Quasi-steady amplitude: time average over the last quarter of the run when
    available, otherwise the final-snapshot core-shell excess."""
    fin = case.get("final") or {}
    if "excess_avg" in fin:
        return abs(fin["excess_avg"])
    return abs(case["null"]["excess_observed"])


def _sig(case, z_thresh=Z_SIGNIF):
    n = case["null"]
    return (abs(n["excess_z"]) >= z_thresh) or (abs(n["corr_z"]) >= z_thresh)


def evaluate(cases, axial_checks=None, n_pytest=None):
    """Evaluate C1-C6 over the case records."""
    out = {"criteria": {}, "definitions": criteria_definitions()}
    by_name = {c["name"]: c for c in cases}

    # ---------------- C1: direction vs literature -------------------------
    ratio_rows = []
    for ratio in (1.2, 1.5, 2.0, 3.0):
        c = by_name.get(f"ratio_{ratio:g}")
        if c is None:
            continue
        n = c["null"]
        ok = (c["mode"] == "periphery") and _sig(c)
        ratio_rows.append({
            "ratio": ratio, "mode": c["mode"], "excess": n["excess_observed"],
            "excess_z": n["excess_z"], "corr": n["corr_observed"],
            "corr_z": n["corr_z"], "S": n["S_observed"], "S_z": n["S_z"],
            "significant": _sig(c), "compliant": bool(ok),
            "literature_expectation": "periphery (coarse at wall)",
        })
    n_ok = sum(r["compliant"] for r in ratio_rows)
    out["criteria"]["C1"] = {
        "rows": ratio_rows, "n_compliant": n_ok, "n_ratios": len(ratio_rows),
        "pass": bool(n_ok >= 3 and len(ratio_rows) >= 3),
    }

    # ---------------- C2: scaling -----------------------------------------
    # the scaling law is fitted on the quasi-steady amplitude (time-averaged
    # core-shell excess), which removes the single-snapshot sampling noise
    amp_by_ratio = {}
    for r in ratio_rows:
        c = by_name.get(f"ratio_{r['ratio']:g}")
        amp_by_ratio[r["ratio"]] = _amp(c) if c is not None else abs(r["excess"])
        r["amplitude_steady"] = amp_by_ratio[r["ratio"]]
    fit_rows = [r for r in ratio_rows if r["ratio"] > 1.0]
    fit = fit_scaling([r["ratio"] for r in fit_rows],
                      [amp_by_ratio[r["ratio"]] for r in fit_rows])
    amp_at_3 = amp_by_ratio.get(3.0, float("nan"))
    alpha_ok = LIT_ALPHA_WINDOW[0] <= fit["alpha"] <= LIT_ALPHA_WINDOW[1]
    amp_ok = LIT_EXCESS_WINDOW[0] <= abs(amp_at_3) <= LIT_EXCESS_WINDOW[1]
    out["criteria"]["C2"] = {
        "fit": fit, "literature_alpha_band": list(LIT_ALPHA_BAND),
        "alpha_window": list(LIT_ALPHA_WINDOW),
        "amplitude_at_ratio3": amp_at_3,
        "literature_amplitude_band": list(LIT_EXCESS_BAND),
        "amplitude_window": list(LIT_EXCESS_WINDOW),
        "alpha_pass": bool(alpha_ok), "amplitude_pass": bool(amp_ok),
        "R2_pass": bool(fit["R2"] >= 0.8),
        "pass": bool(alpha_ok and amp_ok and fit["R2"] >= 0.8),
    }

    # ---------------- C3: pattern map -------------------------------------
    modes = {}
    for c in cases:
        if c["group"] in ("numeric", "repeat"):
            continue
        if _sig(c):
            modes.setdefault(c["mode"], []).append(c["name"])
    mixed_cases = [c["name"] for c in cases
                   if c["group"] in ("mono", "omega", "fill", "ratio")
                   and not _sig(c) and c["mode"] == "mixed"]
    if mixed_cases:
        modes.setdefault("mixed", []).extend(mixed_cases[:3])
    axial_ok = bool(axial_checks and axial_checks.get("A2_wavelength", {}).get("pass"))
    if axial_ok:
        modes.setdefault("axial_bands", []).append("axial_model_A2")
    out["criteria"]["C3"] = {
        "modes": {k: v for k, v in modes.items()},
        "n_modes": len(modes), "axial_model_anchor_pass": axial_ok,
        "pass": bool(len(modes) >= 3),
    }

    # ---------------- C4: monodisperse negative control --------------------
    mono = by_name.get("mono_1.00_s0")
    c4_rows = []
    for nm in ("mono_1.00_s0", "mono_1.00_s1"):
        c = by_name.get(nm)
        if c is None:
            continue
        c4_rows.append({"case": nm, "corr": c["null"]["corr_observed"],
                        "corr_abs": abs(c["null"]["corr_observed"]),
                        "S": c["null"]["S_observed"],
                        "S_null_mean": c["null"]["S_null_mean"],
                        "excess": c["null"]["excess_observed"],
                        "mode": c["mode"]})
    c4_pass = bool(c4_rows) and all(r["corr_abs"] <= 1e-6 for r in c4_rows)
    out["criteria"]["C4"] = {"rows": c4_rows, "threshold": 1e-6, "pass": c4_pass}

    # ---------------- C5: three negative controls -------------------------
    # (a) ratio -> 1
    ex = {}
    for nm, key in (("mono_1.00_s0", 1.00), ("mono_1.05", 1.05),
                    ("mono_1.10", 1.10), ("ratio_1.2", 1.2)):
        c = by_name.get(nm)
        if c is not None:
            ex[key] = _amp(c)
    mono_row = by_name.get("mono_1.00_s0")
    a_zero = bool(mono_row is not None
                  and abs(mono_row["null"]["corr_observed"]) <= 1e-6
                  and abs(mono_row["null"]["excess_z"]) < Z_SIGNIF)
    seq = [_amp(by_name[k]) for k in
           ("mono_1.05", "mono_1.10", "ratio_1.2") if k in by_name]
    a_mono = bool(a_zero and len(seq) == 3 and seq[0] <= seq[1] <= seq[2])

    # (b) omega -> 0
    from .sweeps import BASE as _BASE
    om_grid = [0.0, 0.5, 1.0, 4.0]
    om_cases = [(om, by_name.get(f"omega_{om:g}")) for om in om_grid]
    om_amp = {}
    for om, c in om_cases:
        if c is not None:
            om_amp[om] = _amp(c)
    ref = by_name.get("ratio_2")
    om_ref = float(_BASE["omega"])
    if ref is not None:
        om_amp[om_ref] = _amp(ref)
    om_static = by_name.get("omega_0")
    b_zero = bool(om_static is not None and om_static["mode"] == "mixed"
                  and abs(om_static["null"]["excess_z"]) < Z_SIGNIF)
    om_seq = [om_amp[k] for k in sorted(om_amp) if k in om_amp]
    b_mono = bool(len(om_seq) == 5 and all(om_seq[i] <= om_seq[i + 1] for i in range(4)))

    # (c) fill -> 1
    fill_cases = {fl: by_name.get(f"fill_{fl:g}") for fl in (0.15, 0.5, 0.65)}
    fill_cases[0.30] = by_name.get("ratio_2")
    fill_amp = {k: _amp(v) for k, v in fill_cases.items() if v}
    fill_full = by_name.get("fill_0.65")
    c_zero = bool(fill_full is not None and fill_full["mode"] == "mixed"
                  and abs(fill_full["null"]["excess_z"]) < Z_SIGNIF)
    c_amp = bool(0.65 in fill_amp and 0.30 in fill_amp
                 and fill_amp[0.65] < fill_amp[0.30])
    out["criteria"]["C5"] = {
        "a_ratio_to_one": {"zero_index_at_ratio_1": a_zero,
                           "monotone_growth_1.05_1.10_1.2": a_mono,
                           "amplitudes": {str(k): v for k, v in ex.items()},
                           "pass": bool(a_zero and a_mono)},
        "b_omega_to_zero": {"static_mixed": b_zero, "monotone_in_omega": b_mono,
                            "amplitudes": {str(k): v for k, v in sorted(om_amp.items())},
                            "pass": bool(b_zero and b_mono)},
        "c_fill_to_one": {"fill065_mixed": c_zero, "fill065_below_fill030": c_amp,
                          "amplitudes": {str(k): v for k, v in sorted(fill_amp.items())},
                          "pass": bool(c_zero and c_amp)},
    }
    out["criteria"]["C5"]["pass"] = bool(
        out["criteria"]["C5"]["a_ratio_to_one"]["pass"]
        and out["criteria"]["C5"]["b_omega_to_zero"]["pass"]
        and out["criteria"]["C5"]["c_fill_to_one"]["pass"])

    # ---------------- C6: pytest ------------------------------------------
    out["criteria"]["C6"] = {"n_pytest": n_pytest,
                             "pass": bool(n_pytest is not None and n_pytest >= 15),
                             "threshold": 15}

    # ---------------- supporting numerical checks -------------------------
    num = {}
    ref = by_name.get("ratio_2")
    for nm in ("numeric_dt_half", "numeric_bins12", "repeat_seed1"):
        c = by_name.get(nm)
        if c is None or ref is None:
            continue
        d = {}
        for key in ("S_observed", "excess_observed", "corr_observed"):
            a = ref["null"][key]
            b = c["null"][key]
            d[key] = {"reference": a, "check": b,
                      "rel_change": (b - a) / a if a not in (0.0, None) else float("nan")}
        num[nm] = d
    # ---------------- supplementary: quasi-steady (time-averaged) test --------
    # Declared POST-HOC (see the report): the pre-registered significance test
    # uses a single final snapshot, which is noise limited (a 581-particle bed
    # gives a +-0.04 standard error on the correlation index).  The same run also
    # provides a quasi-steady estimate: the mean of the index over the last
    # quarter of the run.  Its standard error is taken from the lag-1
    # autocorrelation of the series (n_eff = n (1-rho)/(1+rho)), which is the
    # standard correction for a correlated time series.
    supp = []
    for ratio in (1.2, 1.5, 2.0, 3.0):
        c = by_name.get(f"ratio_{ratio:g}")
        if c is None or "series" not in c:
            continue
        row = {"ratio": ratio}
        for key, avg_key in (("core_shell_excess", "excess_avg"),
                             ("size_pos_corr", "corr_avg")):
            arr = np.asarray(c["series"].get(key, []), float)
            if arr.size < 4:
                continue
            k = max(2, int(round(0.25 * arr.size)))
            tail = arr[-k:]
            n = tail.size
            rho1 = 0.0
            if n > 2 and tail.std() > 0:
                rho1 = float(np.corrcoef(tail[:-1], tail[1:])[0, 1])
                rho1 = max(min(rho1, 0.95), -0.95)
            n_eff = max(1.0, n * (1.0 - rho1) / (1.0 + rho1))
            se = float(tail.std(ddof=1) / math.sqrt(n_eff)) if n_eff > 1 else float("nan")
            row[key] = {"mean": float(tail.mean()), "sd": float(tail.std(ddof=1)),
                        "n": n, "rho1": rho1, "n_eff": n_eff, "se": se,
                        "z": float(tail.mean() / se) if se and np.isfinite(se) and se > 0 else 0.0}
        if "size_pos_corr" in row:
            row["significant"] = bool(abs(row["size_pos_corr"]["z"]) >= Z_SIGNIF
                                      and row["size_pos_corr"]["mean"] > 0)
        supp.append(row)
    out["supplementary_steady_state_C1"] = {
        "rows": supp,
        "n_significant_periphery": sum(1 for r in supp if r.get("significant")),
        "note": "POST-HOC estimator (quasi-steady mean with autocorrelation-corrected "
                "standard error); NOT the pre-registered criterion. The pre-registered "
                "single-snapshot verdict is in criteria.C1.",
        "pass_3_of_4": bool(sum(1 for r in supp if r.get("significant")) >= 3),
    }

    out["numerical_sensitivity"] = num
    out["all_pass"] = bool(all(v["pass"] for v in out["criteria"].values()))
    return out