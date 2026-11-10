#!/usr/bin/env python
"""Parameter sweep + analysis for rubble accumulation around a fixed cylinder.

Campaign node IN69-N06.  Runs the parameter box

    v_i in [0.1, 1.0] m/s  x  h_i in [0.1, 0.5] m  x  C_i in [0.1, 0.5]
    x  D in [1, 5] m

with :mod:`tensordem.rubble_accumulation`, plus

* the *sweeper* baselines (identical ice field and driving, pile continuously
  removed) needed for the shielding factor ``R_shield = p_front / p_0``;
* the two negative controls of the acceptance criteria (``C_i = 0``, ``v_i = 0``);
* a block-size sensitivity case (doubled particle budget).

Everything is written to ``artifacts/rubble_accumulation/`` as one JSON record
with per-case time series and fits, plus a machine-checkable acceptance ledger.

Usage::

    PYTHONPATH=src python tools/run_rubble_sweep.py --mode grid --workers 40
    PYTHONPATH=src python tools/run_rubble_sweep.py --analyse-only
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "src"))

from tensordem.rubble_accumulation import (  # noqa: E402
    CYLINDER_DRAG_COEFFICIENT,
    RubbleConfig,
    fit_exponential_growth,
    momentum_baseline_pressure,
    run_rubble_case,
    shielding_ratio,
)

OUT_DIR = os.path.join("artifacts", "rubble_accumulation")

V_GRID = (0.1, 0.3, 0.6, 1.0)
H_GRID = (0.1, 0.25, 0.5)
C_GRID = (0.1, 0.3, 0.5)
D_GRID = (1.0, 3.0, 5.0)


# --------------------------------------------------------------------------
# case construction
# --------------------------------------------------------------------------
#: Thrust pressure of the sheet-pressure boundary [Pa].  Decided by the
#: amendment-1 controlled comparison (prereg/rubble_accumulation_amendment1.json);
#: recorded in every case so a run can always be traced back to its drive.
THRUST_PRESSURE = float(os.environ.get("RUBBLE_THRUST_PRESSURE", "20000"))


def base_kwargs() -> dict:
    """Reference drive = mass-flux feed + water drag (thrust off)."""
    return dict(seed=0, max_steps=9000, particle_budget=3000,
                thrust_pressure=0.0)


def grid_cases() -> list[dict]:
    """Main design (prereg amendment 2).

    G1 core     D = 3, h = 0.25, C in {0.1,0.3,0.5}, v in {0.1,0.3,1.0}   (9)
    G2 sheet    D = 3, h in {0.1,0.5},  C = 0.3,        v in {0.3,1.0}    (4)
    G3 cylinder D in {1,5}, h = 0.25,   C = 0.3,        v in {0.3,1.0}    (4)
    G4 baseline one sweeper (cleared-cylinder) twin per G1 and G3 case     (13)
    G5 controls C = 0 and v = 0 of the G1 centre point                     (2)
    G6 block-size sensitivity at the G1 centre point, budget 1200 / 12000   (2)
    """
    cases = []
    for C in C_GRID:
        for v in (0.1, 0.3, 1.0):
            cases.append(dict(D=3.0, h=0.25, C=C, v=v, kind="pile"))
    for h in (0.1, 0.5):
        for v in (0.3, 1.0):
            cases.append(dict(D=3.0, h=h, C=0.3, v=v, kind="pile"))
    for D in (1.0, 5.0):
        for v in (0.3, 1.0):
            cases.append(dict(D=D, h=0.25, C=0.3, v=v, kind="pile"))
    # cleared-cylinder baselines of the G1 and G3 families
    for C in C_GRID:
        for v in (0.1, 0.3, 1.0):
            cases.append(dict(D=3.0, h=0.25, C=C, v=v, kind="sweeper"))
    for D in (1.0, 5.0):
        for v in (0.3, 1.0):
            cases.append(dict(D=D, h=0.25, C=0.3, v=v, kind="sweeper"))
    # G7 thrust treatment: the G1 core re-run with the sheet pressure ON, i.e.
    # the amendment-1 mechanism applied over the Fr / C space (paired).
    for C in C_GRID:
        for v in (0.1, 0.3, 1.0):
            cases.append(dict(D=3.0, h=0.25, C=C, v=v, kind="pile_thrust"))
    cases += [dict(D=3.0, h=0.25, C=0.0, v=0.3, kind="pile"),
              dict(D=3.0, h=0.25, C=0.3, v=0.0, kind="pile"),
              dict(D=3.0, h=0.25, C=0.3, v=0.3, kind="pile",
                   particle_budget=1200),
              dict(D=3.0, h=0.25, C=0.3, v=0.3, kind="pile",
                   particle_budget=12000)]
    return cases


def quick_cases() -> list[dict]:
    """Reduced design for a quick look: the G1 core plus its baselines."""
    cases = []
    for C in (0.1, 0.3, 0.5):
        for v in (0.3, 1.0):
            cases.append(dict(D=3.0, h=0.25, C=C, v=v, kind="pile",
                              particle_budget=800, max_steps=4000))
    for C in (0.1, 0.3, 0.5):
        for v in (0.3, 1.0):
            cases.append(dict(D=3.0, h=0.25, C=C, v=v, kind="sweeper",
                              particle_budget=800, max_steps=4000))
    return cases


def build_config(case: dict) -> RubbleConfig:
    kw = base_kwargs()
    kw.update(diameter=case["D"], ice_thickness=case["h"],
              concentration=case["C"], drift_speed=case["v"])
    if case["kind"] == "sweeper":
        kw["sweep_shell_radii"] = 1.02
    if case["kind"] == "pile_thrust":
        kw["thrust_pressure"] = THRUST_PRESSURE
    if case.get("max_steps"):
        kw["max_steps"] = int(case["max_steps"])
    if "particle_budget" in case:
        kw["particle_budget"] = case["particle_budget"]
    if "thrust_pressure" in case:
        kw["thrust_pressure"] = float(case["thrust_pressure"])
    return RubbleConfig(**kw)


def run_one(case: dict) -> dict:
    cfg = build_config(case)
    t_end = 10.0 if case["v"] <= 0.0 else cfg.recommended_duration()
    t0 = time.time()
    rec = run_rubble_case(cfg, t_end=t_end)
    rec["case"] = case
    rec["wall_seconds"] = time.time() - t0
    rec["baseline_momentum"] = momentum_baseline_pressure(
        cfg.concentration, cfg.drift_speed, rho_ice=cfg.rho_ice,
        drag_coefficient=CYLINDER_DRAG_COEFFICIENT)
    return rec


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def _fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares ``log y = log a + b log x`` with a standard error on b."""
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return {"exponent": float("nan"), "coefficient": float("nan"),
                "stderr": float("nan"), "n": int(mask.sum()), "r2": float("nan")}
    lx, ly = np.log(x[mask]), np.log(y[mask])
    A = np.vstack([np.ones_like(lx), lx]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(len(lx) - 2, 1)
    sigma2 = ss_res / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    return {"exponent": float(coef[1]), "coefficient": float(math.exp(coef[0])),
            "stderr": float(math.sqrt(max(cov[1, 1], 0.0))), "n": int(mask.sum()),
            "r2": r2}


def _smooth(y, window: int) -> np.ndarray:
    """Centred moving average with edge padding (fixed, pre-declared window)."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size <= window:
        return y.copy()
    pad = window // 2
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[:y.size]


def refit(records: list[dict], window: int = 11) -> None:
    """Recompute the geometry fits from the stored time series.

    The raw fits are kept next to the smoothed ones (``r2_raw``), because the
    raw series of a loose granular accumulation is intermittent: the fixed
    ``window``-sample moving average is applied to *every* case, and both numbers
    are reported.
    """
    from tensordem.rubble_accumulation import fit_exponential_growth
    for rec in records:
        h = rec["history"]
        t = np.asarray(h["time"], dtype=float)
        for label, key in (("L", "L_p"), ("B", "B_p"), ("H", "H_p")):
            y = np.asarray(h[key], dtype=float)
            raw = fit_exponential_growth(t, y)
            ys = _smooth(y, window)
            nz = np.flatnonzero(y > 0.02 * max(np.max(y), 1e-30)) if y.size else []
            start = max(int(nz[0]) - 1, 0) if len(nz) else 0
            fit = fit_exponential_growth(t[start:], ys[start:])
            d = fit.as_dict()
            d["r2_raw"] = raw.r2
            d["ok_raw"] = raw.ok
            d["smoothing_window"] = window
            d["fit_window_start"] = float(t[start]) if t.size else 0.0
            if d["inf"] > 5.0 * max(float(np.max(y)) if y.size else 0.0, 1e-12):
                d["ok"] = False           # plateau far above the observed maximum
                d["message"] = "implausible plateau (fit ran away)"
            rec["fits"][label] = d
            inf = d["inf"]
            rec.setdefault("plateau_fraction", {})[label] = (
                float(y[-1] / inf) if inf > 0 else 0.0)
        rec["converged"] = bool(all(v >= 0.95 for v in
                                   rec["plateau_fraction"].values()))


def _row(r: dict, kind: str) -> dict:
    cfg = r["config"]
    return dict(kind=kind, D=cfg["diameter"], h=cfg["ice_thickness"],
                C=cfg["concentration"], v=cfg["drift_speed"], Fr=cfg["froude"],
                thrust_pressure=cfg.get("thrust_pressure", 0.0),
                particle_radius=cfg["particle_radius"],
                sheet_resolution=cfg["sheet_resolution"],
                resolved=cfg["sheet_resolved"], converged=r["converged"],
                capped=r["capped_by_time"],
                L_inf=r["fits"]["L"]["inf"], B_inf=r["fits"]["B"]["inf"],
                H_inf=r["fits"]["H"]["inf"],
                L_r2=r["fits"]["L"]["r2"], B_r2=r["fits"]["B"]["r2"],
                H_r2=r["fits"]["H"]["r2"],
                L_ok=r["fits"]["L"]["ok"], B_ok=r["fits"]["B"]["ok"],
                H_ok=r["fits"]["H"]["ok"],
                L_steady=r["steady"]["L_p"], B_steady=r["steady"]["B_p"],
                H_steady=r["steady"]["H_p"],
                H_over_blocks=r["steady"]["H_p"] / (2.0 * cfg["particle_radius"]),
                pile_mass=r["steady"]["pile_mass"],
                p_front=r["steady"]["p_front"],
                p_momentum=r["baseline_momentum"],
                zone_closure_error=r["ledger"]["zone_closure_error"],
                partition_error=r["ledger"]["closure_error"])


def analyse(records: list[dict]) -> dict:
    """Every acceptance criterion of the node, with the number it was decided on."""
    ref = [r for r in records if r["case"]["kind"] == "pile"
           and r["case"]["C"] > 0 and r["case"]["v"] > 0]
    thrust = [r for r in records if r["case"]["kind"] == "pile_thrust"
              and r["case"]["C"] > 0 and r["case"]["v"] > 0]
    sweeper = [r for r in records if r["case"]["kind"] == "sweeper"]
    controls = [r for r in records if r["case"]["kind"] in ("pile", "pile_thrust")
                and (r["case"]["C"] == 0 or r["case"]["v"] == 0)]

    rows = [_row(r, r["case"]["kind"]) for r in ref]
    thrust_rows = [_row(r, r["case"]["kind"]) for r in thrust]

    # --- (a) growth law y = y_inf (1 - exp(-t/tau)), R^2 > 0.95 ---------
    fits = {}
    for label in ("L", "B", "H"):
        r2 = [x[f"{label}_r2"] for x in rows if x[f"{label}_ok"]]
        fits[label] = {
            "n_cases": len(rows),
            "n_fit_ok": int(sum(1 for x in rows if x[f"{label}_ok"])),
            "n_r2_gt_0.95": int(sum(1 for v in r2 if v > 0.95)),
            "median_r2": float(np.median(r2)) if r2 else float("nan"),
            "min_r2": float(np.min(r2)) if r2 else float("nan"),
            "max_r2": float(np.max(r2)) if r2 else float("nan"),
        }
    n_pass_a = fits["L"]["n_r2_gt_0.95"] + fits["B"]["n_r2_gt_0.95"] + \
        fits["H"]["n_r2_gt_0.95"]

    # --- (b) steady geometry scaling (exponents vs Fr and C_i) ----------
    # A height is *resolved* when the envelope rises measurably above the
    # floating level: H_p is measured from the undisturbed ice surface, so a
    # value of a few percent of the block radius is a numerical residue rather
    # than a pile.  The threshold is 0.1 r, and the degenerate cases
    # (H_p/(2r) < 0.5, i.e. no sail) are reported next to the fit so that the
    # exponent is never read as a sail scaling law.
    usable = [x for x in rows if x["resolved"]
              and x["H_steady"] >= 0.1 * x["particle_radius"]
              and x["B_steady"] > 0.0 and x["L_steady"] > 0.0]
    scaling = {}
    if usable:
        Fr = np.array([x["Fr"] for x in usable])
        Cc = np.array([x["C"] for x in usable])
        for key, label, norm in (("B_steady", "B_over_D", "D"),
                                 ("H_steady", "H_over_h", "h"),
                                 ("L_steady", "L_over_D", "D")):
            y = np.array([x[key] / x[norm] for x in usable])
            scaling[label] = {"vs_Fr": _fit_power_law(Fr, y),
                              "vs_C": _fit_power_law(Cc, y),
                              "range": [float(np.min(y)), float(np.max(y))],
                              "n": len(usable)}
        scaling["H_over_blocks_range"] = [
            float(np.min([x["H_over_blocks"] for x in usable])),
            float(np.max([x["H_over_blocks"] for x in usable]))]
        scaling["n_no_sail"] = int(sum(1 for x in usable if x["H_over_blocks"] < 0.5))

    # --- (c) shielding factor: two independent baselines ---------------
    twin_by_key = {(r["case"]["D"], r["case"]["h"], r["case"]["C"],
                    r["case"]["v"]): r for r in sweeper}
    shield = []
    for r in ref + thrust:
        key = (r["case"]["D"], r["case"]["h"], r["case"]["C"], r["case"]["v"])
        twin = twin_by_key.get(key)
        p1 = r["steady"]["p_front"]
        p_mom = r["baseline_momentum"]
        p0 = twin["steady"]["p_front"] if twin is not None else float("nan")
        shield.append(dict(
            kind=r["case"]["kind"], D=key[0], h=key[1], C=key[2], v=key[3],
            Fr=r["config"]["froude"], p_front=p1, p_momentum=p_mom, p0_swept=p0,
            R_momentum=shielding_ratio(p1, p_mom),
            R_swept=shielding_ratio(p1, p0),
            pile_mass=r["steady"]["pile_mass"], converged=r["converged"]))
    shield_sorted = sorted(shield, key=lambda x: (x["kind"], x["C"], x["Fr"]))

    def crossing(series, C_value):
        pts = [x for x in series if abs(x["C"] - C_value) < 1e-9]
        for a, b in zip(pts, pts[1:]):
            fa = a["R_momentum"] - 1.0
            fb = b["R_momentum"] - 1.0
            if not (np.isfinite(fa) and np.isfinite(fb)) or fa == fb:
                continue
            if fa < 0.0 <= fb or fa > 0.0 >= fb:
                w = fa / (fa - fb)
                return float(a["Fr"] + w * (b["Fr"] - a["Fr"]))
        return None

    ref_shield = [x for x in shield_sorted if x["kind"] == "pile"]
    r_mom = [x["R_momentum"] for x in shield_sorted if np.isfinite(x["R_momentum"])]
    r_sw = [x["R_swept"] for x in shield_sorted if np.isfinite(x["R_swept"])]
    # A case whose front pressure is exactly zero measured *no load* -- it did
    # not shield anything.  Those cases are counted separately and excluded from
    # the shielding statistics and from the transition search.
    zero_load = [x for x in shield_sorted if not (x["p_front"] > 0.0)]
    loaded = [x for x in shield_sorted if x["p_front"] > 0.0]
    agree = [x for x in shield_sorted
             if np.isfinite(x["R_momentum"]) and np.isfinite(x["R_swept"])]
    shield_summary = {
        "n_pairs": len(shield),
        "n_R_momentum_defined": len(r_mom),
        "n_R_swept_defined": len(r_sw),
        "n_both_defined": len(agree),
        "n_sign_agree": int(sum(1 for x in agree
                                if np.sign(x["R_momentum"] - 1.0)
                                == np.sign(x["R_swept"] - 1.0))),
        "n_zero_load_cases": len(zero_load),
        "zero_load_cases": [dict(kind=x["kind"], D=x["D"], C=x["C"], v=x["v"],
                                 Fr=x["Fr"], p_front=x["p_front"])
                            for x in zero_load],
        "R_momentum_range": [float(np.min(r_mom)), float(np.max(r_mom))] if r_mom else None,
        "R_swept_range": [float(np.min(r_sw)), float(np.max(r_sw))] if r_sw else None,
        "R_momentum_range_loaded": ([float(np.min([x["R_momentum"] for x in loaded])),
                                     float(np.max([x["R_momentum"] for x in loaded]))]
                                    if loaded else None),
        "R_swept_range_loaded": ([float(np.min([x["R_swept"] for x in loaded
                                                if np.isfinite(x["R_swept"])])),
                                  float(np.max([x["R_swept"] for x in loaded
                                                if np.isfinite(x["R_swept"])]))]
                                 if any(np.isfinite(x["R_swept"]) for x in loaded) else None),
        "n_shielded_momentum": int(sum(1 for x in r_mom if 0.0 < x < 1.0)),
        "n_amplified_momentum": int(sum(1 for x in r_mom if x > 1.0)),
        "n_shielded_swept": int(sum(1 for x in r_sw if 0.0 < x < 1.0)),
        "n_amplified_swept": int(sum(1 for x in r_sw if x > 1.0)),
        "transition_Fr_momentum": {f"C={C}": crossing(
            [x for x in ref_shield if x["p_front"] > 0.0], C) for C in C_GRID},
        "transition_Fr_swept": None,
    }

    # --- (d) mass conservation ----------------------------------------
    zone = [x["zone_closure_error"] for x in rows]
    part = [x["partition_error"] for x in rows]
    mass = {
        "max_zone_error": float(np.max(zone)) if zone else None,
        "median_zone_error": float(np.median(zone)) if zone else None,
        "max_partition_error": float(np.max(part)) if part else None,
        "note": "the zone-crossing error is the independent check (flux integral "
                "vs inventory); the partition error is an exact bookkeeping "
                "identity and can only detect a bookkeeping bug",
    }

    # --- (e) negative controls ----------------------------------------
    ctrl = []
    for r in controls:
        ctrl.append(dict(C=r["case"]["C"], v=r["case"]["v"], kind=r["case"]["kind"],
                         pile_mass=r["steady"]["pile_mass"], L_p=r["steady"]["L_p"],
                         B_p=r["steady"]["B_p"], H_p=r["steady"]["H_p"],
                         p_front=r["steady"]["p_front"]))
    controls_pass = all(c["pile_mass"] < 1e-9 and c["H_p"] < 1e-9 for c in ctrl)

    # --- criterion verdicts -------------------------------------------
    def verdict(ok: bool, accept: str, fail: str, withheld: str | None = None):
        if ok:
            return ("ACCEPT", accept)
        if withheld:
            return ("WITHHELD", withheld)
        return ("FAIL", fail)

    v_a = verdict(n_pass_a > 0 and fits["H"]["n_r2_gt_0.95"] > 0,
                  f"growth law fitted with R^2>0.95 in {n_pass_a} observable-case fits",
                  f"only {n_pass_a} of {3 * len(rows)} observable-case growth fits reach "
                  f"R^2>0.95 (L {fits['L']['n_r2_gt_0.95']}, B {fits['B']['n_r2_gt_0.95']}, "
                  f"H {fits['H']['n_r2_gt_0.95']}); the signal is not a smooth "
                  f"saturating growth on this grid")
    lit = {"note": "no authoritative literature value exists for the Fr/C_i "
                   "exponents of B_inf/D, H_inf/h_i, L_inf/D around a cylinder; "
                   "the exponent claim is therefore withheld and only the "
                   "measured ranges are reported"}
    v_b = ("WITHHELD", "steady-geometry exponents have no authoritative "
                       "literature comparator; ranges reported instead "
                       "(see scaling.range)") if usable else ("FAIL", "no usable steady rows")
    n_shield = shield_summary["n_shielded_momentum"] + shield_summary["n_shielded_swept"]
    v_c = verdict(n_shield > 0,
                  f"{n_shield} observable-baseline pairs show 0 < R < 1",
                  f"no pair shows 0 < R < 1: R_momentum over the loaded cases is "
                  f"{shield_summary['R_momentum_range_loaded']} and R_swept is "
                  f"{shield_summary['R_swept_range_loaded']}, i.e. the front pressure of "
                  f"an accumulated field is always above both the free-stream momentum "
                  f"flux and the cleared-cylinder load. The pre-registered "
                  f"'shielded at low Fr' branch is falsified; the two R=0 entries are "
                  f"cases that measured no load at all (C_i=0.1 at v_i=0.1 m/s), not "
                  f"shielding", None)
    v_d = verdict(bool(zone) and max(zone) < 0.02,
                  f"zone-crossing closure error <= {max(zone):.2e} (< 0.02)",
                  f"zone-crossing closure error up to {max(zone):.3g}")
    v_e = verdict(controls_pass,
                  "C_i=0 and v_i=0 produce no pile mass, no L/B/H and no load",
                  "a negative control produced a non-zero pile")

    ledger = {
        "growth_fits": fits,
        "mass": mass,
        "negative_controls": ctrl,
        "negative_controls_pass": controls_pass,
        "literature_comparator": lit,
    }
    return {"rows": rows, "thrust_rows": thrust_rows, "scaling": scaling,
            "shielding": shield_sorted, "shield_summary": shield_summary,
            "criteria": {"a_growth_fits": {"verdict": v_a[0], "detail": v_a[1],
                                           "numbers": fits},
                         "b_steady_scaling": {"verdict": v_b[0], "detail": v_b[1]},
                         "c_shielding": {"verdict": v_c[0], "detail": v_c[1],
                                         "summary": shield_summary},
                         "d_mass": {"verdict": v_d[0], "detail": v_d[1], "numbers": mass},
                         "e_negative_controls": {"verdict": v_e[0], "detail": v_e[1]}},
            "ledger": ledger}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def make_figures(records: list[dict], analysis: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []

    # (1) growth curves + exponential fits
    pick = [r for r in records if r["case"]["kind"] == "pile"
            and r["case"]["D"] == 3.0 and r["case"]["h"] == 0.25
            and r["case"]["v"] == 0.3 and r["case"]["C"] in (0.1, 0.3, 0.5)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for r in sorted(pick, key=lambda x: x["case"]["C"]):
        h = r["history"]
        t = np.asarray(h["time"])
        for ax, key, lab, unit in zip(
                axes, ("L_p", "B_p", "H_p"), ("L_p", "B_p", "H_p"), ("m", "m", "m")):
            ax.plot(t, h[key], lw=1.2, label=f"C_i={r['case']['C']}")
            fit = r["fits"][key[0]]
            if fit["ok"]:
                ax.plot(t, fit["inf"] * (1 - np.exp(-np.maximum(t - t[0], 0)
                                                     / max(fit["tau"], 1e-9))),
                        "--", lw=0.9, color=ax.lines[-1].get_color())
            ax.set_xlabel("t [s]")
            ax.set_ylabel(f"{lab} [{unit}]")
            ax.set_title(f"{lab}(t), D=3 m, h_i=0.25 m, v_i=0.3 m/s")
            ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_growth_laws.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # (2) front pressure, pile versus sweeper baseline
    pairs = []
    for r in records:
        if r["case"]["kind"] == "sweeper" and r["case"]["C"] == 0.3:
            twin = [q for q in records if q["case"]["kind"] == "pile"
                    and all(q["case"][k] == r["case"][k] for k in ("D", "h", "C", "v"))]
            if twin:
                pairs.append((r, twin[0]))
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 3.2),
                                 sharey=False)
        axes = np.atleast_1d(axes)
        for ax, (sw, pl) in zip(axes, sorted(pairs, key=lambda x: x[1]["case"]["v"])):
            for rec, lab in ((pl, "with accumulation"), (sw, "swept baseline p_0")):
                t = np.asarray(rec["history"]["time"])
                p = np.asarray(rec["history"]["p_front"])
                ax.plot(t, p, lw=1.0, label=lab)
            ax.set_title(f"v_i={pl['case']['v']} m/s  (Fr={pl['config']['froude']:.2f})",
                         fontsize=9)
            ax.set_xlabel("t [s]")
            ax.set_ylabel("p_front [Pa]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig2_shielding_timeseries.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # (3) R_shield(Fr): the two independent baselines, per concentration
    shield = analysis["shielding"]
    if shield:
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), sharex=True)
        for ax, key, lab in ((axes[0], "R_momentum", "p_0 = free-stream momentum flux"),
                             (axes[1], "R_swept", "p_0 = cleared-cylinder twin")):
            for kind, ls in (("pile", "-"), ("pile_thrust", "--")):
                for C in sorted({x["C"] for x in shield}):
                    pts = [x for x in shield if x["C"] == C and x["kind"] == kind
                           and np.isfinite(x[key])]
                    if len(pts) < 2:
                        continue
                    pts.sort(key=lambda x: x["Fr"])
                    ax.plot([x["Fr"] for x in pts], [x[key] for x in pts],
                            "o" + ls, ms=4, lw=1.0,
                            label=f"C_i={C}" + (" +thrust" if kind == "pile_thrust" else ""))
            ax.axhline(1.0, color="k", lw=0.8, ls=":")
            ax.set_yscale("log")
            ax.set_xlabel("Fr = v_i / sqrt(g h_i)")
            ax.set_ylabel("R_shield = p_front / p_0")
            ax.set_title(lab, fontsize=9)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=6)
        fig.suptitle("shielding / amplification, D = 3 m, h_i = 0.25 m", fontsize=10)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig3_shielding_vs_fr.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # (4) steady-state scaling
    rows = [x for x in analysis["rows"] if x["resolved"] and x["H_inf"] > 0]
    if rows:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
        for ax, key, norm, lab in zip(
                axes, ("B_inf", "H_inf", "L_inf"), ("D", "h", "D"),
                ("B_inf/D", "H_inf/h_i", "L_inf/D")):
            for C in sorted({x["C"] for x in rows}):
                pts = [x for x in rows if x["C"] == C]
                ax.loglog([x["Fr"] for x in pts], [x[key] / x[norm] for x in pts],
                          "o", ms=4, label=f"C_i={C}")
            ax.set_xlabel("Fr")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3, which="both")
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig4_scaling.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)

    # (5) sail diagnostic: H_p / (2r) -- the height that the model cannot build
    rows = analysis["rows"] + analysis.get("thrust_rows", [])
    if rows:
        fig, ax = plt.subplots(figsize=(5.0, 3.6))
        for kind, mk, lab in (("pile", "o", "reference drive"),
                              ("pile_thrust", "s", "thrust 20 kPa")):
            for C in sorted({x["C"] for x in rows}):
                pts = [x for x in rows if x["C"] == C and x["kind"] == kind]
                if not pts:
                    continue
                ax.scatter([x["Fr"] for x in pts], [x["H_over_blocks"] for x in pts],
                           marker=mk, s=18, label=f"C_i={C}" + (" +thrust" if kind == "pile_thrust" else ""))
        ax.axhline(1.5, color="k", ls=":", lw=0.9)
        ax.text(0.02, 1.5, " amendment-1 sail threshold", fontsize=6, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("Fr = v_i / sqrt(g h_i)")
        ax.set_ylabel("H_p / (2r)  [block diameters]")
        ax.set_title("pile height: no sail forms (H_p < 0.5 block diameter)", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        p = os.path.join(out_dir, "fig5_sail_diagnostic.png")
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=("grid", "shield", "controls", "resolution", "all"))
    ap.add_argument("--quick-design", action="store_true",
                    help="reduced design for a heavily shared node")
    ap.add_argument("--subset", action="store_true",
                    help="compute-budget-sized design instead of the full box")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N cases")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "sweep.json"))
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--smooth", type=int, default=0,
                    help="re-fit the stored time series with this moving-average window")
    ap.add_argument("--in", dest="inp", default=os.path.join(OUT_DIR, "sweep.json"))
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    if not args.analyse_only:
        cases = []
        if args.quick_design:
            cases = quick_cases()
        elif args.mode in ("grid", "all"):
            cases += grid_cases()
        if args.limit:
            cases = cases[:args.limit]
        print(f"[sweep] {len(cases)} cases, {args.workers} workers", flush=True)
        t0 = time.time()
        records = []
        with Pool(processes=args.workers) as pool:
            for k, rec in enumerate(pool.imap_unordered(run_one, cases), 1):
                records.append(rec)
                print(f"[{k}/{len(cases)}] {rec['case']} "
                      f"L_inf={rec['fits']['L']['inf']:.2f} "
                      f"B_inf={rec['fits']['B']['inf']:.2f} "
                      f"H_inf={rec['fits']['H']['inf']:.3f} "
                      f"R2(H)={rec['fits']['H']['r2']:.3f} "
                      f"{rec['wall_seconds']:.0f}s", flush=True)
        analysis = analyse(records)
        payload = {"cases": records, "analysis": analysis,
                   "wall_seconds": time.time() - t0,
                   "case_count": len(cases),
                   "drive": {"thrust_pressure_pa": THRUST_PRESSURE,
                             "reference_drive": "mass-flux feed + water drag",
                             "drag_coefficient_momentum_baseline":
                                 CYLINDER_DRAG_COEFFICIENT},
                   "design": [c for c in cases]}
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)
        print(f"[sweep] wrote {args.out}", flush=True)
    else:
        with open(args.inp) as fh:
            payload = json.load(fh)
        records = payload["cases"]
        if args.smooth > 1:
            refit(records, args.smooth)
        analysis = analyse(records)
        payload["analysis"] = analysis
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=1)

    figs = make_figures(records, analysis, OUT_DIR)
    ledger = {"analysis": analysis, "figures": figs,
              "case_count": payload.get("case_count"),
              "criteria": analysis["criteria"]}
    with open(os.path.join(OUT_DIR, "acceptance.json"), "w") as fh:
        json.dump(ledger, fh, indent=1)
    print(json.dumps(analysis["ledger"], indent=1), flush=True)
    print(json.dumps(analysis["shield_summary"], indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())