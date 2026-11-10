"""Aggregation, acceptance criteria and figures for the segregation campaign.

The analysis reads the per-case JSON records written by
:mod:`tensordem.segregation.campaign`, evaluates the machine-checkable
acceptance criteria of the validation node, and produces both a summary JSON
(the artefact the report cites) and the figures.  No number is typed by hand:
everything here is derived from the records.

Criteria (as stated by the campaign node)
-----------------------------------------
``a``  equal-size negative control ``d_l/d_s = 1``: ``|S_fine| <= 0.02`` and
       ``|v_up|`` indistinguishable from zero;
``b``  ``S`` monotonically increasing over the four diameter ratios and the
       Brazil-nut rise-velocity exponent within 30 % of the literature value;
``c``  mass conservation ``<= 1e-9`` and centre-of-mass consistency;
``d``  vibration threshold ``Gamma_c`` within 30 % of the literature value plus
       the ``Gamma -> inf`` saturation value;
``e``  the closed-form ``dTCG``/``dGM`` mapping agrees with its shard sums to
       ``<= 1e-6``;
``f``  three-seed coefficient of variation of ``S`` ``<= 15 %``.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from .transfer import (
    HoldGeometry,
    distribution_from_record,
    floor_pressure,
    gm_mapping,
    zone_moment_check,
)

# Literature anchors -------------------------------------------------------
#: percolation / kinetic-sieving segregation velocity is linear in the diameter
#: ratio excess (Savage & Lun 1988; Fan & Hill 2011 use the same leading order),
#: i.e. ``v_up ~ (d_l/d_s - 1)^p`` with ``p = 1``.
P_LITERATURE = 1.0
#: onset of Brazil-nut segregation in vertically vibrated beds, ``Gamma_c ~ 1.2``
#: (fluidisation/onset of the heaping-convection regime).  The spread reported in
#: the literature is roughly 1.0-1.5, which the +/-30 % band covers.
GAMMA_C_LITERATURE = 1.2
RATIOS = (1.5, 2.0, 3.0, 4.0)


def load_records(run_dir: str) -> dict[str, dict]:
    """Read every ``*.json`` case record in ``run_dir``."""
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(run_dir, "*.json"))):
        with open(path) as fh:
            rec = json.load(fh)
        if "spec" not in rec or "metrics" not in rec:
            continue
        rec["_path"] = path
        out[rec["spec"]["name"]] = rec
    return out


def _family(records, *, drive="vib", ratio=None, fraction=None, gamma=None,
            frequency=None) -> list[dict]:
    sel = []
    for rec in records.values():
        spec = rec["spec"]
        if drive is not None and spec["drive"] != drive:
            continue
        if ratio is not None and abs(spec["ratio"] - ratio) > 1e-9:
            continue
        if fraction is not None and abs(spec["large_fraction"] - fraction) > 1e-9:
            continue
        if gamma is not None and abs(spec["gamma"] - gamma) > 1e-9:
            continue
        if frequency is not None and abs(spec["frequency"] - frequency) > 1e-9:
            continue
        sel.append(rec)
    return sorted(sel, key=lambda r: (r["spec"]["ratio"], r["spec"]["gamma"],
                                      r["spec"]["seed"]))


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def fit_power_law(x: np.ndarray, y: np.ndarray) -> dict:
    """Fit ``y = c x^p`` in log space and return ``p`` with its standard error."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = (x > 0.0) & (y > 0.0) & np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return {"p": float("nan"), "c": float("nan"), "p_stderr": float("nan"),
                "r2": float("nan"), "n": int(keep.sum())}
    lx = np.log(x[keep])
    ly = np.log(y[keep])
    p, b = np.polyfit(lx, ly, 1)
    pred = p * lx + b
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    n = int(keep.sum())
    stderr = float("nan")
    if n > 2 and ss_tot > 0.0:
        stderr = math.sqrt(ss_res / (n - 2) / float(np.sum((lx - lx.mean()) ** 2)))
    return {
        "p": float(p), "c": float(math.exp(b)), "p_stderr": stderr,
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 1.0, "n": n,
    }


def fit_threshold(gammas: np.ndarray, s_values: np.ndarray) -> dict:
    """Fit ``S(Gamma) = S_sat * (1 - exp(-(Gamma - Gamma_c)/Gamma_0))``.

    ``S`` is set to its clamp at zero below the threshold, which is what the
    closed form does by construction (``max(0, ...)``).  The fit is done with a
    coarse grid search plus a local refinement so that no scipy optimizer
    dependency or initial-guess lottery is introduced.
    """
    g = np.asarray(gammas, dtype=np.float64)
    s = np.asarray(s_values, dtype=np.float64)
    keep = np.isfinite(g) & np.isfinite(s)
    g, s = g[keep], s[keep]
    if g.size < 3:
        return {"gamma_c": float("nan"), "s_sat": float("nan"),
                "gamma_0": float("nan"), "rmse": float("nan"), "n": int(g.size)}

    def model(gg, gc, s_sat, g0):
        return np.where(gg > gc, s_sat * (1.0 - np.exp(-(gg - gc) / max(g0, 1e-6))), 0.0)

    best = None
    for gc in np.linspace(0.2, 2.4, 111):
        for s_sat in np.linspace(max(s.max(), 1e-3) * 0.5, max(s.max(), 1e-3) * 3.0, 26):
            for g0 in np.linspace(0.05, 4.0, 40):
                rmse = float(np.sqrt(np.mean((model(g, gc, s_sat, g0) - s) ** 2)))
                if best is None or rmse < best[0]:
                    best = (rmse, gc, s_sat, g0)
    rmse, gc, s_sat, g0 = best
    # local refinement
    for _ in range(3):
        for dgc in np.linspace(-0.05, 0.05, 11):
            for ds in np.linspace(-0.1 * s_sat, 0.1 * s_sat, 11):
                for dg0 in np.linspace(-0.1 * g0, 0.1 * g0, 11):
                    cand = (gc + dgc, s_sat + ds, max(g0 + dg0, 1e-3))
                    r = float(np.sqrt(np.mean((model(g, *cand) - s) ** 2)))
                    if r < rmse:
                        rmse, (gc, s_sat, g0) = r, cand
    return {"gamma_c": float(gc), "s_sat": float(s_sat), "gamma_0": float(g0),
            "rmse": float(rmse), "n": int(g.size)}


def evaluate_criteria(records: dict[str, dict], *, geometry: HoldGeometry | None = None,
                      displacement_t: float = 20000.0, km_m: float = 9.0) -> dict:
    """Evaluate the acceptance criteria and return a report-ready structure."""
    geo = geometry or HoldGeometry(length_m=20.0, breadth_m=12.0, depth_m=6.0,
                                   n_longitudinal=8)
    out: dict[str, object] = {}

    # ---- (a) equal-size negative control -------------------------------
    control = _family(records, ratio=1.0)
    if control:
        s_fine = [r["metrics"]["segregation"]["S_fine"] for r in control]
        s_bn = [r["metrics"]["segregation"]["S_bn"] for r in control]
        v_up = [r["metrics"]["rise_large"]["v_up_m_s"] for r in control]
        v_ref = [
            r["metrics"]["rise_large"]["v_up_m_s"]
            for r in _family(records, ratio=4.0)
        ] or [0.0]
        thr = max(1.0e-4, 0.05 * abs(float(np.mean(v_ref))))
        out["a_negative_control"] = {
            "n_cases": len(control),
            "seed_S_fine": s_fine,
            "seed_S_bn": s_bn,
            "seed_v_up_m_s": v_up,
            "mean_S_fine": float(np.mean(s_fine)),
            "max_abs_S_fine": float(np.max(np.abs(s_fine))),
            "mean_S_bn": float(np.mean(s_bn)),
            "mean_v_up_m_s": float(np.mean(v_up)),
            "v_up_threshold_m_s": thr,
            "pass": bool(max(abs(v) for v in s_fine) <= 0.02
                         and abs(float(np.mean(v_up))) <= thr),
        }
    else:
        out["a_negative_control"] = {"n_cases": 0, "pass": False,
                                     "note": "no d_l/d_s = 1 case in the run set"}

    # ---- (b) monotonicity in d_l/d_s + rise-velocity exponent ----------
    # The designed campaign sweeps *both* the diameter ratio (at Gamma = 3) and
    # the drive (at d_l/d_s = 2), so a bare ``ratio`` filter silently mixes
    # several Gamma levels in the d_l/d_s = 2 family and dilutes its S.  The
    # scaling law is a fixed-drive statement, so the ratio families are pinned
    # to one common drive level: the largest Gamma present for *every* ratio
    # (i.e. the highest saturated point of the matrix).
    ratio_gammas = []
    for ratio in RATIOS:
        fam_all = _family(records, ratio=ratio, fraction=0.3)
        ratio_gammas.append({round(r["spec"]["gamma"], 9) for r in fam_all})
    common_gammas = set.intersection(*ratio_gammas) if ratio_gammas else set()
    drive_gamma = max(common_gammas) if common_gammas else None

    by_ratio = {}
    for ratio in RATIOS:
        fam = _family(records, ratio=ratio, fraction=0.3, gamma=drive_gamma)
        if not fam:
            continue
        s_vals = [r["metrics"]["segregation"]["S_bn"] for r in fam]
        v_vals = [r["metrics"]["rise_large"]["v_up_m_s"] for r in fam]
        by_ratio[ratio] = {
            "n_cases": len(fam),
            "S_bn_mean": float(np.mean(s_vals)),
            "S_bn_seeds": s_vals,
            "S_fine_mean": float(np.mean([r["metrics"]["segregation"]["S_fine"] for r in fam])),
            "v_up_mean_m_s": float(np.mean(v_vals)),
            "v_up_seeds": v_vals,
            "names": [r["spec"]["name"] for r in fam],
        }
    monotone = None
    if len(by_ratio) >= 2:
        ratios_sorted = sorted(by_ratio)
        s_seq = [by_ratio[r]["S_bn_mean"] for r in ratios_sorted]
        monotone = all(b > a for a, b in zip(s_seq, s_seq[1:]))
    p_fit = {"p": float("nan")}
    if len(by_ratio) >= 3:
        xs = np.array([r - 1.0 for r in sorted(by_ratio)])
        ys = np.array([abs(by_ratio[r]["v_up_mean_m_s"]) for r in sorted(by_ratio)])
        p_fit = fit_power_law(xs, ys)
    p_dev = (
        abs(p_fit["p"] - P_LITERATURE) / P_LITERATURE
        if np.isfinite(p_fit.get("p", float("nan"))) else float("nan")
    )
    out["b_ratio_scaling"] = {
        "drive_gamma": None if drive_gamma is None else float(drive_gamma),
        "by_ratio": by_ratio,
        "strictly_monotone": monotone,
        "power_law": p_fit,
        "p_literature": P_LITERATURE,
        "p_relative_deviation": float(p_dev) if np.isfinite(p_dev) else None,
        "pass": bool(monotone is True and np.isfinite(p_dev) and p_dev <= 0.30),
    }

    # ---- (c) mass conservation and centre of mass ----------------------
    mass_err = [r["metrics"]["mass_conservation_rel"] for r in records.values()]
    zc_err = [r["metrics"]["z_com_consistency_rel"] for r in records.values()]
    if mass_err:
        out["c_mass_conservation"] = {
            "n_cases": len(mass_err),
            "max_mass_conservation_rel": float(np.max(mass_err)),
            "max_z_com_consistency_rel": float(np.max(zc_err)),
            "worst_mass_case": max(records.values(),
                                   key=lambda r: r["metrics"]["mass_conservation_rel"])["spec"]["name"],
            "pass": bool(np.max(mass_err) <= 1e-9 and np.max(zc_err) <= 1e-6),
        }

    # ---- (d) Gamma threshold and saturation ----------------------------
    gscan = _family(records, ratio=2.0, fraction=0.3, frequency=30.0)
    if not gscan:
        gscan = _family(records, ratio=2.0, fraction=0.3)
    if gscan:
        gammas = sorted({r["spec"]["gamma"] for r in gscan})
        s_by_gamma = []
        for gam in gammas:
            vals = [r["metrics"]["segregation"]["S_bn"] for r in gscan
                    if abs(r["spec"]["gamma"] - gam) < 1e-9]
            s_by_gamma.append(float(np.mean(vals)))
        fit = fit_threshold(np.array(gammas), np.array(s_by_gamma))
        dev = (abs(fit["gamma_c"] - GAMMA_C_LITERATURE) / GAMMA_C_LITERATURE
               if np.isfinite(fit["gamma_c"]) else float("nan"))
        out["d_gamma_threshold"] = {
            "gammas": gammas,
            "S_bn_mean": s_by_gamma,
            "fit": fit,
            "gamma_c_literature": GAMMA_C_LITERATURE,
            "gamma_c_relative_deviation": float(dev) if np.isfinite(dev) else None,
            "S_saturation": fit["s_sat"],
            "pass": bool(np.isfinite(dev) and dev <= 0.30),
        }

    # ---- (e) transfer mapping vs shard sums ---------------------------
    transfer_cases = {}
    for name, rec in records.items():
        if rec["spec"]["drive"] != "vib":
            continue
        try:
            dist = distribution_from_record(rec, geo)
        except Exception as exc:  # pragma: no cover - diagnostic path
            transfer_cases[name] = {"error": repr(exc)}
            continue
        shard = zone_moment_check(dist)
        # Homogeneous reference: same mass, uniform fill whose centre of gravity
        # is the *pre-drive* centre of gravity of the bed.  Using the segregated
        # state's own mean height here would make dVCG identically zero by
        # construction (the mapping would compare a state with itself), which is
        # what an earlier revision did -- the reference must not be a function of
        # the state under test.
        state_mass = dist.total_mass_kg
        z_mean = float(dist.z_moment_kg_m.sum() / max(state_mass, 1e-30))
        z_ref = rec.get("initial_z_com_m")
        if not (isinstance(z_ref, (int, float)) and math.isfinite(z_ref) and z_ref > 0.0):
            h_settled = rec.get("settled_bed_height_m")
            z_ref = 0.5 * float(h_settled) if isinstance(h_settled, (int, float)) \
                and math.isfinite(h_settled) and h_settled > 0.0 else z_mean
        uniform = type(dist)(
            geometry=geo, mass_kg=dist.mass_kg,
            z_moment_kg_m=dist.mass_kg * z_ref,
            x_moment_kg_m=dist.mass_kg * (dist.geometry.length_m * 0.5),
            y_moment_kg_m=dist.mass_kg * geo.breadth_m * 0.5,
            z2_moment_kg_m2=dist.mass_kg * z_ref**2,
        )
        void = rec.get("void_fraction", 0.18)
        gm = gm_mapping(uniform, dist, displacement_t=displacement_t, km_m=km_m,
                        void_before=void, void_after=void)
        try:
            press = floor_pressure(dist)
        except ValueError:
            press = {}
        transfer_cases[name] = {
            "shard_check": shard,
            "gm": gm,
            "reference_vcg_m": float(z_ref),
            "segregated_vcg_m": float(z_mean),
            "pressure": press,
        }
    if transfer_cases:
        worst = max(
            (v["shard_check"]["max_rel"] for v in transfer_cases.values()
             if "shard_check" in v),
            default=float("nan"),
        )
        out["e_transfer"] = {
            "worst_shard_rel": float(worst),
            "n_cases": len(transfer_cases),
            "cases": {k: v for k, v in list(transfer_cases.items())[:20]},
            "pass": bool(np.isfinite(worst) and worst <= 1e-6),
        }
        load_errs = [abs(v["pressure"]["total_load_rel_error"])
                     for v in transfer_cases.values() if v.get("pressure")]
        if load_errs:
            out["e_transfer"]["max_floor_load_rel_error"] = float(np.max(load_errs))

    # ---- (f) seed reproducibility -------------------------------------
    seed_stats = {}
    for ratio in sorted(by_ratio):
        vals = by_ratio[ratio]["S_bn_seeds"]
        mean, std = _mean_std(vals)
        cv = std / abs(mean) if mean else float("inf")
        seed_stats[ratio] = {"mean_S_bn": mean, "std_S_bn": std, "cv": cv,
                             "n": len(vals)}
    if seed_stats:
        worst_cv = max(v["cv"] for v in seed_stats.values())
        out["f_seed_reproducibility"] = {
            "by_ratio": seed_stats,
            "max_cv": float(worst_cv),
            "pass": bool(np.isfinite(worst_cv) and worst_cv <= 0.15),
        }

    # ---- global verdict ------------------------------------------------
    keys = [k for k in out if k[0] in "abcdef"]
    out["verdict"] = {
        "criteria": {k: bool(out[k].get("pass", False)) for k in keys},
        "all_pass": all(bool(out[k].get("pass", False)) for k in keys),
        "n_criteria_evaluated": len(keys),
    }
    return out