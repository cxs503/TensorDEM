"""Reduce + adjudicate the CASE-COLUMN-COLLAPSE realisation axis (m20259).

Inputs (all machine-read, nothing retyped)
------------------------------------------
* ``runs/case_column_collapse_ensemble_m19922/cases/*.json``            — the m19922 wave
  (lattice control + ``jitter = 0.05 / 0.10`` members, as pre-registered there)
* ``runs/case_column_collapse_ensemble_admissible_m20259/cases/*.json`` — this mission's
  admissible wave (``jitter = 0.0025 / 0.005``)
* initial packings rebuilt from ``AxisymCase`` (same seed) for the geometry scan

Outputs
-------
``docs/artifacts/column_collapse_ensemble_m20259/{cases_ensemble.csv, summary.json,
manifest_sha256.json, fig_admissibility_scan.png, fig_runout_vs_jitter.png}``

Pre-registered hypotheses of *this* mission (fixed before the first admissible
member was run; thresholds are not retuned after the fact):

``A_ADMISS``  every admissible-wave member starts with ``overlap_pairs == 0``.
``A_BAND``    every admissible member is within 15 % of the Lube (2004) runout law.
``A_SPREAD``  1-sigma / mean of ``r_inf/r_i`` over the admissible members <= 5 %.
``A_SCALE``   |shift| from the lattice to the largest admissible jitter <= 5 %.
``A_GAP``     the m19922 ``jitter`` amplitudes are geometrically inadmissible
              (``overlap_pairs > 0``), so its out-of-band members cannot be read
              as physical realisation uncertainty.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensordem.granular_column.admissibility import (  # noqa: E402
    max_admissible_jitter, overlap_stats)
from tensordem.granular_column.axisym import (  # noqa: E402
    AxisymCase, build_axisymmetric_column, lube_2004_emplacement_time,
    lube_2004_runout_ratio)

RUN_INVALID = os.path.join(ROOT, "runs", "case_column_collapse_ensemble_m19922",
                           "cases")
RUN_ADMISSIBLE = os.path.join(ROOT, "runs",
                              "case_column_collapse_ensemble_admissible_m20259",
                              "cases")
ART_DIR = os.path.join(ROOT, "docs", "artifacts", "column_collapse_ensemble_m20259")

BAND_TOL = 0.15          # anchor tolerance (unchanged)
SPREAD_TOL = 0.05        # A_SPREAD
SCALE_TOL = 0.05         # A_SCALE
SCAN_LEVELS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10)
SCAN_SEED = 101
SCAN_ASPECTS = (0.5, 2.0)

CLOSING_ARM = dict(radius=0.03, density=2500.0, mu=0.5, mu_rolling=0.5,
                   mu_wall=0.8, restitution=0.5, kn=6.0e4, kt_ratio=0.5,
                   grains_per_radius=8.0, packing_gap=0.02, gravity=9.81,
                   dt_scale=1.0, duration_factor=8.0, min_time=0.5,
                   rest_ke_tol=1.0e-4, rest_hold=200, t_reach_fraction=0.99)

COLUMNS = ["wave", "case", "member", "jitter", "seed", "aspect_nominal",
           "aspect_measured", "n_particles", "stop_reason", "runout_ratio_p995",
           "ref_runout_ratio", "err_runout", "t95", "t99", "ref_emplacement_time",
           "err_t95", "err_t99", "peak_front_speed", "overlap_pairs",
           "max_overlap", "admissible"]


def _load(directory: str, wave: str, geom: dict) -> list[dict]:
    rows = []
    if not os.path.isdir(directory):
        return rows
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name)) as fh:
            raw = json.load(fh)
        ens = raw.get("ensemble", {})
        a_nom = ens.get("aspect", raw.get("aspect_nominal"))
        if a_nom is None:
            a_nom = float(name.split("_a")[1].split("_")[0])
        jitter = float(ens.get("jitter", 0.0))
        _case = AxisymCase(**CLOSING_ARM, aspect_ratio=float(a_nom),
                           jitter=jitter, seed=int(ens.get("seed", 0)))
        _pos, _ = build_axisymmetric_column(_case)
        st = overlap_stats(_pos, _case.radius)
        frac = raw.get("t_reach_by_fraction", {})
        ref = lube_2004_runout_ratio(float(a_nom))
        t_ref = float(lube_2004_emplacement_time(
            float(a_nom), AxisymCase(**CLOSING_ARM, aspect_ratio=float(a_nom)).column_radius,
            9.81))
        rows.append({
            "wave": wave,
            "case": raw.get("label", name[:-5]),
            "member": ens.get("member", ""),
            "jitter": jitter,
            "seed": int(ens.get("seed", 0)),
            "aspect_nominal": float(a_nom),
            "aspect_measured": float(raw["aspect_measured"]),
            "n_particles": int(raw["n_particles"]),
            "stop_reason": raw.get("stop_reason", ""),
            "runout_ratio_p995": float(raw["runout_ratio_p995"]),
            "ref_runout_ratio": float(ref),
            "err_runout": float(raw["runout_ratio_p995"]) / float(ref) - 1.0,
            "t95": float(frac.get("0.95", float("nan"))),
            "t99": float(frac.get("0.99", float("nan"))),
            "ref_emplacement_time": t_ref,
            "err_t95": (float(frac["0.95"]) / t_ref - 1.0) if t_ref and math.isfinite(t_ref)
            and "0.95" in frac else float("nan"),
            "err_t99": (float(frac["0.99"]) / t_ref - 1.0) if t_ref and math.isfinite(t_ref)
            and "0.99" in frac else float("nan"),
            "peak_front_speed": float(raw.get("peak_front_speed", float("nan"))),
            "overlap_pairs": (st or {}).get("overlap_pairs", None),
            "max_overlap": (st or {}).get("max_overlap", None),
            "admissible": (st or {}).get("admissible", None),
        })
    return rows


def seed_key(ens: dict):
    return int(ens.get("seed", 0))


def geometry_scan() -> dict:
    """Overlap census of the initial packings, per aspect and jitter level."""
    out = {"scan_seed": SCAN_SEED, "levels": list(SCAN_LEVELS), "aspects": {}}
    for a in SCAN_ASPECTS:
        def build(j, a=a):
            case = AxisymCase(**CLOSING_ARM, aspect_ratio=a, jitter=j,
                              seed=SCAN_SEED)
            pos, _ = build_axisymmetric_column(case)
            return pos

        res = max_admissible_jitter(build, AxisymCase(
            **CLOSING_ARM, aspect_ratio=a, seed=SCAN_SEED).radius, SCAN_LEVELS)
        out["aspects"][f"{a:g}"] = res
    return out


def _stats(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, float("nan")
    return m, math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def adjudicate(rows: list[dict]) -> dict:
    adm = [r for r in rows if r["wave"] == "admissible"]
    inv = [r for r in rows if r["wave"] == "inadmissible"]
    control = [r for r in rows if r["jitter"] == 0.0]

    cells, out = [], {}
    n_bad = sum(1 for r in adm if r["overlap_pairs"] != 0)
    out["A_ADMISS"] = {"pass": bool(adm) and n_bad == 0, "n": len(adm),
                       "required_n": 1,
                       "status": ("pass" if (adm and n_bad == 0) else
                                  ("fail" if adm else "insufficient")),
                       "detail": f"members with initial overlaps: {n_bad}"}
    worst = max((abs(r["err_runout"]) for r in adm), default=float("nan"))
    out["A_BAND"] = {"pass": (worst <= BAND_TOL) if adm else None,
                     "status": ("pass" if adm and worst <= BAND_TOL else
                                ("fail" if adm else "insufficient")),
                     "n": len(adm), "required_n": 1, "detail": worst}
    mean, sigma = _stats([r["runout_ratio_p995"] for r in adm])
    rel = sigma / mean if math.isfinite(sigma) and mean else float("nan")
    out["A_SPREAD"] = {"pass": (rel <= SPREAD_TOL) if math.isfinite(rel) else None,
                       "status": ("pass" if math.isfinite(rel) and rel <= SPREAD_TOL
                                  else ("fail" if math.isfinite(rel) else "insufficient")),
                       "n": len(adm), "required_n": 2,
                       "detail": {"mean": mean, "sigma": sigma, "sigma_rel": rel}}
    if control and adm:
        c = control[0]["runout_ratio_p995"]
        shift = max(abs(r["runout_ratio_p995"] / c - 1.0) for r in adm)
        out["A_SCALE"] = {"pass": shift <= SCALE_TOL,
                          "status": "pass" if shift <= SCALE_TOL else "fail",
                          "n": len(adm) + 1, "required_n": 2, "detail": shift}
    else:
        out["A_SCALE"] = {"pass": None, "status": "insufficient", "n": 0,
                          "required_n": 2, "detail": None}
    jit_inv = [r for r in inv if r["jitter"] > 0.0]
    inadm = [r for r in jit_inv if r["overlap_pairs"] and r["overlap_pairs"] > 0]
    frac = len(inadm) / len(jit_inv) if jit_inv else float("nan")
    out["A_GAP"] = {"pass": bool(jit_inv) and len(inadm) == len(jit_inv),
                    "status": ("pass" if jit_inv and len(inadm) == len(jit_inv) else
                               ("fail" if jit_inv else "insufficient")),
                    "n": len(jit_inv), "required_n": 1,
                    "detail": {"jitter_members_with_overlaps": len(inadm),
                               "jitter_members": len(jit_inv),
                               "fraction": frac}}
    del cells
    return out


def m19922_verdicts(rows: list[dict]) -> dict:
    """Re-adjudicate the m19922 wave with the corrected tri-state logic."""
    inv = [r for r in rows if r["wave"] == "inadmissible"]
    if not inv:
        return {}
    control = [r for r in inv if r["jitter"] == 0.0]
    jit = [r for r in inv if r["jitter"] > 0.0]
    mean, sigma = _stats([r["runout_ratio_p995"] for r in jit])
    rel = sigma / mean if math.isfinite(sigma) and mean else float("nan")
    worst = max(abs(r["err_runout"]) for r in inv)
    shift = (max(abs(r["runout_ratio_p995"] / control[0]["runout_ratio_p995"] - 1.0)
                 for r in jit) if control and jit else float("nan"))
    return {
        "H_BAND": {"pass": worst <= BAND_TOL,
                   "status": "pass" if worst <= BAND_TOL else "fail",
                   "n": len(inv), "required_n": 1, "detail": worst},
        "H_SPREAD": {"pass": (rel <= SPREAD_TOL) if math.isfinite(rel) else None,
                     "status": ("pass" if rel <= SPREAD_TOL else "fail")
                     if math.isfinite(rel) else "insufficient",
                     "n": len(jit), "required_n": 2,
                     "detail": {"mean": mean, "sigma": sigma, "sigma_rel": rel}},
        "H_JITTER": {"pass": (shift <= SPREAD_TOL) if math.isfinite(shift) else None,
                     "status": ("pass" if shift <= SPREAD_TOL else "fail")
                     if math.isfinite(shift) else "insufficient",
                     "n": len(jit), "required_n": 1, "detail": shift},
    }


def write_csv(rows: list[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def figures(rows: list[dict], geom: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for a, res in geom["aspects"].items():
        lv = [r["jitter"] for r in res["levels"]]
        ov = [1000.0 * r["max_overlap"] for r in res["levels"]]
        ax.plot(lv, ov, "o-", label=f"a = {a} (seed {geom['scan_seed']})")
        if res["jitter_max_admissible"] is not None:
            ax.axvline(res["jitter_max_admissible"], ls=":", lw=0.8, color="grey")
    ax.axvspan(0.0, 0.005, color="tab:green", alpha=0.12)
    ax.set_xlabel("positional jitter amplitude  [$r_p$]")
    ax.set_ylabel("deepest initial overlap  [mm]")
    ax.set_title("Initial packing admissibility: jitter vs stored overlap\n"
                 "(green = admissible band, overlap_pairs = 0)")
    ax.legend()
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_admissibility_scan.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for wave, marker, lab in (("inadmissible", "s", "m19922 wave (pre-registered)"),
                              ("admissible", "o", "m20259 admissible wave")):
        sub = sorted((r for r in rows if r["wave"] == wave),
                     key=lambda r: r["jitter"])
        if not sub:
            continue
        x = [r["jitter"] for r in sub]
        y = [100.0 * r["err_runout"] for r in sub]
        ax.plot(x, y, marker + "-", label=lab)
        for r in sub:
            ax.annotate(f"{100 * r['err_runout']:+.0f}%",
                        (r["jitter"], 100 * r["err_runout"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.axhspan(-100 * BAND_TOL, 100 * BAND_TOL, color="tab:blue", alpha=0.10)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("positional jitter amplitude  [$r_p$]")
    ax.set_ylabel("runout error vs Lube (2004)  [%]")
    ax.set_title("a = 0.5: runout error vs initial-state jitter\n"
                 "(shaded = +/-15 % anchor band)")
    ax.legend()
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_runout_vs_jitter.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def main() -> int:
    geom = geometry_scan()
    scan = {}
    for a_key, res in geom["aspects"].items():
        for r in res["levels"]:
            scan[(round(float(a_key), 6), SCAN_SEED)] = r
    rows = _load(RUN_INVALID, "inadmissible", scan)
    rows += _load(RUN_ADMISSIBLE, "admissible", scan)
    if not rows:
        print("no case JSON found", file=sys.stderr)
        return 2
    os.makedirs(ART_DIR, exist_ok=True)
    csv_path = write_csv(rows, os.path.join(ART_DIR, "cases_ensemble.csv"))
    figs = figures(rows, geom, ART_DIR)
    payload = {
        "study": "CASE-COLUMN-COLLAPSE realisation axis: admissibility, not jitter",
        "reference": "Lube et al. 2004 JFM 508:175-199, doi:10.1017/S0022112004009036",
        "tolerance": BAND_TOL,
        "thresholds": {"BAND_TOL": BAND_TOL, "SPREAD_TOL": SPREAD_TOL,
                       "SCALE_TOL": SCALE_TOL},
        "n_cases": len(rows),
        "incomplete_aspects": [a for a in (0.5, 1.0, 2.0)
                               if a not in {r["aspect_nominal"] for r in rows}],
        "geometry_scan": geom,
        "verdicts": adjudicate(rows),
        "m19922_readjudicated": m19922_verdicts(rows),
        "figures": [os.path.relpath(p, ROOT) for p in figs],
        "rows": rows,
    }
    json_path = os.path.join(ART_DIR, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    manifest = {os.path.relpath(p, ROOT): sha256(p) for p in [csv_path, json_path] + figs}
    with open(os.path.join(ART_DIR, "manifest_sha256.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    print(f"{'wave':<13}{'a':>5}{'jitter':>8}{'overlap':>9}{'r_inf/r_i':>11}"
          f"{'err%':>8}{'t95err%':>9}{'stop':>10}")
    for r in sorted(rows, key=lambda r: (r["wave"], r["jitter"], r["seed"])):
        print(f"{r['wave']:<13}{r['aspect_nominal']:>5.2f}{r['jitter']:>8.4f}"
              f"{str(r['overlap_pairs']):>9}{r['runout_ratio_p995']:>11.3f}"
              f"{100 * r['err_runout']:>8.1f}{100 * r['err_t95']:>9.1f}"
              f"{r['stop_reason']:>10}")
    print()
    for a_key, res in geom["aspects"].items():
        print(f"geometry a={a_key}: jitter_max_admissible="
              f"{res['jitter_max_admissible']} first_rejecting="
              f"{res['jitter_first_rejecting']}")
    print()
    for h, v in payload["verdicts"].items():
        print(f"{h}: {v['status'].upper()} (n={v['n']}/{v['required_n']}) {v['detail']}")
    print("m19922 re-adjudicated:")
    for h, v in payload["m19922_readjudicated"].items():
        print(f"  {h}: {v['status'].upper()} (n={v['n']}/{v['required_n']}) {v['detail']}")
    print(f"artifacts -> {os.path.relpath(ART_DIR, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())