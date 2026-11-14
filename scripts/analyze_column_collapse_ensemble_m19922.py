"""Reducer / machine gate for the CASE-COLUMN-COLLAPSE ensemble wave (m19922).

Reads the per-case JSON written by
``benchmarks/case_column_collapse_ensemble_m19922.py``, recomputes every
reference value from the Lube (2004) formulas (no target number is typed by
hand), emits

* ``docs/artifacts/column_collapse_ensemble_m19922/cases_ensemble.csv``
* ``docs/artifacts/column_collapse_ensemble_m19922/summary.json``
* ``docs/artifacts/column_collapse_ensemble_m19922/manifest_sha256.json``
* four figures

and adjudicates the pre-registered hypotheses (see the benchmark module
docstring).  Import-clean: the pure functions below take plain dicts so the
pytest gate can drive them with synthetic input.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tensordem.granular_column.axisym import (lube_2004_emplacement_time,  # noqa: E402
                                             lube_2004_runout_ratio)

# --- pre-registered thresholds (fixed before the runs, see module docstring) --
SPREAD_TOL = 0.05      # H_SPREAD: sigma(r_inf/r_i) / mean <= 5 %
BAND_TOL = 0.15        # H_BAND:   every member inside the anchor tolerance
REPRO_TOL = 0.005      # H_REPRO:  deterministic control reproduces the archive
JITTER_TOL = 0.05      # H_JITTER: lattice -> 10 % disorder changes runout < 5 %
SIGN_ASPECT = 0.5      # H_SIGN:   shallow column, time law must stay out of band
SIGN_FLOOR = 0.15      #           and on the late side for every member

#: Minimum membership before an *ensemble* statistic may be adjudicated.
#: A one-member "ensemble" has no scatter to estimate: its sigma is
#: undefined, not zero, and a NaN must never be read as a pass (m20259).
MIN_JITTER_MEMBERS = 2
MIN_MEMBERS = 2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_DIR = os.path.join(ROOT, "runs", "case_column_collapse_ensemble_m19922")
CASE_DIR = os.path.join(RUN_DIR, "cases")
ART_DIR = os.path.join(ROOT, "docs", "artifacts", "column_collapse_ensemble_m19922")
#: Archived closing-arm scalars from the round-5 deposit audit (machine-read
#: provenance for H_REPRO; the numbers are *not* retyped here).
ARCHIVE_CSV = os.path.join(ROOT, "docs", "artifacts",
                           "column_collapse_deposit_audit_m18385",
                           "cases_deposit_audit.csv")

CSV_COLUMNS = [
    "case", "member", "jitter", "seed", "aspect_nominal", "aspect_measured",
    "n_particles", "dt", "n_steps", "stop_reason",
    "runout_ratio_p995", "runout_ratio_max", "ref_runout_ratio", "err_runout",
    "ref_runout_ratio_measured_a", "err_runout_measured_a",
    "t90", "t95", "t99", "ref_emplacement_time", "err_t90", "err_t95", "err_t99",
    "energy_residual", "peak_front_speed", "wall_clock_s",
]


def _rel(measured: float, reference: float) -> float:
    if reference == 0.0 or not math.isfinite(reference):
        return float("nan")
    return (measured - reference) / reference


def reduce_case(raw: dict) -> dict:
    """One per-case JSON -> the scalar row that goes into the archive CSV."""
    case = raw["case"]
    a_nom = float(case["aspect_ratio"])
    a_meas = float(raw["aspect_measured"])
    # Reference values are *evaluated* from the Lube (2004) law, both on the
    # nominal aspect ratio (primary, as in rounds 1-6) and on the measured one
    # (self-consistent convention raised by m17534).
    ref_nom = lube_2004_runout_ratio(a_nom)
    ref_meas = lube_2004_runout_ratio(a_meas)
    ref_time = lube_2004_emplacement_time(a_nom, float(case["column_radius"]) if
                                          "column_radius" in case else
                                          float(case["grains_per_radius"]) * 2.0 *
                                          float(case["radius"]),
                                          float(case["gravity"]))
    runout = float(raw["runout_ratio_p995"])
    frac = raw.get("t_reach_by_fraction", {})
    row = {
        "case": case.get("label", ""),
        "member": raw.get("ensemble", {}).get("member", ""),
        "jitter": float(raw.get("ensemble", {}).get("jitter", case.get("jitter", 0.0))),
        "seed": int(raw.get("ensemble", {}).get("seed", case.get("seed", 0))),
        "aspect_nominal": a_nom,
        "aspect_measured": a_meas,
        "n_particles": int(raw["n_particles"]),
        "dt": float(raw["dt"]),
        "n_steps": int(raw.get("n_steps", 0)),
        "stop_reason": raw.get("stop_reason", ""),
        "runout_ratio_p995": runout,
        "runout_ratio_max": float(raw["runout_ratio_max"]),
        "ref_runout_ratio": ref_nom,
        "err_runout": _rel(runout, ref_nom),
        "ref_runout_ratio_measured_a": ref_meas,
        "err_runout_measured_a": _rel(runout, ref_meas),
        "t90": float(frac.get("0.9", float("nan"))),
        "t95": float(frac.get("0.95", float("nan"))),
        "t99": float(frac.get("0.99", float("nan"))),
        "ref_emplacement_time": ref_time,
        "energy_residual": float(raw["energy"]["residual_fraction_of_potential"]),
        "peak_front_speed": float(raw.get("peak_front_speed", float("nan"))),
        "wall_clock_s": float(raw.get("wall_clock_s", float("nan"))),
    }
    for key, col in (("0.9", "t90"), ("0.95", "t95"), ("0.99", "t99")):
        row[f"err_{col}"] = _rel(row[col], ref_time)
    return row


def load_rows(case_dir: str = CASE_DIR) -> list[dict]:
    rows = []
    for path in sorted(glob.glob(os.path.join(case_dir, "*.json"))):
        with open(path) as fh:
            rows.append(reduce_case(json.load(fh)))
    rows.sort(key=lambda r: (r["aspect_nominal"], r["jitter"], r["seed"]))
    return rows


def _mean_sigma(values: list[float]) -> tuple[float, float]:
    """Sample mean and (unbiased) sample sigma; sigma is NaN for n < 2.

    A single realisation has no scatter to estimate.  Returning 0.0 there used
    to make a one-member "ensemble" report a 0 % spread and pass H_SPREAD,
    i.e. an unmeasured quantity was reported as a measured zero (m20259).
    """
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
    return m, math.sqrt(var)


def summarize(rows: list[dict], band: float = BAND_TOL,
              spread_tol: float = SPREAD_TOL) -> dict:
    """Per-aspect ensemble statistics (jittered members only for sigma)."""
    aspects = sorted({r["aspect_nominal"] for r in rows})
    out = {}
    for a in aspects:
        grp = [r for r in rows if r["aspect_nominal"] == a]
        jit = [r for r in grp if r["jitter"] > 0.0]
        control = [r for r in grp if r["jitter"] == 0.0]
        errs = [r["err_runout"] for r in grp]
        mean, sigma = _mean_sigma([r["runout_ratio_p995"] for r in jit])
        err_mean, err_sigma = _mean_sigma(errs)
        spread_rel = (sigma / mean) if math.isfinite(mean) and mean else float("nan")
        per_aspect = {
            "n_members": len(grp),
            "n_jitter": len(jit),
            "n_control": len(control),
            "runout_mean": mean,
            "runout_sigma": sigma,
            "runout_sigma_rel": spread_rel,
            "runout_min": min(r["runout_ratio_p995"] for r in grp),
            "runout_max": max(r["runout_ratio_p995"] for r in grp),
            "err_mean": err_mean,
            "err_sigma": err_sigma,
            "err_worst_abs": max(abs(e) for e in errs),
            "n_in_band": sum(1 for e in errs if abs(e) <= band),
            "all_in_band": all(abs(e) <= band for e in errs),
            # None (not False) when the scatter itself is undefined: "not
            # measurable" and "measured too wide" are different statements.
            "spread_ok": (None if not math.isfinite(spread_rel)
                          else bool(spread_rel <= spread_tol)),
            "spread_testable": len(jit) >= MIN_JITTER_MEMBERS,
            "t99_errors": [r["err_t99"] for r in grp],
            "t95_errors": [r["err_t95"] for r in grp],
            "t90_errors": [r["err_t90"] for r in grp],
            "err_t99_mean": _mean_sigma([r["err_t99"] for r in grp])[0],
            "err_t95_mean": _mean_sigma([r["err_t95"] for r in grp])[0],
            "control_runout": (control[0]["runout_ratio_p995"] if control else None),
            "jitter_axis": sorted({r["jitter"] for r in grp}),
            "stop_reasons": sorted({r["stop_reason"] for r in grp}),
        }
        # H_JITTER: lattice -> maximal disorder, per aspect ratio.
        if control and jit:
            jitter_max = max(jit, key=lambda r: r["jitter"])
            per_aspect["jitter_shift"] = abs(_rel(jitter_max["runout_ratio_p995"],
                                                  control[0]["runout_ratio_p995"]))
        else:
            per_aspect["jitter_shift"] = float("nan")
        out[a] = per_aspect
    return out


def repro_check(rows: list[dict], archive_csv: str = ARCHIVE_CSV,
                tol: float = REPRO_TOL) -> dict:
    """Compare the ``jitter = 0`` control members against the round-5 archive.

    The archived closing-arm rows are selected mechanically from the deposit
    audit CSV: same nominal aspect ratio, ``grains_per_radius == 8``,
    ``dt_scale == 1.0``, ``mu_wall == 0.8``, ``mu == mu_rolling == 0.5`` and a
    ``w5b`` case id.  Nothing is retyped.
    """
    per_aspect, ok_all, used = {}, True, []
    if not os.path.exists(archive_csv):
        return {"available": False, "per_aspect": {}, "all_match": None,
                "tolerance": tol, "used": []}
    with open(archive_csv) as fh:
        archive = list(csv.DictReader(fh))
    with open(archive_csv, "rb") as fh:
        archive_sha = hashlib.sha256(fh.read()).hexdigest()
    for r in rows:
        if r["jitter"] != 0.0:
            continue
        a = r["aspect_nominal"]
        cands = [x for x in archive
                 if x.get("case", "").startswith("w5b")
                 and abs(float(x["a_nominal"]) - a) < 1e-9
                 and abs(float(x["grains_per_radius"]) - 8.0) < 1e-9
                 and abs(float(x["dt_scale"]) - 1.0) < 1e-9
                 and abs(float(x["mu_wall"]) - 0.8) < 1e-9
                 and abs(float(x["mu"]) - 0.5) < 1e-9
                 and abs(float(x["mu_rolling"]) - 0.5) < 1e-9]
        if not cands:
            per_aspect[a] = {"archived_case": None, "delta": float("nan"),
                             "match": None}
            continue
        cand = cands[0]
        archived = float(cand["stored_runout_ratio"])
        delta = _rel(r["runout_ratio_p995"], archived)
        match = abs(delta) <= tol
        ok_all = ok_all and match
        used.append({"aspect": a, "archived_case": cand["case"],
                     "archived_sha256": cand.get("sha256", ""),
                     "archived_runout_ratio": archived,
                     "this_runout_ratio": r["runout_ratio_p995"],
                     "delta": delta, "match": match})
        per_aspect[a] = {"archived_case": cand["case"],
                         "archived_runout_ratio": archived,
                         "current_runout_ratio": r["runout_ratio_p995"],
                         "delta": delta, "match": match}
    return {"available": True, "archive_csv": os.path.relpath(archive_csv, ROOT),
            "archive_sha256": archive_sha, "tolerance": tol,
            "per_aspect": per_aspect, "used": used, "n_checked": len(used),
            "all_match": ok_all if used else None}


def _cell(passed, n: int, need: int, detail) -> dict:
    """One adjudication cell: True / False / None(=insufficient evidence)."""
    if n < need:
        return {"pass": None, "status": "insufficient", "n": n,
                "required_n": need, "detail": detail}
    if passed is None:
        return {"pass": None, "status": "insufficient", "n": n,
                "required_n": need, "detail": detail}
    ok = bool(passed)
    return {"pass": ok, "status": "pass" if ok else "fail", "n": n,
            "required_n": need, "detail": detail}


def _combine(cells: list[dict], tolerance: float, detail: dict) -> dict:
    """A fail anywhere dominates; insufficient evidence dominates a pass."""
    statuses = [c["status"] for c in cells]
    if "fail" in statuses:
        status = "fail"
    elif "insufficient" in statuses or not cells:
        status = "insufficient"
    else:
        status = "pass"
    return {"pass": {"pass": True, "fail": False}.get(status),
            "status": status, "tolerance": tolerance, "detail": detail,
            "n": sum(c["n"] for c in cells),
            "required_n": sum(c["required_n"] for c in cells)}


def hypothesis_verdicts(summary: dict, repro: dict) -> dict:
    """Mechanical adjudication of the five pre-registered hypotheses.

    Every cell is tri-state.  ``pass`` is ``None`` when the statistic needed by
    the hypothesis is undefined or its membership is below the pre-registered
    minimum; a NaN therefore reads as *insufficient*, never as a pass (m20259).
    Thresholds are untouched: only the missing-evidence branch changed.
    """
    aspects = sorted(summary)

    cells_spread, cells_band, cells_jitter, detail_spread = [], [], [], {}
    for a in aspects:
        s = summary[a]
        span = s.get("runout_sigma_rel", float("nan"))
        detail_spread[a] = span
        ok = None if not math.isfinite(span) else span <= SPREAD_TOL
        cells_spread.append(_cell(ok, int(s.get("n_jitter", 0)),
                                  MIN_JITTER_MEMBERS, span))
        band_ok = all_in = bool(s["all_in_band"])
        cells_band.append(_cell(band_ok, int(s["n_members"]), 1,
                                s["err_worst_abs"]))
        shift = s.get("jitter_shift", float("nan"))
        has_axis = bool(s.get("n_control", 0)) and bool(s.get("n_jitter", 0))
        jit_ok = (None if (not has_axis or not math.isfinite(shift))
                  else shift <= JITTER_TOL)
        cells_jitter.append(_cell(jit_ok, int(s.get("n_jitter", 0)), 1, shift))

    sign_aspect = [a for a in aspects if abs(a - SIGN_ASPECT) < 1e-9]
    sign_ok, sign_detail, sign_n = None, {}, 0
    sign_cells = []
    for a in sign_aspect:
        errs = [e for e in summary[a]["t99_errors"] if math.isfinite(e)]
        sign_detail[a] = errs
        sign_n = len(errs)
        sign_ok = bool(errs) and all(e > SIGN_FLOOR for e in errs)
        sign_cells.append(_cell(sign_ok, len(errs), 1, errs))

    n_repro = int(repro.get("n_checked") or 0)
    repro_ok = repro.get("all_match")
    repro_cell = _cell(repro_ok, n_repro, 1, repro_ok)

    return {
        "H_REPRO": _combine([repro_cell], REPRO_TOL, "jitter=0 control vs round-5 "
                            "archived ratio"),
        "H_SPREAD": _combine(cells_spread, SPREAD_TOL, detail_spread),
        "H_BAND": _combine(cells_band, BAND_TOL,
                           {a: summary[a]["err_worst_abs"] for a in aspects}),
        "H_SIGN": _combine(sign_cells, SIGN_FLOOR, sign_detail),
        "H_JITTER": _combine(cells_jitter, JITTER_TOL,
                             {a: summary[a]["jitter_shift"] for a in aspects}),
    }


def write_csv(rows: list[dict], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_COLUMNS})
    return path


def write_manifest(paths: list[str], out_path: str) -> dict:
    manifest = {}
    for p in paths:
        if os.path.exists(p):
            with open(p, "rb") as fh:
                manifest[os.path.relpath(p, ROOT)] = hashlib.sha256(fh.read()).hexdigest()
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    return manifest


def _figures(rows: list[dict], summary: dict, out_dir: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    made = []
    aspects = sorted(summary)

    # Fig 1: runout ratio vs aspect ratio, member scatter + law + band
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    grid = [0.3 + 0.02 * i for i in range(151)]
    ax.plot(grid, [lube_2004_runout_ratio(a) for a in grid], "k-", lw=1.6,
            label="Lube et al. (2004) law")
    ax.fill_between(grid, [lube_2004_runout_ratio(a) * 0.85 for a in grid],
                    [lube_2004_runout_ratio(a) * 1.15 for a in grid],
                    color="0.8", alpha=0.5, label="±15 % band")
    for a in aspects:
        grp = [r for r in rows if r["aspect_nominal"] == a]
        ax.plot([r["aspect_nominal"] for r in grp],
                [r["runout_ratio_p995"] for r in grp], "o", ms=4,
                color="tab:blue", alpha=0.75)
    ax.plot(aspects, [summary[a]["runout_mean"] for a in aspects], "s-",
            color="tab:red", ms=5, label="ensemble mean (jittered)")
    ax.set_xlabel("aspect ratio $a=h_i/r_i$ (-)")
    ax.set_ylabel("$r_\\infty / r_i$ (-)")
    ax.set_title("COLUMN-COLLAPSE: ensemble vs Lube (2004)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_runout_ensemble.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # Fig 2: per-member relative error with ensemble mean ± 1 sigma
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhspan(-15, 15, color="0.85", label="±15 % acceptance")
    for a in aspects:
        grp = [r for r in rows if r["aspect_nominal"] == a]
        ax.plot([r["aspect_nominal"] for r in grp],
                [100 * r["err_runout"] for r in grp], "o", ms=4, color="tab:blue",
                alpha=0.75)
    ax.errorbar(aspects, [100 * summary[a]["err_mean"] for a in aspects],
                yerr=[100 * summary[a]["err_sigma"] for a in aspects],
                fmt="s-", color="tab:red", capsize=4, label="mean ± 1σ")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("aspect ratio $a$ (-)")
    ax.set_ylabel("runout error (%)")
    ax.set_title("realisation scatter vs the 15 % acceptance band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_runout_error_ensemble.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # Fig 3: emplacement-time error, three arrival definitions, per member
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhspan(-15, 15, color="0.85")
    for key, marker in (("err_t90", "o"), ("err_t95", "s"), ("err_t99", "^")):
        ax.plot([r["aspect_nominal"] for r in rows],
                [100 * r[key] for r in rows], marker, ms=4, alpha=0.6,
                label=key.replace("err_", "") + " arrival")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("aspect ratio $a$ (-)")
    ax.set_ylabel("emplacement-time error (%)")
    ax.set_title("time law: overshoot is sign-stable across realisations")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_time_error_ensemble.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)

    # Fig 4: runout vs jitter amplitude (the disorder axis)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for a in aspects:
        grp = sorted([r for r in rows if r["aspect_nominal"] == a],
                     key=lambda r: r["jitter"])
        ax.plot([r["jitter"] for r in grp], [r["runout_ratio_p995"] for r in grp],
                "o-", ms=4, label=f"a = {a:g}")
    ax.set_xlabel("initial positional disorder  jitter (σ / r_p)")
    ax.set_ylabel("$r_\\infty / r_i$ (-)")
    ax.set_title("lattice vs disordered packings")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = os.path.join(out_dir, "fig_jitter_axis.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig); made.append(p)
    return made


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case-dir", default=CASE_DIR)
    ap.add_argument("--out-dir", default=ART_DIR)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    rows = load_rows(args.case_dir)
    if not rows:
        print(f"no per-case JSON under {args.case_dir}", file=sys.stderr)
        return 2
    summary = summarize(rows)
    repro = repro_check(rows)
    verdicts = hypothesis_verdicts(summary, repro)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = write_csv(rows, os.path.join(args.out_dir, "cases_ensemble.csv"))
    figs = [] if args.no_figures else _figures(rows, summary, args.out_dir)
    payload = {
        "study": "CASE-COLUMN-COLLAPSE ensemble (initial-realization uncertainty)",
        "reference": "Lube et al. 2004 JFM 508:175-199, doi:10.1017/S0022112004009036",
        "tolerance": BAND_TOL,
        "thresholds": {"SPREAD_TOL": SPREAD_TOL, "BAND_TOL": BAND_TOL,
                       "REPRO_TOL": REPRO_TOL, "JITTER_TOL": JITTER_TOL,
                       "SIGN_ASPECT": SIGN_ASPECT, "SIGN_FLOOR": SIGN_FLOOR},
        "n_cases": len(rows),
        # aspects pre-registered by the wave design but never run in
        # this branch (declared, not silently missing; m20259)
        "incomplete_aspects": [a for a in (0.5, 1.0, 2.0)
                               if a not in {r["aspect_nominal"] for r in rows}],
        "aspects": sorted(summary),
        "summary": {f"{a:g}": summary[a] for a in summary},
        "repro": repro,
        "verdicts": verdicts,
        "rows": rows,
        "figures": [os.path.relpath(p, ROOT) for p in figs],
    }
    json_path = os.path.join(args.out_dir, "summary.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    write_manifest([csv_path, json_path] + figs,
                   os.path.join(args.out_dir, "manifest_sha256.json"))

    for a in summary:
        s = summary[a]
        print(f"a={a:g}: mean r/r_i={s['runout_mean']:.3f} ± {s['runout_sigma']:.3f} "
              f"({100 * s['runout_sigma_rel']:.1f} %), worst |err|={100 * s['err_worst_abs']:.1f} %, "
              f"in-band {s['n_in_band']}/{s['n_members']}, jitter shift="
              f"{100 * s['jitter_shift']:.1f} %")
    for h, v in verdicts.items():
        print(f"{h}: {v['status'].upper()} (n={v['n']}/{v['required_n']})")
    print(f"artifacts -> {os.path.relpath(args.out_dir, ROOT)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())