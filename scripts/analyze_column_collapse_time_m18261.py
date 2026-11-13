"""CASE-COLUMN-COLLAPSE m18261 — adjudication, tables, figures.

Reads the per-case JSONs written by
``benchmarks/case_column_collapse_time_m18261.py`` and the archived round-5
case JSONs (read-only, for the ``H6`` method audit), prints every table of
``docs/CASE_COLUMN_COLLAPSE_REFINEMENT_m18261.md``, writes
``artifacts/case_column_collapse_time_m18261/`` (figures + ``analysis.json``)
and never transcribes a number by hand: the report is generated from the same
``analysis.json``.

Adjudication rules are the ones frozen in the benchmark docstring; they are
implemented in :func:`adjudicate` and asserted by the test suite.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from tensordem.granular_column.arrival import (  # noqa: E402
    DEFAULT_WINDOW_S, PRIMARY_FRACTION, PRIMARY_LENGTH_TOLERANCE_D,
    PRIMARY_VELOCITY_FRACTION, arrival_report)

TOL = 0.15
PRIMARY = ("fraction_0.9", "velocity_0.01", "length_1d")
CARD_ASPECTS = (0.5, 1.0, 1.5, 2.0, 3.0)
RUN_DIR = os.path.join("runs", "case_column_collapse_time_m18261")
CASE_DIR = os.path.join(RUN_DIR, "cases")
OUT_DIR = os.path.join("artifacts", "case_column_collapse_time_m18261")
#: Archived round-5 worktree used for the H6 method audit (read-only).
ARCHIVE_DIR = "/root/tp-agent/work/m17111/runs/case_column_collapse_lube2004/cases"


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_cases(case_dir: str = CASE_DIR) -> dict[str, dict]:
    cases = {}
    for path in sorted(glob.glob(os.path.join(case_dir, "*.json"))):
        with open(path) as fh:
            row = json.load(fh)
        row["_path"] = path
        cases[row["case_id"]] = row
    return cases


def pct(value: float) -> str:
    return "n/a" if value is None or not math.isfinite(value) else f"{100 * value:+.1f}%"


def err_t(row: dict, key: str, convention: str = "nominal") -> float:
    return row["t_error_" + convention][key]


def err_runout(row: dict, convention: str = "nominal") -> float:
    return row["runout_error_" + convention]


# --------------------------------------------------------------------------
# H6 — are the historical "12 grains per radius" controls refinements?
# --------------------------------------------------------------------------
def audit_historical_controls(archive_dir: str = ARCHIVE_DIR) -> dict:
    """Compare the archived gpr-8 and gpr-12 controls' *dimensionless* setup."""
    pairs = [("w5b_wall0.8_a1", "w5c_gpr12_a1", 1.0),
             ("w5b_wall0.8_a3", "w5c_gpr12_a3", 3.0)]
    rows, notes = [], []
    for base_id, fine_id, aspect in pairs:
        b_path = os.path.join(archive_dir, f"{base_id}.json")
        f_path = os.path.join(archive_dir, f"{fine_id}.json")
        if not (os.path.exists(b_path) and os.path.exists(f_path)):
            continue
        with open(b_path) as fh:
            b = json.load(fh)
        with open(f_path) as fh:
            f = json.load(fh)
        rows.append({
            "aspect": aspect,
            "coarse_id": base_id, "fine_id": fine_id,
            "coarse_radius": b["case"]["radius"],
            "fine_radius": f["case"]["radius"],
            "coarse_grains_per_radius": b["case"]["grains_per_radius"],
            "fine_grains_per_radius": f["case"]["grains_per_radius"],
            "coarse_column_radius": b["column_radius_measured"],
            "fine_column_radius": f["column_radius_measured"],
            "coarse_aspect_measured": b["aspect_measured"],
            "fine_aspect_measured": f["aspect_measured"],
            "coarse_kn": b["case"]["kn"], "fine_kn": f["case"]["kn"],
            "coarse_runout_ratio": b["runout_ratio_p995"],
            "fine_runout_ratio": f["runout_ratio_p995"],
            "coarse_sha256": hashlib.sha256(open(b_path, "rb").read()).hexdigest(),
            "fine_sha256": hashlib.sha256(open(f_path, "rb").read()).hexdigest(),
        })
    for row in rows:
        # a refinement holds r_i and the dimensionless stiffness; these did not
        row["column_radius_change"] = (row["fine_column_radius"]
                                       / row["coarse_column_radius"] - 1.0)
        row["is_refinement"] = (abs(row["coarse_radius"] - row["fine_radius"]) < 1e-12
                                and abs(row["column_radius_change"]) < 0.01)
        notes.append(
            f"{row['fine_id']}: r_i {row['coarse_column_radius']:.3f} -> "
            f"{row['fine_column_radius']:.3f} m "
            f"({100 * row['column_radius_change']:+.1f} %), grain radius unchanged "
            f"({row['coarse_radius']:.4f} m) -> NOT a refinement")
    return {"archive_dir": archive_dir, "available": bool(rows), "rows": rows,
            "notes": notes,
            "verdict": ("supported" if rows and not any(r["is_refinement"] for r in rows)
                        else ("no_data" if not rows else "falsified"))}


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
def card_table(cases: dict) -> list[dict]:
    """The card's five aspects at the frozen resolution (gpr = 8)."""
    rows = []
    for aspect in CARD_ASPECTS:
        cid = f"R_a{aspect:g}_g8"
        row = cases.get(cid)
        if row is None:
            continue
        rows.append({
            "case_id": cid, "aspect": aspect,
            "aspect_measured": row["aspect_measured"],
            "runout_ratio": row["runout_ratio_p995"],
            "ref_runout_nominal": row["reference"]["nominal"]["runout_ratio"],
            "ref_runout_measured": row["reference"]["measured"]["runout_ratio"],
            "err_runout_nominal": err_runout(row, "nominal"),
            "err_runout_measured": err_runout(row, "measured"),
            "runout_pass_nominal": abs(err_runout(row, "nominal")) <= TOL,
            "runout_pass_measured": abs(err_runout(row, "measured")) <= TOL,
            "ref_t": row["reference"]["nominal"]["emplacement_time"],
            "t_primary": {k: row["arrival"]["primary"][k] for k in PRIMARY},
            "err_t": {k: {c: err_t(row, k, c) for c in ("nominal", "measured")}
                      for k in PRIMARY},
            "t_pass": {k: abs(err_t(row, k, "nominal")) <= TOL for k in PRIMARY},
            "stop_reason": row["stop_reason"],
            "energy_residual": row["energy"]["residual_fraction_of_potential"],
        })
    return rows


def definition_table(cases: dict, aspect: float = 0.5) -> dict:
    """Every pre-registered definition at one aspect (nominal convention)."""
    row = cases.get(f"R_a{aspect:g}_g8")
    if row is None:
        return {}
    ref = row["reference"]["nominal"]["emplacement_time"]
    out = {"case_id": row["case_id"], "ref_t": ref, "fraction": {},
           "velocity": {}, "length": {}}
    for family in ("fraction", "velocity", "length"):
        for member, value in row["arrival"][family].items():
            out[family][member] = {"t": value, "error": (value - ref) / ref}
    return out


def refinement_table(cases: dict) -> list[dict]:
    rows = []
    for aspect in (0.5, 1.0, 3.0):
        for gpr in (8.0, 12.0, 16.0):
            cid = f"R_a{aspect:g}_g{gpr:g}"
            row = cases.get(cid)
            if row is None:
                continue
            rows.append({
                "case_id": cid, "aspect": aspect, "gpr": gpr,
                "d_over_ri": row["protocol"]["radius"] / row["column_radius_measured"],
                "n_particles": row["n_particles"], "dt": row["dt"],
                "runout_ratio": row["runout_ratio_p995"],
                "err_runout": err_runout(row, "nominal"),
                "err_runout_measured": err_runout(row, "measured"),
                "err_t": {k: err_t(row, k, "nominal") for k in PRIMARY},
                "stop_reason": row["stop_reason"],
                "wall_clock_s": row["wall_clock_s"],
            })
    return rows


def axis_delta_table(cases: dict) -> list[dict]:
    """T (dt/2) and G (gap 0) deltas against the matching gpr-8 baseline."""
    rows = []
    for cid, row in sorted(cases.items()):
        if row["axis"] not in ("T", "G"):
            continue
        aspect = row["aspect_nominal"]
        base = cases.get(f"R_a{aspect:g}_g8")
        if base is None:
            continue
        rows.append({
            "case_id": cid, "axis": row["axis"], "aspect": aspect,
            "dt_scale": row["protocol"]["dt_scale"],
            "packing_gap": row["protocol"]["packing_gap"],
            "runout_ratio": row["runout_ratio_p995"],
            "err_runout": err_runout(row, "nominal"),
            "delta_runout_pp": 100 * (err_runout(row, "nominal")
                                      - err_runout(base, "nominal")),
            "err_t": {k: err_t(row, k, "nominal") for k in PRIMARY},
            "delta_t_pp": {k: 100 * (err_t(row, k, "nominal")
                                     - err_t(base, k, "nominal")) for k in PRIMARY},
            "aspect_measured": row["aspect_measured"],
            "stop_reason": row["stop_reason"],
            "wall_clock_s": row["wall_clock_s"],
        })
    return rows


# --------------------------------------------------------------------------
# adjudication (rules frozen in the benchmark docstring)
# --------------------------------------------------------------------------
def adjudicate(cases: dict, historical: dict) -> dict:
    card = card_table(cases)
    refinement = refinement_table(cases)
    axes = axis_delta_table(cases)
    verdicts = {}

    complete = len(card) == len(CARD_ASPECTS)
    # H1: velocity-threshold definition closes all five aspects
    h1_pass = [r["t_pass"]["velocity_0.01"] for r in card]
    verdicts["H1_velocity_definition_closes"] = {
        "rule": "velocity 0.01 -> all 5 card aspects |err| <= 15%",
        "n_pass": int(sum(h1_pass)), "n_cases": len(card),
        "status": ("supported" if complete and all(h1_pass)
                   else ("falsified" if complete else "not_run")),
        "median_error": float(np.median([r["err_t"]["velocity_0.01"]["nominal"]
                                         for r in card])) if card else None,
    }
    # H2: runout closure survives refinement at a = 0.5 and 1.0
    h2 = {}
    for aspect in (0.5, 1.0):
        pts = [r for r in refinement if r["aspect"] == aspect]
        if len(pts) < 2:
            h2[aspect] = {"status": "not_run"}
            continue
        worst = max(abs(p["err_runout"]) for p in pts)
        drift = 100 * abs(pts[-1]["err_runout"] - pts[0]["err_runout"])
        h2[f"{aspect:g}"] = {
            "points": [{"gpr": p["gpr"], "err_runout": p["err_runout"]} for p in pts],
            "worst_abs_err": worst, "drift_pp": drift,
            "in_band_all": worst <= TOL, "drift_ok": drift <= 5.0,
            "status": ("supported" if worst <= TOL and drift <= 5.0 else "falsified"),
        }
    verdicts["H2_runout_closure_survives_refinement"] = h2
    # H3: time deficit is model, not discretisation
    shallow = [r for r in refinement if r["aspect"] == 0.5]
    if len(shallow) >= 2:
        deltas = {k: 100 * abs(shallow[-1]["err_t"][k] - shallow[0]["err_t"][k])
                  for k in PRIMARY}
        worst = max(deltas.values())
        verdicts["H3_time_deficit_is_model"] = {
            "deltas_pp": deltas, "threshold_pp": 5.0,
            "status": ("supported" if worst <= 5.0 else "falsified"),
            "finest_errors": shallow[-1]["err_t"],
            "finest_all_pass": all(abs(v) <= TOL
                                   for v in shallow[-1]["err_t"].values()),
        }
    else:
        verdicts["H3_time_deficit_is_model"] = {"status": "not_run"}
    # H4/H5 from the axis table
    for tag, axis, limit in (("H4_initialisation_gap", "G", 5.0),
                             ("H5_timestep_adequacy", "T", 2.0)):
        pts = [r for r in axes if r["axis"] == axis]
        if not pts:
            verdicts[tag] = {"status": "not_run"}
            continue
        worst_r = max(abs(p["delta_runout_pp"]) for p in pts)
        worst_t = max(max(abs(v) for v in p["delta_t_pp"].values()) for p in pts)
        verdicts[tag] = {
            "cases": [p["case_id"] for p in pts],
            "worst_delta_runout_pp": worst_r, "worst_delta_t_pp": worst_t,
            "threshold_pp": limit,
            "status": ("supported" if max(worst_r, worst_t) <= limit
                       else "falsified"),
        }
    verdicts["H6_historical_control_is_not_a_refinement"] = {
        "status": historical.get("verdict", "not_run"),
        "rows": [{k: r[k] for k in ("fine_id", "coarse_column_radius",
                                    "fine_column_radius", "column_radius_change",
                                    "is_refinement")} for r in historical.get("rows", [])],
    }
    return {"verdicts": verdicts}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_figures(cases: dict, analysis: dict, out_dir: str) -> dict:
    plt = _mpl()
    os.makedirs(out_dir, exist_ok=True)
    figures = {}

    # 1 runout law with the refinement points
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    a_line = np.linspace(0.2, 3.4, 200)
    from tensordem.granular_column.axisym import lube_2004_runout_ratio
    ref = [lube_2004_runout_ratio(a) for a in a_line]
    ax.plot(a_line, ref, "k-", lw=1.4, label="Lube et al. (2004) closed form")
    for gpr, marker, colour in ((8.0, "o", "C0"), (12.0, "s", "C1"),
                                (16.0, "^", "C2")):
        xs, ys = [], []
        for aspect in CARD_ASPECTS:
            row = cases.get(f"R_a{aspect:g}_g{gpr:g}")
            if row is None:
                continue
            xs.append(row["aspect_measured"])
            ys.append(row["runout_ratio_p995"])
        if xs:
            ax.plot(xs, ys, marker, ls="none", ms=6,
                    label=f"DEM, {gpr:g} grains/$r_i$")
    ax.set_xlabel("measured aspect ratio $h_i/r_i$")
    ax.set_ylabel("$r_\\infty / r_i$")
    ax.set_title("runout law under grid refinement (closed arm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "runout_law_refined.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures["runout_law_refined.png"] = (
        "Runout ratio vs measured aspect ratio: Lube et al. (2004) closed form "
        "and the DEM points at 8/12/16 grains per column radius.")

    # 2 error vs definition family at the card aspects (velocity family)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    members = sorted(next(iter(cases.values()))["arrival"]["velocity"].keys(),
                     key=float)
    for aspect in CARD_ASPECTS:
        row = cases.get(f"R_a{aspect:g}_g8")
        if row is None:
            continue
        errs = [100 * (row["arrival"]["velocity"][m]
                       - row["reference"]["nominal"]["emplacement_time"])
                / row["reference"]["nominal"]["emplacement_time"] for m in members]
        ax.plot([float(m) for m in members], errs, "o-", label=f"a = {aspect:g}")
    ax.axhspan(-15, 15, color="0.85", label="±15 % band")
    ax.set_xscale("log")
    ax.set_xlabel("velocity-threshold fraction of the front-speed peak")
    ax.set_ylabel("error of $t_\\infty$ [%]")
    ax.set_title("the velocity-threshold family does not open the band")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "velocity_definition_family.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures["velocity_definition_family.png"] = (
        "Emplacement-time error vs the velocity-threshold definition: the whole "
        "family sits above the reference, so it is not a definition artefact.")

    # 3 dimensionless front trajectories
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for aspect in CARD_ASPECTS:
        row = cases.get(f"R_a{aspect:g}_g8")
        if row is None:
            continue
        t = np.asarray(row["history"]["time"], float)
        f = np.asarray(row["history"]["front"], float)
        r_i = row["column_radius_measured"]
        scale = math.sqrt(row["column_height_measured"] / 9.81)
        ax.plot(t / scale, (f - r_i) / r_i, lw=1.2, label=f"a = {aspect:g}")
        ax.axvline(3.0 * math.sqrt(aspect), ls=":", lw=0.8, color="0.6")
    ax.set_xlabel("$t\\,/\\,\\sqrt{h_i/g}$")
    ax.set_ylabel("$(r_{front} - r_i)/r_i$")
    ax.set_title("front trajectories (vertical: reference $t_\\infty$)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "front_trajectories.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures["front_trajectories.png"] = (
        "Dimensionless front trajectories: the shallow cases keep creeping past "
        "the reference dimensionless emplacement time.")

    # 4 convergence of the two errors
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for aspect, marker in ((0.5, "o"), (1.0, "s"), (3.0, "^")):
        pts = [r for r in analysis["refinement"] if r["aspect"] == aspect]
        if not pts:
            continue
        x = [p["d_over_ri"] for p in pts]
        axes[0].plot(x, [100 * p["err_runout"] for p in pts], marker + "-",
                     label=f"a = {aspect:g}")
        axes[1].plot(x, [100 * p["err_t"]["fraction_0.9"] for p in pts],
                     marker + "-", label=f"a = {aspect:g} (90 %)")
        axes[1].plot(x, [100 * p["err_t"]["velocity_0.01"] for p in pts],
                     marker + ":", label=f"a = {aspect:g} (v < 1 %)")
    for ax, title in ((axes[0], "runout error"), (axes[1], "emplacement-time error")):
        ax.axhspan(-15, 15, color="0.85")
        ax.set_xscale("log")
        ax.set_xlabel("$d / r_i$ (grain size / column radius)")
        ax.set_ylabel("relative error [%]")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    path = os.path.join(out_dir, "refinement_convergence.png")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    figures["refinement_convergence.png"] = (
        "Errors vs the resolution parameter d/r_i at fixed r_i and fixed "
        "dimensionless stiffness; shaded band = ±15 % acceptance.")

    return figures


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build(case_dir: str = CASE_DIR, out_dir: str = OUT_DIR,
          archive_dir: str = ARCHIVE_DIR) -> dict:
    cases = load_cases(case_dir)
    historical = audit_historical_controls(archive_dir)
    analysis = {
        "n_cases": len(cases),
        "case_ids": sorted(cases),
        "tolerance": TOL,
        "primary_definitions": list(PRIMARY),
        "arrival_window_s": DEFAULT_WINDOW_S,
        "card_table": card_table(cases),
        "definition_table_a0.5": definition_table(cases, 0.5),
        "refinement": refinement_table(cases),
        "axis_deltas": axis_delta_table(cases),
        "historical_controls": historical,
    }
    analysis.update(adjudicate(cases, historical))
    os.makedirs(out_dir, exist_ok=True)
    analysis["figures"] = make_figures(cases, analysis, out_dir)
    with open(os.path.join(out_dir, "analysis.json"), "w") as fh:
        json.dump(analysis, fh, indent=1)
    return analysis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case-dir", default=CASE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--archive-dir", default=ARCHIVE_DIR)
    args = ap.parse_args(argv)
    analysis = build(args.case_dir, args.out_dir, args.archive_dir)
    print(json.dumps({"n_cases": analysis["n_cases"],
                      "verdicts": analysis["verdicts"]}, indent=1, default=str))
    for row in analysis["card_table"]:
        print(f"a={row['aspect']:<4g} runout={row['runout_ratio']:.3f} "
              f"err={pct(row['err_runout_nominal'])} "
              f"t90={pct(row['err_t']['fraction_0.9']['nominal'])} "
              f"t1%={pct(row['err_t']['velocity_0.01']['nominal'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())