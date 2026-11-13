#!/usr/bin/env python3
"""CASE-COLUMN-COLLAPSE — deposit / archive audit of the round-5 corpus (m18385).

Anchor: ``CASE-COLUMN-COLLAPSE`` (算例池·颗粒柱坍塌; ref Lube et al. 2004 JFM 508,
tolerance < 15 %).  This node is a **verification-debt** node: it does not run a
new DEM case, it re-checks the *archived* runs from their raw grain coordinates.

Pre-registered hypotheses (written before the script was first executed)
-----------------------------------------------------------------------
Let ``n_archived`` be ``len(positions)`` and ``n_particles`` the recorded grain
count, ``R_out = r_p995 + radius`` the deposit outer radius and
``A_m = |sum_i r_i exp(i m theta_i)| / sum_i r_i`` the mass-weighted azimuthal
moment of the footprint.

* ``H_INT``  report <-> raw-data integrity: for every case with
  ``n_archived == n_particles`` the stored ``runout_radius_p995`` /
  ``runout_radius_max`` are reproduced from the archived coordinates to
  ``<= 1e-9`` relative.  Falsified => reported numbers do not trace to the
  archived data.
* ``H_ARCH`` archival completeness: no case archives fewer grains than it
  reports.  Falsified => list the offenders and quantify what a regular
  subsample does to the metric (``H_SUB``).
* ``H_SYM``  solver symmetry: horizontal COM drift ``<= 5e-3 R_out`` in every
  case (the initial HCP cylinder is coaxial, so an axisymmetric collapse cannot
  translate its COM).
* ``H_ANISO`` statistical axisymmetry of the footprint: primary criterion
  ``median_over_cases(A_1 / floor_q95(A_1)) <= 1`` (the azimuth-shuffled null
  floor of the same radial distribution); secondary criterion
  ``max_over_cases(A_1 / floor_q95(A_1)) <= 5``.  Same two tiers for ``A_2``.
* ``H_FLOOR`` floor contact sanity: ``min_i(z_i) - radius >= -0.20 radius``
  (contact overlap bounded) and no grain below ``-0.20 radius``.
* ``H_SUB``  a regular 4x subsample changes ``r_p995`` by ``<= 1 % of R_out``
  in every fully archived case — the condition under which the subsampled
  archives (``H_ARCH`` offenders) may still be quoted to that precision.
* ``H_METRIC`` (declared *after* a single exploratory case, before the
  corpus-wide evaluation — ordering stated for honesty): the runout metric
  ``r_p995 + radius`` is an extreme order statistic whose 1-sigma sampling
  noise at a 25 % random subsample is ``>= 3 % of R_out`` in the median case,
  i.e. a disagreement below 3 % cannot be certified with this metric at all.

Usage
-----
    PYTHONPATH=src python3 scripts/analyze_column_collapse_deposit_audit_m18385.py \
        [--cases-dir DIR ...] [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.granular_column.axisym import (  # noqa: E402
    lube_2004_emplacement_time, lube_2004_runout_ratio, radial_extent)
from tensordem.granular_column.deposit_audit import (  # noqa: E402
    azimuthal_moments, azimuthal_shuffle_floor, azimuthal_uniform_floor, decimation_bias,
    edge_effective_count, floor_clearance, horizontal_com_drift, shape_factor,
    subsample_noise)

DEFAULT_CASES_DIRS = ["/root/tp-agent/work/m17111/runs/case_column_collapse_lube2004/cases"]
TOL_INTERNAL = 1e-9
TOL_SYM = 5e-3
TOL_ANISO_PRIMARY = 1.0
TOL_ANISO_SECONDARY = 5.0
TOL_FLOOR = -0.20
TOL_SUB = 1e-2
TOL_METRIC_NOISE = 0.03
ORDERS = (1, 2, 6)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(dirs):
    files = []
    for d in dirs:
        if not os.path.isdir(d):
            raise SystemExit(f"cases dir not found: {d}")
        for name in sorted(os.listdir(d)):
            if name.endswith(".json") and name.startswith("w5"):
                files.append(os.path.join(d, name))
    if not files:
        raise SystemExit(f"no case JSON in {dirs}")
    return files


def analyse_case(path: str) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    case = d["case"]
    radius = float(case["radius"])
    pos = np.asarray(d.get("positions") or [], dtype=np.float64)
    n_particles = int(d.get("n_particles", 0))
    row = {
        "case": case["label"],
        "file": os.path.basename(path),
        "sha256": sha256(path),
        "a_nominal": float(case["aspect_ratio"]),
        "a_measured": float(d["aspect_measured"]),
        "grains_per_radius": float(case["grains_per_radius"]),
        "radius": radius,
        "mu": float(case["mu"]),
        "mu_rolling": float(case["mu_rolling"]),
        "mu_wall": ("" if case.get("mu_wall") is None else float(case["mu_wall"])),
        "mu_law": case.get("mu_law", "constant"),
        "dt_scale": float(case.get("dt_scale", 1.0)),
        "n_particles": n_particles,
        "n_archived": int(pos.shape[0]),
        "stop_reason": d.get("stop_reason", ""),
        "stored_runout_radius_p995": float(d["runout_radius_p995"]),
        "stored_runout_radius_max": float(d["runout_radius_max"]),
        "stored_runout_ratio": float(d["runout_ratio_p995"]),
        "reference_runout_ratio": float(d["reference_runout_ratio"]),
        "reference_emplacement_time": float(d["reference_emplacement_time"]),
        "t_reach": float(d.get("t_reach", float("nan"))),
        "t90": float(d.get("t_reach_by_fraction", {}).get("0.9", float("nan"))),
        "t95": float(d.get("t_reach_by_fraction", {}).get("0.95", float("nan"))),
        "energy_residual_fraction": float(
            d.get("energy", {}).get("residual_fraction_of_potential", float("nan"))),
    }
    row["archived_stride"] = (n_particles / pos.shape[0]) if pos.shape[0] else float("nan")
    row["reference_runout_recomputed"] = lube_2004_runout_ratio(row["a_measured"])
    row["reference_time_recomputed"] = lube_2004_emplacement_time(
        row["a_measured"], float(d["column_radius_measured"]))
    row["runout_error_pct"] = 100.0 * (
        row["stored_runout_ratio"] / row["reference_runout_recomputed"] - 1.0)
    row["time_error_pct"] = 100.0 * (row["t95"] / row["reference_time_recomputed"] - 1.0)

    if pos.shape[0] == 0:
        row["has_positions"] = False
        return row
    row["has_positions"] = True

    ext = radial_extent(pos, radius)
    row["recomputed_runout_radius_p995"] = ext["p995"]
    row["recomputed_runout_radius_max"] = ext["max"]
    row["p995_rel_diff_stored"] = abs(ext["p995"] - row["stored_runout_radius_p995"]) / \
        row["stored_runout_radius_p995"]
    row["max_rel_diff_stored"] = abs(ext["max"] - row["stored_runout_radius_max"]) / \
        row["stored_runout_radius_max"]
    row["r_outer"] = ext["p995"]

    com = horizontal_com_drift(pos, radius)
    mom = azimuthal_moments(pos, orders=ORDERS)
    floor = azimuthal_uniform_floor(pos, orders=ORDERS, n_trials=64)
    perm = azimuthal_shuffle_floor(pos, orders=ORDERS, n_trials=64)
    fl = floor_clearance(pos, radius)
    shp = shape_factor(pos, radius, n_total=n_particles)
    dec = decimation_bias(pos, radius, strides=(2, 4), orders=ORDERS)
    eff = edge_effective_count(pos, radius)
    noise = subsample_noise(pos, radius, fractions=(0.5, 0.25, 0.1), n_trials=200)

    row["com_drift"] = com["drift"]
    row["com_drift_over_outer"] = com["drift_over_outer"]
    for m in ORDERS:
        row[f"a{m}"] = mom[f"a{m}"]
        row[f"floor_a{m}_q95"] = floor[f"a{m}"]["q95"]
        row[f"floor_a{m}_max"] = floor[f"a{m}"]["max"]
        row[f"permfloor_a{m}_mean"] = perm[f"a{m}"]["mean"]
        row[f"a{m}_over_floor"] = mom[f"a{m}"] / floor[f"a{m}"]["q95"]
    row["n_eff"] = eff["n_eff"]
    row["n_eff_fraction"] = eff["n_eff_fraction"]
    for frac in ("0.5", "0.25", "0.1"):
        row[f"noise{frac}_std_over_outer"] = noise[frac]["std_over_outer"]
    row["min_clearance_over_radius"] = fl["min_clearance_over_radius"]
    row["frac_below_minus_20pct"] = fl["frac_below_minus_20pct"]
    row["h_max"] = shp["h_max"]
    row["cone_factor"] = shp["cone_factor"]
    row["h_max_over_r_outer"] = shp["h_max"] / shp["r_outer"]
    row["n_used_for_volume"] = shp["n_used_for_volume"]
    for stride, rec in dec.items():
        row[f"dec{stride}_delta_p995_over_outer"] = rec["delta_p995_over_outer"]
        row[f"dec{stride}_max_delta_p995_over_outer"] = rec["max_delta_p995_over_outer"]
        row[f"dec{stride}_moment_delta_a1"] = rec["moment_delta"]["a1"]
    return row


def _median(vals):
    vals = [v for v in vals if v == v]
    return float(np.median(vals)) if vals else float("nan")


def _max(vals):
    vals = [v for v in vals if v == v]
    return float(max(vals)) if vals else float("nan")


def adjudicate(rows) -> dict:
    full = [r for r in rows if r["has_positions"] and r["n_archived"] == r["n_particles"]]
    short = [r for r in rows if r["has_positions"] and r["n_archived"] < r["n_particles"]]
    withpos = [r for r in rows if r["has_positions"]]

    h_int = {
        "criterion": "max rel. diff of stored vs recomputed r_p995 / r_max <= 1e-9 (full archives)",
        "n_full_archives": len(full),
        "max_p995_rel_diff": _max([r["p995_rel_diff_stored"] for r in full]),
        "max_max_rel_diff": _max([r["max_rel_diff_stored"] for r in full]),
        "max_p995_rel_diff_short": _max([r["p995_rel_diff_stored"] for r in short]),
        "short_cases": [r["case"] for r in short],
    }
    h_int["verdict"] = ("supported" if h_int["max_p995_rel_diff"] <= TOL_INTERNAL
                        and h_int["max_max_rel_diff"] <= TOL_INTERNAL else "falsified")

    h_arch = {
        "criterion": "n_archived == n_particles for every case",
        "n_cases": len(withpos),
        "offenders": [{"case": r["case"], "n_particles": r["n_particles"],
                       "n_archived": r["n_archived"],
                       "archived_stride": r["archived_stride"]} for r in short],
    }
    h_arch["verdict"] = "supported" if not short else "falsified"

    h_sym = {
        "criterion": f"max horizontal COM drift / R_out <= {TOL_SYM}",
        "max_drift_over_outer": _max([r["com_drift_over_outer"] for r in withpos]),
        "case_of_max": max(withpos, key=lambda r: r["com_drift_over_outer"])["case"],
        "median_drift_over_outer": _median([r["com_drift_over_outer"] for r in withpos]),
        "max_drift_full_archives": _max([r["com_drift_over_outer"] for r in full]),
        "case_of_max_full_archives": max(
            full, key=lambda r: r["com_drift_over_outer"])["case"],
        "verdict_full_archives": ("supported"
                                  if _max([r["com_drift_over_outer"] for r in full]) <= TOL_SYM
                                  else "falsified"),
        "max_drift_subsampled_archives": _max([r["com_drift_over_outer"] for r in short]),
    }
    h_sym["verdict"] = "supported" if h_sym["max_drift_over_outer"] <= TOL_SYM else "falsified"

    h_aniso = {"criterion": (f"primary median(A_m/floor_q95) <= {TOL_ANISO_PRIMARY}; "
                             f"secondary max(A_m/floor_q95) <= {TOL_ANISO_SECONDARY}")}
    ok = True
    for m in (1, 2):
        med = _median([r[f"a{m}_over_floor"] for r in withpos])
        mx = _max([r[f"a{m}_over_floor"] for r in withpos])
        h_aniso[f"a{m}_median_ratio"] = med
        h_aniso[f"a{m}_max_ratio"] = mx
        h_aniso[f"a{m}_case_of_max"] = max(withpos, key=lambda r: r[f"a{m}_over_floor"])["case"]
        ok = ok and med <= TOL_ANISO_PRIMARY and mx <= TOL_ANISO_SECONDARY
    h_aniso["criterion_met"] = bool(ok)
    h_aniso["verdict"] = "supported" if ok else "falsified"

    h_floor = {
        "criterion": f"min clearance / radius >= {TOL_FLOOR}; no grain below {TOL_FLOOR} r",
        "min_clearance_over_radius": _min([r["min_clearance_over_radius"] for r in withpos]),
        "case_of_min": min(withpos, key=lambda r: r["min_clearance_over_radius"])["case"],
        "max_frac_below_minus_20pct": _max([r["frac_below_minus_20pct"] for r in withpos]),
    }
    h_floor["verdict"] = ("supported" if h_floor["min_clearance_over_radius"] >= TOL_FLOOR
                          and h_floor["max_frac_below_minus_20pct"] == 0.0 else "falsified")

    h_sub = {
        "criterion": f"stride-4 p995 spread <= {TOL_SUB} of R_out, every full archive",
        "n_full_archives": len(full),
        "max_dec4_delta": _max([r["dec4_delta_p995_over_outer"] for r in full]),
        "max_dec2_delta": _max([r["dec2_delta_p995_over_outer"] for r in full]),
        "case_of_max_dec4": max(full, key=lambda r: r["dec4_delta_p995_over_outer"])["case"],
    }
    h_sub["verdict"] = "supported" if h_sub["max_dec4_delta"] <= TOL_SUB else "falsified"

    h_metric = {
        "criterion": (f"median over full archives of 1-sigma(r_p995 | 25% random "
                      f"subsample) / R_out >= {TOL_METRIC_NOISE}"),
        "median_noise25_over_outer": _median(
            [r["noise0.25_std_over_outer"] for r in full]),
        "max_noise25_over_outer": _max([r["noise0.25_std_over_outer"] for r in full]),
        "case_of_max_noise25": max(full, key=lambda r: r["noise0.25_std_over_outer"])["case"],
        "median_noise50_over_outer": _median(
            [r["noise0.5_std_over_outer"] for r in full]),
        "median_noise10_over_outer": _median(
            [r["noise0.1_std_over_outer"] for r in full]),
        "median_n_eff": _median([float(r["n_eff"]) for r in full]),
        "min_n_eff": _min([float(r["n_eff"]) for r in full]),
    }
    h_metric["verdict"] = ("supported"
                           if h_metric["median_noise25_over_outer"] >= TOL_METRIC_NOISE
                           else "falsified")

    return {"H_INT": h_int, "H_ARCH": h_arch, "H_SYM": h_sym,
            "H_ANISO": h_aniso, "H_FLOOR": h_floor, "H_SUB": h_sub,
            "H_METRIC": h_metric,
            "n_cases": len(rows), "n_with_positions": len(withpos)}


def _min(vals):
    vals = [v for v in vals if v == v]
    return float(min(vals)) if vals else float("nan")


def write_csv(rows, path):
    keys = [k for k in rows[0].keys()]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if isinstance(r.get(k), float) and r[k] != r[k] else r.get(k, ""))
                        for k in keys})


def make_figures(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    withpos = [r for r in rows if r["has_positions"]]
    figpaths = []

    # ---- fig 1: azimuthal anisotropy vs measured aspect ratio, with null floor
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, m in zip(axes, (1, 2)):
        a = [r["a_measured"] for r in withpos]
        amp = [r[f"a{m}"] for r in withpos]
        flr = [r[f"floor_a{m}_q95"] for r in withpos]
        ax.scatter(a, amp, s=42, label=f"measured $A_{m}$", color="#1f4e79")
        ax.scatter(a, flr, s=30, marker="_", color="#c00000",
                   label="uniform-angle null floor (q95)")
        for r in withpos:
            if r["n_archived"] < r["n_particles"]:
                ax.annotate(r["case"], (r["a_measured"], r[f"a{m}"]),
                            textcoords="offset points", xytext=(4, 6), fontsize=7)
        ax.set_xlabel("measured aspect ratio $a=h_i/r_i$ (-)")
        ax.set_ylabel(f"$A_{m}$ (-)")
        ax.set_title(f"azimuthal moment m={m}: deposit footprint vs null floor")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_azimuthal_anisotropy.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figpaths.append(p)

    # ---- fig 2: decimation bias vs archived grain count
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for stride, color in ((2, "#2e75b6"), (4, "#c00000")):
        n = [r["n_archived"] for r in withpos if f"dec{stride}_delta_p995_over_outer" in r]
        v = [100.0 * r[f"dec{stride}_delta_p995_over_outer"]
             for r in withpos if f"dec{stride}_delta_p995_over_outer" in r]
        ax.scatter(n, v, s=40, color=color, label=f"stride {stride} subsample")
    ax.axhline(100 * TOL_SUB, ls="--", color="k", lw=1,
               label=f"H_SUB limit = {100*TOL_SUB:.0f} % of $R_{{out}}$")
    ax.set_xscale("log")
    ax.set_xlabel("archived grain count $n$ (-)")
    ax.set_ylabel(r"$r_{p995}$ spread induced by subsampling [% of $R_{out}$]")
    ax.set_title("Metric robustness to archive subsampling")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_decimation_bias.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figpaths.append(p)

    # ---- fig 3: horizontal COM drift
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    a = [r["a_measured"] for r in withpos]
    v = [1e3 * r["com_drift_over_outer"] for r in withpos]
    ax.scatter(a, v, s=42, color="#375623")
    ax.axhline(1e3 * TOL_SYM, ls="--", color="k", lw=1,
               label=f"H_SYM limit = {1e3*TOL_SYM:.0f}e-3 $R_{{out}}$")
    ax.set_xlabel("measured aspect ratio $a$ (-)")
    ax.set_ylabel(r"horizontal COM drift [$10^{-3}R_{out}$]")
    ax.set_title("Symmetry preservation of the axisymmetric collapse")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_com_drift.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figpaths.append(p)

    # ---- fig 4: dimensionless footprint profiles (descriptive)
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for r in withpos:
        if r["mu_wall"] != 0.8 or r["mu_law"] != "constant":
            continue
        d = json.load(open(os.path.join(os.path.dirname(r["_path"]), r["file"])))
        pos = np.asarray(d["positions"], dtype=np.float64)
        rr = np.hypot(pos[:, 0], pos[:, 1]) / r["r_outer"]
        hist, edges = np.histogram(rr, bins=30, range=(0, 1.05))
        centres = 0.5 * (edges[1:] + edges[:-1])
        width = edges[1] - edges[0]
        ax.plot(centres, hist / (hist.sum() * width), marker="", lw=1.4,
                label=f"a={r['a_measured']:.2f}")
    ax.set_xlabel(r"$r/R_{out}$ (-)")
    ax.set_ylabel("normalised surface density (-)")
    ax.set_title("Final footprint density, wall-friction arm (descriptive)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig_footprint_profiles.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figpaths.append(p)
    # ---- fig 5: runout error vs Lube law with the metric's own noise band
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    full = [r for r in withpos if r["n_archived"] == r["n_particles"]]
    a = [r["a_measured"] for r in full]
    err = [r["runout_error_pct"] for r in full]
    band = [100.0 * r["noise0.25_std_over_outer"] for r in full]
    ax.axhspan(-15, 15, color="#d9e2f3", alpha=0.8, label="Lube 2004 acceptance +-15 %")
    ax.errorbar(a, err, yerr=band, fmt="o", ms=5, lw=1, capsize=3,
                color="#1f4e79", label="DEM vs law, err +- 1 sigma metric noise (25 % subsample)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("measured aspect ratio $a$ (-)")
    ax.set_ylabel("runout $r_\\infty/r_i$ relative error [%]")
    ax.set_title("Runout agreement and the metric's own resolution")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    p5 = os.path.join(outdir, "fig_runout_error_vs_metric_noise.png")
    fig.savefig(p5, dpi=140)
    plt.close(fig)
    figpaths.append(p5)
    return figpaths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-dir", action="append", default=None,
                    help="directory holding archived case JSONs (repeatable)")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "docs", "artifacts", "column_collapse_deposit_audit_m18385"))
    args = ap.parse_args(argv)
    dirs = args.cases_dir or DEFAULT_CASES_DIRS
    outdir = os.path.abspath(args.out)
    os.makedirs(outdir, exist_ok=True)

    files = discover(dirs)
    rows = []
    for path in files:
        row = analyse_case(path)
        row["_path"] = path
        rows.append(row)
    rows.sort(key=lambda r: (r["a_nominal"], r["case"]))

    verdicts = adjudicate(rows)
    csv_path = os.path.join(outdir, "cases_deposit_audit.csv")
    write_csv(rows, csv_path)
    manifest = {r["file"]: r["sha256"] for r in rows}
    with open(os.path.join(outdir, "manifest_sha256.json"), "w") as fh:
        json.dump({"cases_dir": dirs, "files": manifest}, fh, indent=2, sort_keys=True)

    figpaths = make_figures(rows, outdir)
    figs = {os.path.basename(p): sha256(p) for p in figpaths}

    clean = [{k: v for k, v in r.items() if k != "_path"} for r in rows]
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump({"verdicts": verdicts, "figures_sha256": figs, "cases": clean},
                  fh, indent=2, default=str, sort_keys=True)

    print(json.dumps({"n_cases": len(rows), "out": outdir,
                      "verdicts": {k: v.get("verdict") for k, v in verdicts.items()
                                   if isinstance(v, dict) and "verdict" in v}},
                     indent=2))
    for k, v in verdicts.items():
        if isinstance(v, dict) and "verdict" in v:
            detail = {kk: vv for kk, vv in v.items() if kk not in ("criterion", "verdict")}
            print(f"{k}: {v['verdict']} | {json.dumps(detail, default=str)[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())