#!/usr/bin/env python
"""CASE-DEM-BIAXIAL-FRICTION round 3: constant-volume (isochoric) analysis.

Consumes the raw stress paths written by
``scripts/m20537_probe_isochoric.py`` (``runs/m20537/iso_*.json``), locates the
critical state with :mod:`tensordem.biaxial_critical_state` using the
**logarithmic** strain measure (the only measure for which an exactly
volume-preserving cell reports ``eps_v = 0``) and the isochoric boundary
closure check, then judges the result against the references declared in
round 2 *before* any of these runs existed:

* primary   -- Rowe (1962) D = 0 limit ``phi_cv = arctan(mu)``; 28.81 deg for
  mu = 0.55; tolerance 8 % (case-pool acceptance);
* secondary -- Bolton (1986) angular quartz sand ``phi_crit = 33 deg`` (the
  shape-gap reference, reported but not the pass criterion).

Writes

    artifacts/CASE-DEM-BIAXIAL-FRICTION/run_iso.json
    artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_isochoric.png

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_isochoric.py
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensordem.biaxial_critical_state import (  # noqa: E402
    critical_state_from_stress_path,
    judge_against_reference,
    phi_deg_from_stresses,
    rowe_stress_dilatancy,
)

TOLERANCE = 0.08
MU_ANCHOR = 0.55
PHI_ROWE_DEG = math.degrees(math.atan(MU_ANCHOR))   # 28.81 deg
PHI_BOLTON_QUARTZ_DEG = 33.0
REF_SOURCES = {
    "rowe_limit": ("Rowe (1962) Proc. R. Soc. Lond. A 269:500-527, stress-dilatancy "
                   "relation at D=0: phi_cv = arctan(mu) = 28.81 deg for mu = 0.55"),
    "bolton_quartz": ("Bolton (1986) Geotechnique 36(1):65-78: angular quartz sand "
                      "critical-state angle phi_crit = 33 deg"),
}
PROBE_DIR = os.environ.get("PROBE_DIR", os.path.join(REPO, "runs", "m20537"))
OUT_DIR = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION")
# committed, downsampled copy of the probe stress paths: the analysis (and the
# provenance regression test) must run with no cluster artefacts on disk
DATA_DIR = os.path.join(REPO, "benchmarks", "case_dem_biaxial_friction",
                        "results_m22748")
BUNDLE = os.path.join(DATA_DIR, "iso_paths.json.gz")

PATH_KEYS = ("axial_strain", "axial_strain_log", "sigma_v", "sigma_h", "deviator",
             "volumetric_strain", "volumetric_strain_log")


def analyse_one(raw: dict) -> dict:
    cfg = dict(raw["config"])
    sp = {k: np.asarray(v, dtype=float) for k, v in raw["stress_path"].items()
          if k in PATH_KEYS}
    mu = float(cfg.get("mu", MU_ANCHOR))
    p0 = float(cfg["confining_stress"])
    cs = critical_state_from_stress_path(
        sp, sigma3_target=p0, strain_measure="log", boundary_control="isochoric")
    # the Rowe cross-check needs samples above its min_strain; clamp it to the
    # record length (a run that stops at eps_a = 0.45 cannot host a window
    # starting at 0.52)
    rowe_floor = min(cs.window[0], 0.85 * float(np.max(sp["axial_strain"])))
    try:
        rowe = rowe_stress_dilatancy(sp, mu=mu, min_strain=rowe_floor)
    except ValueError as exc:                     # pragma: no cover - defensive
        rowe = {"error": str(exc), "min_strain": rowe_floor}
    # stress-definition closure: wall traction vs Love-Weber virial, same window
    mask = (np.asarray(sp["axial_strain_log"]) >= cs.window[0]) & \
           (np.asarray(sp["axial_strain_log"]) <= cs.window[1])
    phi_wall = float(np.nanmean(phi_deg_from_stresses(sp["sigma_v"], sp["sigma_h"])[mask]))
    internal = {}
    if "sigma_v_internal" in raw["stress_path"]:
        sv_i = np.asarray(raw["stress_path"]["sigma_v_internal"], dtype=float)
        sh_i = np.asarray(raw["stress_path"]["sigma_h_internal"], dtype=float)
        phi_int = float(np.nanmean(phi_deg_from_stresses(sv_i, sh_i)[mask]))
        internal = {"phi_internal_deg": phi_int,
                    "phi_wall_deg": phi_wall,
                    "relative_difference": ((phi_int - phi_wall) / phi_wall
                                            if phi_wall else float("nan"))}
    mean_p = 0.5 * (sp["sigma_v"] + sp["sigma_h"])
    return {
        "tag": raw["tag"],
        "config": cfg,
        "specimen": raw.get("specimen", {}),
        "loading": raw.get("loading", {}),
        "elapsed_s": raw.get("elapsed_s"),
        "peak_phi_deg": float(raw.get("peak", {}).get(
            "friction_angle_deg", float(np.max(phi_deg_from_stresses(
                sp["sigma_v"], sp["sigma_h"]))))),
        "critical_state": cs.as_dict(),
        "rowe": rowe,
        "stress_definition_closure": internal,
        "mean_pressure_in_window_kPa": float(mean_p[mask].mean() / 1e3),
        "mean_pressure_initial_kPa": float(p0 / 1e3),
        "judge_primary_rowe": judge_against_reference(cs.phi_cv_deg, PHI_ROWE_DEG,
                                                     TOLERANCE),
        "judge_secondary_bolton": judge_against_reference(cs.phi_cv_deg,
                                                         PHI_BOLTON_QUARTZ_DEG,
                                                         TOLERANCE),
        "curves": {
            "axial_strain": sp["axial_strain"].tolist(),
            "phi_deg": phi_deg_from_stresses(sp["sigma_v"], sp["sigma_h"]).tolist(),
            "q_kPa": (sp["deviator"] / 1e3).tolist(),
            "p_kPa": (mean_p / 1e3).tolist(),
            "sigma_h_kPa": (sp["sigma_h"] / 1e3).tolist(),
            "volumetric_strain_log": sp["volumetric_strain_log"].tolist(),
        },
    }


def render(payload: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["cases"]
    anchor = next((r for r in rows if r["tag"].startswith("anchor")), rows[0])
    cur = anchor["curves"]
    eps = np.asarray(cur["axial_strain"])
    phi = np.asarray(cur["phi_deg"])
    q = np.asarray(cur["q_kPa"])
    p = np.asarray(cur["p_kPa"])
    sh = np.asarray(cur["sigma_h_kPa"])
    es = np.asarray(cur["volumetric_strain_log"])
    cs = anchor["critical_state"]

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5))

    ax = axes[0]
    ax.plot(eps, phi, color="#1f77b4", lw=0.9, label=r"$\phi$ (wall traction)")
    ax.axhline(PHI_ROWE_DEG, color="green", ls="--",
               label=f"Rowe D=0 limit arctan$\\mu$ = {PHI_ROWE_DEG:.2f} deg")
    ax.axhspan(PHI_ROWE_DEG * (1 - TOLERANCE), PHI_ROWE_DEG * (1 + TOLERANCE),
               color="green", alpha=0.12, label="8 % band")
    ax.axhline(PHI_BOLTON_QUARTZ_DEG, color="gray", ls=":",
               label=f"angular quartz sand {PHI_BOLTON_QUARTZ_DEG:.0f} deg (Bolton)")
    ax.axvline(cs["window"][0], color="orange", lw=1.0, ls="-.")
    ax.axvspan(cs["window"][0], cs["window"][1], color="orange", alpha=0.2,
               label="critical-state window")
    ax.axhline(cs["phi_cv_deg"], color="red", lw=0.8,
               label=f"measured $\\phi_{{cv}}$ = {cs['phi_cv_deg']:.2f} deg")
    ax.set_xlabel(r"axial strain $\varepsilon_a$ [-]")
    ax.set_ylabel(r"mobilised $\phi$ [deg]")
    ax.set_title(f"anchor 24x48, $\\mu$=0.55, constant volume\n"
                 f"peak {anchor['peak_phi_deg']:.1f} deg -> $\\phi_{{cv}}$ "
                 f"{cs['phi_cv_deg']:.1f} deg")
    ax.set_ylim(0, 55)
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eps, q, color="#d62728", lw=0.9, label="$q = \\sigma_1-\\sigma_3$ [kPa]")
    ax.plot(eps, p, color="#1f77b4", lw=0.9, label="$p = (\\sigma_1+\\sigma_3)/2$ [kPa]")
    ax.plot(eps, sh, color="#7f7f7f", lw=0.7, ls=":",
            label="$\\sigma_h$ (measured lateral) [kPa]")
    ax.axvspan(cs["window"][0], cs["window"][1], color="orange", alpha=0.2)
    ax.set_xlabel(r"$\varepsilon_a$ [-]")
    ax.set_ylabel("stress [kPa]")
    ax2 = ax.twinx()
    ax2.plot(eps, es, color="#2ca02c", lw=0.8, label=r"$\varepsilon_v$ (log)")
    ax2.set_ylabel(r"logarithmic $\varepsilon_v$ [-]", color="#2ca02c")
    ax.set_title("constant-volume path: $q$, $p$, $\\varepsilon_v=0$ by construction\n"
                 f"$d\\varepsilon_v/d\\varepsilon_a$ = {cs['d_eps_v_d_eps_a']:+.2e} (log)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    tags = [r["tag"] for r in rows]
    vals = [r["critical_state"]["phi_cv_deg"] for r in rows]
    colors = ["#1f77b4" if r["judge_primary_rowe"]["passed"] else "#ff7f0e" for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=5.5)
    ax.axhline(PHI_ROWE_DEG, color="green", ls="--", label=f"Rowe limit {PHI_ROWE_DEG:.2f} deg")
    ax.axhspan(PHI_ROWE_DEG * (1 - TOLERANCE), PHI_ROWE_DEG * (1 + TOLERANCE),
               color="green", alpha=0.12)
    ax.axhline(PHI_BOLTON_QUARTZ_DEG, color="gray", ls=":", label="angular quartz 33 deg")
    for bar, r in zip(bars, rows):
        err = r["judge_primary_rowe"]["relative_error"] * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}\n{err:.1f}%", ha="center", fontsize=6)
    ax.set_ylabel(r"$\phi_{cv}$ [deg]")
    ax.set_title("critical-state angle per configuration\n(relative error vs Rowe limit)")
    top = max(vals + [PHI_BOLTON_QUARTZ_DEG]) * 1.35
    ax.set_ylim(0, top)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def _sig(x: float, digits: int = 7) -> float:
    """Round to ``digits`` significant figures (keeps the JSON small)."""
    if x == 0.0 or not math.isfinite(x):
        return float(x)
    return float(f"{x:.{digits}g}")


def load_runs(probe_dir: str) -> tuple[list[dict], list[str]]:
    """Load probe records from a directory of ``iso_*.json`` files.

    Falls back to the committed, downsampled bundle
    (``results_m22748/iso_paths.json.gz``) when the probe directory is empty, so
    the published numbers can be recomputed on a clean checkout.
    """
    paths = sorted(glob.glob(os.path.join(probe_dir, "iso_*.json*")))
    if not paths and os.path.exists(BUNDLE):
        paths = [BUNDLE]
    runs: list[dict] = []
    for path in paths:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fh:
            data = json.load(fh)
        runs.extend(data if isinstance(data, list) else [data])
    runs.sort(key=lambda r: r["tag"])
    return runs, [os.path.basename(p) for p in paths]


def compact_runs(runs: list[dict], out_path: str, max_samples: int = 1000) -> str:
    """Write a gzip bundle of the stress paths, decimated to ``max_samples``.

    Every recorded array is subsampled with a fixed stride (the last sample is
    always kept) and rounded to 7 significant figures.  This is what makes the
    round-3 numbers auditable from the repository without a 120 k-step raw
    artefact: the decimated path re-analyses to the published ``phi_cv`` within
    the reported uncertainty (see
    ``tests/test_biaxial_isochoric_provenance.py``).
    """
    bundle = []
    for raw in runs:
        sp = raw["stress_path"]
        n = len(np.asarray(sp["axial_strain"]))
        stride = max(1, int(math.ceil(n / float(max_samples))))
        idx = np.arange(0, n, stride)
        if idx[-1] != n - 1:
            idx = np.append(idx, n - 1)
        rec = {k: raw[k] for k in ("tag", "config", "specimen", "loading", "peak",
                                   "note", "elapsed_s") if k in raw}
        rec["stress_path"] = {k: [_sig(float(v)) for v in np.asarray(arr)[idx]]
                              for k, arr in sp.items()
                              if len(np.asarray(arr)) == n}
        rec["compaction"] = {"stride": int(stride), "n_original": int(n),
                             "n_kept": int(idx.size),
                             "note": "fixed-stride decimation of the probe record; "
                                     "the last sample is always kept"}
        bundle.append(rec)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with gzip.open(out_path, "wt") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    return out_path


def markdown_table(cases: list[dict]) -> str:
    """Sensitivity table (markdown) of the analysed configurations."""
    head = ("| run | grid | phi_peak [deg] | phi_cv [deg] | dEv/dEa | q drift | "
            "window spread [deg] | sigma_h [kPa] | p [kPa] | converged | "
            "err vs Rowe |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for r in cases:
        c = r["critical_state"]
        cfg = r["config"]
        rows.append(
            f"| `{r['tag']}` | {cfg['columns']}x{cfg['rows']} | "
            f"{r['peak_phi_deg']:.2f} | **{c['phi_cv_deg']:.2f}** | "
            f"{c['d_eps_v_d_eps_a']:+.2e} | {c['q_drift']:.3f} | "
            f"{c['window_spread_deg']:.2f} | {c['sigma_h_cv_Pa'] / 1e3:.0f} | "
            f"{r['mean_pressure_in_window_kPa']:.0f} | "
            f"{'yes' if c['converged'] else 'no'} | "
            f"{r['judge_primary_rowe']['relative_error'] * 100:.1f} % |")
    return head + "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe-dir", default=os.environ.get(
        "PROBE_DIR", os.path.join(REPO, "runs", "m20537")),
        help="directory holding the iso_*.json probe records "
             "(empty => committed bundle)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--compact", metavar="PATH",
                    help="write a downsampled gzip bundle of the consumed runs")
    ap.add_argument("--max-samples", type=int, default=1000,
                    help="samples per configuration kept by --compact")
    ap.add_argument("--markdown", action="store_true",
                    help="print the sensitivity table as markdown")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    runs, names = load_runs(args.probe_dir)
    if not runs:
        print(f"no isochoric runs under {args.probe_dir}")
        return 2
    if args.compact:
        print("compacted:", compact_runs(runs, args.compact, args.max_samples))
    cases = [analyse_one(raw) for raw in runs]
    payload = {
        "case_key": "CASE-DEM-BIAXIAL-FRICTION",
        "round": 3,
        "method": ("unbonded plane-strain biaxial, CONSTANT VOLUME (kinematic "
                   "volume-preserving lateral walls, servo released after "
                   "consolidation), frictionless walls; critical state = last 20 % of "
                   "the sheared path above eps_a = 0.20, log strain measure"),
        "references": REF_SOURCES,
        "tolerance": TOLERANCE,
        "anchor_mu": MU_ANCHOR,
        "provenance": {"probe_dir": os.path.relpath(args.probe_dir, REPO)
                       if os.path.isdir(args.probe_dir) else args.probe_dir,
                       "raw_runs": names,
                       "compaction": [r.get("compaction") for r in runs]},
        "cases": cases,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "run_iso.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=1,
                  default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    png = ""
    if not args.no_figure:
        png = render(payload, os.path.join(args.out_dir, "biaxial_isochoric.png"))

    print(f"{'tag':30s} {'peak':>6s} {'phi_cv':>7s} {'dEv/dEa':>9s} {'p[win]':>7s} "
          f"{'q_drift':>8s} {'jump':>6s} {'err(rowe)':>9s} {'conv':>5s}")
    for r in cases:
        c = r["critical_state"]
        print(f"{r['tag']:30s} {r['peak_phi_deg']:6.2f} {c['phi_cv_deg']:7.2f} "
              f"{c['d_eps_v_d_eps_a']:+9.2e} {r['mean_pressure_in_window_kPa']:7.1f} "
              f"{c['q_drift']:8.3f} {c['window_spread_deg']:6.2f} "
              f"{r['judge_primary_rowe']['relative_error'] * 100:8.1f}% "
              f"{str(c['converged']):>5s} flags={c['flags']}")
    if args.markdown:
        print()
        print(markdown_table(cases))
    print("artefacts:", json_path, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
