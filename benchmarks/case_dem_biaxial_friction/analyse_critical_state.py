#!/usr/bin/env python
"""CASE-DEM-BIAXIAL-FRICTION round 2: critical-state analysis + artefacts.

Consumes the raw stress paths recorded by ``scripts/m20304_probe_cs2.py``
(one JSON per configuration, produced on the compute cluster), locates the
critical state with :mod:`tensordem.biaxial_critical_state`, judges it against
the two declared references and writes

    artifacts/CASE-DEM-BIAXIAL-FRICTION/run.json
    artifacts/CASE-DEM-BIAXIAL-FRICTION/biaxial_critical_state.png

Declared references (fixed before the analysis, see
``docs/RESEARCH_CASE-DEM-BIAXIAL-FRICTION_R2.md`` §3):

* primary   -- Rowe (1962) D = 0 limit: ``phi_cv = arctan(mu)`` (28.81 deg for
  mu = 0.55), tolerance 8 %;
* secondary -- Bolton (1986) angular quartz sand ``phi_crit = 33 deg``,
  reported for the shape-gap discussion, not used as the pass criterion.

Usage:
    PYTHONPATH=src python benchmarks/case_dem_biaxial_friction/analyse_critical_state.py
"""
from __future__ import annotations

import glob
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
    "rowe_limit": ("Rowe (1962) Proc. R. Soc. Lond. A 269:500-527, stress-"
                   "dilatancy relation at D=0: phi_cv = arctan(mu) = 28.81 deg for mu=0.55"),
    "bolton_quartz": ("Bolton (1986) Geotechnique 36(1):65-78: angular quartz "
                      "sand critical-state angle phi_crit = 33 deg"),
}
PROBE_DIR = os.environ.get("PROBE_DIR", os.path.join(REPO, "runs", "m20304"))
OUT_DIR = os.path.join(REPO, "artifacts", "CASE-DEM-BIAXIAL-FRICTION")


def analyse_one(path: str, raw: dict | None = None, tag_prefix: str = "") -> dict:
    if raw is None:
        raw = json.load(open(path))
    cfg = dict(raw["config"])
    if "stress_path" in raw:
        sp = raw["stress_path"]
    else:                       # probe-format record: history in kPa
        h = raw["history"]
        sp = {"axial_strain": h["eps"],
              "sigma_v": [v * 1e3 for v in h["sv_kPa"]],
              "sigma_h": [v * 1e3 for v in h["sh_kPa"]],
              "deviator": [v * 1e3 for v in h["q_kPa"]],
              "volumetric_strain": h["ev"]}
        cfg.setdefault("mu", cfg.get("mu", 0.55))
        cfg.setdefault("confining_stress", raw.get("config", {}).get("confining_stress", 100e3))
    cfg.setdefault("mu", 0.55)
    target = float(cfg["confining_stress"])
    sp = {k: (np.asarray(v, dtype=float) if k in
              ("axial_strain", "sigma_v", "sigma_h", "deviator", "volumetric_strain")
              else v) for k, v in sp.items()}
    cs = critical_state_from_stress_path(sp, sigma3_target=target)
    ev = raw.get("elapsed_s")
    rowe = rowe_stress_dilatancy(sp, mu=float(cfg["mu"]),
                                 min_strain=cs.window[0])
    judge_primary = judge_against_reference(cs.phi_cv_deg, PHI_ROWE_DEG, TOLERANCE)
    judge_secondary = judge_against_reference(cs.phi_cv_deg, PHI_BOLTON_QUARTZ_DEG,
                                              TOLERANCE)
    return {
        "tag": tag_prefix + raw.get("tag", ""), "config": cfg, "elapsed_s": ev,
        "specimen": raw.get("specimen", {}),
        "loading": raw.get("loading", {"steps": len(sp["axial_strain"]),
                                       "final_axial_strain": float(sp["axial_strain"][-1])}),
        "peak": raw.get("peak", {"friction_angle_deg": float(np.max(
            phi_deg_from_stresses(sp["sigma_v"], sp["sigma_h"])))}),
        "critical_state": cs.as_dict(),
        "rowe": rowe,
        "judge_primary_rowe": judge_primary,
        "judge_secondary_bolton": judge_secondary,
        "stress_path_png_data": {
            "axial_strain": sp["axial_strain"],
            "phi_deg": phi_deg_from_stresses(sp["sigma_v"], sp["sigma_h"]).tolist(),
            "q_kPa": (np.asarray(sp["deviator"]) / 1e3).tolist(),
            "sigma_h_kPa": (np.asarray(sp["sigma_h"]) / 1e3).tolist(),
            "volumetric_strain": sp["volumetric_strain"],
        },
    }


def render(payload: dict, path: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = payload["cases"]
    anchor = next((r for r in rows if r["tag"].startswith("anchor")), rows[0])
    sp = anchor["stress_path_png_data"]
    eps = np.asarray(sp["axial_strain"])
    phi = np.asarray(sp["phi_deg"])
    q = np.asarray(sp["q_kPa"])
    ev = np.asarray(sp["volumetric_strain"])
    cs = anchor["critical_state"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    ax = axes[0]
    ax.plot(eps, phi, color="#1f77b4", lw=0.9, label=r"$\phi$ (wall traction)")
    ax.axhline(PHI_ROWE_DEG, color="green", ls="--",
               label=f"Rowe limit arctan$\\mu$ = {PHI_ROWE_DEG:.2f} deg")
    ax.axhspan(PHI_ROWE_DEG * (1 - TOLERANCE), PHI_ROWE_DEG * (1 + TOLERANCE),
               color="green", alpha=0.12, label="8 % band")
    ax.axhline(PHI_BOLTON_QUARTZ_DEG, color="gray", ls=":",
               label=f"angular quartz sand {PHI_BOLTON_QUARTZ_DEG:.0f} deg (Bolton)")
    ax.axvspan(cs["window"][0], cs["window"][1], color="orange", alpha=0.2,
               label="critical-state window")
    ax.set_xlabel(r"axial strain $\varepsilon_a$ [-]")
    ax.set_ylabel(r"mobilised $\phi$ [deg]")
    ax.set_title(f"anchor: $\\phi_{{cv}}$ = {cs['phi_cv_deg']:.2f} deg "
                 f"(peak {anchor['peak']['friction_angle_deg']:.2f})")
    ax.set_ylim(0, 55)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(eps, q, color="#d62728", lw=0.9, label="deviator $q$ [kPa]")
    ax.set_xlabel(r"$\varepsilon_a$ [-]")
    ax.set_ylabel("$q$ [kPa]", color="#d62728")
    ax2 = ax.twinx()
    ax2.plot(eps, ev, color="#2ca02c", lw=0.9, label=r"$\varepsilon_v$")
    ax2.set_ylabel(r"volumetric strain $\varepsilon_v$ [-]", color="#2ca02c")
    ax.axvspan(cs["window"][0], cs["window"][1], color="orange", alpha=0.2)
    ax.set_title(f"stress/volume path: d$\\varepsilon_v$/d$\\varepsilon_a$ = "
                 f"{cs['d_eps_v_d_eps_a']:+.4f} in window")
    ax.grid(alpha=0.3)

    ax = axes[2]
    tags = [r["tag"] for r in rows]
    vals = [r["critical_state"]["phi_cv_deg"] for r in rows]
    colors = ["#1f77b4" if r.get("judge_primary_rowe", {}).get("passed") else "#ff7f0e"
              for r in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, edgecolor="white")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([t.replace("_", "\n") for t in tags], fontsize=7)
    ax.axhline(PHI_ROWE_DEG, color="green", ls="--",
               label=f"Rowe limit {PHI_ROWE_DEG:.2f} deg")
    ax.axhspan(PHI_ROWE_DEG * (1 - TOLERANCE), PHI_ROWE_DEG * (1 + TOLERANCE),
               color="green", alpha=0.12)
    ax.axhline(PHI_BOLTON_QUARTZ_DEG, color="gray", ls=":",
               label="angular quartz 33 deg")
    for bar, r in zip(bars, rows):
        err = r["judge_primary_rowe"]["relative_error"] * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{bar.get_height():.1f}\n{err:.1f}%", ha="center", fontsize=7)
    ax.set_ylabel(r"$\phi_{cv}$ [deg]")
    ax.set_title("critical-state angle per configuration")
    ax.set_ylim(0, 42)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main() -> int:
    paths = sorted(glob.glob(os.path.join(PROBE_DIR, "*.json")))
    if not paths:
        print(f"no raw runs under {PROBE_DIR}")
        return 2
    cases = []
    for path in paths:
        raw = json.load(open(path))
        if isinstance(raw, dict) and "config" in raw and "history" in raw or (
                isinstance(raw, dict) and "stress_path" in raw):
            cases.append(analyse_one(path, raw))
        else:                    # collection file: {tag: record}
            for tag, rec in raw.items():
                if isinstance(rec, dict) and "error" not in rec:
                    cases.append(analyse_one(path, rec, tag_prefix=tag + ":"))
    payload = {
        "case_key": "CASE-DEM-BIAXIAL-FRICTION",
        "round": 2,
        "method": ("unbonded plane-strain biaxial, frictionless lateral servo walls, "
                   "critical state = last 20 % of the sheared path above eps_a = 0.20"),
        "references": REF_SOURCES,
        "tolerance": TOLERANCE,
        "anchor_mu": MU_ANCHOR,
        "cases": cases,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    json_path = os.path.join(OUT_DIR, "run.json")
    with open(json_path, "w") as fh:
        json.dump(payload, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    png = render(payload, os.path.join(OUT_DIR, "biaxial_critical_state.png"))

    print(f"{'tag':32s} {'phi_peak':>8s} {'phi_cv':>7s} {'dEv/dEa':>8s} "
          f"{'sh':>7s} {'err(rowe)':>9s} {'err(33)':>8s} {'conv':>5s}")
    for r in cases:
        c = r["critical_state"]
        print(f"{r['tag']:32s} {r['peak']['friction_angle_deg']:8.2f} "
              f"{c['phi_cv_deg']:7.2f} {c['d_eps_v_d_eps_a']:+8.4f} "
              f"{c['sigma_h_cv_Pa'] / 1e3:7.1f} "
              f"{r['judge_primary_rowe']['relative_error'] * 100:8.1f}% "
              f"{r['judge_secondary_bolton']['relative_error'] * 100:7.1f}% "
              f"{str(c['converged']):>5s}  flags={c['flags']}")
    print("artefacts:", json_path, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())