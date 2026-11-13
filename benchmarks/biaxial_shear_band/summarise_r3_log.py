"""Build the R3 ensemble artefact from the delivered benchmark log (no hand typing).

The R3 ensemble run was launched as a background job and reaped by the harness
shell timeout before it could write its JSON; every completed specimen had
already printed its provenance line to ``logs/r3_ensemble.log``.  This script
parses those raw log lines (the primary artefact) into
``artifacts/biaxial_shear_band/r3_ensemble.json`` plus the ensemble figure, so
the numbers in the report are machine-transcribed, never retyped.

Parsed line format (written by run_r3_ensemble.py):
``[done] <name> radon=<deg> tensor=<deg> resolved=<bool> (<wall>s)``
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

THETA_REF = 60.0
TOL = 0.15
LINE = re.compile(r"^\[done\] (\S+) radon=([-\d.]+) tensor=([-\d.]+) "
                  r"resolved=(True|False) \((\d+)s\)")
#: read from the R3 design in run_r3_ensemble.py (same seeds/specs)
SPEC = {
    "anchor_s20261031": ("anchor", True, 100.0e3, 24, 48, 20261031),
    "anchor_s11": ("anchor", True, 100.0e3, 24, 48, 11),
    "anchor_s23": ("anchor", True, 100.0e3, 24, 48, 23),
    "anchor_s37": ("anchor", True, 100.0e3, 24, 48, 37),
    "anchor_s41": ("anchor", True, 100.0e3, 24, 48, 41),
    "sens_p200_s11": ("confining_pressure", False, 200.0e3, 24, 48, 11),
    "sens_p200_s23": ("confining_pressure", False, 200.0e3, 24, 48, 23),
    "sens_r32x64_s31": ("resolution", False, 100.0e3, 32, 64, 20261031),
}


def stats(values):
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if arr.size == 0:
        return {"n": 0}
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    if arr.size > 1:
        from scipy import stats as st
        half = float(st.t.ppf(0.975, arr.size - 1)) * std / math.sqrt(arr.size)
    else:
        half = math.nan
    return {"n": int(arr.size), "mean_deg": mean, "std_deg": std,
            "ci95_half_width_deg": half,
            "ci95_deg": [mean - half, mean + half] if np.isfinite(half) else None,
            "relative_error": abs(mean - THETA_REF) / THETA_REF,
            "within_tolerance": bool(abs(mean - THETA_REF) / THETA_REF <= TOL)}


def main() -> int:
    log_path = "logs/r3_ensemble.log"
    rows = []
    with open(log_path) as handle:
        for line in handle:
            match = LINE.match(line.strip())
            if not match:
                continue
            name, radon, tensor, resolved, wall = match.groups()
            axis, member, p, cols, rws, seed = SPEC.get(
                name, ("unknown", False, math.nan, 0, 0, 0))
            rows.append({"name": name, "axis": axis, "ensemble_member": member,
                         "confining_stress": p, "columns": cols, "rows": rws,
                         "seed": seed,
                         "theta_radon_mean_deg": float(radon),
                         "theta_tensor_mean_deg": float(tensor),
                         "resolved_production": resolved == "True",
                         "wall_time_s": float(wall)})
    anchor = [r for r in rows if r["ensemble_member"]]
    sens = [r for r in rows if not r["ensemble_member"]]
    radon = [r["theta_radon_mean_deg"] for r in anchor]
    tensor = [r["theta_tensor_mean_deg"] for r in anchor]
    resolved = sum(1 for r in anchor if r["resolved_production"])
    st_radon, st_tensor = stats(radon), stats(tensor)
    verdict = {
        "theta_ref_deg": THETA_REF, "tolerance": TOL,
        "acceptance_band_deg": [THETA_REF * (1 - TOL), THETA_REF * (1 + TOL)],
        "ensemble_radon": st_radon, "ensemble_tensor": st_tensor,
        "specimens_resolved": resolved, "specimens_total": len(anchor),
        "operator_mean_abs_difference_deg": float(np.mean(
            [abs(a - b) for a, b in zip(radon, tensor)])) if radon else math.nan,
        "operator_max_abs_difference_deg": float(np.max(
            [abs(a - b) for a, b in zip(radon, tensor)])) if radon else math.nan,
        "pass_rule": "|mean-60|/60 <= 0.15 and >= 4/5 specimens resolved",
        "passed_radon": bool(st_radon.get("within_tolerance") and resolved >= 4),
        "passed_tensor": bool(st_tensor.get("within_tolerance") and resolved >= 4),
        "provenance": {
            "source_log": log_path,
            "note": ("ensemble run reaped by the harness 120 s shell cap before "
                     "writing its JSON; the log lines are the primary artefact and "
                     "are parsed here. Cumulative-window measurement and the 32x64 "
                     "resolution point are WITHHELD (never completed)."),
            "completed_specimens": len(rows), "planned_specimens": len(SPEC),
        },
        "sensitivity": [{"name": r["name"], "axis": r["axis"],
                         "confining_stress": r["confining_stress"],
                         "columns": r["columns"], "rows": r["rows"],
                         "theta_radon_deg": r["theta_radon_mean_deg"],
                         "theta_tensor_deg": r["theta_tensor_mean_deg"],
                         "resolved": r["resolved_production"],
                         "relative_error_radon": abs(r["theta_radon_mean_deg"] - THETA_REF) / THETA_REF}
                        for r in sens],
    }
    payload = {"case": "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE", "round": "R3",
               "specification": {"cell": "plane-strain biaxial 24 x 48 discs",
                                 "confining_stress_Pa": 100.0e3,
                                 "wall_friction": 0.55,
                                 "consolidation_steps": 4000,
                                 "freeze_bond_fraction": 0.35,
                                 "strain_rate_per_s": 0.05, "strain_target": 0.12,
                                 "servo_gain": 3.0e-4,
                                 "anchor": "Desrues & Viggiani 2004 IJNAMG 28:279-321, "
                                           "dense Hostun sand 55-60 deg (midpoint 60)"},
               "verdict": verdict, "specimens": rows}
    out = "artifacts/biaxial_shear_band/r3_ensemble.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=2, default=float)
    print(json.dumps(verdict, indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    x = np.arange(len(anchor))
    labels = [r["name"].replace("anchor_", "") for r in anchor]
    axes[0].bar(x - 0.2, radon, 0.4, label="Radon focus (production)")
    axes[0].bar(x + 0.2, tensor, 0.4, label="structure tensor (cross-check)")
    axes[0].axhline(THETA_REF, color="k", lw=1.2, label="anchor 60°")
    axes[0].axhspan(THETA_REF * (1 - TOL), THETA_REF * (1 + TOL), color="k",
                    alpha=0.1, label="±15 %")
    axes[0].axhline(st_radon["mean_deg"], color="C0", ls="--", lw=1.2,
                    label=f"ensemble mean {st_radon['mean_deg']:.2f}°")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=25)
    axes[0].set_ylabel("band inclination from σ₃ (deg)")
    axes[0].set_title("(a) 5 independent specimens, 24×48, p=100 kPa")
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].scatter(radon, tensor, s=50)
    axes[1].plot([45, 90], [45, 90], "k--", lw=1)
    axes[1].axvspan(51, 69, color="green", alpha=0.12)
    axes[1].axhspan(51, 69, color="green", alpha=0.12)
    axes[1].set_xlabel("Radon focus (deg)"); axes[1].set_ylabel("structure tensor (deg)")
    axes[1].set_title("(b) operator agreement, identical snapshots")
    axes[1].grid(alpha=0.3)
    fig.suptitle("CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE — R3 anchor ensemble "
                 f"({resolved}/{len(anchor)} specimens resolved)")
    fig.tight_layout()
    fig.savefig("artifacts/biaxial_shear_band/r3_ensemble.png", dpi=140)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
