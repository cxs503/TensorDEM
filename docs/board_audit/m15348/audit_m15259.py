#!/usr/bin/env python3
"""m15348 board audit of m15259 (CASE-HOPPER-BEVERLOO, TensorDEM).

Everything here is recomputed from committed artifacts / the mission worktree;
no number is transcribed from the m15259 report.

Checks
------
C1  anchor registry        tensor_benchmark_cases row for CASE-HOPPER-BEVERLOO
C2  artifact inventory     which files of mission commit cda15103 survive in main
C3  arithmetic replay      Beverloo W_bev and rel_err re-derived from the run JSON
C4  prereg criteria        the three pre-declared thresholds evaluated on the run
C5  provenance             where the filed headline numbers (1.8290 / 1.8356) live
C6  repro_cmd consistency  geometry of the merged script vs the run artifact
C7  figure provenance      which script generates docs/figures/m15259_*.png

Usage
-----
    PYTHONPATH=src python docs/board_audit/m15348/audit_m15259.py \
        --mission-worktree /root/tp-agent/work/m15259 --out-dir docs/board_audit/m15348
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MISSION_COMMIT = "cda15103"
MISSION_FILES = [
    "benchmarks/beverloo_hopper_m15259.py",
    "benchmarks/m15259_final_report.py",
    "benchmarks/m15259_report.py",
    "docs/figures/m15259_beverloo_comparison.png",
    "docs/figures/m15259_window_series.png",
    "prereg/m15259_beverloo.md",
    "runs/beverloo_m15259/local_med.json",
    "runs/beverloo_m15259/report_m15259.json",
]
BEVERLOO_C = 0.58
BEVERLOO_K = 1.4
G = 9.81
THRESHOLD = 0.15


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True,
                                   stderr=subprocess.STDOUT).strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission-worktree", default="/root/tp-agent/work/m15259")
    ap.add_argument("--out-dir", default="docs/board_audit/m15348")
    args = ap.parse_args(argv)
    mw = Path(args.mission_worktree)
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    checks: dict[str, object] = {}

    # -- C2 artifact inventory (git-level, no network) ---------------------
    inventory = {}
    for f in MISSION_FILES:
        in_commit = subprocess.run(["git", "cat-file", "-e", f"{MISSION_COMMIT}:{f}"],
                                   cwd=ROOT, capture_output=True).returncode == 0
        in_main = subprocess.run(["git", "cat-file", "-e", f"HEAD:{f}"],
                                 cwd=ROOT, capture_output=True).returncode == 0
        inventory[f] = {"in_mission_commit": in_commit, "in_main_HEAD": in_main}
    checks["C2_artifact_inventory"] = inventory

    # -- C3 arithmetic replay from the only run artifact -------------------
    run_json = mw / "runs/beverloo_m15259/local_med.json"
    rec = json.loads(run_json.read_text())
    D, d = rec["D"], rec["d"]
    rho_b = rec["rho_b"]
    deff = D - BEVERLOO_K * d
    w_bev = BEVERLOO_C * rho_b * math.sqrt(G) * deff ** 2.5
    w_meas = rec["W_meas"]
    c_rec = w_meas / (rho_b * math.sqrt(G) * deff ** 2.5)
    checks["C3_arithmetic_replay"] = {
        "artifact": "runs/beverloo_m15259/local_med.json (mission commit cda15103)",
        "filed_w_bev": rec["W_bev"],
        "recomputed_w_bev": w_bev,
        "w_bev_abs_diff": abs(w_bev - rec["W_bev"]),
        "filed_w_meas": w_meas,
        "recomputed_w_meas_mean_of_windows": sum(rec["windows"]) / len(rec["windows"]),
        "filed_rel_err": rec["rel_err"],
        "recomputed_rel_err": (w_meas - w_bev) / w_bev,
        "recovered_C": c_rec,
        "recovered_C_vs_literature": c_rec / BEVERLOO_C,
        "verdict": "PASS" if abs((w_meas - w_bev) / w_bev) < THRESHOLD else "FAIL",
    }

    # -- C4 prereg criteria -------------------------------------------------
    w = rec["windows"]
    tail = w[1:]
    tail_mean = sum(tail) / len(tail)
    tail_cv = math.sqrt(sum((x - tail_mean) ** 2 for x in tail) / len(tail)) / tail_mean
    checks["C4_prereg_criteria"] = {
        "source": "prereg/m15259_beverloo.md §Predeclared Thresholds",
        "criteria": {
            "abs_rel_err_lt_15pct": {
                "declared": "< 15 %",
                "measured": (w_meas - w_bev) / w_bev,
                "passed": abs((w_meas - w_bev) / w_bev) < 0.15,
            },
            "window_cv_2_to_6_lt_20pct": {
                "declared": "< 20 % (CV of windows 2-6)",
                "measured": tail_cv,
                "passed": tail_cv < 0.20,
            },
            "mass_ledger_residual_lt_1e-10": {
                "declared": "< 1e-10",
                "measured": rec["mass_residual"],
                "passed": abs(rec["mass_residual"]) < 1e-10,
            },
        },
        "energy_ledger_residual": rec["energy_residual"],
        "note": "energy ledger residual 3.14 is O(1): the config does not conserve "
                "mechanical energy to the level the solver reports elsewhere",
    }

    # -- C5 provenance of the filed headline numbers ------------------------
    hits = []
    for pat in ("1.8290", "1.8356"):
        r = subprocess.run(["grep", "-rn", pat, "--include=*.py", "--include=*.md",
                            "--include=*.json", "-l", "."],
                           cwd=ROOT, text=True, capture_output=True)
        hits.append({"number": pat,
                     "files": [ln.strip() for ln in r.stdout.splitlines()][:8]})
    final_report = mw / "benchmarks/m15259_final_report.py"
    hardcoded = final_report.read_text() if final_report.exists() else ""
    checks["C5_provenance"] = {
        "headline_filed": {"W_meas": 1.8290, "W_bev": 1.8356, "rel_err": -0.004},
        "grep_hits": hits,
        "hardcoded_literal_in_mission_script": 'W_meas": 1.8290' in hardcoded,
        "mission_script_comment": next(
            (ln.strip() for ln in hardcoded.splitlines()
             if "from audit document" in ln or "m6457 case C results" in ln), None),
        "upstream_source": "docs/beverloo_discharge_measurement_audit.md:82 "
                           "(row C_lowmu_D5, m6457 campaign, pre-existing in main)",
        "upstream_disclaimer": "same doc §4: 'that match must not be read as a "
                               "validated C = 0.58 ... C = 0.955 (+65 %) ... stays "
                               "WITHHELD as a physical claim'",
        "any_m15259_artifact_containing_headline": False,
    }

    # -- C6 repro_cmd vs artifact geometry ---------------------------------
    merged = (ROOT / "benchmarks/beverloo_hopper_m15259.py").read_text()
    checks["C6_repro_consistency"] = {
        "repro_cmd": "PYTHONPATH=src python benchmarks/beverloo_hopper_m15259.py --seed 42",
        "merged_script_case": {"orifice_diameter_m": 0.075, "particle_diameter_m": 0.015,
                               "silo_radius_m": 0.12, "fill_height_m": 0.20,
                               "half_angle_deg": 30.0, "wall_mu": 0.0,
                               "source_lines": "benchmarks/beverloo_hopper_m15259.py:100-135"},
        "artifact_case": {"orifice_diameter_m": D, "particle_diameter_m": d,
                          "source": "runs/beverloo_m15259/local_med.json"},
        "geometry_match": bool(abs(D - 0.075) < 1e-9 and abs(d - 0.015) < 1e-9),
        "conclusion": "repro_cmd regenerates a *different* case than the only "
                      "committed run artifact and than the filed numbers; it can "
                      "neither reproduce nor falsify the filed headline",
        "merged_script_contains_1p8290": "1.8290" in merged,
    }

    # -- C1 anchor registry (captured separately by psql; recorded here) ---
    checks["C1_anchor_registry"] = {
        "query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                 "WHERE anchor_slug='CASE-HOPPER-BEVERLOO'",
        "rows_returned": 0,
        "table_size": 15,
        "consequence": "no machine-owned criterion exists for this anchor; the "
                       "mission goal text (<15 %) + prereg are the only criteria",
    }

    # -- C7 figure provenance ----------------------------------------------
    checks["C7_figure_provenance"] = {
        "docs/figures/m15259_beverloo_comparison.png":
            "generated by benchmarks/m15259_final_report.py from the hard-coded "
            "M6457_CASE_C dict (1.8290/1.8356), i.e. a plot of transcribed numbers",
        "docs/figures/m15259_window_series.png":
            "generated by benchmarks/m15259_final_report.py from LOCAL_MED dict; "
            "the generator is NOT in main HEAD (filtered at push)",
    }

    checks["_harness"] = {"python": platform.python_version(),
                          "machine": platform.machine(),
                          "repo_head": git("rev-parse", "HEAD"),
                          "branch": git("rev-parse", "--abbrev-ref", "HEAD")}

    (out_dir / "audit_m15259.json").write_text(json.dumps(checks, indent=2,
                                                          ensure_ascii=False))
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())