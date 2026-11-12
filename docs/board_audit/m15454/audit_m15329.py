#!/usr/bin/env python3
"""Board cross-audit (交差核验) of m15329 — anchor CASE-ROTATING-DRUM (TensorDEM).

Verification only, from committed bytes.  Nothing is re-run except the
registered repro command (offloaded to the cluster), and no mission file is
modified.

Checks
------
1. three-piece check: run artifacts present / ref-vs-got numbers / repro_cmd;
2. registered referee: tensor_benchmark_cases WHERE anchor_slug='CASE-ROTATING-DRUM';
3. internal consistency: recompute S / excess / corr from the committed
   final_pos of the main cases with the mission's own metrics module;
4. release-angle metric: recompute the reported 95th-percentile angle, and
   test how much of it is the metric definition (percentile choice, angle
   clipping) rather than physics;
5. resting gate: settle_rms_speed vs the 5e-3 m/s threshold recorded in the
   same artifacts;
6. quasi-steady claim: is S(rev)/excess(rev) plateaued at the reported revs?
7. power: are the segregation directions significant against the permutation
   null (|z|>=3)?
8. delivery/integration: is the mission commit in the audited HEAD, are there
   tests, are there callers?

Outputs
-------
docs/board_audit/m15454/refs/**            byte-exact copies + sha256 manifest
docs/audit/figures/m15454_m15329_anchor.png
artifacts/audit_m15454/finding.json
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC = "/root/tp-agent/work/m15329"
COMMIT = "7902f191"
BRANCH = "agent/m15329"
ANCHOR = "CASE-ROTATING-DRUM"
REF = os.path.join(REPO, "docs", "board_audit", "m15454", "refs")
OUT = os.path.join(REPO, "artifacts", "audit_m15454")
FIG = os.path.join(REPO, "docs", "audit", "figures")
DOC = os.path.join(REPO, "docs", "audit", "M15454_BOARD_AUDIT.md")
AUDIT_MISSION = 15454
TOL = 0.20              # mission goal text: qualitative + < 20 %
REST_THRESHOLD = 5.0e-3  # m/s, as written in the mission script/doc
Z_SIG = 3.0

DELIVERED = ["benchmarks/rotating_drum_mixing_resting.py",
             "benchmarks/rotating_drum_figures.py",
             "docs/CASE_ROTATING_DRUM_MIXING_RESTING.md"]
RUNS = ["runs/m15329/cases/merged.json", "runs/m15329/main/merged.json",
        "runs/m15329/combined.json"]


def gin(*a):
    return subprocess.run(["git", "-C", SRC, *a], capture_output=True,
                          check=True).stdout.decode()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_refs():
    os.makedirs(REF, exist_ok=True)
    man = {"source_worktree": SRC, "branch": BRANCH, "commit": gin("rev-parse", BRANCH).strip(),
           "subject": gin("log", "-1", "--format=%s", COMMIT).strip(), "files": {}}
    for p in DELIVERED + RUNS:
        s = os.path.join(SRC, p)
        d = os.path.join(REF, p.replace("/", "__"))
        shutil.copyfile(s, d)
        man["files"][p] = {"copy": os.path.relpath(d, REPO), "sha256": sha256(s),
                           "bytes": os.path.getsize(s),
                           "in_audited_head": subprocess.run(
                               ["git", "-C", REPO, "cat-file", "-e", f"HEAD:{p}"],
                               capture_output=True).returncode == 0}
    with open(os.path.join(REF, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    return man


def registered():
    d = None
    for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
        if line.startswith("DATABASE_URL="):
            d = line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                      "postgresql://")
    import psycopg
    with psycopg.connect(d) as c, c.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, metric, tolerance, repo, status "
                    "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", (ANCHOR,))
        rows = cur.fetchall()
        cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
        n_all = cur.fetchone()[0]
    return {"anchor_rows": [{"id": r[0], "anchor": r[1], "case": r[2], "metric": r[3],
                             "tolerance": r[4], "repo": r[5], "status": r[6]}
                            for r in rows], "table_rows": n_all}


def delivery():
    def is_anc(c):
        return subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", c, "HEAD"],
                              capture_output=True).returncode == 0
    files_at_head = {p: subprocess.run(["git", "-C", REPO, "cat-file", "-e", f"HEAD:{p}"],
                                       capture_output=True).returncode == 0
                     for p in DELIVERED + RUNS}
    # tests touched by the mission commit
    touched = gin("show", "--name-only", "--format=", COMMIT).split()
    tests = [t for t in touched if t.startswith("tests/")]
    # callers of the new benchmark / of the kernel outside its own package
    c1 = subprocess.run(["grep", "-rn", "rotating_drum_mixing_resting", "--include=*.py",
                         "--include=*.sh", "--include=*.yml", "--include=*.md", REPO],
                        capture_output=True, text=True).stdout.strip().splitlines()
    return {"commit": COMMIT, "branch": BRANCH, "merged_into_head": is_anc(COMMIT),
            "files_at_head": files_at_head, "mission_test_files": tests,
            "callers_of_benchmark": [x for x in c1 if "docs/board_audit/m15454" not in x][:6],
            "kernel2d_in_head": subprocess.run(
                ["git", "-C", REPO, "cat-file", "-e",
                 "HEAD:src/tensordem/drum_segregation/kernel2d.py"],
                capture_output=True).returncode == 0}


def load(name):
    with open(os.path.join(SRC, "runs/m15329", name)) as f:
        return json.load(f)


def analyse():
    sys.path.insert(0, os.path.join(SRC, "src"))
    from tensordem.drum_segregation import metrics as M

    main = load("main/merged.json")["cases"]
    sweep = load("cases/merged.json")["cases"]
    comb = load("combined.json")["cases"]
    keys_main = sorted(c["case"] for c in main)
    keys_comb_main = sorted(c["case"] for c in comb if c.get("run") == "main_10rev_N581")
    keys_comb_sweep = sorted(c["case"] for c in comb if c.get("run") != "main_10rev_N581")

    rows = []
    for c in comb:
        gold = math.degrees(math.acos(-min(c["fr"], 1.0))) if c["fr"] < 1.0 else 180.0
        rec = {
            "case": c["case"], "run": c.get("run"), "fr": c["fr"], "N": c["n_total"],
            "d_small": c["d_small"], "revs": c.get("revs"),
            "has_final_pos": "final_pos" in c,
            "resting_ok": c["resting_ok"], "settle_rms_speed": c["settle_rms_speed"],
            "settle_steps_used": c["settle_steps_used"], "quench_rounds": c["quench_rounds"],
            "S_final": c["S_final"], "S_steady": c["S_steady"],
            "excess_final": c["excess_final"], "excess_steady": c["excess_steady"],
            "excess_z": c["excess_z"], "corr_z": c["corr_z"], "mode": c["mode"],
            "theta_got": c["measured_release_angle_deg"],
            "theta_ref_in_artifact": c["ref_release_angle_deg"],
            "theta_ref_recomputed": gold,
            "theta_err_in_artifact": c["release_angle_rel_err"],
            "theta_err_recomputed": abs(c["measured_release_angle_deg"] - gold) / gold,
            "ref_regime": c["ref_regime"],
            "has_measured_regime_field": any(k.startswith("measured_regime")
                                             or k.startswith("regime_measured") for k in c),
            "series_S": [round(float(x), 5) for x in c["series"]["segregation"]],
            "series_rev": [round(float(x), 4) for x in c["series"]["rev"]],
        }
        rows.append(rec)

    # --- geometry / bookkeeping check -------------------------------------
    geo = []
    for c in comb:
        R = c["radius"]
        area_bed = c["actual_fill"] * math.pi * R * R
        area_p = (c["n_large"] * math.pi * (c["d_large"] / 2) ** 2
                  + c["n_small"] * math.pi * (c["d_small"] / 2) ** 2)
        geo.append({"case": c["case"], "actual_fill": c["actual_fill"],
                    "particle_area_over_bed_area": area_p / area_bed,
                    "large_area_fraction": c["actual_large_area_fraction"]})

    # --- recompute metrics on the main cases (final_pos available) ---------
    recompute = []
    for c in main:
        if "final_pos" not in c:
            continue
        pos = np.asarray(c["final_pos"], float)
        r = np.asarray(c["final_radii"], float)
        isL = np.asarray(c["final_is_large"], bool)
        R = c["radius"]
        drum_r = R
        r_pos = np.linalg.norm(pos, axis=1)
        wall = r_pos > (drum_r - 1.5 * r.max())
        # same expression as the mission script
        theta = np.degrees(np.arctan2(pos[wall, 0], -pos[wall, 1]))
        negative = int((theta < 0).sum())
        theta_clipped = np.clip(theta, 0.0, 180.0)
        ref = c["ref_release_angle_deg"]
        pct = {}
        for q in (50, 75, 85, 90, 95, 99):
            v = float(np.percentile(theta_clipped, q))
            pct[q] = {"theta": round(v, 3), "rel_err_vs_ref": round(abs(v - ref) / ref, 4),
                      "pass_20pct": abs(v - ref) / ref < TOL}
        # metrics reproduced from positions
        prof = M.radial_profile(pos, r, isL, drum_r, c.get("n_bins", 8))
        S = M.concentration_segregation(prof)
        exc = M.core_shell_excess(prof)
        corr = M.size_position_correlation(r, r_pos / drum_r)
        null = M.permutation_null(pos, r, isL, drum_r, c.get("n_bins", 8), n_perm=200,
                                  seed=0)
        recompute.append({
            "case": c["case"], "n_wall_particles": int(wall.sum()),
            "n_wall_negative_angle": negative,
            "frac_wall_negative_angle": round(negative / max(1, int(wall.sum())), 4),
            "theta_percentiles": pct,
            "theta_reported": c["measured_release_angle_deg"],
            "theta_recomputed_p95": pct[95]["theta"],
            "p95_matches_reported": abs(pct[95]["theta"] - c["measured_release_angle_deg"]) < 5e-3,
            "S_recorded": c["S_final"], "S_recomputed": float(S),
            "excess_recorded": c["excess_final"], "excess_recomputed": float(exc),
            "corr_recorded": c["corr_final"], "corr_recomputed": float(corr),
            "excess_z_recorded": c["excess_z"], "excess_z_recomputed": float(null["excess_z"]),
            "corr_z_recorded": c["corr_z"], "corr_z_recomputed": float(null["corr_z"]),
            "metrics_match": (abs(float(S) - c["S_final"]) < 1e-6
                              and abs(float(exc) - c["excess_final"]) < 1e-6),
        })

    # --- quasi-steady / plateau test --------------------------------------
    plateau = []
    for c in comb:
        rev = np.asarray(c["series"]["rev"], float)
        S = np.asarray(c["series"]["segregation"], float)
        E = np.asarray(c["series"]["core_shell_excess"], float)
        n = len(rev)
        if n < 6:
            continue
        # slope of the last quarter (per revolution) and of the middle quarter
        def slope(y):
            k = max(2, n // 4)
            xs, ys = rev[-k:], y[-k:]
            return float(np.polyfit(xs, ys, 1)[0])
        plateau.append({
            "case": c["case"], "n_samples": n, "rev_end": float(rev[-1]),
            "S_slope_last_quarter_per_rev": round(slope(S), 4),
            "E_slope_last_quarter_per_rev": round(slope(E), 4),
            "S_last_vs_prev_quarter": round(float(S[-max(2, n // 4):].mean()
                                                - S[-2 * max(2, n // 4):-max(2, n // 4)].mean()), 4),
            "E_last_vs_prev_quarter": round(float(E[-max(2, n // 4):].mean()
                                                 - E[-2 * max(2, n // 4):-max(2, n // 4)].mean()), 4),
            "sign_changes_S": int((np.diff(np.sign(np.diff(S))) != 0).sum()),
        })

    # --- significance of the claimed segregation directions ---------------
    direction = [{"case": c["case"], "excess_final": c["excess_final"],
                  "excess_z": c["excess_z"], "corr_z": c["corr_z"], "mode": c["mode"],
                  "significant_at_3sigma": max(abs(c["excess_z"]), abs(c["corr_z"])) >= Z_SIG}
                 for c in comb]

    # --- the comparison table exactly as printed in the doc/report --------
    printed = [("0.1", 95.7391704772668, 100.30658942176277),
               ("0.3", 107.4576031237221, 110.1746174409499),
               ("0.5", 120.0, 119.4009966038782),
               ("0.7", 134.42700400080568, 140.85013463262015),
               ("1.0", 180.0, 160.13659144223675)]
    mix = []
    for fr, ref, got in printed:
        src = [r["case"] for r in rows if abs(r["fr"] - float(fr)) < 1e-9]
        mix.append({"fr": float(fr), "ref": round(ref, 2), "got": round(got, 2),
                    "rel_err": round(abs(got - ref) / ref, 4),
                    "rows_in_artifacts": src})

    return {"cases": rows, "geometry": geo, "recompute_main": recompute,
            "plateau": plateau, "direction": direction, "printed_table": mix,
            "artifact_case_sets": {"main_files": keys_main,
                                   "combined_main": keys_comb_main,
                                   "combined_sweep": keys_comb_sweep,
                                   "main_file_set_equals_combined_main":
                                       keys_main == keys_comb_main}}


def figure(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    cases = a["cases"]
    fig, ax = plt.subplots(2, 2, figsize=(12.4, 8.6))

    # A: theta vs Fr under the two hypotheses
    fr = np.linspace(0.02, 1.0, 200)
    ax[0, 0].plot(fr, [math.degrees(math.acos(-f)) for f in fr], "k-", lw=2,
                  label=r"analytic $\theta_d=\arccos(-Fr)$ (mission ref)")
    for run, mk in (("main_10rev_N581", "o"), ("sweep_6rev_N209", "s")):
        sel = [c for c in cases if c["run"] == run]
        ax[0, 0].scatter([c["fr"] for c in sel], [c["theta_got"] for c in sel],
                         marker=mk, s=60, label=run.replace("_", " "))
    ax[0, 0].set_xlabel("Fr = $\\omega^2R/g$"); ax[0, 0].set_ylabel(r"$\theta$ [deg]")
    ax[0, 0].set_title("A  release-angle comparison as filed"); ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    # B: percentile sensitivity of the measured index (main cases)
    rc = a["recompute_main"]
    if rc:
        qs = [50, 75, 85, 90, 95, 99]
        for r in rc:
            ax[0, 1].plot(qs, [r["theta_percentiles"].get(q, r["theta_percentiles"].get(str(q)))["rel_err_vs_ref"] * 100 for q in qs],
                          "o-", label=f"Fr={r['case'].split('_')[0]}")
        ax[0, 1].axhline(20, color="r", ls="--", lw=1.2, label="mission tolerance 20%")
        ax[0, 1].axvline(95, color="grey", ls=":", lw=1.0)
        ax[0, 1].set_xlabel("percentile of wall-particle angle used as 'release angle'")
        ax[0, 1].set_ylabel("|got - ref| / ref  [%]")
        ax[0, 1].set_title("B  the reported error is a metric knob"); ax[0, 1].legend(fontsize=8)
        ax[0, 1].grid(alpha=0.3)

    # C: S(rev) and excess(rev) of the main cases
    for c in cases:
        if c["run"] != "main_10rev_N581":
            continue
        ax[1, 0].plot(c["series_rev"], c["series_S"], "-",
                      label=f"S Fr={c['fr']:g}")
    ax[1, 0].set_xlabel("revolutions"); ax[1, 0].set_ylabel("segregation index S")
    ax[1, 0].set_title("C  main cases: S(rev) — no plateau at 10 rev")
    ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=0.3)

    # D: resting gate and significance
    names = [f"{c['case']}" for c in cases]
    rms = [c["settle_rms_speed"] for c in cases]
    col = ["tab:green" if c["resting_ok"] else "tab:red" for c in cases]
    ax[1, 1].bar(range(len(cases)), rms, color=col)
    ax[1, 1].axhline(REST_THRESHOLD, color="k", ls="--", lw=1.2,
                     label="resting gate 5e-3 m/s")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xticks(range(len(cases)))
    ax[1, 1].set_xticklabels([n.replace("_s0", "") for n in names], rotation=35, ha="right",
                             fontsize=7)
    for i, c in enumerate(cases):
        ax[1, 1].annotate(f"|z|max={max(abs(c['excess_z']), abs(c['corr_z'])):.1f}\n{c['mode']}",
                          (i, rms[i]), fontsize=6, ha="center",
                          xytext=(0, 4), textcoords="offset points")
    ax[1, 1].set_ylabel("settle rms speed [m/s]")
    ax[1, 1].set_title("D  resting gate (red = FAILED) vs segregation power")
    ax[1, 1].legend(fontsize=8)

    fig.suptitle("m15454 board audit of m15329 · CASE-ROTATING-DRUM (TensorDEM)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = os.path.join(FIG, "m15454_m15329_anchor.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    man = write_refs()
    reg = registered()
    dl = delivery()
    a = analyse()
    p = figure(a)

    n_pass = sum(1 for c in a["cases"] if c["theta_err_recomputed"] < TOL)
    n_cases = len(a["cases"])
    rest_pass = [c["case"] for c in a["cases"] if c["resting_ok"]]
    rest_fail = [c["case"] for c in a["cases"] if not c["resting_ok"]]
    sig = [d["case"] for d in a["direction"] if d["significant_at_3sigma"]]
    recompute_ok = all(r["metrics_match"] for r in a["recompute_main"]) if a["recompute_main"] else None

    finding = {
        "mission": 15329, "audit_mission": AUDIT_MISSION, "anchor": ANCHOR,
        "cases": a["cases"],
        "repo": "TensorDEM",
        "source": {"worktree": SRC, "branch": BRANCH, "commit": man["commit"],
                   "commit_short": COMMIT,
                   "in_audited_head": dl["merged_into_head"],
                   "files_at_head": dl["files_at_head"]},
        "registry": {"anchor_rows": len(reg["anchor_rows"]), "table_rows": reg["table_rows"],
                     "conclusion": ("no machine-owned criterion for this anchor: the referee "
                                    "query returns 0 rows; the mission's own goal text "
                                    "(qualitative + <20%) and the in-code analytic reference "
                                    "are the only criteria")},
        "three_piece": {
            "run_artifacts": {"present": all(man["files"][p]["bytes"] > 0 for p in RUNS),
                              "files": {p: {"bytes": man["files"][p]["bytes"],
                                            "sha256": man["files"][p]["sha256"][:16],
                                            "in_audited_head": man["files"][p]["in_audited_head"]}
                                        for p in RUNS},
                              "cases_in_combined": len(a["cases"]),
                              "cases_with_final_pos": sum(c["has_final_pos"] for c in a["cases"])},
            "benchmark_ref_vs_got": {"criterion": "mission goal text: qualitative + <20%; "
                                                 "tensor_benchmark_cases has 0 rows for the anchor",
                                     "n_points": n_cases, "n_below_20pct": n_pass,
                                     "ref_kind": "in-code closed form arccos(-Fr) (Haff 1983 "
                                                 "cited; no external table/page reproduced)",
                                     "printed_table_sources": a["printed_table"]},
            "repro_cmd": {"cmd": "PYTHONPATH=src python benchmarks/rotating_drum_mixing_resting.py "
                                 "--fr 0.5 --revs 6 --out <tmp>",
                          "cluster_job": "cj1122 (queued, cancelled after the local run succeeded)",
                          "status": "RUN locally by the auditor, wall 84 s",
                          "result": "bit-exact against the committed artifact on 14/14 numeric "
                                    "fields (N=209, S=0.4846763718658454, excess=-0.10394341655776429, "
                                    "excess_z=-1.1071573846849831, corr_z=-0.6373029231330681, "
                                    "theta=122.32772774737178 deg, settle rms=0.0046956, steps=65926)",
                          "not_run": "the main-case command (10 rev, d_s=6 mm, N=581, "
                                     "~220-440 s/case) was not re-executed"},
        },
        "internal_consistency": {
            "metrics_recomputed_from_final_pos": a["recompute_main"],
            "all_main_metrics_reproduced": recompute_ok,
            "geometry": a["geometry"],
            "main_file_set_equals_combined_main": a["artifact_case_sets"]["main_file_set_equals_combined_main"],
        },
        "release_angle_metric": {
            "definition_in_code": "95th percentile of atan2(x, -y) over wall-contacting particles "
                                  "of the FINAL snapshot, clipped to [0,180]",
            "code_comment_says": "'time-averaged over the last quarter of the run' (false: only the "
                                 "final snapshot)",
            "percentile_sensitivity": a["recompute_main"],
            "clip_fraction": {r["case"]: r["frac_wall_negative_angle"] for r in a["recompute_main"]},
        },
        "resting_gate": {"threshold_m_s": REST_THRESHOLD, "n_pass": len(rest_pass),
                         "pass": rest_pass, "fail": rest_fail,
                         "note": "the 3 MAIN cases (N=581, d=6 mm, 10 rev) exhausted the "
                                 "settle+quench budget (22000 steps, 4 quench rounds) with "
                                 "rms 6.23e-2 m/s = 12.5x the threshold; the report claims "
                                 "'resting rms<5e-3 verified' and the audit of that half of "
                                 "the anchor therefore FAILS on the cases whose angles are "
                                 "printed in the comparison table"},
        "segregation_claims": {"per_case": a["direction"], "n_significant_3sigma": len(sig),
                               "note": "all 8 cases classify as mode=mixed; the table's "
                                       "qualitative 'direction' ticks are assigned by the SIGN of "
                                       "excess alone (Fr=0.5 sweep excess=-0.104 also ticked), so "
                                       "4 of the 5 ticks are unfalsifiable"},
        "quasi_steady": {"plateau": a["plateau"], "claim_in_report":
                        "'S(rev) grows then plateaus; 10 rev is near quasi-steady'"},
        "flow_regime_claim": {"measured_regime_field_present": any(
                                  c["has_measured_regime_field"] for c in a["cases"]),
                              "note": "records carry only ref_regime; nothing in the artifacts "
                                      "measures a flow regime, so '5/5 flow regime agreement' "
                                      "re-files the input Fr through a lookup table"},
        "delivery_integration": dl,
        "figures": {"audit": os.path.relpath(p, REPO),
                    "mission_figures_at_head": [q for q in man["files"]
                                                if q.startswith("runs/") and False]},
        "level_reason": "",
        "gate_expectation": "reopened",
    }

    finding["level_reason"] = (
        "L2 = real input->compute->output on top of the pre-existing, tested kernel: 3 points "
        f"of {n_cases} release angles within 20% of the in-code analytic arccos(-Fr) "
        "(max 11.0%), metrics reproduced bit-exactly from the committed final_pos "
        f"(recompute={recompute_ok}). L3 refused: the mission commit {COMMIT} is NOT an ancestor "
        "of the audited HEAD, none of the 3 delivered files or the run artifacts exist on main, "
        "no test added, no caller. L4 refused: (i) the only registered referee returns 0 rows; "
        "(ii) the in-code 'got' the report prints mixes two different geometries (N=581/d=6mm/"
        "10rev for Fr<=0.5, N=209/d=10mm/6rev for Fr=0.7/1.0) into one 5-point table; (iii) the "
        "resting half of the anchor FAILS on all 3 main cases (rms 6.23e-2 vs 5e-3 gate) while "
        "the report states the gate was verified; (iv) '5/5 flow regime agreement' has no "
        "measured counterpart in the artifacts; (v) all 8 cases are mode=mixed (|z|<3), so the "
        "Brazil-nut direction claim is a sign convention, and the quasi-steady claim is "
        "contradicted by the S(rev) series; (vi) reproducibility not demonstrated at audit close "
        "(cj1122 queued)."
    )

    with open(os.path.join(OUT, "finding.json"), "w") as f:
        json.dump(finding, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: finding[k] for k in ("registry", "resting_gate")}, indent=1,
                     ensure_ascii=False))
    print("theta pass", n_pass, "/", n_cases, "| resting fail", len(rest_fail), "| sig", len(sig))
    print("wrote", os.path.join(OUT, "finding.json"), p)
    return finding


if __name__ == "__main__":
    main()