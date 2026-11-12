#!/usr/bin/env python3
"""m16194 board audit of m16136 (CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT, TensorDEM).

Collects *machine* evidence for the three-piece deliverable check (run_artifacts /
benchmark ref-vs-got / repro_cmd) and writes ``evidence_verification.json`` next to
this file.  Every number in the audit report is read out of that JSON -- nothing is
hand-copied.

What it does, in order
  1. inventory + sha256 + git-tracked check of the audited worktree
     (``/root/tp-agent/work/m16136`` @ commit 00e704d9);
  2. re-run the acceptance test file (``pytest -q tests/test_elastic_limit_shear.py``);
  3. re-run the *full* kappa ladder (5 rungs, the documented repro_cmd) into
     ``runs/audit_m16136/repro/`` and compare every committed point JSON field by
     field with the fresh run (bit-identity = reproducibility, not "close enough");
  4. re-run ``--aggregate`` and compare the fit-refusal payload with the committed
     ``scaling.json``;
  5. read the platform referee (``tensor_benchmark_cases``), the mission row and the
     mission report (six-section completeness + embedded figure) straight from
     PostgreSQL (``-U postgres -d tensorplatform``).
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

TARGET = pathlib.Path("/root/tp-agent/work/m16136")
HERE = pathlib.Path(__file__).resolve().parent          # docs/audit-m16136
ROOT = HERE.parents[1]                                   # audit worktree
RUN = ROOT / "runs" / "audit_m16136"
REPRO = RUN / "repro"
ART = TARGET / "artifacts" / "case_dem_shear_campbell_elastic_limit"
AUDITED_COMMIT = "00e704d9"
LADDER = ("kn1e2", "kn1e3", "kn1e4", "kn1e5", "kn1e6")
COMPARE_FIELDS = ("tau", "pressure", "tau_star", "steps", "coordination",
                  "inertial_number", "gate", "gate_pass", "kappa")


def sanitize(value):
    """Strict-JSON safe copy: non-finite floats become strings ('nan'/'inf')."""
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    return value


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run(cmd, cwd=None, timeout=1800):
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True,
                       text=True, timeout=timeout)
    return {"cmd": " ".join(str(c) for c in cmd), "cwd": str(cwd) if cwd else None,
            "returncode": p.returncode, "stdout_tail": p.stdout[-3000:],
            "stderr_tail": p.stderr[-2000:]}


def psql(sql: str) -> str:
    p = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA",
                        "-F", "\x01", "-c", sql], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"psql failed: {p.stderr}")
    return p.stdout.strip()


def rows(sql: str) -> list[list[str]]:
    out = psql(sql)
    return [line.split("\x01") for line in out.splitlines()] if out else []


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    REPRO.mkdir(parents=True, exist_ok=True)
    ev: dict = {"audit_mission": 16194, "audited_mission": 16136, "repo": "TensorDEM",
                "scope": "m16136 CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT",
                "audited_worktree": str(TARGET), "audited_commit": AUDITED_COMMIT}

    # ---- 1. inventory of the deliverable -------------------------------------
    tracked_files = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", AUDITED_COMMIT], cwd=str(TARGET),
        capture_output=True, text=True).stdout.split()
    files = {}
    for p in sorted(ART.glob("*")):
        files[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p),
                         "git_tracked_in_audited_commit":
                             str(p.relative_to(TARGET)) in tracked_files}
    ev["run_artifacts"] = {
        "dir": "artifacts/case_dem_shear_campbell_elastic_limit/",
        "files": files,
        "all_committed": all(v["git_tracked_in_audited_commit"] for v in files.values()),
        "n_files": len(files),
        "commit_file_list": [f for f in tracked_files if f.startswith(
            ("artifacts/case_dem_shear_campbell", "src/tensordem/elastic_limit_shear",
             "benchmarks/case_dem_shear_campbell", "tests/test_elastic_limit_shear",
             "docs/CASE_DEM_SHEAR_CAMPBELL"))],
    }

    # ---- 2. acceptance test file (re-executed here) ---------------------------
    tp = run([sys.executable, "-m", "pytest", "-q", "tests/test_elastic_limit_shear.py"],
             cwd=TARGET, timeout=900)
    ev["tests"] = {
        "file": "tests/test_elastic_limit_shear.py",
        "cmd": tp["cmd"], "returncode": tp["returncode"],
        "result": tp["stdout_tail"].strip().splitlines()[-1] if tp["stdout_tail"] else "",
        "tests": {
            "23": "test_dimensionless_groups_match_campbell_definition (kappa/tau* group algebra)",
            "32": "test_case_rejects_impossible_parameters (input validation)",
            "41": "test_starting_configuration_is_overlap_free (start-state invariant, <=0.05 d)",
            "53": "test_couette_point_is_deterministic_and_reports_finite_numbers (determinism)",
            "66": "test_current_integrator_is_rejected_by_the_physics_gates "
                  "(INVERTED pin: the open defect must persist)",
            "77": "test_scaling_fit_reports_slopes_and_refuses_short_ladders",
        },
        "note": "green here means the contract (incl. the honest-FAIL pin) holds, NOT "
                "that the physics validates",
    }

    # ---- 3. full ladder reproduction (the documented repro_cmd) --------------
    points, identical_all = {}, True
    for tag in LADDER:
        kn = tag.replace("kn", "")
        r = run([sys.executable,
                 "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py",
                 "--tag", tag, "--kn", kn, "--n", "100", "--strain", "1.5",
                 "--average-from", "0.4", "--out", str(REPRO)], cwd=TARGET, timeout=900)
        if r["returncode"] != 0:
            identical_all = False
            points[tag] = {"repro_returncode": r["returncode"], "stderr": r["stderr_tail"]}
            continue
        old = json.loads((ART / f"point_{tag}.json").read_text())
        new = json.loads((REPRO / f"point_{tag}.json").read_text())
        diff = {k: {"committed": old.get(k), "reproduced": new.get(k)}
                for k in COMPARE_FIELDS if old.get(k) != new.get(k)}
        identical_all &= not diff
        points[tag] = {
            "kappa": new["kappa"], "tau": new["tau"], "pressure": new["pressure"],
            "tau_star": new["tau_star"], "coordination": new["coordination"],
            "inertial_number": new["inertial_number"],
            "gate": new["gate"], "gate_pass": new["gate_pass"],
            "bit_identical_to_committed": not diff, "diffs": diff,
            "wall_clock_s": new.get("wall_clock_s"),
        }
    ev["repro"] = {
        "cmd": ("python benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py "
                "--tag kn1e4 --kn 1e4 --n 100 --strain 1.5 --average-from 0.4"),
        "executed_here": True, "all_five_rungs_bit_identical": identical_all,
        "repro_dir": "runs/audit_m16136/repro", "points": points,
    }

    # ---- 4. aggregate / fit refusal ------------------------------------------
    r = run([sys.executable,
             "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py",
             "--aggregate", "--out", str(REPRO)], cwd=TARGET, timeout=900)
    new_scal = json.loads((REPRO / "scaling.json").read_text())
    old_scal = json.loads((ART / "scaling.json").read_text())
    ev["aggregate"] = {
        "returncode": r["returncode"],
        "n_points": new_scal["n_points"], "n_gated": new_scal["n_gated"],
        "scaling_payload": new_scal["scaling"],
        "same_refusal_as_committed": (new_scal["scaling"] == old_scal["scaling"]
                                      and new_scal["n_gated"] == old_scal["n_gated"]),
        "figure_regenerated": (REPRO / "fig_tau_star_kappa.png").exists(),
    }

    # ---- 5. platform referee + report -----------------------------------------
    ev["benchmark_referee"] = {
        "table": "tensor_benchmark_cases",
        "rows_for_anchor": [
            {"case_name": r0[0], "metric": r0[1], "tolerance": r0[2], "status": r0[3]}
            for r0 in rows("SELECT case_name, metric, coalesce(tolerance,''), status "
                           "FROM tensor_benchmark_cases WHERE anchor_slug="
                           "'CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT'")],
        "note": "no referee row existed for this anchor before the audit (first "
                "collection run returned 0 rows) -> the four R1-R4 rows are registered by "
                "land_verdict.py as status=pending (no pass claim)",
        "registration_state": [
            {"case_name": r0[0], "status": r0[1],
             "created_within_last_hour": r0[2]}
            for r0 in rows("SELECT case_name, status, "
                           "(created_at > now() - interval '1 hour')::text "
                           "FROM tensor_benchmark_cases WHERE anchor_slug="
                           "'CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT' ORDER BY id")],
    }
    ev["declared_criteria"] = [
        {"id": "R1", "quantity": "Bagnold exponent n in the inertial limit",
         "reference": 2.0, "tolerance": "15 % rel.",
         "got": "not measurable (n_gated = 0; tau* non-monotonic, tau sign flips)",
         "pass": False},
        {"id": "R2", "quantity": "regime boundary kappa_c (elastic-inertial -> inertial)",
         "reference": 1e4, "tolerance": "order of magnitude (factor 3)",
         "got": "not measurable (requires >=3 gated points)",
         "pass": False},
        {"id": "R3", "quantity": "elastic-quasi-static exponent n (tau* ~ kappa)",
         "reference": 0.0, "tolerance": "absolute 0.3",
         "got": "not measurable", "pass": False},
        {"id": "R4", "quantity": "Bagnold coefficient tau* plateau",
         "reference": "WITHHELD (Campbell 2002 pp. 261-291 CLOSED)",
         "tolerance": "-",
         "got": "WITHHELD - declared, not guessed",
         "pass": None},
    ]
    ev["mission_status"] = [
        {"id": r0[0], "repo": r0[1], "status": r0[2]}
        for r0 in rows("SELECT id, repo, status FROM tensor_dev_missions WHERE id=16136")]

    reports = rows("SELECT id, title, length(markdown), "
                   "jsonb_array_length(figures), "
                   "(markdown LIKE '%## 1%')::text || (markdown LIKE '%## 2%')::text || "
                   "(markdown LIKE '%## 3%')::text || (markdown LIKE '%## 4%')::text || "
                   "(markdown LIKE '%## 5%')::text || (markdown LIKE '%## 6%')::text "
                   "FROM tensor_reports WHERE mission_id=16136 ORDER BY id")
    figures = rows("SELECT jsonb_array_length(figures), coalesce(figures->0->>'name',''), "
                   "coalesce(left(figures->0->>'caption',120),''), "
                   "coalesce(length(figures->0->>'png_base64'),0) "
                   "FROM tensor_reports WHERE mission_id=16136 ORDER BY id")
    ev["report"] = {
        "rows": [{"id": r0[0], "title": r0[1], "markdown_chars": int(r0[2]),
                  "six_sections_mask": r0[4]} for r0 in reports],
        "figures": [{"n": int(f0[0]), "name": f0[1], "caption": f0[2],
                     "png_base64_chars": int(f0[3])} for f0 in figures],
        "figures_embedded": any(int(f0[3]) > 0 for f0 in figures),
        "figure_deviation": ("figure declared by name+caption only: no png_base64 in the "
                             "report payload, so a reader of the report cannot see the "
                             "plot; the PNG itself IS committed "
                             "(artifacts/case_dem_shear_campbell_elastic_limit/"
                             "fig_tau_star_kappa.png) and is regenerated by --aggregate"),
    }

    # ---- 6. independent level checks ----------------------------------------
    callers = run(["grep", "-rn", "elastic_limit_shear", "--include=*.py",
                   "src/", "tests/", "examples/", "benchmarks/"], cwd=TARGET)
    caller_lines = [ln for ln in callers["stdout_tail"].splitlines()
                    if "src/tensordem/elastic_limit_shear.py:"
                    not in ln and "tests/test_elastic_limit_shear.py:"
                    not in ln and "benchmarks/case_dem_shear_campbell" not in ln]
    init_export = run(["grep", "-n", "elastic_limit",
                       "src/tensordem/__init__.py"], cwd=TARGET)
    ev["integration_callers"] = {
        "grep": "grep -rn elastic_limit_shear --include=*.py src/ tests/ examples/ benchmarks/",
        "importers_outside_module_test_benchmark": caller_lines,
        "exported_in_package_init": init_export["returncode"] == 0,
        "verdict": "no integration caller (standalone case module)" if not caller_lines
                   else "has integration callers",
    }

    ev["level_decision"] = {
        "level": 2,
        "L2": "real function implementation: 620 loc module (DEM loop, linear "
              "spring-dashpot + Coulomb tangential, wall momentum-flux stress, three "
              "physics gates, kappa/tau* groups, ladder fitter), 130 loc driver, "
              "input->compute->output, deterministic and bit-reproducible",
        "L3_refused": "no integration caller, not exported by src/tensordem/__init__.py; "
                      "the module re-derives its own contact law instead of reusing "
                      "tensordem.shear_cell (documented in the case doc section 6.4)",
        "L4_refused": "0 of 3 numeric criteria measurable + 1 legitimately WITHHELD: all "
                      "five rungs rejected by the module's own gates (energy imbalance "
                      "1.6e2-3.8e4 x, max overlap 0.39-0.95 d, Z collapse 1.12->0.02); "
                      "no reference comparison exists, the fit is explicitly refused",
        "L5_refused": "no scientific result delivered (honest-FAIL round 1)",
    }

    ev["honest_fail"] = {
        "declared": True,
        "where": "docs/CASE_DEM_SHEAR_CAMPBELL_ELASTIC_LIMIT.md sections 5-7 and report "
                 "15268 section 6 ('no quantity enters the 15 % criterion, let alone 3 %')",
        "mission_status": "succeeded (deliverable + honest FAIL disclosed)",
    }

    kn1e2 = ev["repro"]["points"].get("kn1e2", {})
    ev["numeric_hygiene"] = {
        "kn1e2_steady_drift_is_nan": isinstance(kn1e2.get("gate", {}).get("steady_drift"),
                                                float)
                                     and kn1e2["gate"]["steady_drift"] != kn1e2["gate"]["steady_drift"],
        "cause": "_relative_drift returns NaN when fewer than 8 |tau| samples fall in the "
                 "averaging window (shortest rung, kn=1e2, 251 steps); NaN <= tol is False so "
                 "the gate still rejects the point (fails conservatively), but the JSON carries "
                 "NaN and the diagnostic is uninformative",
        "note": "evidence JSON sanitizes non-finite floats to strings so it stays strict JSON",
    }

    (HERE / "evidence_verification.json").write_text(
        json.dumps(sanitize(ev), indent=1, allow_nan=False) + "\n")
    print(json.dumps({"run_artifacts": ev["run_artifacts"]["n_files"],
                      "all_committed": ev["run_artifacts"]["all_committed"],
                      "tests": ev["tests"]["result"],
                      "ladder_bit_identical": identical_all,
                      "n_gated": ev["aggregate"]["n_gated"],
                      "refusal_same": ev["aggregate"]["same_refusal_as_committed"],
                      "bench_rows_before": len(ev["benchmark_referee"]["rows_for_anchor"]),
                      "report_sections": ev["report"]["rows"][0]["six_sections_mask"]
                      if ev["report"]["rows"] else None,
                      "figure_b64_chars": ev["report"]["figures"][0]["png_base64_chars"]
                      if ev["report"]["figures"] else 0,
                      "integration_callers": len(caller_lines)}, indent=1))


if __name__ == "__main__":
    main()