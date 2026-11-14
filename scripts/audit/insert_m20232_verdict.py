#!/usr/bin/env python3
"""Insert the m20232 board-audit verdict for m20055 / CASE-DEM-SILO-BEVERLOO.

Evidence numbers are read from ``docs/board_audit/m20232/audit_m20055_beverloo.json``
(produced by ``scripts/audit/audit_m20055_beverloo.py``) so that nothing is
hand-copied; the qualitative pointers are fixed strings that name real files,
lines and artefacts.

    python scripts/audit/insert_m20232_verdict.py [--dry-run]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs" / "board_audit" / "m20232" / "audit_m20055_beverloo.json"
MISSION_ID = 20232
REPO = "TensorDEM"
SCOPE = "m20055 CASE-DEM-SILO-BEVERLOO"
LEVEL = 2

REPRO_NOTE = (
    "mission's repro.sh is valid bash but the report quotes it as `PYTHONPATH=src python "
    "benchmarks/m20055_beverloo/repro.sh` (python on a shell script) => not executable as written; "
    "the underlying path was re-verified by this audit: (a) pytest re-run 22 passed 13.33 s; "
    "(b) independent estimator replay scripts/audit/audit_m20055_beverloo.py reproduces the filed "
    "W=1.8175329 kg/s to 2.2e-16 from the raw discharge-event record; (c) fresh DEM replay of D4 on the "
    "compute gate (cj2578, 1 core, stage_dir=/root/tp-agent/work/m20055) => CJ2578_RESULT"
)


def build() -> dict:
    a = json.loads(AUDIT.read_text())
    est = a["estimators"]
    est_span = [min(e["rel_pct"] for e in est.values() if e["pass_10pct"] is False),
                max(e["rel_pct"] for e in est.values())]
    return {
        "impl_files": [
            "src/tensordem/silo_discharge.py:260-315 steady_rate() NEW (uncommitted diff, +156/-26)",
            "src/tensordem/silo_discharge.py:70-78 packing='settled' / plug_during_settle / settle_loose_gap",
            "src/tensordem/silo_discharge.py:142-257 _plate_forces() closed-plate gate + rim circles",
            "benchmarks/m20055_beverloo/run_case.py:37-80,165-210 (case matrix + entry)",
            "benchmarks/m20055_beverloo/analyse.py:36-72 (quantile sensitivity, reads ref JSON only)",
            "prereg/m20055_beverloo.md (predeclaration, frozen before the matrix ran)",
        ],
        "entry": ("PYTHONPATH=src python benchmarks/m20055_beverloo/run_case.py --case D4 "
                  "(the filed artefact records the shard form: --shard 1 --nshards 6); "
                  "then analyse.py / render_doc.py / repro.sh"),
        "run_artifacts": [
            "runs/m20055_beverloo/D4.json - the ONLY artefact of the 6 predeclared cases "
            "(byte copy sha256 ad594807536c5e28... in docs/board_audit/m20232/refs/)",
            "runs/m20055_beverloo/summary.json (n_runs=1, all_pass_10pct=false, power_law={})",
            "runs/m20055_beverloo/report_tables.md",
            "docs/figures/m20055_beverloo_flow_law.png (n=1 => self-anchored, not exponent evidence)",
            "docs/figures/m20055_beverloo_errors.png",
            "MISSING: D5 (declared primary), D6, D5_mu15, D5_mu45, D5_dthalf",
        ],
        "tests": (
            "tests/test_silo_discharge_steady.py (10 tests, real invariants: constant-rate recovery to "
            "1e-12; plateau recovery between a slow start and a slow tail to 2%; degenerate safety "
            "0.0/NaN; plugged plate supports at 0.999r while the open plate does not; plug leaves the "
            "solid plate unchanged; settled packing born with min pair distance > 2r; legacy HCP "
            "unchanged; invalid packing rejected; no discharge before the orifice opens; legacy "
            "open-from-step-0 protocol unchanged) + pre-existing tests/test_silo_discharge_beverloo.py "
            "(3) => 22 passed in 13.33 s, re-run by this audit on 1 core. No test asserts a stored JSON "
            "value, so the suite is green because the physics holds, not because a number was copied."
        ),
        "integration_callers": (
            "NONE for the new capability: steady_rate/packing='settled' are called only from "
            "benchmarks/m20055_beverloo/{run_case,analyse}.py and the mission's own test; no API/CLI/"
            "export/solver caller. The reused DEM kernel tensordem.granular3d_core is pre-existing "
            "(m5481) and silo_discharge itself is called only by the mission scripts "
            "(benchmarks/m15259_final_report.py references it as prose). L3 refused on inspection."
        ),
        "benchmark": {
            "anchor": "CASE-DEM-SILO-BEVERLOO (Beverloo 1961 Eq.4 p.265)",
            "ref_kg_s": a["beverloo_ref"]["W_ref_kg_s"],
            "got_kg_s": a["filed"]["W_got_kg_s"],
            "rel_pct": a["filed"]["rel_pct"],
            "recovered_C": est["quantile_0.20-0.80 (predeclared)"]["C_recovered"],
            "literature_C_band": [0.55, 0.60],
            "own_pass_10pct": a["filed"]["pass_10pct"],
            "own_pass_15pct": a["filed"]["pass_15pct"] if "pass_15pct" in a["filed"] else False,
            "verdict": "FAIL: +25.0% is outside the case's own 10% gate AND outside the platform 15% "
                       "acceptability bound; recovered C=0.725 sits outside the spherical 0.55-0.60 band",
            "estimator_robustness": (
                "8 estimators on the same raw record: 5 quantile bands + whole-stream mean + "
                "longest +/-5 pct plateau give %+.1f..%+.1f pct (all FAIL); the only inside-tolerance "
                "number, -5.5 pct, is the legacy fixed-window mean that averages across the plugged "
                "phase (flow_rate_windows_kg_s[0]=0 while the first discharge event is at t=0.665 s "
                ">= settle_time=0.60 s) => it is an artefact of the protocol defect, not agreement"
                % (est_span[0], est_span[1])),
            "registered_referee": (
                "NONE: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
                "anchor_slug='CASE-DEM-SILO-BEVERLOO' => 0 rows (127 rows total, none %SILO%/%BEVERLOO%); "
                "the mission record's own server_quality_audit is {gate_valid:false, "
                "gate_missing_warning:true}, so the anchor could not have been machine-judged"),
            "domain_edge": "the only filed point D/d=4 sits at the lower edge of the reference's declared "
                           "applicability (ref_beverloo_1961.json validation_domain.D_over_d_min=4.0)",
        },
        "repro_cmd": REPRO_NOTE,
        "external_anchor": (
            "Beverloo W.A., Leniger H.A., van de Velde J., Chem. Eng. Sci. 15(3-4) 260-269 (1961), "
            "DOI 10.1016/0009-2509(61)80059-4, Eq.(4) p.265: W = C rho_b sqrt(g) (D - k d)^(5/2), "
            "C=0.58 (0.55-0.60), k=1.4 (1.2-1.6); machine-read from "
            "benchmarks/m20055_beverloo/ref_beverloo_1961.json, whose provenance field honestly records "
            "'cited, NOT re-digitized' (no paper copy, no network in the worktree)"
        ),
        "level_reason": (
            "L2 = real code input->compute->output, proven: geometry+DISORDERED bed -> granular3d_core "
            "contacts -> discharge-event stream (763 events, 766 particles, 70945 steps) -> quantile "
            "steady rate -> error vs the machine-read Beverloo value; the filed W is reproduced from the "
            "raw record to 2.2e-16 by this audit. L1 also holds (22 real invariant tests). L3 refused: no "
            "caller/export/API outside the mission's own benchmark. L4 refused: anchor FAIL at +25.0% "
            "(>15% unacceptable, >>3% target), only 1 of 6 predeclared cases has an artefact (D5 primary, "
            "D6 and both sensitivity arms are missing so the 5/2 exponent test and the dt/mu sensitivity "
            "tables are empty), and no referee row exists in tensor_benchmark_cases. The +25% is NOT an "
            "estimator artefact: 7 of 8 estimators land at +19.6..+28.8%, 7.5 sigma above the sub-window "
            "standard error."
        ),
        "findings": {
            "blocking": [
                "B1 anchor FAIL +25.0% (own artefact pass_10pct=false, pass_15pct=false; C_rec=0.725)",
                "B2 5 of 6 predeclared cases have no artefact: report section 5(a) dt-convergence and "
                "5(b) friction-sensitivity tables are header-only empty; power_law={} so the 5/2 exponent "
                "is never tested (cj2460 landed only shard 1, cj2555 never fetched)",
                "B3 the report's headline verdict names the primary case D5 (report lines 3 and 86: "
                "D/d = 5 is announced as the deciding case) but no D5 artefact exists on disk or in the "
                "DB => unsupported claim; the audited evidence covers D/d=4 only, which sits at the edge "
                "of the correlation's validity domain",
            ],
            "medium": [
                "M1 no registered referee row for the anchor (0 rows) => the anchor cannot be "
                "machine-judged at all; the mission's own server_quality_audit gate_valid=false",
                "M2 the Beverloo C/k coefficients are cited not re-digitised (honestly declared); the "
                "0.55-0.60 band alone is +/-4% on W",
                "M3 the predeclared steadiness gate CV<0.15 is too weak at this system size: "
                "steady_rate_cv=0.060 passes while the 0.25 s binned rate swings 1.20-2.12 kg/s "
                "(CV=0.126) and the longest +/-5% plateau is 0.75 s => 'steady' is not established and a "
                "10% agreement target is at the resolution limit of n_discharged=766",
                "M4 reproducibility identity incomplete: artefact git_sha='unknown', all 8 delivered "
                "paths uncommitted (head_commit NULL), nshards and job id not recorded in the artefact",
                "M5 the report's own one-command repro line runs a bash script through python => not "
                "executable as written (repro.sh itself is valid)",
            ],
            "low": [
                "L1 flow_law figure is degenerate with n=1: the '5/2 (anchored)' curve is anchored on the "
                "single D4 point, so it is decorative",
                "L2 solid plate thickness 0.5 d and R_silo/D=3.25 are unswept; rolling friction mu_r=0.02 "
                "is fixed and unswept although it is the predeclared suspect for a high discharge rate",
                "L3 silo_discharge.py module docstring still advertises 'C ~= 0.58, k ~= 1.4 for glass "
                "beads' as verified; this mission's own result (C_rec=0.725) refutes that for the shipped "
                "protocol",
            ],
            "credit": [
                "REAL DEM, no formula shortcut: 1525 particles, 70945 steps, dt/T_contact=0.0354",
                "the density-basis defect found by the m15418 board audit is FIXED here: W_ref uses the "
                "measured bed rho_b=1298.485 (phi=0.519) at the instant of opening, not a hard-coded "
                "phi=0.64; this audit recomputed W_ref=1.454480 kg/s from the ref JSON + that rho_b (no "
                "inflation factor)",
                "plug integrity is measured, not asserted: first discharge at t=0.665 s > settle_time=0.60 s",
                "the 3 pre-existing silo_discharge_beverloo tests still pass against the rewritten plate "
                "contact geometry (+156/-26 lines, backward compatible)",
            ],
        },
        "audit_artifacts": [
            "docs/board_audit/m20232/AUDIT_CASE-DEM-SILO-BEVERLOO.md",
            "docs/board_audit/m20232/audit_m20055_beverloo.json",
            "scripts/audit/audit_m20055_beverloo.py",
            "scripts/audit/fig_m20055_beverloo.py",
            "docs/board_audit/m20232/fig_m20055_rate_vs_time.png",
            "docs/board_audit/m20232/fig_m20055_estimators.png",
            "docs/board_audit/m20232/refs/ (byte copies + sha256 of every reviewed mission file)",
        ],
        "independence_limitations": (
            "audited from a different worktree (m20232, main-line clone) while the mission's 8 files are "
            "uncommitted, so every mission artefact was reviewed as a sha256-pinned byte copy; this audit "
            "re-ran the test suite and one DEM case, not the 5 missing cases; the Beverloo 1961 original "
            "table could not be re-digitised (no paper, no network)"
        ),
        "checks_not_run": [
            "D5/D6/D5_mu15/D5_mu45/D5_dthalf: no artefacts exist; rerunning them is a remediation order, "
            "so the dt-convergence and mu-sensitivity claims are UNKNOWN, not passed",
            "Beverloo 1961 original data points: not re-digitised (no paper, no outbound network)",
            "the mission result's local 553-particle pilot (+3.0%, CV 0.166): no artefact and no recorded "
            "parameters => NOT reproducible; note its CV would fail the mission's own 0.15 stability gate",
        ],
        "disposition": "approve with notes at L2 (honest-FAIL already declared by the mission); the "
                       "anchor stays FAIL and needs a separate remediation order: run+retrieve the 5 "
                       "missing cases, register the referee row (rel_err(W), tol 0.10), strengthen the "
                       "steadiness gate and report the resolution limit, record git_sha/nshards/job id, "
                       "fix the repro line, add the honest caveat to the module docstring",
        "repo_head_audited": "base 5101c393 (m20055 worktree, branch agent/m20055, 8 dirty paths, "
                             "head_commit NULL); audit worktree m20232 on agent/m20232",
        "gate_expectation": "confirmed at L2 by JUDGE_SCHEME (evidence pointers are real files/lines/"
                            "artefacts) while the anchor stays FAIL; JUDGE_SCHEME does not read benchmark "
                            "status, so L2 confirmation does not imply agreement with Beverloo",
    }


def sql_literal(ev: dict) -> str:
    return json.dumps(ev, ensure_ascii=False).replace("'", "''")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ev = build()
    (ROOT / "verdict_m20232.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    stmt = (
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)\n"
        f"VALUES ('{REPO}', '{SCOPE}', {MISSION_ID}, {LEVEL}, '{sql_literal(ev)}'::jsonb, 'board-audit')\n"
        "ON CONFLICT (repo, scope, verdict_by) DO UPDATE\n"
        "  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,\n"
        "      evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL,\n"
        "      gate_note = NULL;\n"
    )
    (ROOT / "verdict_m20232.sql").write_text(stmt)
    print(stmt[:400] + " ...\n")
    if "--dry-run" in argv:
        print("[dry-run] verdict_m20232.sql written, not executed")
        return 0
    res = subprocess.run(
        ["psql", "-U", "postgres", "-h", "127.0.0.1", "-d", "tensorplatform",
         "-v", "ON_ERROR_STOP=1", "-f", str(ROOT / "verdict_m20232.sql")],
        capture_output=True, text=True, env={"PGPASSWORD": "x", "PATH": "/usr/bin:/bin"})
    print(res.stdout, res.stderr)
    return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())