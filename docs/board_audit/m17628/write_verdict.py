#!/usr/bin/env python3
"""Write the m17628 board-audit verdict for m17503 into tensor_maturity_verdicts,
then dry-run the platform's own maturity gate on the evidence (no side effects).

Idempotent: replaces an earlier board-audit row of the same scope/mission.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

SCOPE = "m17503 XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE"
REPO = "TensorDEM"
LEVEL = 3
HERE = pathlib.Path(__file__).resolve().parent
ver = json.loads((HERE / "verification.json").read_text())
probe = (HERE / "refs/probe/probe_overlap_out.txt").read_text().strip().splitlines()


def _probe(arm):
    d = {}
    for tok in arm.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


leg, lee = _probe(probe[0]), _probe(probe[1])

EVIDENCE = {
    "impl_files": [
        "docs/board_audit/m17628/refs/impl/mts_contact.py",
        "docs/board_audit/m17628/refs/impl/xval_matrix.py",
        "docs/board_audit/m17628/refs/impl/make_report_artifacts.py",
    ],
    "entry": "docs/board_audit/m17628/refs/impl/xval_matrix.py",
    "entry_note": "main(): CLI -> arm matrix + pair-level invariants; auditor re-ran "
                  "'PYTHONPATH=src python scripts/m17503/xval_matrix.py --gammas 10 --gamma-end 0.2 --no-sens' "
                  "exit 0, 33.8 s, tau/P +0.1082 / -0.0415, P 3.315e+00 / 1.045e+03 Pa identical to the artifacts",
    "run_artifacts": [
        "docs/board_audit/m17628/refs/artifacts/results.json",
        "docs/board_audit/m17628/refs/artifacts/tables.json",
        "docs/board_audit/m17628/refs/artifacts/xval_periodic_sllod.png",
        "docs/board_audit/m17628/verification.json",
        "docs/board_audit/m17628/refs/probe/probe_overlap_out.txt",
        "docs/board_audit/m17628/fig_m17628.png",
    ],
    "tests": [
        "tests/test_lees_edwards_sllod.py — auditor re-ran 2026-11-13: 5 passed, 1 xfailed in 5.98 s "
        "(I1 minimum image vs 9x9 brute-force lattice search atol=1e-15; I2 affine velocity continuity "
        "across the y boundary; I3 frame invariance xfail documents the unrepaired image selection)",
        "tests/test_lees_edwards_sllod.py + tests/test_mts_contact.py — auditor re-ran: 30 passed, 1 xfailed in 64.93 s "
        "(subject wrote '31 passed'; actual is 31 collected = 30 passed + 1 xfailed)",
    ],
    "integration_callers": "shear_cell_run",
    "integration_callers_detail": [
        "src/tensordem/mts_contact.py:1422 shear_cell_run(..., lees_edwards: bool = False) — the flag is threaded through "
        "the module's public entry (exported via __all__, mts_contact.py:128) down to MTSSystem/_gas_system; "
        "the result dict carries lees_edwards/strain_end (mts_contact.py:1556)",
        "benchmarks/mts_contact_gate.py:142 calls shear_cell_run (product/benchmark caller exists) but does NOT pass "
        "lees_edwards=True -> the Lees-Edwards path is a first-class in-solver API that no product path exercises (F6)",
    ],
    "benchmark": {
        "anchor": "XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE",
        "registered_in_tensor_benchmark_cases": False,
        "referee_sql": "SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug="
                       "'XVAL-DEM-PERIODIC-SLLOD-CROSSCHECK-AVAILABLE' -> 0 rows",
        "criterion": "relative deviation of tau/P between the legacy periodic-SLLOD arm and the Lees-Edwards arm <15% "
                     "(owner precision line, self-declared by the subject; no board referee row exists)",
        "ref_vs_got": {
            "gamma_dot_10": {"tau_over_p_legacy": 0.10822444334959849, "tau_over_p_le": -0.04146591450779702,
                             "rel_diff_pct": 361.0, "within_15pct": False},
            "gamma_dot_20": {"tau_over_p_legacy": 0.0474065703003783, "tau_over_p_le": -0.07556051996840124,
                             "rel_diff_pct": 162.7, "within_15pct": False},
            "gamma_dot_40": {"tau_over_p_legacy": 0.07253679224305867, "tau_over_p_le": -0.0131,
                             "rel_diff_pct": 654.8, "within_15pct": False},
            "pressure_ratio_same_phi": {"legacy_Pa": 3.315, "lees_edwards_Pa": 1045.1},
            "verdict": "FAIL at all three shear rates (163-655% >> 15%)",
            "tables_json_recomputed_from_results_json": ver["checks"]["C2_cross_table"]["all_rows_match"],
        },
    },
    "repro_cmd": "PYTHONPATH=src pytest tests/test_lees_edwards_sllod.py -q -> 5 passed, 1 xfailed (5.98 s); "
                 "PYTHONPATH=src python scripts/m17503/xval_matrix.py --gammas 10 --gamma-end 0.2 --no-sens -> "
                 "tau/P +0.1082 / -0.0415 reproduced; PYTHONPATH=src python scripts/m17503/xval_matrix.py "
                 "--only-invariants -> invariants bit-identical to artifacts (n_contacts 335/338, "
                 "frame_invariance 1.363e-02/3.568e-01, rate_error 9.3432/8.8240). "
                 "Caveat: the report's printed repro block omits the --only-invariants call, so following it literally "
                 "leaves invariants={} and sensitivity=[] in results.json (F3)",
    "external_anchor": "WITHHELD — the subject states the OSS implementations (LAMMPS fix deform triclinic + fix nvt/sllod, "
                       "ESPResSo LEES_EDWARDS, HOOMD shear) were NOT read line-by-line in this environment "
                       "(knowledge engine :8021 offline, no DEM repo in ~/oss-cache, oss_solutions endpoint 404); "
                       "the audit did not do the source-level crosscheck either -> no external anchor, so L5 is out of reach "
                       "and the L4 benchmark leg rests on a self-declared bar only",
    "three_piece_verification": {
        "run_artifacts": "PASS — results.json 3561 B / tables.json 1200 B / png 93739 B (PNG magic ok), sha256 in "
                         "verification.json#C1",
        "benchmark_ref_vs_got": "FAIL — 0 referee rows for the anchor; self-declared <15% bar missed by 163-655%; "
                                "external anchor WITHHELD",
        "repro_cmd": "PASS (with the F3 caveat on the printed sequence) — auditor reran the shipped code and reproduced "
                     "the gamma=10 arm pair and the invariant block bit-for-bit",
    },
    "report_six_sections": {
        "all_present": ver["checks"]["C4_six_sections"]["all_present"],
        "explicit_3pct_answer": ver["checks"]["C4_six_sections"]["explicit_3pct_verdict"],
        "honest_fail_table": "R1-R4 conclusion table + failure modes F1-F3 at the head of the report",
    },
    "level_rationale": "L3: the repair is real code inside the main contact solver (triclinic minimum image over the "
                       "sheared lattice, co-moving pair search mapped back to the true lattice, image-corrected affine "
                       "velocity in _forces, virial from the true branch vector, strain advanced in both integrators), "
                       "exposed through the module's public entry shear_cell_run/MTSSystem with the flag default-off so "
                       "all shipped paths are unchanged (zero-shear regression test asserts bit-identical pressure/"
                       "shear/coord), and covered by 6 invariant tests of which I1 is checked against a 9x9 brute-force "
                       "lattice search and I3 is an honest xfail recording the unrepaired image selection. "
                       "NOT L4: no tensor_benchmark_cases referee row (0 rows), the self-declared <15% bar fails at "
                       "163/163/655%, and the auditor's independent probe shows the Lees-Edwards arm's stress is dominated "
                       "by near-singular interpenetration, so no trustworthy reference truth exists for a benchmark pass.",
    "findings": {
        "F1_no_referee_and_criterion_missed": "blocking for L4 — 0 rows in tensor_benchmark_cases for the anchor; "
                                              "rel. deviation of tau/P 361.0/162.7/654.8% vs the <15% bar",
        "F2_le_arm_dominated_by_deep_overlap": "blocking for the 'crosscheck reference' claim — auditor probe "
                                               f"(n=64, phi=0.80, gamma=10, 909 steps, no relax stage): legacy max "
                                               f"overlap {leg['ov_max']} of radius, 0 contacts >10% r, tau/P="
                                               f"{leg['tau/P']}; Lees-Edwards max overlap {lee['ov_max']}, "
                                               f"{lee['n_ov>10%']} contacts >10% r, P={lee['P']} Pa, tau={lee['tau']} Pa "
                                               f"(tau/P={lee['tau/P']}, i.e. negative friction). Consistent with the "
                                               "subject's own F1 (frame-invariance residual 0.357 on the static jammed "
                                               "cell) but not quantified by the delivered artifacts",
        "F3_artifact_provenance": "medium — results.json wall_clock_total=0.0249 s is the timer of the last incremental "
                                  "write (the 6-case matrix needs ~95 s: auditor measured 33.8 s for one rate/2 arms); "
                                  "sensitivity=[] while the report quotes a 2.4% substep sensitivity that has no artifact "
                                  "and cannot be rerun; the printed repro block omits --only-invariants",
        "F4_overstated_exactness": "low — '934% analytic exact, non-statistical' is a deterministic evaluation of "
                                   "Ly/<|l_y|> over the measured straddling-contact set (legacy 9.3432 vs LE 8.8240), "
                                   "not a configuration-independent analytic constant",
        "F5_regression_count": "low — subject wrote '31 passed'; actual 30 passed + 1 xfailed (31 collected)",
        "F6_no_product_caller": "low — lees_edwards=True only in scripts/m17503 and tests; benchmarks/mts_contact_gate.py:142 "
                                "calls shear_cell_run without the flag",
    },
    "confirmed_true_claims": [
        "I1 tilted-lattice minimum image agrees with a brute-force 9x9 lattice search at atol=1e-15 (auditor re-ran)",
        "I2 the image-corrected affine velocity makes the straddling contact see shear_rate*l_y exactly (single-contact level)",
        "I3 legacy arm provably breaks y-translation invariance (1.363e-02) and the Lees-Edwards arm's 0.357 residual is "
        "declared as xfail instead of hidden",
        "tables.json/figure are rendered from results.json by make_report_artifacts.py — the report's digits are not hand-typed "
        "(auditor re-derived all rows: match <0.05 pp)",
        "the flag is default-off and the zero-shear path is bit-identical (regression test passes; 25 pre-existing "
        "test_mts_contact.py tests still green)",
    ],
    "not_run": [
        "full-repository pytest (only the two files touched by the subject were re-run)",
        "the other two shear rates (gamma=20/40) were not re-executed; only gamma=10 was reproduced end-to-end",
        "time-step sensitivity (the sensitivity artifact is empty, see F3)",
        "OSS source-level crosscheck (LAMMPS/ESPResSo/HOOMD) — WITHHELD by both subject and auditor",
        "bit-for-bit regression against pre-m17503 main (no stored baseline artifact with the same protocol)",
        "cross-architecture (x86 / GitHub CI) reproduction — loongarch64 host only",
    ],
    "disposition": "approve with notes — L3 confirmed; rework suggestions: (1) move the image argmin to the true tilted "
                   "lattice and key friction history by physical contact to clear F1/I3; (2) make wall_clock_total "
                   "cumulative, ship the substep-sensitivity artifact, fix the printed repro sequence; (3) report max "
                   "overlap / deep-overlap count next to tau/P so the Lees-Edwards arm's trustworthiness is visible; "
                   "(4) register a tensor_benchmark_cases row before claiming L4 for any XVAL anchor",
}

SQL_CHECK = ("SELECT id, level, coalesce(gate_state,'') FROM tensor_maturity_verdicts "
             "WHERE mission_id=17628")


def psql(sql: str) -> str:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
                        "-tA", "-F", "|", "-v", "ON_ERROR_STOP=1", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"psql failed: {r.stderr[:500]}")
    return r.stdout.strip()


def main() -> int:
    psql(f"DELETE FROM tensor_maturity_verdicts WHERE mission_id=17628 AND verdict_by='board-audit'")
    payload = json.dumps(EVIDENCE, ensure_ascii=False)
    assert "$$" not in payload
    psql("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
         f"VALUES ('{REPO}', $s${SCOPE}$s$, 17628, {LEVEL}, $j${payload}$j$::jsonb, 'board-audit')")
    print("inserted:", psql(SQL_CHECK))

    # dry-run the platform's own referee on this evidence (read-only)
    import sys
    sys.path.insert(0, "/root/tp-agent")
    import maturity_gate as mg
    lvl, why = mg.judge(LEVEL, EVIDENCE, REPO, 17628)
    print("gate dry-run -> gate_level", lvl)
    for w in why:
        print("  ", w)
    (HERE / "gate_dryrun.json").write_text(json.dumps(
        {"claimed": LEVEL, "gate_level": lvl, "why": why}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())