"""Assemble and record the board maturity verdict for m17498.

Level rule applied (board standard):
  L2 = real input -> compute -> output implementation with executable tests
  L3 = wired into the production solver/platform (a caller outside the mission
       files, an export, a CLI/API/harness entry)
  L4 = benchmark validated: complete case + error analysis + reproducible, with
       the platform referee able to recompute it (tensorbench_results row)

m17498 satisfies L2.  L3 is refused (commit 1d7a0b6e is not an ancestor of
main, `grep -rn shear_band_structure_tensor` over main HEAD returns 0 hits, no
`src/tensordem/__init__.py` export, no CLI/API/harness caller).  L4 is refused
even though ref vs got (60 vs 53.832 deg, 10.28 %) sits inside the <15 %
case tolerance, because (a) the platform referee has no measured row
(tensorbench_results: 0 rows for mission 17498 -> L4 verdict endpoint returns
CLAIMED_NOT_PROVEN), (b) no raw per-specimen field (npz / cache) was persisted,
so got is only re-derivable from a 6-line log rather than from the data, and
(c) 2 of 8 pre-registered specimens plus the cumulative-window and 32x64
points are WITHHELD by the author's own report.

Run:  python3 docs/board_audit/m17626/verdict_m17498.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit_m17498.json")
OUT = os.path.join(HERE, "verdict_m17498.json")

MISSION_AUDITED = 17498
MISSION_AUDITOR = 17626
SCOPE = "m17498 CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE"
LEVEL = 2


def build() -> dict:
    audit = json.load(open(AUDIT))
    prov = audit["provenance"]["artifacts"]
    arith = audit["arithmetic"]
    ref = audit["referee"]
    op = audit.get("operator", {})
    sha = {k: v["sha256"] for k, v in prov.items() if v.get("exists")}

    evidence = {
        "impl_files": [
            "src/tensordem/shear_band_structure_tensor.py:1 "
            "measure_band_structure_tensor() (244 lines, new in 1d7a0b6e)",
            "src/tensordem/biaxial_shear_band.py:154 radon_band_angle() "
            "(binning='equal_count', the R2 aspect-bias fix, already in main "
            "at 7a5ea5d2)",
            "benchmarks/biaxial_shear_band/run_r3_ensemble.py:1 (pre-registered "
            "5-specimen ensemble + 2 sensitivity axes)",
            "benchmarks/biaxial_shear_band/run_r3_estimator_audit.py:1",
            "benchmarks/biaxial_shear_band/summarise_r3_log.py:1",
        ],
        "entry": (
            "CLI only: benchmarks/biaxial_shear_band/run_r3_ensemble.py:363 "
            "(--out/--figure/--workers/--cache-dir/--only). No library entry: "
            "src/tensordem/__init__.py exports neither measure_band_structure_tensor "
            "nor the band measurement."
        ),
        "run_artifacts": [
            f"artifacts/biaxial_shear_band/r3_ensemble.json (4216 B, sha256 {sha.get('artifacts/biaxial_shear_band/r3_ensemble.json')})",
            f"artifacts/biaxial_shear_band/r3_ensemble.png (106794 B, sha256 {sha.get('artifacts/biaxial_shear_band/r3_ensemble.png')})",
            f"artifacts/biaxial_shear_band/r3_estimator_audit.json (39218 B, sha256 {sha.get('artifacts/biaxial_shear_band/r3_estimator_audit.json')})",
            f"logs/r3_ensemble.log (389 B, 6 lines = the primary artefact, sha256 {sha.get('logs/r3_ensemble.log')})",
        ],
        "tests": [
            "tests/test_shear_band_structure_tensor.py (9 tests: known-angle "
            "30/45/60/70 deg on the case lattice, aspect neutrality, uniform-"
            "deformation rejection, degenerate input, empty-cell fill, shape "
            "mismatch)",
            "auditor rerun 2026-11-13 on the m17498 worktree: 5/9 executed, "
            "5 passed (60.0 deg known-angle case 17.6 s; the four fast guards "
            "27.1 s); the full 9-test file did not fit the 110 s shell cap at "
            "load 216/128 cores -> remainder not_run locally",
            "tests/test_biaxial_shear_band.py (15 tests, production operator, "
            "author-run only = not_run by the auditor)",
        ],
        "integration_callers": (
            "NONE. `grep -rn shear_band_structure_tensor` over main HEAD c6b2b86f "
            "returns 0 hits; commit 1d7a0b6e is not an ancestor of main "
            "(git merge-base --is-ancestor = false); no __init__ export, no "
            "CLI/API/harness caller -> L3 refused."
        ),
        "benchmark": {
            "registry": "tensor_benchmark_cases id=59, anchor_slug="
                        "CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE, metric='shear band "
                        "inclination vs anchor (deg)', tolerance='<15% (owner "
                        "standard)', status='passed' -- note status was written "
                        "equal to created_at (2026-11-13 06:13:49), i.e. it is a "
                        "registration default, not a measured pass",
            "ref": {
                "value_deg": ref["ref_deg"],
                "source": ref["ref_source"],
                "file": "docs/BIAXIAL_SHEAR_BAND.md:30-31",
            },
            "got": {"value_deg": ref["got_deg"],
                    "statistic": "ensemble mean over 5 independent specimens, "
                                 "Radon focus (production operator)"},
            "abs_error_deg": ref["abs_error_deg"],
            "rel_error": ref["rel_error"],
            "tolerance_rel": ref["tolerance_rel"],
            "within_owner_15pct": ref["within_owner_15pct"],
            "within_3pct": ref["within_3pct"],
            "vs_literature_lower_bound_55_deg": ref["vs_literature_lower_bound_55"],
            "ci95_deg": ref["ci95_deg"],
            "ci95_inside_tolerance_band": ref["ci95_inside_tolerance_band"],
            "platform_referee": {
                "tensorbench_results_rows": 0,
                "l4_verdict_endpoint": "CLAIMED_NOT_PROVEN",
                "failed_checks": ["execution_success", "source_cluster_job",
                                  "cluster_job_bound", "ref_got_error_present",
                                  "within_tolerance"],
            },
            "auditor_recomputation": {
                "recomputed_mean_deg": arith["recomputed_radon"]["mean"],
                "published_mean_deg": arith["published_radon"]["mean"],
                "delta_mean_deg": arith["delta_mean_deg"],
                "delta_std_deg": arith["delta_std_deg"],
                "log_vs_json_specimen_mismatches": len(
                    arith["per_specimen_log_vs_json_mismatches"]),
            },
            "specimen_coverage": arith["specimens_completed_vs_planned"],
        },
        "repro_cmd": (
            "cd /root/tp-agent/work/m17498 && PYTHONPATH=src python3 "
            "benchmarks/biaxial_shear_band/run_r3_ensemble.py --workers 5 "
            "(~95 s per specimen, 8 specimens; the 120 s run_shell cap reaps it "
            "-> per-specimen cache/npz), then PYTHONPATH=src python3 "
            "benchmarks/biaxial_shear_band/summarise_r3_log.py. Auditor's "
            "independent re-derivation: PYTHONPATH=src python3 "
            "docs/board_audit/m17626/audit_m17498.py (recomputes mean/std/CI "
            "from the log and re-measures synthetic known-angle bands on the "
            "24x48 lattice)."
        ),
        "external_anchor": {
            "citation": "Desrues & Viggiani 2004, Int. J. Numer. Anal. Meth. "
                        "Geomech. 28:279-321 -- dense Hostun sand plane-strain "
                        "biaxial shear band 55-60 deg",
            "repo_pointer": "docs/BIAXIAL_SHEAR_BAND.md:30-31",
            "value_used_deg": 60.0,
            "value_provenance": "inherited from the pre-declared case sheet; the "
                                "original paper table was not re-read in R3 "
                                "(author-declared WITHHELD) and was not re-read "
                                "by the auditor either",
        },
        "auditor_checks": {
            "operator_unbiasedness": {
                "what": "production Radon on a synthetic band of known angle "
                        "injected into the exact 24x48 / 1128-particle case "
                        "lattice, noise-free, slip 0.02 m",
                "rows": op.get("rows"),
                "max_abs_error_deg": op.get("max_abs_error_deg"),
                "conclusion": "<=1.0 deg over 45/60/70 deg -> the 53.8 deg "
                              "reading is not the R1/R2 aspect-rescaling "
                              "artefact (that defect produced 44 deg from a "
                              "60 deg band)",
            },
            "residual_risk": [
                "no raw field snapshot retained (npz=[] and r3_cache/ empty) -> "
                "the noise level of the real incremental fields, the factor the "
                "author's own audit identifies as decisive beyond 0.5 %, was "
                "never measured on the actual data",
                "the two operators disagree by 13.0 deg on average / 30.5 deg "
                "max on identical snapshots; the structure-tensor member s11 "
                "reads 83.03 deg (38.4 % off) -> operator-dependent, not "
                "converged",
                "the 200 kPa sensitivity specimen is unresolved (73.50 deg, "
                "22.5 %) -> the physical trend across confinement is unknown",
            ],
            "three_deliverables": {
                "run_artifacts": "present and hashed (see run_artifacts), but the "
                                 "ensemble JSON was machine-transcribed from 6 log "
                                 "lines after the run was reaped, and no per-"
                                 "specimen field was persisted",
                "ref_vs_got": "present and inside the <15 % tolerance; audited "
                              "arithmetic reproduces mean/std to 0.0 deg and CI "
                              "to 1.8e-4 deg",
                "repro_cmd": "present and executable in form (CLI args verified, "
                             "dependencies importable, operator re-measured by "
                             "the auditor); end-to-end ensemble rerun not_run "
                             "(~8 min at load>200; cluster job cj1619 queued "
                             "behind 383)",
            },
        },
        "level_reason": (
            "L2 = real implementation (DEM biaxial cell on RidgeMaterial, 1128 "
            "particles, 4000 consolidation steps; new structure-tensor operator; "
            "9 executable tests; artefacts hashed and arithmetic reproduced by "
            "the auditor). L3 refused: not in main, zero callers/exports. L4 "
            "refused: no platform-recomputed benchmark row (CLAIMED_NOT_PROVEN), "
            "no raw fields persisted (got only re-derivable from a 6-line log), "
            "2/8 pre-registered specimens + cumulative window + 32x64 WITHHELD, "
            "and 10.3 % error fails the 3 % bar even though it passes the 15 % "
            "owner bar."
        ),
        "audit_artifacts": [
            "docs/board_audit/m17626/AUDIT.md",
            "docs/board_audit/m17626/audit_m17498.py",
            "docs/board_audit/m17626/audit_m17498.json",
            "docs/board_audit/m17626/verdict_m17498.json",
            "docs/board_audit/m17626/m17626_audit.png",
        ],
    }
    return evidence


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    evidence = build()
    with open(OUT, "w") as fh:
        json.dump(evidence, fh, indent=2, default=float)
    print("wrote", OUT)
    if args.dry_run:
        return 0
    # psycopg2 in this sandbox is a stub -> drive psql through a SQL file with
    # dollar-quoted JSON (no shell interpolation of the payload).
    import subprocess
    import tempfile

    payload = json.dumps(evidence)
    sql = (
        "INSERT INTO tensor_maturity_verdicts\n"
        "    (repo, scope, mission_id, level, evidence, verdict_by)\n"
        f"VALUES ('TensorDEM', $s${SCOPE}$s$, {MISSION_AUDITOR}, {LEVEL},\n"
        f"        $json${payload}$json$::jsonb, 'board-audit')\n"
        "ON CONFLICT (repo, scope, verdict_by) DO UPDATE\n"
        "    SET mission_id = EXCLUDED.mission_id,\n"
        "        level = EXCLUDED.level,\n"
        "        evidence = EXCLUDED.evidence\n"
        "RETURNING id;\n"
        "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state,\n"
        "       gate_level FROM tensor_maturity_verdicts\n"
        " WHERE repo='TensorDEM' AND scope=$s$" + SCOPE + "$s$;\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as fh:
        fh.write(sql)
        path = fh.name
    proc = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform",
                           "-v", "ON_ERROR_STOP=1", "-f", path],
                          capture_output=True, text=True)
    print(proc.stdout[-2000:])
    if proc.returncode != 0:
        print("psql stderr:", proc.stderr[-2000:])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())