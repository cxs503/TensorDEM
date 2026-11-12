#!/usr/bin/env python3
"""Insert the m15454 board-audit verdict for m15329 (TensorDEM, CASE-ROTATING-DRUM).

Reads artifacts/audit_m15454/finding.json (produced by audit_m15329.py) and writes
one row into tensor_maturity_verdicts.  Idempotent: re-running updates the row
and resets gate_state to 'pending'.
"""
from __future__ import annotations

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
FIND = os.path.join(REPO, "artifacts", "audit_m15454", "finding.json")
LEVEL = 2
SCOPE = "m15329 CASE-ROTATING-DRUM"
AUDIT_MISSION = 15454


def dsn():
    for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                         "postgresql://")
    raise SystemExit("no DATABASE_URL")


def build_evidence(f):
    rec = f["internal_consistency"]["metrics_recomputed_from_final_pos"]
    ev = {
        "impl_files": [
            "benchmarks/rotating_drum_mixing_resting.py:44-89 (ref_release_angle_deg=arccos(-Fr) "
            "closed form + FR_BANDS), :91-108 build_cfg, :113-148 wall-contact angle measurement, "
            ":176-225 record, :228-283 CLI/merge  [copied byte-exact to "
            "docs/board_audit/m15454/refs/benchmarks__rotating_drum_mixing_resting.py]",
            "benchmarks/rotating_drum_figures.py:1-116 (reads runs/m15329/combined.json -> 4 png)",
            "src/tensordem/drum_segregation/kernel2d.py (PRE-EXISTING m-p176w2-n08 2-D soft-sphere "
            "kernel, present at audited HEAD; m15329 only calls DrumConfig/simulate)",
        ],
        "entry": "PYTHONPATH=src python benchmarks/rotating_drum_mixing_resting.py "
                 "--fr 0.1 0.3 0.5 --revs 10 --d-small 0.006 --out runs/m15329/main ; "
                 "second command --fr 0.1 0.3 0.5 0.7 1.0 --revs 6 --out runs/m15329/cases",
        "run_artifacts": [
            "runs/m15329/main/merged.json (163627 B, sha256 0be5a3149c465190…, 3 cases, final_pos present)",
            "runs/m15329/cases/merged.json (14375 B, sha256 0975902304598b63…, 5 cases, final_pos stripped)",
            "runs/m15329/combined.json (177815 B, sha256 7334c0ebf438be6a…, 8 cases)",
            "runs/m15329/figures/fig1..fig4.png (4 png in the mission commit)",
            "refs: docs/board_audit/m15454/refs/** + MANIFEST.json",
            "DELIVERY: none of these files exists at audited HEAD fc5cb532 "
            "(delivery_integration.files_at_head all false)",
        ],
        "tests": "0 test files in mission commit 7902f191 (git show --name-only); no assertion "
                 "on resting_ok/settle_rms semantics, no unit test of arccos(-Fr); pre-existing "
                 "tests/test_drum_seg_verification.py belongs to m-p176w2-n08, not to this mission",
        "integration_callers": "0: grep -rn rotating_drum_mixing_resting over the audited HEAD "
                               "returns no caller outside the audit copy; not a CLI, not exported, "
                               "commit not an ancestor of HEAD (git merge-base --is-ancestor "
                               "7902f191 HEAD = false) -> L3 refused",
        "benchmark": {
            "registry": "tensor_benchmark_cases WHERE anchor_slug='CASE-ROTATING-DRUM' -> 0 rows "
                        "(table has 16 rows, none drum/rotating) -> no machine-owned criterion",
            "criterion_used": "mission goal text: qualitative + <20%",
            "ref": {"kind": "in-code closed form theta_d = acos(-Fr) (deg), Fr=omega^2 R/g",
                    "values_deg": {"0.1": 95.739, "0.3": 107.458, "0.5": 120.0,
                                   "0.7": 134.427, "1.0": 180.0},
                    "external_anchor_citation": "Haff 1983 JFM 132:401-430 as written by the "
                                                "author (NOT verified against the journal in this "
                                                "audit -> WITHHELD); no external table/page "
                                                "reproduced"},
            "got": {"main_N581_d6mm_10rev": {"0.1": 100.307, "0.3": 110.175, "0.5": 119.401},
                    "sweep_N209_d10mm_6rev": {"0.1": 101.036, "0.3": 99.430, "0.5": 122.328,
                                              "0.7": 140.850, "1.0": 160.137}},
            "rel_err": {"0.1": 0.0477, "0.3": 0.0253, "0.5": 0.0050, "0.7": 0.0478, "1.0": 0.1104},
            "tolerance": 0.20,
            "status": "PASS on 8/8 points (max 11.04%) for the printed quantity, BUT the printed "
                      "5-row table mixes two geometries: Fr<=0.5 from the N=581/d=6mm/10rev main "
                      "run, Fr=0.7/1.0 from the N=209/d=10mm/6rev sweep -> '5/5' corresponds to "
                      "no single reproducible experiment",
            "blocking_failures": {
                "resting_gate": "main cases settle_rms_speed=6.2336e-2 m/s = 12.5x the 5e-3 "
                                "threshold with settle budget exhausted (22000 steps, 4 quench "
                                "rounds, resting_ok=false), while the report states 'resting "
                                "rms<5e-3 m/s verified'; sweep cases pass at 4.696e-3",
                "flow_regime_claim": "'5/5 qualitative flow-regime agreement' has no measured "
                                     "counterpart: records store only ref_regime (a lookup of the "
                                     "input Fr through FR_BANDS) -> unfalsifiable",
                "segregation_direction": "all 8 cases classify mode=mixed, max |z| = 2.10 "
                                         "(and 1.19 for main Fr=0.1); the table ticks the direction "
                                         "for +0.090/+0.081/-0.035/+0.071/+0.057 -> sign "
                                         "convention, 4 of 5 ticks unfalsifiable",
                "quasi_steady_claim": "'S(rev) plateaus by 10 rev' contradicted by its own series: "
                                      "14-16 sign changes in 23 samples, last-quarter slope "
                                      "-0.120/+0.066/-0.028 per rev for Fr=0.1/0.3/0.5",
                "metric_definition": "release angle = 95th percentile of a clipped wall-contact "
                                     "angle distribution; 22-28% of wall particles have negative "
                                     "angle clipped to 0; percentile sweep on Fr=0.1 main gives "
                                     "p50 30.6deg (68% err), p75 23.5%, p85 13.1%, p90 4.2%, "
                                     "p95 4.8%, p99 12.3% -> the reported accuracy is a percentile "
                                     "knob, not a detachment event",
            },
            "internal_consistency": "main-case S/excess/corr/excess_z/corr_z recomputed bit-exactly "
                                    f"from final_pos of {len(rec)}/3 main cases "
                                    "(metrics_match=true); particle_area/bed_area = 1.0000 (8/8) "
                                    "so fill and counts are consistent",
        },
        "repro_cmd": "PYTHONPATH=src python benchmarks/rotating_drum_mixing_resting.py --fr 0.5 "
                     "--revs 6 --out <tmp> executed BY THE AUDITOR (wall 84 s): 14/14 numeric "
                     "fields bit-identical to the committed case_fr0.5_rev6_s0.json. NOT run: the "
                     "main-case command (10 rev, d_s=6 mm, 218-440 s/case).",
        "external_anchor": "none verified in this audit: the only 'external' anchor is the in-code "
                           "closed form; Haff 1983 is cited by the author but not reproduced here "
                           "(WITHHELD); registry empty; segregation magnitude has no cited source "
                           "(the report's '0.1-0.5 literature band' carries no reference)",
        "level_reason": f["level_reason"],
        "audit_artifacts": [
            "docs/audit/M15454_BOARD_AUDIT.md",
            "docs/board_audit/m15454/audit_m15329.py",
            "docs/board_audit/m15454/refs/MANIFEST.json",
            "artifacts/audit_m15454/finding.json",
            "docs/audit/figures/m15454_m15329_anchor.png",
        ],
        "repo_head_audited": "TensorDEM fc5cb532 (branch agent/m15454); mission commit 7902f191 "
                             "on agent/m15329 is NOT an ancestor of HEAD",
        "gate_expectation": "reopened",
    }
    return ev


def main():
    f = json.load(open(FIND))
    ev = build_evidence(f)
    import psycopg
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, "
            "verdict_by) VALUES (%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET mission_id=EXCLUDED.mission_id, "
            "level=EXCLUDED.level, evidence=EXCLUDED.evidence, created_at=now(), "
            "gate_state='pending', gate_level=NULL, gate_note=NULL "
            "RETURNING id, repo, scope, mission_id, level, gate_state",
            ("TensorDEM", SCOPE, AUDIT_MISSION, LEVEL, json.dumps(ev, ensure_ascii=False),
             "board-audit"))
        row = cur.fetchone()
    print("verdict row:", row)


if __name__ == "__main__":
    main()