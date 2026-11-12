#!/usr/bin/env python3
"""m16194: land the board-audit verdict for m16136 (TensorDEM, CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT).

Inputs (machine evidence, no hand-typed numbers):
  - docs/audit-m16136/evidence_verification.json   (written by collect_evidence.py)
Outputs:
  - runs/audit_m16136/verdict_payload.json   (exact evidence JSON inserted)
  - runs/audit_m16136/insert_verdict.sql     (replayable SQL)
  - runs/audit_m16136/register_case.sql      (referee rows in tensor_benchmark_cases,
                                              status=pending -- no pass claim)
Side effect: executes both SQL files against tensorplatform (gate_state left pending so
the maturity gate adjudicates confirmed / reopened).

Evidence pointers are real: file:line, artifact path, test name, DB row ids.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent          # docs/audit-m16136
ROOT = HERE.parents[1]
RUN = ROOT / "runs" / "audit_m16136"
RUN.mkdir(parents=True, exist_ok=True)
EV = json.loads((HERE / "evidence_verification.json").read_text())

REVIEWED = "m16136 @ 00e704d9 (TensorDEM)"
ART = "artifacts/case_dem_shear_campbell_elastic_limit"

evidence = {
    "impl_files": [
        "src/tensordem/elastic_limit_shear.py:1-620 (new module; 620 loc)",
        "src/tensordem/elastic_limit_shear.py:198 kappa_of, :203 tau_star_of (dimensionless groups)",
        "src/tensordem/elastic_limit_shear.py:264 _relax_overlaps (overlap-free POCA start)",
        "src/tensordem/elastic_limit_shear.py:331-557 simulate_couette_shear "
        "(DEM loop, pair_forces :364, wall_forces :424)",
        "src/tensordem/elastic_limit_shear.py:491-496 wall momentum-flux stress "
        "tau=-Fx(top)/Lx, P=Fy(top)/Lx, two-wall cross-check",
        "src/tensordem/elastic_limit_shear.py:522-543 physics gates "
        "(energy_balance_rel<=0.30, max_overlap/d<=0.60, steady_drift<=0.30)",
        "src/tensordem/elastic_limit_shear.py:574-620 shear_stress_scaling "
        "(beta=dln tau*/dln kappa, n=2(1-beta), refuses <3 gated points)",
        f"{ART}/reference_evidence.json (Campbell 2002 abstract quotes + arXiv:1712.02693, sha256)",
    ],
    "entry": [
        "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py:100 main "
        "(one point per invocation; --aggregate at :55)",
    ],
    "run_artifacts": {
        "dir": ART,
        "committed_in_audited_commit": EV["run_artifacts"]["all_committed"],
        "n_files": EV["run_artifacts"]["n_files"],
        "files": {k: v["sha256"][:16] for k, v in EV["run_artifacts"]["files"].items()},
        "figure_embedded_in_report": EV["report"]["figures_embedded"],
        "figure_deviation": EV["report"]["figure_deviation"],
    },
    "tests": {
        "cmd": "python -m pytest -q tests/test_elastic_limit_shear.py",
        "result_here": EV["tests"]["result"],
        "inverted_pin": "tests/test_elastic_limit_shear.py:66 "
                        "test_current_integrator_is_rejected_by_the_physics_gates "
                        "(asserts the open defect persists -> green != physics validated)",
        "invariants_pinned": [
            "tests/test_elastic_limit_shear.py:23 kappa/tau* group algebra (kappa ~ gdot^-2)",
            "tests/test_elastic_limit_shear.py:41 start state overlap <= 0.05 d, inside walls",
            "tests/test_elastic_limit_shear.py:53 determinism + finite observables",
            "tests/test_elastic_limit_shear.py:77 scaling fitter refuses a <3 point ladder",
        ],
    },
    "integration_callers": EV["integration_callers"],
    "benchmark": {
        "referee_table": "tensor_benchmark_cases",
        "rows_for_anchor_before_audit": EV["benchmark_referee"]["rows_for_anchor"],
        "criteria": EV["declared_criteria"],
        "pass_count": "0/3 measurable criteria met, 1 (R4) legitimately WITHHELD",
        "compare": "no reference comparison exists: n_gated=0 so the ladder fitter "
                   "refuses (scaling.json 'fewer than 3 points passed the physics gates')",
        "gate_diagnostics_all_five_rungs": {
            tag: {"energy_balance_rel": p["gate"]["energy_balance_rel"],
                  "max_overlap_over_d": p["gate"]["max_overlap_over_d"],
                  "steady_drift": p["gate"]["steady_drift"],
                  "z": p["coordination"], "tau": p["tau"]}
            for tag, p in EV["repro"]["points"].items() if "gate" in p},
    },
    "repro_cmd": {
        "single_point": EV["repro"]["cmd"],
        "full_ladder": "for kn in 1e2 1e3 1e4 1e5 1e6; do python "
                       "benchmarks/case_dem_shear_campbell_elastic_limit/run_case.py "
                       "--tag kn$kn --kn $kn --n 100 --strain 1.5 --average-from 0.4 "
                       "--out <dir>; done; python .../run_case.py --aggregate --out <dir>",
        "re_executed_by_audit": True,
        "all_five_rungs_bit_identical": EV["repro"]["all_five_rungs_bit_identical"],
        "aggregate_refusal_reproduced": EV["aggregate"]["same_refusal_as_committed"],
        "evidence": "runs/audit_m16136/repro/ (fresh point_kn*.json + scaling.json + png)",
    },
    "external_anchor": {
        "doi": "10.1017/S002211200200109X",
        "citation": "Campbell, C. S. (2002) Granular shear flows at the elastic limit, "
                    "J. Fluid Mech. 465, 261-291",
        "access": "full text CLOSED (Unpaywall best_oa_location=null, "
                  "SemanticScholar openAccessPdf=CLOSED) -> coefficient leg R4 WITHHELD, "
                  "not guessed; only abstract + open-access secondary source (arXiv:1712.02693) quoted",
        "quotes_sha256": EV["run_artifacts"]["files"]["reference_evidence.json"]["sha256"],
    },
    "report": {
        "id": 15268,
        "six_sections_mask": EV["report"]["rows"][0]["six_sections_mask"]
                             if EV["report"]["rows"] else None,
        "markdown_chars": EV["report"]["rows"][0]["markdown_chars"]
                          if EV["report"]["rows"] else None,
        "figures": EV["report"]["figures"],
        "error_statement": "section 6 answers explicitly: no quantity enters the 15 % "
                           "criterion, let alone 3 % (R1/R2/R3 not measurable, R4 WITHHELD)",
    },
    "level_rationale": EV["level_decision"],
    "honest_fail": EV["honest_fail"],
    "audited_commit": "00e704d9",
    "audited_worktree": "/root/tp-agent/work/m16136",
}

LEVEL = 2
SCOPE = "m16136 CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT"

payload = {"repo": "TensorDEM", "scope": SCOPE, "mission_id": 16194,
           "level": LEVEL, "verdict_by": "board-audit", "evidence": evidence}
(RUN / "verdict_payload.json").write_text(json.dumps(payload, indent=1) + "\n")

ev_sql = json.dumps(evidence, ensure_ascii=False).replace("'", "''")
sql = f"""INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
VALUES ('TensorDEM', '{SCOPE}', 16194, {LEVEL}, '{ev_sql}'::jsonb, 'board-audit')
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET level = EXCLUDED.level, mission_id = EXCLUDED.mission_id,
      evidence = EXCLUDED.evidence, created_at = now(), gate_state = 'pending'
RETURNING id, level, gate_state;
"""
(RUN / "insert_verdict.sql").write_text(sql)

# Referee rows: the anchor had NO row in tensor_benchmark_cases. Register the four
# criteria the case itself declared a priori; status 'pending' = not validated, no pass
# claim. R4 carries the WITHHELD reference in its text.
crit = [
    ("R1_bagnold_exponent_inertial_limit",
     "n = d ln tau / d ln gdot in the inertial limit (tau ~ rho d^2 gdot^2)",
     "Campbell 2002 JFM 465:261 (abstract) + Bagnold 1954", "15 % rel."),
    ("R2_regime_boundary_kappa_c",
     "kappa_c for the elastic-inertial -> inertial transition",
     "Campbell 2002 regime map, quoted by arXiv:1712.02693", "factor 3 (order of magnitude)"),
    ("R3_elastic_quasistatic_exponent",
     "n for tau* ~ kappa (elastic-quasi-static limit)",
     "Campbell 2002 (abstract): stresses scale purely elastically", "absolute 0.3"),
    ("R4_bagnold_coefficient_plateau",
     "tau* plateau coefficient (WITHHELD: JFM 2002 pp.261-291 CLOSED, no OA copy)",
     "Campbell 2002 JFM 465:261 tables (not accessible in sandbox)", "WITHHELD"),
]
vals = ",\n".join(
    f"('CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT', '{c}', '{s}', '{m}', '{t}', "
    f"'TensorDEM', 'pending')" for c, m, s, t in crit)
case_sql = f"""-- registered by m16194 board audit: anchor had no referee row at all
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT v.anchor_slug, v.case_name, v.ref_source, v.metric, v.tolerance, v.repo, v.status
FROM (VALUES
{vals}
) AS v(anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases b
                  WHERE b.anchor_slug = v.anchor_slug AND b.case_name = v.case_name);
"""
(RUN / "register_case.sql").write_text(case_sql)


def psql_file(p: pathlib.Path):
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-v",
                        "ON_ERROR_STOP=1", "-f", str(p)], capture_output=True, text=True)
    print(f"--- {p.name}: rc={r.returncode}\n{r.stdout.strip()}\n{r.stderr.strip()}")
    return r.returncode


rc1 = psql_file(RUN / "register_case.sql")
rc2 = psql_file(RUN / "insert_verdict.sql")
raise SystemExit(0 if rc1 == 0 and rc2 == 0 else 1)