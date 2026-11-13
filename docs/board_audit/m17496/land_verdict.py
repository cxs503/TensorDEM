#!/usr/bin/env python3
"""Land the board-audit maturity verdict for m17455 (TensorDEM)."""
from __future__ import annotations

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
AUDIT_MISSION = 17496
MISSION = 17455
ANCHOR = "XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT"

EVIDENCE = {
    "impl_files": (
        "src/tensordem/binding_contract.py:64 BINDING_CONTRACT (declarative 2-D/3-D "
        "surface), :257 FROZEN_FINGERPRINT sha256:0cbfed8c..., :307 verify() "
        "introspection over live classes/instances (42 frozen checks), :434 "
        "assert_conforms(), :480 probe_semantics() 19 behavioural probes, :656 "
        "_mutants() 7 controlled drift classes, :706 mutation_matrix(), :736 "
        "substep_rule_check(); sha256 6478c4d23792ba17 (37425 B). "
        "Present on disk in /root/tp-agent/work/m17455 only — NOT committed"
    ),
    "entry": (
        "scripts/xval_library_binding_contract.py:281 main() — "
        "`PYTHONPATH=src python3 scripts/xval_library_binding_contract.py --outdir DIR`; "
        "sha256 f19559af80958730 (16471 B)"
    ),
    "run_artifacts": (
        "artifacts/xval_library_binding_contract/result.json sha256 abe0efb1f134d48b "
        "(19254 B; 8-criterion verdict table + impulse sweep n=1..128 + 7-row mutation "
        "matrix + 19 probes + environment) + fig_impulse_error/mutation_matrix/coverage/"
        "substep_boundary.png + runs/xval_library_binding_contract/run.json "
        "sha256 d7b08cb3e6aea520. All on disk in work/m17455, untracked (git status '??')"
    ),
    "tests": (
        "tests/test_binding_contract.py (20 tests; re-run by the auditor: "
        "`PYTHONPATH=src python3 -m pytest tests/test_binding_contract.py -q` -> "
        "20 passed in 7.00 s). Tests assert invariants, not mere execution: pinned "
        "fingerprint equality, json round-trip hash stability, live-surface conformance, "
        "binder-consumed field subset, 2-D/3-D restart asymmetry, negative assertions "
        "`not hasattr(IceDEM,'restore')`"
    ),
    "integration_callers": (
        "NONE. grep -rln binding_contract src/ benchmarks/ scripts/ tests/ -> only "
        "tests/test_binding_contract.py and scripts/xval_library_binding_contract.py; "
        "no solver/platform module imports it; no CI/gate consumes assert_conforms(); "
        "git log --all --grep=17455 -> 0 commits, HEAD:src/tensordem/binding_contract.py "
        "absent on agent/m17455 and on repo HEAD"
    ),
    "benchmark": (
        "REFEREE EMPTY: SELECT ... FROM tensor_benchmark_cases WHERE "
        "anchor_slug='XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT' -> 0 rows (54 registered "
        "cases in total; the closest XVAL row is XVAL-FEM-TPB1976). No registered "
        "ref-vs-got entry exists to machine-adjudicate L4. Mission's own prereg "
        "prereg/XVAL_DEM_LIBRARY_BINDING_CONTRACT.json: 8/8 criteria MET, but the "
        "numeric 'reference' is analytic n*dt*F of the same linear integrator "
        "(self-consistency at 2.66e-16, not an independent external benchmark), and the "
        "1.0 drift-detection rate is measured on the detector author's own 7 mutation "
        "classes (baseline presence-only control 1/7 = 0.143)"
    ),
    "repro_cmd": (
        "AUDITOR REPRODUCED: `PYTHONPATH=src python3 scripts/xval_library_binding_contract.py "
        "--outdir /tmp/m17496_repro` -> exit 0; deep diff of the fresh result.json vs the "
        "archived artifact = 0 numeric differences (only wall-clock fields differ), i.e. "
        "every reported number (impulse rel-err, clock, restart drift, mutation matrix, "
        "probe table) is bitwise reproducible"
    ),
    "external_anchor": (
        "OSS practice verified byte-exact (zero transcription): TensorPhysics "
        "src/tensorphysics/adapters/tensordem.py = 113 lines (doc claims 113), "
        "tests/test_adapter_inventory_meta.py = 43 lines (doc claims 43) and does contain "
        "`assert not hasattr(Solver,'restore')`; docs/adapter-contracts.md and "
        "docs/adapter-inventory.md exist. Caveat: this anchor compares mechanism/practice, "
        "not a physical quantity, so the owner 15 % accuracy gate has no physical subject"
    ),
    "level_rationale": (
        "L2 (independent functional implementation). L1 met (analytic identities + "
        "invariant tests). L3 NOT met: zero caller outside tests/scripts, deliverable "
        "never committed (auto-push had nothing to push; mission result head_commit=null). "
        "L4 NOT met: registered referee empty and the numeric reference is self-consistent "
        "analytic identity, not an external benchmark. L5 not claimed."
    ),
    "audit_findings": {
        "blocking": [
            "F1 delivery: all 7 deliverable paths are untracked on agent/m17455 "
            "(git status '??'), absent from repo HEAD, and no m17455 commit exists — "
            "the evidence chain survives only in one worktree, so nothing was pushed"
        ],
        "non_blocking": [
            "F2 verify() treats a contract whose fingerprint != FROZEN_FINGERPRINT as "
            "status=CONFORMS (breaking=0) and only sets frozen_fingerprint_match=false; "
            "reproduced: edit units N->kN -> CONFORMS + fp_match False, so assert_conforms() "
            "will not raise for a locally edited contract",
            "F3 IceDEM3D surface only 40 % frozen (8/20) — disclosed by the mission in doc "
            "§6.2 item 6; binder-consumed 3-D members outside the 8 can still drift silently",
            "F4 drift-detection rate is measured on mutation classes authored by the same "
            "module; no independent drift corpus, so 1.0 is an upper bound, not a field rate",
            "F5 impulse identity is exact by construction for a linear free-flight "
            "integrator (2.66e-16 at every n); it certifies clock/load bookkeeping, not "
            "contact physics — the mission says so (§6.2/⑥) and this is not a defect"
        ],
        "overstated_claims": [
            "mission result headline implies a delivered ecosystem capability; in fact the "
            "module is not committed and has no consumer, so 'TensorDEM 新增供应方契约' is a "
            "worktree-local increment until merged",
        ],
    },
    "audit_artifacts": [
        "docs/board_audit/m17496/audit_m17455.py",
        "docs/board_audit/m17496/refs/MANIFEST.json (11 byte-exact copies + sha256)",
        "artifacts/audit_m17496/finding.json",
        "docs/audit/figures/m17496_m17455_audit.png",
        "docs/BOARD_AUDIT_M17496.md",
    ],
}


def main():
    d = None
    for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
        if line.startswith("DATABASE_URL="):
            d = line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                      "postgresql://")
    import psycopg
    with psycopg.connect(d) as c, c.cursor() as cur:
        cur.execute("SELECT id, level, gate_state FROM tensor_maturity_verdicts "
                    "WHERE mission_id=%s AND scope LIKE %s",
                    (AUDIT_MISSION, f"%{MISSION}%"))
        prev = cur.fetchall()
        if prev:
            print("existing row(s):", prev)
        cur.execute(
            "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, "
            "verdict_by) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            ("TensorDEM", f"{MISSION} {ANCHOR}", AUDIT_MISSION, 2,
             json.dumps(EVIDENCE, ensure_ascii=False), "board-audit"))
        vid = cur.fetchone()[0]
        c.commit()
        cur.execute("SELECT id, repo, scope, mission_id, level, verdict_by, gate_state, "
                    "gate_level, gate_note FROM tensor_maturity_verdicts WHERE id=%s", (vid,))
        print("landed:", cur.fetchone())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())