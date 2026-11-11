#!/usr/bin/env python3
"""Board-audit verdict for m14948 (H1-BROKEN-ICE-DEM) — INSERT only, no ALTER.

Cross-check (交差核验) of the three-piece acceptance for the claimed 'succeeded'
mission:
  1. run_artifacts  -> MISSING from worktree (artifacts/h1_broken_ice_dem/ absent;
     only pr166 present). run_validation.py produces results.json on-demand but the
     committed artifacts (results.json + 3 figures) are not in the tree.
  2. benchmark ref vs got -> C1 buoyancy PASS (Archimedes 0.894634 vs 0.894555,
     0.0088% / 0.0033%); C2 repose FAIL (literature band 25-40 deg vs 0-16.27 deg,
     not settled, fail_stage=C2_repose_not_settled per mission's own doc);
     C3 channel PASS (demo, t50=0.4485s, c0=0.5931->c1=0.4880).
  3. repro_cmd 可执行 -> PRESENT (benchmarks/h1_broken_ice_dem/run_validation.py,
     executed rc=0, produced results.json; but full repose run still fails).

Level = functional substance. Real independent function implementation exists
(h1_cases.py 316 loc + dynamics.py linear_drag + study.py case3_repose fix) = L2.
But the mission's own core acceptance criterion C2 (rubble repose vs literature
band) is an honest FAIL, the test suite is RED (1/6 failing:
test_linear_drag_removes_energy asserts energy_residual<1.0 but gets 212.08),
and the run_artifacts are not committed -> L4 (complete benchmark validation)
not confirmable. Verdict L2, gate should reopen (返修).
"""
from __future__ import annotations

import json

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
MISSION = 15029
BY = "board-audit"

EVIDENCE = {
    "impl_files": "src/tensordem/polyhedron_floe3d/h1_cases.py (316 loc, real: buoyancy_equilibrium_case / repose_case / channel_clearing_case / validate); src/tensordem/polyhedron_floe3d/dynamics.py (linear_drag/angular_drag added); src/tensordem/polyhedron_floe3d/study.py (case3_repose release-grid + duplicate-definition fix)",
    "entry": "benchmarks/h1_broken_ice_dem/run_validation.py (executable: PYTHONPATH=src python ... --quick -> rc=0, wrote artifacts/h1_broken_ice_dem/results.json)",
    "run_artifacts": "MISSING from worktree: artifacts/h1_broken_ice_dem/{results.json,fig1_buoyancy.png,fig2_repose.png,fig3_channel.png} not present (only artifacts/pr166); produced on-demand by run_validation.py but not committed",
    "tests": "tests/test_h1_broken_ice_dem.py -> 6 collected, 1 FAILED (test_linear_drag_removes_energy asserts energy_residual<1.0 but got 212.08; 5 passed in 26.38s)",
    "integration_callers": "h1_cases.py imports from tensordem.polyhedron_floe3d.{contact,dynamics,field,geometry,study} (production DEM solver path); run_validation.py is the benchmark driver",
    "benchmark": "C1 buoyancy: ref=Archimedes rho_i/rho_w=0.894634, got=0.894555 (rel_err 0.0088%), draft 0.107360 vs 0.107356 (0.0033%) -> PASS <5%; C2 repose: ref=literature band 25-40deg, got=0-16.27deg (not settled, contacts 0-1, fail_stage=C2_repose_not_settled per docs/H1_BROKEN_ICE_DEM_ROUND2.md) -> FAIL; C3 channel: ref=concentration time series, got=t50=0.4485s, c0=0.5931->c1=0.4880, mass_residual=0 -> PASS (demo, full 900-step run)",
    "repro_cmd": "PYTHONPATH=src python benchmarks/h1_broken_ice_dem/run_validation.py --repose-bodies 27 --repose-steps 3000 --channel-floes 24 --channel-steps 900 (executed --quick rc=0; full repose run still fails - not settled)",
    "external_anchor": "Archimedes rho_ice/rho_water (C1, analytic, independent of plan form); ice-rubble repose literature band 25-40deg (C2, WITHHELD - no local experimental dataset, mission's own doc declares this)",
    "level_reason": "functional substance = real independent function implementation (L2: h1_cases.py + dynamics.py + study.py, input->compute->output). L4 refused: the mission's own core acceptance criterion C2 (rubble repose vs literature band) is an honest FAIL (not settled, 0-16.27deg), the test suite is RED (1/6 failing: energy_residual 212.08 vs <1.0 asserted), and run_artifacts (results.json + 3 figures) are not committed -> complete benchmark validation not confirmable. L3 not granted as a full capability: only C1/C3 sub-cases pass; the '碎冰场 DEM 动力学' anchor's central observable (repose) does not validate.",
}

LEVEL = 2


def main() -> None:
    conn = psycopg.connect(DSN)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tensor_maturity_verdicts "
        "(repo, scope, mission_id, level, evidence, verdict_by) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        ("TensorDEM", "m14948 H1-BROKEN-ICE-DEM", MISSION, LEVEL,
         json.dumps(EVIDENCE), BY),
    )
    conn.commit()
    cur.execute(
        "SELECT id, repo, scope, mission_id, level, gate_state, gate_level "
        "FROM tensor_maturity_verdicts WHERE mission_id=%s AND scope LIKE '%%H1-BROKEN-ICE-DEM%%'",
        (MISSION,),
    )
    print("inserted:", cur.fetchone())
    conn.close()


if __name__ == "__main__":
    main()
