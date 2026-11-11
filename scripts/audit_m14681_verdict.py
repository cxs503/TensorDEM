#!/usr/bin/env python3
"""Board-audit verdict for m14681 (GAP-DEM-CLUMP) — INSERT only, no ALTER.

Cross-check (交差核验) of the three-piece acceptance:
  1. run_artifacts 文件  -> MISSING (filtered by gate in merge 38ed2598, 5 files)
  2. benchmark ref vs got -> PRESENT (docs/GAP_DEM_CLUMP_TABLES.md, verified genuine
     by re-running C1/C2 legs: C1=0.0, C2 a/b=3 rel_err_production=0.0031)
  3. repro_cmd 可执行    -> PRESENT (benchmarks/gap_dem_clump/run_gap_dem_clump.py,
     C1/C2 legs reproduce the committed numbers)

Level = functional substance.  The analytic inertia kernel is real (L2) and
integrated with the production ClumpParticle (L3); the benchmark is a complete
example with error analysis and is reproducible.  But the run_artifacts
(results.json / criteria.json / gap_dem_clump.png / repro_cmd.txt) were
filtered out of the worktree by the gate, so the committed benchmark evidence
is incomplete -> L3 (not L4).
"""
from __future__ import annotations

import json

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
MISSION = 14979
BY = "board-audit"

EVIDENCE = {
    "impl_files": "src/tensordem/clump_inertia.py (374 loc, real analytic union-inertia kernel: sphere_cap_moments / lens_moments / union_mass_properties)",
    "entry": "src/tensordem/clump_inertia.py::union_mass_properties (input member centers+radii -> output inertia tensor about COM, inclusion-exclusion to 3rd order)",
    "run_artifacts": "MISSING: artifacts/GAP-DEM-CLUMP/{results.json,criteria.json,gap_dem_clump.png,repro_cmd.txt} were in original commit 981a9c1e but filtered by gate in merge 38ed2598 ('5 个文件被门禁滤除'); not present in worktree (git ls-files empty for GAP-DEM-CLUMP)",
    "tests": "tests/test_clump_inertia_analytic.py -> 20 passed (pytest, 12.96s); pins analytic kernel to exact sphere / tangent dumbbell / 384^3 voxel / 3rd-order triple, and production clump to analytic <1%",
    "integration_callers": "compare_with_voxel (clump_inertia.py) integrates with tensordem.particle.clump.ClumpParticle (production voxel grid=72); test_production_clump_inertia_within_one_percent_of_analytic pins production to analytic",
    "benchmark": "benchmarks/gap_dem_clump/run_gap_dem_clump.py (C1/C2/C3 criteria); re-ran C1/C2 legs -> C1 sphere err=0.0 (bit-exact), C2 a/b=3 rel_err_production=0.0031, rel_err_fine_voxel=2.5e-04, matching docs/GAP_DEM_CLUMP_TABLES.md; C3 honest-FAIL (repose trend non-monotone, max_delta +2.17deg < 3deg) with mechanism diagnostic",
    "repro_cmd": "python benchmarks/gap_dem_clump/run_gap_dem_clump.py --out <dir> (script present & executable; C1/C2 legs verified to reproduce committed numbers)",
    "external_anchor": "sphere I=2/5 mr^2 (bit-exact), tangent dumbbell textbook tensor (bit-exact), spherical-cap moments (elementary, vs dense quadrature 2e-6); literature repose band 22-25deg (sphere) / 35-45deg (ellipsoidal) quoted in docs/CLUMP_POLYHEDRON_PARTICLE.md",
    "level_reason": "functional substance = real analytic kernel (L2) + integration with production ClumpParticle (L3) + reproducible benchmark with error analysis; but run_artifacts (results.json/criteria.json/png/repro_cmd.txt) were filtered out of the worktree by the gate -> committed benchmark evidence incomplete, so L4 not confirmable from committed artifacts",
}

LEVEL = 3

def main() -> None:
    conn = psycopg.connect(DSN)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO tensor_maturity_verdicts "
        "(repo, scope, mission_id, level, evidence, verdict_by) "
        "VALUES (%s, %s, %s, %s, %s, %s)",
        ("TensorDEM", "m14681 GAP-DEM-CLUMP", MISSION, LEVEL,
         json.dumps(EVIDENCE), BY),
    )
    conn.commit()
    cur.execute(
        "SELECT id, repo, scope, mission_id, level, gate_state, gate_level "
        "FROM tensor_maturity_verdicts WHERE mission_id=%s AND scope LIKE '%%GAP-DEM-CLUMP%%'",
        (MISSION,),
    )
    print("inserted:", cur.fetchone())
    conn.close()


if __name__ == "__main__":
    main()
