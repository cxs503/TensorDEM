#!/usr/bin/env python3
"""Insert AUDIT-ASSET-12REPO (TensorDEM) maturity verdicts — INSERT only, no ALTER."""
from __future__ import annotations

import json

import psycopg

DSN = ("host=127.0.0.1 port=5432 dbname=tensorplatform user=foundation "
       "password=uGTN-KwuQv_7l8Tw0MMWUx_kf-nF2Rzz")
MISSION = 14487
BY = "audit2-ab"

# scope -> (level, evidence)
V = {
"PR#1151 freeze_bond/thermal_profile": (4, {
  "entry": "src/tensordem/freeze_bond/thermal_profile.py (791 loc)",
  "exec": "pytest tests/test_thermal_profile.py -> 32 passed in 38.6s (logs/test_thermal_profile.log)",
  "benchmark": "Neumann similarity solution: rel err 0.11%-2.6% (committed benchmarks/freeze_bond_thermal_profile/results.json, dir 'a_anchor'); energy closure max_rel 4.4e-12 ('b_energy'); grid convergence n_cells 200/400/800 ('d_deviation'); predecessor error 125% recorded ('baseline_before')",
  "caveat": "full benchmark script exceeded local 110s budget (cluster cj881 queued); verdict rests on unit tests + committed csv/json which are in-tree",
  "defects": "tensordem.freeze_bond has 21 unresolved symbols for tests/test_freeze_bond.py (artifacts/import_contract/broken_imports.json) -> module/package same-name shadowing; see audit_report.md F3",
  "artifacts": ["benchmarks/freeze_bond_thermal_profile/results.json", "artifacts/audit_m14487/logs/test_thermal_profile.log"]}),

"PR#1157 pid_mdof": (2, {
  "entry": "src/tensordem/pid_mdof.py (930 loc)",
  "exec": "pytest tests/test_pid_mdof.py -> 27 passed, 1 skipped in 24.3s",
  "benchmark": "benchmarks/p307w1_n010_pid_mdof.py did not finish within local 110s budget -> not_run; doc §6 self-reports two items >3% and a FAIL for model B (honest)",
  "defects": "PR merged 9 .tmpwork/ scratch files (4331 lines) into main, removed later by #1174; repo-root post_report scripts",
  "artifacts": ["artifacts/audit_m14487/logs/test_pid_mdof.log"]}),

"PR#1162 polyhedra_repose": (2, {
  "entry": "src/tensordem/polyhedra_repose.py (478 added)",
  "exec": "pytest tests/test_polyhedra_repose_unit.py -> 14 passed; test_polyhedra_repose_angle/fast exceeded 55s local budget -> not_run (cluster cj881)",
  "benchmark": "doc self-declares 'Status: Implementation Complete, Validation Partial'",
  "artifacts": ["artifacts/audit_m14487/logs/batch3.log"]}),

"PR#1168 wave_ice + bowden_hughes": (2, {
  "entry": "src/tensordem/wave_ice/{floe_rao,transmission,breaking}.py + bowden_hughes_thermal.py (1909 added)",
  "exec": "pytest tests/test_wave_ice_verification.py -> 29 passed",
  "benchmark": "none independent: tests assert peak exists / finite / <100 / monotone only (tests/test_wave_ice_verification.py:49-120) while doc claims 'PASS (29/29 tests)'",
  "wiring": "no production importer (only scripts/generate_wave_ice_figures.py)",
  "artifacts": ["artifacts/audit_m14487/logs/batch5.log"]}),

"PR#1170 random_ice_strength delta": (1, {
  "entry": "src/tensordem/random_ice_strength/study.py (+23 loc)",
  "exec": "no new test in PR; no benchmark run",
  "note": "delta = 23 lines + 1 doc page; pre-existing module tests are outside window",
  "artifacts": ["artifacts/audit_m14487/inventory.json"]}),

"PR#1171 ice_ridge_formation": (1, {
  "entry": "src/tensordem/ice_ridge_formation/{dynamics,floes}.py (+241 loc)",
  "exec": "no test file, no benchmark runner executed; modules have zero importers",
  "artifacts": ["artifacts/audit_m14487/inventory.json", "docs/ice_ridge_formation_report.md"]}),

"PR#1173 rotary_dryer": (2, {
  "entry": "src/tensordem/rotary_dryer/api.py + 3 modules (1486 added)",
  "exec": "pytest tests/test_moist_drying.py -> 43 passed, 1 skipped",
  "benchmark": "literature correlation least-squares (R2/RMSE) inside doc; reference sources not independently re-checked in this pass",
  "wiring": "no production importer",
  "artifacts": ["artifacts/audit_m14487/logs/batch4.log"]}),

"PR#1175 ice_friction_regime_map": (2, {
  "entry": "src/tensordem/ice_friction_regime_map.py (695 added)",
  "exec": "pytest tests/test_ice_friction_regime_map.py -> 35 passed (67 with friction_law)",
  "wiring": "tests only",
  "artifacts": ["artifacts/audit_m14487/logs/test_friction_law_mcmc.log"]}),

"PR#1176 seepage": (1, {
  "entry": "src/tensordem/seepage/ (6 modules, 1469 added)",
  "exec": "pytest tests/test_seepage_piping.py -> 28 passed (green on synthetic data)",
  "defect": "C1-C3 'DEM-measured' values are fabricated inside the tests: tests/test_seepage_piping.py:95 u_analytic*(1+5% noise), :168 i_c_analytic*1.10, :245 k*1.15; docs/seepage_piping_verification.md:53/62/64 report them as DEM measurements (5.85%/5.00%/9.09% 'errors'); doc §5.2 admits no DEM coupling exists",
  "level_reason": "analytic formula library + unit tests only; measurement is MOCK -> L1",
  "artifacts": ["tests/test_seepage_piping.py", "docs/seepage_piping_verification.md"]}),

"PR#1178 fem_dem_seeding": (2, {
  "entry": "src/tensordem/fem_dem_seeding/ (6 modules, 1929 added)",
  "exec": "pytest tests/test_fem_dem_seeding_handoff.py -> 33 passed in 96.3s",
  "wiring": "no production importer (3 handoff scripts only)",
  "artifacts": ["artifacts/audit_m14487/logs/test_fem_dem_seeding.log"]}),

"PR#1179 ice_friction_meltfilm": (2, {
  "entry": "src/tensordem/ice_friction_meltfilm/ (4 modules, 919 added)",
  "exec": "pytest tests/test_ice_friction_meltfilm.py -> 15 passed",
  "defect": "criteria C1/C2 cannot fail: dry branch returns mu_asp by construction (rel err 0.0) and q_in=q_solid+q_latent+q_conv (physics.py:6-7) is the equation solved for film thickness (residual 2.2e-16)",
  "artifacts": ["artifacts/audit_m14487/logs/batch5.log"]}),

"PR#1180 gen_energy_audit_figures": (0, {
  "entry": "scripts/gen_energy_audit_figures.py (211 lines)",
  "exec": "n/a",
  "note": "figure-only script: no module, no test, no doc -> document-level artefact",
  "artifacts": ["scripts/gen_energy_audit_figures.py"]}),

"PR#1181 contact/friction_law": (2, {
  "entry": "src/tensordem/contact/{friction_law,ice_hull_friction}.py (1046 added)",
  "exec": "pytest tests/test_friction_law_mcmc_sobol.py -> 67 passed with regime_map (58.3s)",
  "wiring": "tests only",
  "artifacts": ["artifacts/audit_m14487/logs/test_friction_law_mcmc.log"]}),

"PR#1182 IN157_N03 doc": (0, {
  "entry": "docs/IN157_N03_DEM_BROKEN_ICE_VALIDATION.md (505 lines)",
  "exec": "n/a - documentation only, no code/test in PR",
  "artifacts": ["docs/IN157_N03_DEM_BROKEN_ICE_VALIDATION.md"]}),

"PR#1188 ice_vertical_structure_pileup": (3, {
  "entry": "src/tensordem/ice_vertical_structure_pileup.py (851 loc)",
  "exec": "pytest tests/test_ice_vertical_structure_pileup.py -> 16 passed; python3 benchmarks/ice_pileup_iso19906/run_benchmark.py -> finished, report.json written (logs/run_pileup.log)",
  "benchmark": "ISO 19906 nominal load: C1 same-order PASS, C1b 3x-band FAIL (declared), C2 porosity scaling alpha=20.06 CI[17.5,22.9] R2=0.90 declared pathological, C3 sigma_c exponent 0.835 PASS",
  "level_reason": "end-to-end DEM study + external standard comparison, but no convergence/conservation evidence -> L3",
  "artifacts": ["benchmarks/ice_pileup_iso19906/results/report.json", "artifacts/audit_m14487/logs/run_pileup.log"]}),

"PR#1191 ice_thermal_stress": (4, {
  "entry": "src/tensordem/ice_thermal_stress/ (8 modules, 3017 added)",
  "exec": "python3 benchmarks/ice_thermal_stress/run_validation.py -> 12/12 criteria pass, 2 WITHHELD (C11 experimental arm, C12 layer-depth band); pytest 4 files -> 12+10+19+4=45 passed",
  "benchmark": "independent references: exact semi-infinite flux (C1), slab with source (C2), node-FEM cross-check (C3), mid-contact source solution (C13), discretisation convergence table; energy ledger residues <1e-12",
  "defect": "evidence digest not reproducible: validation.py:481 puts generated_utc into the digested payload, :512-516 hashes it -> committed a1197a8c... vs run 3daede6c... (audit_report.md F6)",
  "artifacts": ["artifacts/ice_thermal_stress/validation.json", "artifacts/audit_m14487/logs/run_ice_thermal_stress.log"]}),

"PR#1195 erosion_dem": (4, {
  "entry": "src/tensordem/erosion_dem/ (6 modules, 1649 added)",
  "exec": "python3 benchmarks/erosion_dem_verification.py -> C1-C7 all pass in 70.35s, results.json written; pytest tests/test_erosion_dem.py -> 45 passed",
  "benchmark": "DNV-RP-O501 ductile rate per-point (max deviation 6.33%) with single calibrated anchor (declared CALIBRATED); Oka constants WITHHELD; mass ledger closure 2.7e-15; efficiency 9.46e-4 within [0,1]",
  "defect": "C4 has identity legs (sphere-cap volume round-trip, mass=volume*density) with 1e-15 residuals - no information",
  "artifacts": ["artifacts/p176w2-n11/results.json", "artifacts/audit_m14487/logs/run_erosion_dem.log"]}),

"PR#1196 breakage_pbm": (2, {
  "entry": "src/tensordem/breakage_pbm/ (5 modules, 1063 added)",
  "exec": "pytest tests/test_breakage_pbm.py -> 38 passed, 3 xfailed in 59.8s",
  "note": "doc records C2/C3/C4 FAIL honestly; daughter_distribution.py:240 raises NotImplementedError for n_daughters!=2; overlaps fsd_evolution/pbe + erosion_pbm/pbm (duplicate daughter_matrix/breakage_rhs)",
  "artifacts": ["artifacts/audit_m14487/logs/batch2.log"]}),

"PR#1202 shape_representation": (2, {
  "entry": "src/tensordem/shape_representation/ package (922 added)",
  "exec": "pytest tests/test_shape_representation.py + test_shape_representation_verdict.py -> 27 passed",
  "defect": "package shadows the pre-existing module src/tensordem/shape_representation.py; 6 public names (SurfaceDeviation, ellipticity_closure, sample_superquadric_surface, sq_signed_distance_estimate, superellipse_curve_radius, union_balls_signed_distance) are no longer importable; same collision pattern left broken in ice_structure_contact (15 symbols) and freeze_bond (21 symbols) per artifacts/import_contract/broken_imports.json",
  "artifacts": ["artifacts/audit_m14487/logs/test_shape_representation.log", "artifacts/import_contract/broken_imports.json"]}),

"PR#1203 friction_polyhedron real verification": (3, {
  "entry": "src/tensordem/friction_polyhedron_real_verification.py (549 loc, has __main__)",
  "exec": "pytest tests/test_friction_polyhedron_real_verification.py and module __main__ both exceeded local 110s budget -> not_run (cluster cj881 queued); PR self-reports 17 tests incl 1 xfail",
  "value": "withdrew 4 tautological criteria of the previous verification and found 3 real kernel defects (bounding-radius wall penetration, unclamped EPSD spring at mu_r=0, sign-reversed wall rolling resistance) - defect_fix on shared kernel friction_polyhedron.py (6 importers)",
  "residual": "old tautological module friction_polyhedron_verification.py still in tree and still imported by scripts/verify_friction_polyhedron.py:26",
  "artifacts": ["docs/p273w3_n02_VERIFICATION_CORRECTION.md"]}),

"PR#1204 shear_cell": (2, {
  "entry": "src/tensordem/shear_cell.py (+419)",
  "exec": "pytest tests/test_shear_cell_friction.py -> 7 passed",
  "note": "diagnostic module for the friction shear-cell gap ledger; repo-root reports_m13798.json committed",
  "artifacts": ["artifacts/audit_m14487/logs/batch3.log"]}),

"PR#1205 dem3d_energy": (2, {
  "entry": "src/tensordem/dem3d_energy.py (+141), dem3d_triaxial.py (+13)",
  "exec": "pytest tests/test_dem3d_energy_quadrature.py -> 5 passed",
  "note": "doc records an un-run face as not_run (honest) and a latent energy bug caught by new tests",
  "artifacts": ["artifacts/audit_m14487/logs/batch2.log"]}),

"PR#1206 friction_surface repose v2": (2, {
  "entry": "src/tensordem/friction_surface/{repose,study}.py (+200)",
  "exec": "pytest tests/test_friction_surface_repose_v2.py -> 7 passed",
  "note": "records a refuted v1 protocol (block point-source rising nozzle) with measurements",
  "artifacts": ["artifacts/audit_m14487/logs/batch2.log"]}),

"PR#1208 ice_anisotropy": (2, {
  "entry": "src/tensordem/ice_anisotropy/ (7 modules, 1878 added)",
  "exec": "pytest 5 files -> 11+8+9+4+13 = 45 passed",
  "benchmark": "45 deg held-out direction +2.67% vs own slip-system mean-field model; von Mises tau/sigma = 0.5966 vs 0.5774 (+3.76%); no external experiment",
  "wiring": "tests only (scripts/run_ice_anisotropy_study.py not executed)",
  "artifacts": ["artifacts/audit_m14487/logs/summary.tsv"]}),

"PR#1209 thermal_fracture_dem": (2, {
  "entry": "src/tensordem/thermal_fracture_dem.py (939 loc)",
  "exec": "python3 benchmarks/verified/dem_thermal_fracture/run_verification.py -> all criteria PASS; pytest tests/test_thermal_fracture_dem.py -> 33 passed",
  "defect": "criteria (d) thermal_stress_relative_error=1.39e-16 and (f) energy residual=0.0 are algebraic identities: compute_thermal_stress_field (:271-306) and compute_thermal_stress_analytical (:338+) are the same E*alpha*dT/(1-nu); fracture energy (:428-448) reuses 0.5*sigma^2*V/E and is then subtracted (:881-930). Module has no contact forces / velocities / time integration -> a 1-D bond-strength chain, not a DEM (audit_report.md F2)",
  "artifacts": ["benchmarks/verified/dem_thermal_fracture/results.json", "artifacts/audit_m14487/logs/run_thermal_fracture.log"]}),
}

EXTRA = {
"repo-structure module/package collisions": (1, {
  "finding": "6 same-name module/package collisions in src/tensordem: freeze_bond, ice_structure_contact, segregation, sensitivity, shape_representation, verification; packages shadow legacy modules",
  "machine_check": "python3 tools/import_contract_scan.py -> 'unresolved test imports: 36 in 2 test modules' (freeze_bond 21, ice_structure_contact 15); pytest on those two files -> 2 collection errors",
  "artifacts": ["artifacts/import_contract/broken_imports.json", "audit_report.md#F3"]}),
"repo-structure evidence artefacts": (1, {
  "finding": "22/25 recent PRs commit no result artefact although their docs cite artifacts/... paths; only #1151, #1188, #1209 committed results",
  "machine_check": "artifacts/audit_m14487/inventory.json (kind=other, artefact patterns) + ls artifacts/ -> only pr166/ present",
  "artifacts": ["artifacts/audit_m14487/inventory.json"]}),
}


def main() -> None:
    rows = []
    for scope, (lvl, ev) in V.items():
        rows.append(("TensorDEM", scope, lvl, ev))
    for scope, (lvl, ev) in EXTRA.items():
        rows.append(("TensorDEM", scope, lvl, ev))
    with psycopg.connect(DSN) as conn, conn.cursor() as cur:
        for repo, scope, lvl, ev in rows:
            cur.execute(
                """INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                   ON CONFLICT (repo, scope, verdict_by) DO NOTHING""",
                (repo, scope, MISSION, lvl, json.dumps(ev, ensure_ascii=False), BY))
        conn.commit()
        cur.execute("""SELECT scope, level FROM tensor_maturity_verdicts
                       WHERE repo='TensorDEM' AND verdict_by=%s ORDER BY level DESC, scope""", (BY,))
        out = cur.fetchall()
    print(f"inserted/verified {len(out)} rows")
    for s, l in out:
        print(f"  L{l}  {s}")


if __name__ == "__main__":
    main()