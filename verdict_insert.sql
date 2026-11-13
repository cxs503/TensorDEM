-- Verdict for m18189 XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT
-- Audit ID: 18307, Verdict by: board-audit, Date: 2026-11-13
-- Level: L4 (基准验证：完整算例+误差分析+可复现)

INSERT INTO tensor_maturity_verdicts (
    repo, scope, mission_id, level, evidence, verdict_by
) VALUES (
    'TensorDEM',
    'm18189 XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT',
    18307,
    4,
    '{
        "impl_files": "src/tensordem/deposition_packing.py:782-920 (stiffness_limit, extrapolation_selfconsistency)",
        "entry": "benchmarks/xval_dem_round2_deposition_rigid_limit.py",
        "run_artifacts": "artifacts/xval_dem_round2_deposition_rigid_limit/results.json, table_main.md, figures/",
        "tests": "tests/test_rcp_extrapolation_order.py (14/14 passed)",
        "integration_callers": "benchmarks/case_rcp_deposition/run_case.py",
        "benchmark": "prereg/XVAL_DEM_ROUND2_DEPOSITION_RIGID_LIMIT.json; 40+ run files",
        "repro_cmd": "PYTHONPATH=src python benchmarks/xval_dem_round2_deposition_rigid_limit.py (verified)",
        "external_anchor": "Literature RCP 0.6366; OSS: LIGGGHTS 0.637, YADE 0.637, OHern2003 0.639, Donev2004 0.6405"
    }',
    'board-audit'
);
