-- Verdict for m20585 CASE-DEM-SILO-BEVERLOO (diagnostic/research mission)
-- Audit ID: 20620, Verdict by: board-audit, Date: 2026-11-14
-- Level: L0 (文档成果：诊断报告+分析脚本，无计算结果)

INSERT INTO tensor_maturity_verdicts (
    repo, scope, mission_id, level, evidence, verdict_by
) VALUES (
    'TensorDEM',
    'm20585 CASE-DEM-SILO-BEVERLOO',
    20620,
    0,
    '{
        "impl_files": [
            "benchmarks/m20585_beverloo/analyse.py:211 lines (analysis script, never executed)",
            "benchmarks/m20585_beverloo_kn_sensitivity/run_kn_sweep.py:168 lines (kn sweep script, never executed)",
            "docs/CASE-DEM-SILO-BEVERLOO_m20585.md:166 lines (diagnostic report, uncommitted)",
            "docs/oss_solutions_CASE-DEM-SILO-BEVERLOO.json:92 lines (oss solutions, uncommitted)"
        ],
        "entry": "No entry point executed; cluster jobs cj2798/cj2800/cj2811/cj2812 failed (SSH timeout/queuing)",
        "run_artifacts": "NONE - zero JSON results, zero figures, zero summary; mission report admits ''本单集群计算因 SSH 超时/排队未产出完整结果''",
        "tests": "NONE specific to m20585",
        "integration_callers": "NONE - scripts never executed",
        "benchmark": {"anchor": "CASE-DEM-SILO-BEVERLOO", "ref_kg_s": null, "got_kg_s": null, "rel_pct": null, "verdict": "NO RESULT from m20585; report cites m20055 D4 (+25.0% FAIL) and m15259 (-0.4% PASS) from other missions", "registered_referee": "NONE (0 rows in tensor_benchmark_cases)"},
        "repro_cmd": "Scripts exist but never executed: (1) PYTHONPATH=src python benchmarks/m20585_beverloo_kn_sensitivity/run_kn_sweep.py; (2) PYTHONPATH=src python benchmarks/m20585_beverloo/analyse.py",
        "external_anchor": ["Beverloo et al. 1961 Chem. Eng. Sci. 15, 260-269", "Nedderman 1992 Granular Materials §5.3", "m20055 D4 +25.0% FAIL (kn=1e5)", "m15259 -0.4% PASS (kn=1e6)"]
    }',
    'board-audit'
)
ON CONFLICT (repo, scope, verdict_by) DO UPDATE
  SET mission_id = EXCLUDED.mission_id, level = EXCLUDED.level,
      evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL,
      gate_note = NULL;
