-- Verdict for m21538 GAP-ICE-ONEWAY-CLAIM one-way DEM-FEM coupling verification
-- Audit ID: 21564, Verdict by: board-audit, Date: 2026-11-15
-- Level: L4 (基准验证：完整算例+误差分析+可复现)

INSERT INTO tensor_maturity_verdicts (
    repo, scope, mission_id, level, evidence, verdict_by
) VALUES (
    'TensorDEM',
    'm21538 GAP-ICE-ONEWAY-CLAIM',
    21564,
    4,
    '{
        "impl_files": "src/tensordem/dem_fem_gates.py:318L; src/tensordem/fem_euler_bernoulli.py:467L; examples/dem_fem_oneway/run.py:648L; examples/dem_fem_oneway/validate.py:81L; examples/dem_fem_oneway/independent_gates.py:362L; tests/test_dem_fem_independent_gates.py:152L",
        "entry": "examples/dem_fem_oneway/run.py; examples/dem_fem_oneway/independent_gates.py; examples/dem_fem_oneway/validate.py",
        "run_artifacts": "examples/dem_fem_oneway/results.json; artifacts/dem_fem_coupling_ledger.json; artifacts/dem_fem_coupling/independent_gates.json; docs/figures/dem_fem_independent_gates.png; docs/figures/dem_fem_dt_convergence.png",
        "tests": "tests/test_dem_fem_independent_gates.py: 11/11 passed (8.97s)",
        "integration_callers": "run.py imports IceDEM3D+.EulerBernoulliBeam+dem_fem_gates; independent_gates.py imports dem_fem_gates; tests import dem_fem_gates",
        "benchmark": "GAP-ICE-ONEWAY-CLAIM numerical_verification_only; 8 external gates all PASS worst|rel|=1.84%; 2 self_comparison gates PASS; dt convergence ratio=1.000; separability verified; energy ledger complete",
        "repro_cmd": "cd /srv/tensor/agent/work/m21538 && PYTHONPATH=src python examples/dem_fem_oneway/run.py && PYTHONPATH=src python examples/dem_fem_oneway/validate.py && PYTHONPATH=src python examples/dem_fem_oneway/independent_gates.py && python -m pytest tests/test_dem_fem_independent_gates.py -v",
        "external_anchor": "Hookean/Hertz impact closed forms (Johnson Contact Mechanics ch.4); Euler-Bernoulli beam natural frequencies ω_n=(nπ/L)²√(EI/ρA); modal static deflection PL³/48EI; continuum modal superposition (Duhamel) independent implementation"
    }',
    'board-audit'
);
