# MATURITY_EVIDENCE — m17960 (TensorDEM)

Anchor: `XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT` · rework round 2 of m17806/m17871
Branch: `agent/m17960`

## Why the previous rounds stopped at L1

Both predecessors reported artifacts, but `git status` was `?? artifacts/...`: the run
artifacts were **never committed**. The maturity gate resolves evidence paths inside the
ecosystem checkout / audit worktree, so untracked files count as missing
(`run_artifacts(工件缺失)`), leaving only `tests` → L1. This round commits every
deliverable to the branch.

## Evidence by gate key (machine-checkable pointers)

| key | pointer |
|---|---|
| `impl_files` | `src/tensordem/binding_contract.py` (`benchmark_physics_3d`, `free_flight_case_3d`, `isolated_3d_model`, `BENCHMARK_3D_ANCHORS`; 1426 lines, no placeholder) |
| `entry` | `scripts/run_binding_contract_3d_benchmark.py`; also `src/tensordem/cli.py` → `tensordem.cli binding-contract` (non-zero exit on any failed row) |
| `run_artifacts` | `artifacts/xval_library_binding_contract/benchmark_physics_3d.json` (15 rows, env, fingerprint), `.../benchmark_physics_3d_table.md`, `.../result.json` (XVAL round-1+2 report), `.../fig_benchmark_3d_convergence.png`, `.../fig_benchmark_3d_rows.png` |
| `tests` | `tests/test_binding_contract_3d_benchmark.py` (9 tests); `tests/test_binding_contract.py` (29 tests, regression) |
| `integration_callers` | `src/tensordem/dem3d.py:1808` (`load_state_dict` → `assert_entry_conforms(..., "IceDEM3D")`), `src/tensordem/dem.py:8` (2-D gate), `src/tensordem/cli.py:220` (`binding-contract` runs `benchmark_physics_3d`), `scripts/xval_library_binding_contract.py:303` |
| `benchmark` | `benchmark_physics_3d`: ref vs got per row, e.g. impulse-momentum `6.85296542e-03` vs `6.85296542e-03` (rel 2.53e-16), symplectic closed form rel 3.09e-13, continuum 7.14e-03 → 3.57e-03 → 1.79e-03 with order 1.0/1.0, energy drift 8.64e-03, Hertz force `737.165267 N` vs `737.165267 N` (0.0), restart 0.0 m; worst row 8.642e-03 ≪ 0.15 |
| `repro_cmd` | `python scripts/run_binding_contract_3d_benchmark.py` (~14 s, single core, no GPU) |
| `external_anchor` | `BENCHMARK_3D_ANCHORS` in `binding_contract.py`: Hairer/Lubich/Wanner, *Geometric Numerical Integration* (2006) §I.1 (semi-implicit Euler closed form); Hertz (1882) *J. reine angew. Math.* 92:156-171 (contact kernel); Newton's second law (impulse-momentum) |

## Checks actually run

```
python -m pytest tests/test_binding_contract_3d_benchmark.py -q   -> 9 passed (13.4 s)
python -m pytest tests/test_binding_contract.py -q                -> 29 passed (41.1 s)
python scripts/run_binding_contract_3d_benchmark.py               -> 15/15 rows MET, worst 8.642e-03, 13.5 s
python scripts/xval_library_binding_contract.py                   -> wrote result.json with benchmark_physics_3d section, 15.0 s
python -m tensordem.cli binding-contract --benchmark-energy-steps 200 \
       --benchmark-energy-steps-3d 200 --out <tmp>                -> verify=CONFORMS, probes=19/19, benchmark=11/11, benchmark_3d=15/15
```

All numbers above are taken from the committed JSON artifacts, not transcribed.

## Honest scope limits

- The rows anchor **binding semantics** (dt, clock, per-step load consumption, restart
  payload, Hertz constants, first-order consistency), not contact physics vs experiment.
- Energy-drift row is a bounded-stability monitor (2 % pre-registered), not an accuracy claim.
- The 3-D frozen surface has no `load_time` stamp, so the clock-guard path is only
  exercised in 2-D; the 3-D rows check the clock directly.
- Models are tiny (60 particles) by design: this is a contract anchor, not a convergence study.