# Maturity Evidence — m17871 XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT

## Anchor
`XVAL-DEM-LIBRARY-BINDING-FROZEN-CONTRACT`
Title: TensorPhysics 的 library-binding-frozen-contract 做法
Feature-ledger status: open | acceptance: <15%

## L2 Requirements

### ✅ impl_files (Non-placeholder Implementation)
| File | Lines | Purpose |
|------|-------|---------|
| `src/tensordem/binding_contract.py` | 584 | Frozen contract: schema, fingerprint, verify, probes, mutation matrix, substep rule, benchmark |

Key contents:
- `CONTRACT_SCHEMA` / `CONTRACT_VERSION` / `BINDING_CONTRACT` dict (frozen surface for IceDEM + IceDEM3D)
- `FROZEN_FINGERPRINT` (sha256 of the contract dict)
- `fingerprint()` — runtime hash; `verify()` — 42 introspection checks
- `probe_semantics()` — 10 behavioural probes (impulse-momentum, load non-accumulation, etc.)
- `mutation_matrix()` — 7 drift classes (param narrowed/widened, method removed/renamed, etc.)
- `substep_rule_check()` — integer-multiple substep boundary
- `benchmark_physics()` — 11-row ref-vs-got numeric benchmark

### ✅ entry (Entry Point)
| File | Lines | Purpose |
|------|-------|---------|
| `scripts/xval_library_binding_contract.py` | ~350 | Full XVAL run: verify + probes + mutations + substep + impulse + discriminator + baseline + restart + coverage + figures |

Usage:
```bash
PYTHONPATH=src python3 scripts/xval_library_binding_contract.py
```

### ✅ run_artifacts (Real Run Artifacts)
Produced by running the entry script:
```
artifacts/xval_library_binding_contract/
├── result.json              (19 KB — full XVAL report, all numbers programmatic)
├── benchmark_physics.json   (11-row ref-vs-got benchmark)
├── verify_report.json       (42/42 checks CONFORMS)
├── fig_impulse_error.png    (47 KB)
├── fig_mutation_matrix.png  (39 KB)
├── fig_coverage.png         (35 KB)
└── fig_substep_boundary.png (27 KB)
```

## L3 Requirements

### ✅ integration_callers (Main solver / platform side real calls)
| File | Line | Call |
|------|------|------|
| `src/tensordem/dem.py` | 8, 430 | `from .binding_contract import assert_entry_conforms` → `assert_entry_conforms(self, "IceDEM")` in `exchange_state()` |
| `src/tensordem/dem3d.py` | 1808, 1810 | `from .binding_contract import assert_entry_conforms` → `assert_entry_conforms(self, "IceDEM3D")` in `exchange_state()` |
| `src/tensordem/cli.py` | 220-236, 341-348 | CLI subcommand `binding-contract` with `--verify`, `--probes`, `--benchmark` |

## Test Evidence
```
$ PYTHONPATH=src python3 -m pytest tests/test_binding_contract.py -v
29 passed, 1 warning in 25.53s
```

## Verdict Summary (all 8 criteria MET)
| Metric | Reference | Got | Tolerance | Met |
|--------|-----------|-----|-----------|-----|
| impulse-momentum rel error (max) | analytic n·dt·F | 2.66e-16 | 0.15 | ✅ |
| 3-D impulse-momentum rel error | analytic n·dt·F | 3.56e-16 | 0.15 | ✅ |
| ledger vs analytic impulse | analytic n·dt·F | 1.77e-16 | 0.15 | ✅ |
| native clock vs n·dt | analytic n·dt | 1.91e-16 | 0.15 | ✅ |
| restart drift | 0 m (bitwise) | 0.0 | 0.0 | ✅ |
| drift-detector classification | 7/7 | 1.0 | 1.0 | ✅ |
| semantic probe pass rate | all | 1.0 | 1.0 | ✅ |
| contract conformance | all frozen | 1.0 | 1.0 | ✅ |

## Benchmark (ref vs got, 11 rows, all pass)
| Row | Reference | Got rel error | Pass |
|-----|-----------|---------------|------|
| impulse_momentum_2d | analytic | 0.0 | ✅ |
| free_flight_vs_symplectic | semi-implicit Euler closed form | 6.77e-14 | ✅ |
| free_flight_vs_continuum_dt_x1 | continuum 0.5·a·t² | 0.0455 | ✅ |
| free_flight_vs_continuum_dt_x0.5 | continuum 0.5·a·t² | 0.0227 | ✅ |
| free_flight_vs_continuum_dt_x0.25 | continuum 0.5·a·t² | 0.0114 | ✅ |
| energy_drift_conservative | E(0) analytic | 0.0139 | ✅ |
| hertz_reduced_modulus | E/(2(1-ν²)) | 0.0 | ✅ |
| hertz_effective_radius | R/2 | 0.0 | ✅ |
| hertz_normal_force | (4/3)E*√R*·δ^1.5 | 0.0 | ✅ |
| clock_n_dt | n·dt | 0.0 | ✅ |
| restart_roundtrip_drift | bitwise 0 | 0.0 | ✅ |

Worst relative error: 4.55% (free_flight_vs_continuum, first-order semi-implicit Euler vs continuum — converges at order 1.0 as expected).
All rows < 15% owner tolerance.

## Repro Command
```bash
cd /root/tp-agent/work/m17871
PYTHONPATH=src python3 scripts/xval_library_binding_contract.py
PYTHONPATH=src python3 -m pytest tests/test_binding_contract.py -v
```
