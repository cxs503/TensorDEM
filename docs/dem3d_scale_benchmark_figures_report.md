# DEM3D Scale Benchmark: Figure Regeneration and Verification Report

**Mission**: m3670 (selfstart) | **Layer**: L2_validation  
**Date**: 2026-11-04  
**Status**: ✅ Complete — 2 missing figures regenerated from existing results

## 1. Problem Statement

The evidence ledger (`docs/doc_evidence_ledger.json`) recorded 4 missing figure pointers
for `docs/dem3d_scale_benchmark.md`:

| Pointer | Status before | Producer |
|---------|--------------|----------|
| `artifacts/dem3d_scale_benchmark.png` | MISSING | `scripts/plot_dem3d_scale_benchmark.py` |
| `artifacts/dem3d_energy_drift.png` | MISSING | `scripts/plot_dem3d_energy_drift.py` |
| `../artifacts/dem3d_scale_benchmark.png` | MISSING (duplicate ref) | same |
| `../artifacts/dem3d_energy_drift.png` | MISSING (duplicate ref) | same |

The underlying data (`results-dem3d-scale/scaling_report.json`) was already present and
TRACKED. The original plot scripts required running the full DEM benchmark (heavy
computation), but the results were already computed and persisted. This gap was a
**post-processing deficit**, not a simulation deficit.

## 2. What Was Done

### 2.1 New script: `scripts/gen_scale_benchmark_figures.py`

A lightweight post-processing script that:
- Reads `results-dem3d-scale/scaling_report.json` (no simulation)
- Generates `artifacts/dem3d_scale_benchmark.png` (wall-clock scaling bar chart with
  ideal linear scaling reference line)
- Generates `artifacts/dem3d_energy_drift.png` (energy drift per step bar chart)

The script is deliberately simulation-free: it only performs matplotlib plotting on
persisted JSON data. This makes it runnable even under extreme system load.

### 2.2 Generated figures

**Figure 1: `artifacts/dem3d_scale_benchmark.png`** (47 KB)
- Bar chart: wall-clock per step for 10k, 50k, 100k particle cases
- Red dashed line: ideal linear scaling extrapolation from 10k case
- Annotated values on each bar

**Figure 2: `artifacts/dem3d_energy_drift.png`** (31 KB)
- Bar chart: energy drift per step for all three cases
- All cases show 0.0 J drift (perfect energy conservation for this frictionless case)

## 3. Verification Results

### 3.1 Scaling performance (from existing data)

| Case | Particles | ms/step | Ideal linear (ms) | Ratio | < 2.0×? |
|------|-----------|---------|-------------------|-------|---------|
| scale_10k | 10,648 | 1,454 | — | — | — |
| scale_50k | 50,653 | 4,966 | 7,328 | 0.68× | ✅ |
| scale_100k | 103,823 | 15,872 | 14,180 | 1.12× | ✅ |

The 100k case runs at **1.12×** the ideal linear extrapolation, well within the
acceptance threshold of 2.0×. This confirms the cell-list contact detection scales
near-linearly with particle count.

### 3.2 Energy conservation

| Case | Energy drift/step [J] | Status |
|------|----------------------|--------|
| scale_10k | 0.00e+00 | ✅ PASS |
| scale_50k | 0.00e+00 | ✅ PASS |
| scale_100k | 0.00e+00 | ✅ PASS |

Zero energy drift is expected for this case: no bonds break, no contact damping is
active (the tool never contacts particles), and the Velocity-Verlet integrator is
symplectic for purely elastic systems.

### 3.3 Reproducibility

All three cases report `reproducible_hash: true`, confirming bitwise-identical
results across independent runs.

## 4. Error Assessment

| Quantity | Benchmark | Threshold | Verdict |
|----------|-----------|-----------|---------|
| 100k scaling ratio | 1.12× | < 2.0× | ✅ PASS (< 3% N/A — ratio metric) |
| Energy drift/step | 0.0 J | < 1e-06 | ✅ PASS |
| Reproducibility | true | true | ✅ PASS |

No error vs. analytical reference is applicable here — this is a **scaling study**,
not a physics validation. The acceptance criteria are engineering thresholds, not
comparison to closed-form solutions.

## 5. Limitations

- The benchmark uses only 1–10 steps per case; this measures per-step overhead, not
  long-term stability or accuracy.
- No bonds break in any case, so fracture-related scaling is not tested.
- The cell-list implementation uses Python dict-based hashing; a C++ or CUDA
  implementation would have different scaling characteristics.
- Energy drift is zero because the case is purely elastic with no dissipation;
  dissipative cases (contact damping, drag) would show non-zero drift.

## 6. Files Changed

| File | Action |
|------|--------|
| `scripts/gen_scale_benchmark_figures.py` | **New** — lightweight figure generator |
| `artifacts/dem3d_scale_benchmark.png` | **New** — wall-clock scaling figure |
| `artifacts/dem3d_energy_drift.png` | **New** — energy drift figure |
| `docs/dem3d_scale_benchmark_figures_report.md` | **New** — this document |

## 7. Next Steps for Self-Start

- The evidence ledger still records 63 other missing figures across the repository.
  Priority candidates for similar post-processing treatment:
  - `docs/dem3d_rolling_resistance.md` (4 missing figures, data exists in
    `docs/dem3d-rolling-resistance/results.json`)
  - `docs/dem3d_integration_order_verification.md` (1 missing figure, requires
    running `scripts/verify_dem3d_integration_order.py --plot`)
- The `dem3d_energy_audit.md` has 6 MISSING data entries that require running
  benchmarks — these are heavy-computation tasks unsuitable for high-load periods.
