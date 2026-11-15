# Maturity Gate Evidence - Mission m17721

## L2 Requirements (Independent Function)

### ✅ impl_files (Non-placeholder Implementation)
**Location**: `src/tensordem/contact/rolling_static.py`

| Module | Lines | Purpose |
|--------|-------|---------|
| `rolling_static.py` | 156 | Coulomb + viscous rolling laws, COM lever, saturation, stopping speed |
| `clump_contact.py` | ~200 | ContactParams with rolling_model/rolling_lever fields |

**Evidence**:
```bash
$ wc -l src/tensordem/contact/rolling_static.py
156 src/tensordem/contact/rolling_static.py
```

All functions are pure, tested, and non-placeholder:
- `rolling_lever_arm()` — moment arm calculation (member vs COM)
- `rolling_limit()` — Coulomb torque limit
- `rolling_torque()` — torque computation (viscous or Coulomb)
- `rolling_saturation()` — fraction of limit delivered
- `stopping_speed()` — w_eps threshold

### ✅ entry (Entry Point)
**Location**: `scripts/run_xval_round2_method_selection.py`

**Usage**:
```bash
python scripts/run_xval_round2_method_selection.py [--outdir DIR]
```

**Output**:
- `artifacts/xval_round2_method_selection/xval_report.json` (11 KB)
- `artifacts/xval_round2_method_selection/saturation_curves.png` (76 KB)
- `artifacts/xval_round2_method_selection/xval_comparison.png` (44 KB)

### ✅ run_artifacts (Real Run Artifacts)
**Location**: `artifacts/xval_round2_method_selection/`

| Artifact | Size | Content |
|----------|------|---------|
| `xval_report.json` | 11 KB | Full XVAL report with 6 criteria, benchmark data, OSS comparison |
| `saturation_curves.png` | 76 KB | Coulomb vs viscous saturation curves vs speed |
| `xval_comparison.png` | 44 KB | OSS feature comparison bar chart |

## L3 Requirements (Integration)

### ✅ integration_callers (Main solver/platform calls)

| Caller | File | Lines | Usage |
|--------|------|-------|-------|
| clump_dem.py | `src/tensordem/particle/clump_dem.py` | 63, 313, 319, 378, 383 | `rolling_static.rolling_lever_arm()`, `rolling_static.rolling_torque()` |
| dem3d.py | `src/tensordem/dem3d.py` | 720, 1495 | Torque accumulation |
| repose_protocol_v2.py | `benchmarks/gap_dem_clump/` | import | Benchmark protocol |
| run_round2_scan.py | `benchmarks/gap_dem_clump/` | import | Shape-effect scan |
| test_rolling_static.py | `tests/` | 13 tests | Unit verification |

**Evidence** (grep):
```bash
$ grep -n "rolling_static\|rolling_lever_arm\|rolling_torque" src/tensordem/particle/clump_dem.py
63:from ..contact import rolling_static
313:            lever_p = rolling_static.rolling_lever_arm(...)
319:            tau_roll = rolling_static.rolling_torque(...)
378:                lever_w = rolling_static.rolling_lever_arm(...)
383:                tau_roll = rolling_static.rolling_torque(...)
```

## L4 Requirements (Benchmark + Reproducibility)

### ✅ benchmark (ref vs got numerical comparison)

| Criterion | Reference | Got | Error | Pass? |
|-----------|-----------|-----|-------|-------|
| (a) Sphere baseline | com == member | 0.0 error | 0.0 | ✅ |
| (b) Coulomb high speed sat | 1.0 | 1.0 | 0.0% | ✅ |
| (c) Coulomb low speed sat | 0.0 | 5.6e-6 | <0.01% | ✅ |
| (d) Viscous at eps_w | 0.5 | 0.5 | 0.0% | ✅ |
| (e) Stopping speed | analytical | 180.0 | 0.0% | ✅ |
| (f) Clump anisotropy | >2.0 | 3.0 | — | ✅ |

**All 6/6 criteria pass.**

### ✅ repro_cmd (Reproducible command)

```bash
cd /root/tp-agent/work/m17721
python scripts/run_xval_round2_method_selection.py
```

Produces identical artifacts on re-run.

## Test Summary

| Test File | Cases | Status |
|-----------|-------|--------|
| `test_rolling_static.py` | 13 | ✅ All pass |

```bash
$ python -m pytest tests/test_rolling_static.py -v
======================== 13 passed, 1 warning in 7.64s ========================
```

## Maturity Level: L4 ✅

- ✅ **impl_files**: 156 lines of real implementation (rolling_static.py)
- ✅ **entry**: `scripts/run_xval_round2_method_selection.py`
- ✅ **run_artifacts**: JSON report + 2 figures
- ✅ **integration_callers**: clump_dem.py, dem3d.py, benchmarks, tests
- ✅ **benchmark**: 6 criteria with ref vs got numerical comparison
- ✅ **repro_cmd**: Single command produces all artifacts
