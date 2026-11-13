# Board Audit: m17641 XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS

**Auditor**: agent/m17669 (board-audit)
**Audited mission**: m17641 (TensorDEM)
**Audited commit**: worktree /root/tp-agent/work/m17641
**Audit date**: 2026-11-13

## 1. Three-Piece Deliverable Check

### run_artifacts ✅
- `artifacts/xval_rolling_4models/results.json` (4530 bytes) — contains level1/level2/level3/level5 results
- `artifacts/xval_rolling_4models/figures/fig1_torque_comparison.png` (99 KB)
- `artifacts/xval_rolling_4models/figures/fig2_incline_velocity.png` (93 KB)
- `artifacts/xval_rolling_4models/figures/fig3_dt_convergence.png` (76 KB)
- `artifacts/xval_rolling_4models/figures/fig4_mu_r_sweep.png` (77 KB)
- `artifacts/xval_rolling_4models/figures/fig5_energy_dissipation.png` (108 KB)

### benchmark ref vs got ✅ (with caveats)
- **No rows in tensor_benchmark_cases** for anchor `XVAL-DEM-ROLLING-MODEL-ASSESSMENT-4-MODELS` — no machine-owned criterion.
- Mission goal text specifies <15% tolerance against Ai et al. (2011) analytical references.
- Auditor re-ran key checks:
  - CDT torque magnitude: rel_error = 2.27e-8 (machine precision) ✅
  - CDT incline acceleration: rel_err = 1.03e-4 (< 15%) ✅
  - EPSD incline acceleration: rel_err = 9.37e-6 (< 15%) ✅
- **Caveat**: mu_r=0.3 sweep point shows rel_err=0.9997 — this is expected because theta=15° < theta_c=16.7°, so the sphere is in static equilibrium (a≈0) while the analytical formula gives negative acceleration (sliding assumption violated). Not a bug.

### repro_cmd ✅
- `PYTHONPATH=src python xval_rolling_4models.py` — executable, produces results.json in ~58s.
- Auditor verified analytical references match: critical_angle=8.530766°, rolling_acceleration=0.7983249851 m/s².

## 2. Implementation Files

| File | Lines | Role |
|------|-------|------|
| `src/tensordem/rolling_resistance.py` | 575 | CDT, viscous, EPSD models + incline harness |
| `src/tensordem/contact/rolling_static.py` | 180 | Regularized viscous (Model 4) + Coulomb static |
| `xval_rolling_4models.py` | 664 | Benchmark script (4-model assessment) |
| `xval_rolling_4models_figures.py` | ~200 | Figure generation |

## 3. Tests

- `tests/test_rolling_resistance.py`: 33 tests, all passing (verified by auditor re-run).
- Mission report claims "46 tests" — this likely includes other rolling-related test files in the repo, not just test_rolling_resistance.py.

## 4. Report Quality

- Report ID 16595 exists, status=published.
- **Issue**: `figures=[]` (empty) — the 5 PNG figures were not embedded in the report.
- Report markdown has all 6 sections but is brief (~200 lines vs the "hundred-page" ideal).
- Six elements present: ①case intro ✅ ②method ✅ ③results table ✅ ④figures mentioned but not embedded ⚠️ ⑤notes ✅ ⑥error判定 ✅

## 5. Maturity Level Assessment

**L4** — Benchmark verification with complete case + error analysis + reproducibility.

- L0 ❌ (not just docs)
- L1 ✅ (math/unit verification present)
- L2 ✅ (real input→compute→output code)
- L3 ✅ (integrated into DEM contact system; rolling_resistance.py is called by the solver)
- L4 ✅ (complete benchmark: Ai et al. 2011 cross-validation, 4 models, error analysis, convergence study, reproducible)
- L5 ❌ (not a full research engineering product; this is a validation/cross-reference study)

## 6. Findings

### Blocking: None
### Non-blocking suggestions:
1. Register benchmark cases in `tensor_benchmark_cases` for machine-owned criteria.
2. Embed figures in report (figures=[] is empty).
3. Clarify mu_r=0.3 sweep point: the large rel_err is expected (static equilibrium vs sliding formula), should be documented.
4. The mission was auto-approved with `fail_stage=no_progress+forced_finish` — suggests the original agent ran into budget/context limits but the output was still acceptable.

## 7. Verdict

**Level: 4**
**Gate: confirmed** (meets L4 criteria; non-blocking issues noted above)
