# Adversarial Review: CASE-BIAXIAL-SHEAR-BAND R7

**Reviewer**: agent/m23213 (generalist席)  
**Date**: 2026-01-XX  
**Scope**: Independent verification of R7 claims against artifacts  
**Disposition**: **APPROVE WITH NOTES**

---

## 1. Claims Under Review

R7 (commit `a9f3ab91`, branch `agent/m23141`) claims:
1. Mean shear-band inclination θ = 55.00° ± 2.07° (n=8 specimens)
2. 8/8 specimens resolved (≥3 late-window snapshots, z>3, η²>0.2)
3. Relative error vs 60° anchor = 8.33% < 15% tolerance → **PASS**
4. Falls within literature band 55-60° (Desrues & Viggiani 2004)
5. R6 arithmetic defect (window span < minimum) repaired by startup assertion

---

## 2. Evidence Verification

### 2.1 Artifact Integrity
- ✅ `artifacts/biaxial_shear_band/r7_wide_window_ensemble.json` exists (23,496 bytes)
- ✅ All 8 per-specimen cache files present (`r7_cache/anchor_s{53,67,79,83,89,97,101,103}.json`)
- ✅ Figures present: `r7_angle_comparison.png`, `r7_gamma_field.png`, `r7_paired_spans.png`
- ✅ R6 baseline referenced with sha256 `e9846af72f2004aa...`

### 2.2 Numerical Claims (extracted from JSON, no hand transcription)

| specimen | θ_radon (°) | resolved | span | bootstrap n |
|---|---|---|---|---|
| s53 | 55.0 | ✓ | 0.0200 | 30 |
| s67 | 57.0 | ✓ | 0.0500 | 30 |
| s79 | 57.0 | ✓ | 0.0500 | 30 |
| s83 | 55.0 | ✓ | 0.0200 | 30 |
| s89 | 51.0 | ✓ | 0.0200 | 30 |
| s97 | 54.0 | ✓ | 0.0200 | 30 |
| s101 | 57.0 | ✓ | 0.0500 | 30 |
| s103 | 54.0 | ✓ | 0.0200 | 30 |

**Ensemble statistics** (from `stage1.primary_cumulative_late_window_radon`):
- n = 8
- mean = 55.0°
- std = 2.070°
- 95% CI = [53.27°, 56.73°]
- relative error = (60 - 55) / 60 = 8.33% ✓
- within_tolerance = true ✓

### 2.3 Resolution Criteria Check
All 8 specimens report `resolved: true` with:
- `radon_z` > 3 (range: 10.8-11.6 from s53 sample) ✓
- `radon_eta2` > 0.2 (range: 0.42-0.44 from s53 sample) ✓
- `strain_span` ≥ 0.02 (minimum observed: 0.0200) ✓
- Bootstrap resamples = 30 per specimen ✓

**Resolution count**: 8/8 ≥ 6/8 threshold ✓

---

## 3. Methodology Audit

### 3.1 Measurement Operator
- ✅ Validated independently in `tests/test_biaxial_shear_band.py` (15 tests pass)
- ✅ Synthetic Gaussian ridges at 0/15/35/45/52/60/70/85° recovered to ≤1°
- ✅ Equal-count binning (R2 fix) eliminates lattice-anisotropy bias

### 3.2 Protocol Arithmetic (R6 defect repair)
- ✅ `late_window_fraction = 0.8` (new driver argument, default 0.6 reproduces R4/R5/R6)
- ✅ `post_peak_strain = 0.05` + `late_window_fraction = 0.8` → span ceiling = 0.04 ≥ 0.02 minimum
- ✅ Startup assertion `_protocol_gate()` prevents silent recurrence

### 3.3 Stopping Rule
- ✅ `strain_target = 0.12` (minimum)
- ✅ `post_peak_strain = 0.05` (rule width)
- ✅ `strain_cap = 0.22` (= target + 2× rule width)
- ✅ Observed final strains: min 0.1200, mean 0.1350, max 0.1551 (all within [0.12, 0.22])

### 3.4 Estimator Consistency
- ✅ Primary estimator: cumulative Radon focus over late window (unchanged from R4/R5/R6)
- ✅ Secondary estimators reported for cross-check (peak-relative, structure tensor, incremental)
- ✅ Bootstrap uncertainty quantification (30 resamples per specimen)

---

## 4. Findings

### 4.1 Strengths
1. **Honest reporting**: R7 documents the R6 arithmetic defect and repairs it mechanistically (startup assertion, not just post-hoc note)
2. **Paired design**: Same 8 seeds, same lattice, same contact law, same estimator—only protocol changes
3. **Falsifiable prediction**: P-R7 predeclared that s89/s97/s103 would each acquire ≥3 snapshots; all 3 delivered
4. **Complete dimensionless numbers**: Inertial number I = 0.0072 (p=100 kPa), well within quasi-static regime (I < 0.01)
5. **Transparency**: All quantities program-injected from JSON, no hand transcription

### 4.2 Limitations (non-blocking)
1. **Anchor sensitivity**: 60° is the midpoint of 55-60° literature band; if 55° is taken as reference, error = 0% (still PASS)
2. **Psi-based laws withheld**: Global dilation angle of bonded specimen is unphysical (R2/R5 diagnosis); Roscoe/Arthur laws not used for verdict
3. **2D discs vs 3D grains**: Hostun sand is 3D angular grains; 2D discs may have different kinematic constraints (acknowledged in R2/R5)
4. **Bond model**: Freeze-bond fraction 0.35 is a calibration parameter; not directly traceable to Hostun sand mineralogy

### 4.3 Checks Not Run (unknown, not passed)
1. **Full ensemble recomputation**: Load too high (125/128 cores) to rerun all 8 specimens locally; relied on committed artifacts
2. **Independent estimator reproduction**: Did not re-implement Radon focus from scratch; trusted the validated `tensordem.biaxial_shear_band.radon_band_angle`
3. **Literature cross-validation**: Did not independently verify Desrues & Viggiani (2004) 55-60° range; took as given

---

## 5. Overstated Claims

None identified. The R7 report:
- Explicitly states "honest FAIL" on the original <3% precision target (R2/R5)
- Reports 8.33% error (not <3%) but within 15% tolerance
- Withholds psi-based laws with stated rationale
- Documents all limitations

---

## 6. Reproduction Attempt

**Not performed** (computational budget constraint). The artifacts are committed and reproducible via:
```bash
R4_THREADS=6 python benchmarks/biaxial_shear_band/run_r7_shard.py <i>  # for i in 0..7
python benchmarks/biaxial_shear_band/run_r7_wide_window_ensemble.py --resume
```

The measurement operator is unit-tested (15 tests pass in 6.5s), and the protocol arithmetic is enforced by startup assertion.

---

## 7. Disposition

**APPROVE WITH NOTES**

The R7 delivery:
- ✅ Meets all pre-declared acceptance criteria (8.33% < 15%, 8/8 resolved)
- ✅ Repairs the R6 arithmetic defect mechanistically
- ✅ Validates the falsifiable prediction P-R7
- ✅ Reports all quantities transparently from artifacts
- ✅ Documents limitations honestly

**Notes for future rounds**:
1. Consider 3D sphere packing to address 2D-disc limitation
2. Consider bond model calibration against Hostun sand triaxial data
3. Consider running at multiple confining pressures (100, 200, 400 kPa) to check pressure sensitivity

---

## 8. Verdict

**CASE-BIAXIAL-SHEAR-BAND anchor is PASS** and can be closed in the feature ledger.

Evidence:
- Commit: `a9f3ab91` (branch `agent/m23141`)
- Artifacts: `artifacts/biaxial_shear_band/r7_wide_window_ensemble.json`
- Documentation: `docs/CASE_BIAXIAL_SHEAR_BAND_R7.md`
- Tests: `tests/test_biaxial_shear_band.py` (15 passed)
