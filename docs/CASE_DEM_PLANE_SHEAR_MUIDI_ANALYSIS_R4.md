# CASE-DEM-PLANE-SHEAR-MUIDI — Round 4 Analysis: Error Diagnosis & Remediation Path

> **Anchor**: `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open)
> **Repository**: TensorDEM
> **Date**: 2026-11-12
> **Author**: agent/m16633 (selfstart paper seat)
> **Prior rounds**: R1 (m14960), R2 (m16427), R3 (m16497/m16568)

## ① Executive Summary

The plane-shear μ(I) rheology calibration has been through 3 rounds of work.
The μ(I) curve is reproduced within 10% tolerance at low-to-moderate inertial
numbers (I < 0.04) but **fails at high I** (errors of −13% to −20% at I > 0.1).
The φ(I) curve is **systematically ~10% low** across all I values because the
bed solid fraction is depressed by the thin-bed geometry (only ~6 particle
layers) and the sweep artifact is stale (lacks `phi_bed_steady`).

**Verdict**: The case is **not yet verified** against the acceptance criterion
(<10% error on both μ and φ across I = 1e-3..0.3). Two independent issues
contribute: (A) insufficient particle count / bed thickness causing boundary
artefacts, and (B) stale sweep data missing steady-state measurements.

## ② Current State (from sweep.json, programmatic read)

### 2.1 Device Configuration

| Parameter | Value | Note |
|---|---|---|
| P (normal stress) | 62,500 Pa | Servo-held |
| kn (contact stiffness) | 62,500 N/m | P/kn = 1.0 (soft contact) |
| μ_contact | 0.5 | Calibrated micro-friction |
| μ_rolling | 0.05 | Calibrated rolling resistance |
| e (restitution) | 0.3 | |
| r (particle radius) | 0.005 m | d = 10 mm |
| ρ (density) | 2500 kg/m³ | Glass beads |
| Box (W×D×H) | 0.12×0.05×0.06 m | 12d × 5d × 6d |
| N particles | 294 | **Very small system** |
| Shear strain | 0.2 | Total accumulated strain |
| Seed | 5279 | Fixed across all I targets |

### 2.2 Results: μ(I) Comparison

| I (measured) | μ (got) | μ (ref) | μ error | Pass (<10%)? |
|---|---|---|---|---|
| 1.01e-3 | 0.3295 | 0.3616 | **−8.9%** | ✓ (barely) |
| 3.09e-3 | 0.3289 | 0.3649 | **−9.9%** | ✓ (barely) |
| 1.08e-2 | 0.3850 | 0.3768 | **+2.2%** | ✓ |
| 3.46e-2 | 0.4074 | 0.4096 | **−0.5%** | ✓ |
| 1.18e-1 | 0.4302 | 0.4954 | **−13.2%** | ✗ |
| 3.28e-1 | 0.4916 | 0.6108 | **−19.5%** | ✗ |

### 2.3 Results: φ(I) Comparison

| I (measured) | φ (got) | φ (ref) | φ error | Pass (<10%)? |
|---|---|---|---|---|
| 1.01e-3 | 0.5218 | 0.5802 | **−10.1%** | ✗ |
| 3.09e-3 | 0.5218 | 0.5806 | **−10.1%** | ✗ |
| 1.08e-2 | 0.5218 | 0.5821 | **−10.4%** | ✗ |
| 3.46e-2 | 0.5218 | 0.5862 | **−11.0%** | ✗ |
| 1.18e-1 | 0.5218 | 0.5969 | **−12.6%** | ✗ |
| 3.28e-1 | 0.5218 | 0.6114 | **−14.6%** | ✗ |

**Critical observation**: φ (got) is **identical** (0.5218) for all 6 I values.
This is the settled-phase bed fraction (`phi_bed`), not the steady-state
shear-phase fraction (`phi_bed_steady`). The sweep.json was generated before
the `phi_bed_steady` field was added to `simulate_servo_shear`, so the φ(I)
curve is fundamentally broken — it shows no dilatancy because it uses a
frozen pre-shear measurement.

## ③ Root Cause Analysis

### 3.1 φ(I) Failure: Stale Artifact + Thin Bed

**Root cause 1 (data)**: The sweep.json artifact was produced by `run_sweep.py`
before the `phi_bed_steady` / `plate_height_steady` fields were added to
`simulate_servo_shear` (R2 fix). The test file has a fallback to `phi_bed`,
but `phi_bed` is the settled height *before* shear, which is I-independent
by construction.

**Root cause 2 (physics)**: Even with `phi_bed_steady`, the bed is only ~6
particle layers thick (fill_height = 0.06 m, d = 0.01 m). Prior work
(m14960) measured a 19% low packing at 3 layers. The thin bed amplifies
wall effects and prevents the development of a proper bulk shear band.

**Evidence**: The coordination number Z decreases from 4.02 to 3.30 across
the sweep, indicating the bed is under-coordinated (bulk random packing
typically has Z ≈ 6 for frictional spheres).

### 3.2 μ(I) Failure at High I: Boundary Effects + Soft Contact

**Root cause 1 (boundary)**: The box is only 5d wide in the gradient direction
(y). At high shear rates, the shear band occupies a significant fraction of
the box width, and the smooth side walls impose an artificial velocity profile.

**Root cause 2 (soft contact)**: P/kn = 1.0 means the contact stiffness equals
the applied pressure. This is extremely soft — particles overlap by ~5% of
their diameter under load. The Hertzian contact law is not used; the linear
spring-dashpot model with such soft contacts underestimates the collisional
stress contribution at high I.

**Root cause 3 (particle count)**: N = 294 is far too small for a statistically
representative granular flow. The reference simulations (da Cruz 2005) used
thousands of particles. Finite-size effects cause the measured μ to saturate
below the true asymptotic value.

### 3.3 Why μ is Low at Low I Too

At I = 1e-3, μ is 8.9% below the reference. This is because μ_contact = 0.5
was calibrated to match μ at I ≈ 0.01, but the μ(I) curve shape depends on
the interplay of friction, rolling resistance, and contact stiffness. The
calibrated parameters (μ_p = 0.5, μ_r = 0.05) give the right μ at one point
but the wrong curve shape.

## ④ Remediation Path

### 4.1 Immediate (low compute): Re-sweep with phi_bed_steady

Re-run the 6-point sweep with the current `simulate_servo_shear` code that
returns `phi_bed_steady`. This will fix the φ(I) curve measurement without
changing any physics. Expected outcome: φ(I) will show dilatancy (increase
with I), but the absolute values may still be low due to the thin bed.

**Estimated cost**: ~8 minutes CPU (6 points × ~80 s each).

### 4.2 Medium (moderate compute): Thicker bed + more particles

Increase the box depth from 5d to 10d and the fill height from 6d to 12d.
This increases N from 294 to ~2400. The `run_parallel.py` script already
supports parallel execution across I targets.

**Expected outcome**: φ(I) should approach the reference values (0.58–0.64);
μ(I) errors at high I should decrease due to reduced wall effects.

**Estimated cost**: ~60 minutes CPU (6 points × ~600 s each, parallelizable).

### 4.3 Long (high compute): Stiffness arm + system-size convergence

Run the `stiff` arm (kn = 10×, P/kn = 0.1) and a system-size convergence
study (N = 300, 1000, 3000, 10000) to establish the asymptotic μ(I) curve.

**Estimated cost**: Several hours CPU.

## ⑤ Recommendation for Next Round

**Priority order**:
1. **Re-sweep with phi_bed_steady** (R4a) — fixes the broken φ(I) measurement,
   low cost, high value. This alone will make the φ(I) curve meaningful.
2. **Thick bed re-sweep** (R4b) — addresses the thin-bed artefact, moderate
   cost. Should bring φ within 5% and μ within 10% at all I.
3. **Stiffness arm** (R4c) — tests whether soft contacts are the limiting
   factor for μ at high I.

**Acceptance criterion update**: The current criterion (<10% on μ and φ) is
appropriate. However, the φ reference constants (φ_c = 0.58, φ_j = 0.64) are
not from the cited da Cruz 2005 paper (which focuses on μ(I)); they are from
GDR MiDi 2004 but the exact values are not directly quoted in the open-access
version. This should be noted as a provenance limitation.

## ⑥ Artifact Inventory

| Artifact | Path | Status |
|---|---|---|
| Sweep data (stale) | `benchmarks/plane_shear_mu_i/runs/sweep.json` | Needs regeneration |
| Analysis R3 | `benchmarks/plane_shear_mu_i/analyse_r3.py` | Current |
| Test suite | `benchmarks/plane_shear_mu_i/test_plane_shear_mu_i.py` | Passes (with stale data) |
| Documentation R3 | `docs/CASE_DEM_PLANE_SHEAR_MUIDI_R3.md` | Current |
| This analysis | `docs/CASE_DEM_PLANE_SHEAR_MUIDI_ANALYSIS_R4.md` | **New** |

## ⑦ Error Budget Summary

| Quantity | Best error | Worst error | Target | Status |
|---|---|---|---|---|
| μ(I) at I < 0.04 | 0.5% | 9.9% | <10% | ⚠️ Marginal |
| μ(I) at I > 0.1 | 13.2% | 19.5% | <10% | ✗ Fail |
| φ(I) (all I) | 10.1% | 14.6% | <10% | ✗ Fail (stale data) |

**Overall**: CASE-DEM-PLANE-SHEAR-MUIDI is **not verified**. Two of six μ
points and all six φ points fail the <10% criterion. The φ failures are
partially artefactual (stale data); the μ failures at high I are physical
(system-size and boundary effects).
