# TensorDEM Seepage–Piping Initiation Criterion Closed-Loop Verification

> **Evidence grade (revised, mission m16381).** `src/tensordem/seepage/` is an
> *analytic framework*: it contains no particle–fluid coupling. The C1–C3
> "measured" inputs in the test suite are therefore **synthetic surrogates**
> (the analytic reference times a typed factor / a seeded noise draw), not DEM
> measurements. Their grade is **WITHHELD** (no measurement source exists in this
> repository) and the rows below are relabelled accordingly; C4–C5 are analytic
> identities and bookkeeping, graded **DERIVED**. This page previously reported
> C1–C3 as passing "DEM-measured" checks with 5.85 % / 5.00 % / 9.09 % errors —
> those numbers are properties of the constants typed into the test, not of the
> code (defect F1 of `audit_report.md`, closed here; machine check:
> `tools/self_verification_scan.py`, see `docs/SELF_VERIFICATION_SCAN.md`).

## 1. Case Introduction

### Physical Background
Seepage flow through saturated granular soils generates pore water pressures that reduce the effective stress carrying capacity of the soil skeleton. When upward seepage becomes sufficiently strong, the **critical hydraulic gradient** is reached and the soil loses all shear strength — the onset of **piping** (boiling/quick condition). For gap-graded soils, **suffusion** (internal erosion of fine particles) can initiate at gradients significantly below the Terzaghi piping threshold.

### Geometry & Parameters
| Parameter | Value | Unit |
|---|---|---|
| Column height | 0.10 | m |
| Particle diameter | 0.20 | mm |
| Solid density (quartz) | 2650 | kg/m³ |
| Porosity | 0.40 | — |
| Fine fraction | 0.20 | — |
| Water density | 1000 | kg/m³ |
| Water viscosity | 1.0×10⁻³ | Pa·s |

### Acceptance Criteria
| Criterion | Description | Threshold | Evidence grade |
|---|---|---|---|
| C1 | Pore pressure vs analytic hydrostatic | < 10% | **WITHHELD** (surrogate input) |
| C2 | Critical gradient vs Terzaghi i_c=(G_s-1)/(1+e) | < 15% | **WITHHELD** (surrogate input) |
| C3 | Permeability vs Kozeny-Carman | < 20% | **WITHHELD** (surrogate input) |
| C4 | Suffusion trend vs Skempton-Brogan | Trend consistent | **DERIVED** (analytic identity) |
| C5 | Mass conservation (solid+liquid) | < 2% | **DERIVED** (bookkeeping) |
| C6 | pytest ≥ 12 all green | 28 passed | REAL (regression surface) |

## 2. Methodology

### Link 1: Effective Stress Principle
Terzaghi's effective stress principle σ' = σ − u is verified by comparing the DEM-measured pore pressure profile against the analytic hydrostatic solution u(z) = γ_w·z for a saturated column with top drainage. The SaturatedColumn class computes γ_sat, γ', and the full stress profile.

### Link 2: Critical Hydraulic Gradient
The Terzaghi critical gradient i_c = γ'/γ_w = (G_s−1)/(1+e) is the condition where seepage force equals submerged weight. The PipingAnalysis class computes i_c, factor of safety FS = i_c/i, and seepage force per volume j = i·γ_w.

### Link 3: Kozeny-Carman Permeability
The Kozeny-Carman equation k_int = φ³d²/(180(1−φ)²) relates intrinsic permeability to porosity and grain size. Hydraulic conductivity K = k_int·γ_w/μ_w. The KozenyCarmanModel class supports porosity sweeps and comparison with measured values.

### Link 4: Suffusion Internal Erosion
Skempton & Brogan (1994) showed suffusion initiates at i_c,suff = α·i_c,Terzaghi where α depends on fine fraction F. For F < 0.35: α = 0.3 + 0.7F; for F ≥ 0.35: α = 1.0. Erosion rate follows dm_f/dt = C_e·(i − i_c,suff)^β.

### Link 5: Mass Conservation
Total mass audit: initial_total = final_solid + final_fluid + fine_mass_lost. Error must be < 2%.

## 3. Results

### C1: Pore Pressure Profile — **WITHHELD**
| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| Max deviation of the surrogate | 5.85% | 10% | judging logic OK · **evidence WITHHELD** |

The test input is the analytic hydrostatic profile multiplied by `(1 + 0.05·N(0,1))`
with a seeded generator (`tests/test_seepage_piping.py`, declared in-band as
`self-verification: synthetic`). The 5.85 % is the magnitude of the draw, not a
physical discrepancy: it would be 5.85 % for any implementation of the chain,
including a broken one. No DEM or experimental pore-pressure profile exists in
this repository.

### C2: Critical Hydraulic Gradient — **WITHHELD**
| Metric | Value | Threshold | Verdict |
|---|---|---|---|
| i_c (Terzaghi) | 0.990 | — | analytic reference |
| i_c (surrogate) | 1.040 | — | typed `×1.10` |
| Deviation | 5.0% | 15% | judging logic OK · **evidence WITHHELD** |

For quartz sand (G_s = 2.65, n = 0.40) Terzaghi gives i_c = 0.99; the test feeds
`i_c_analytic × 1.10`, so the reported 5.0 % deviation is the typed offset. The
negative control (`×1.25` → FAIL) shows the gate reacts, nothing more.

### C3: Kozeny-Carman Permeability — **WITHHELD**
| Porosity | k_surrogate (m/s) | k_KC (m/s) | Deviation |
|---|---|---|---|
| 0.35 | 6.76×10⁻⁶ | 6.15×10⁻⁶ | 9.1% |
| 0.40 | 1.18×10⁻⁵ | 1.08×10⁻⁵ | 9.1% |
| 0.45 | 2.01×10⁻⁵ | 1.82×10⁻⁵ | 9.1% |
| **Max** | — | — | 9.1 % is the typed `×1.15` · **evidence WITHHELD** |

Every column is `KozenyCarmanModel.conductivity(phi)` (the model under test)
multiplied by 1.15, so the deviation is by construction constant in porosity.
The K-C formula itself is standard textbook physics and is separately exercised
by unit tests; what was missing was a *measurement*, and it still is.

### C4: Suffusion Trend
| Metric | Value | Pass |
|---|---|---|
| i_c,suffusion | 0.475 | — |
| Below threshold: zero erosion | ✅ | ✅ |
| Above threshold: positive erosion | ✅ | ✅ |
| Monotonically increasing | ✅ | ✅ |
| Trend consistent | ✅ | ✅ |

The suffusion threshold (i_c,suff = 0.475) is significantly lower than the Terzaghi piping gradient (i_c = 0.99) due to the fine fraction F = 0.20 giving α = 0.48.

### C5: Mass Conservation
| Metric | Value | Threshold | Pass |
|---|---|---|---|
| Initial total | 15.0 kg | — | — |
| Final total (accounted) | 15.0 kg | — | — |
| Mass error | 0.0% | 2% | ✅ |

### C6: Test Suite
28 pytest cases, all passed.

## 4. Figures

See artifacts/seepage_piping/ for:
- fig1_pore_pressure.png: Pore pressure profile comparison
- fig2_critical_gradient.png: Critical gradient vs porosity
- fig3_kozeny_carman.png: Permeability vs porosity
- fig4_suffusion.png: Suffusion erosion rate vs gradient

## 5. Limitations & Notes

1. **Analytic reference**: All four links use closed-form analytic solutions as reference — no experimental data was used. The verification chain validates the *mathematical consistency* of the seepage-piping framework.

2. **DEM coupling**: The current implementation provides the analytic reference framework. Full DEM particle-fluid coupling (where DEM particles experience seepage forces and fluid flow is solved on the pore network) would be the next step for end-to-end validation. **Consequence (added m16381)**: until that coupling exists, C1–C3 cannot be graded REAL; the test suite must keep its surrogate inputs declared, and `tools/self_verification_scan.py` fails the merge gate if an undeclared `measured_*` value is derived from the model under test.

3. **Suffusion model**: The Skempton-Brogan envelope is an empirical correlation. The erosion coefficient C_e and exponent β are calibrated parameters, not first-principles predictions.

4. **Monodisperse assumption**: Kozeny-Carman is applied to monodisperse spheres. Real soils have grain size distributions that affect both permeability and suffusion behavior.

5. **Steady-state only**: All analyses assume steady-state seepage. Transient seepage (consolidation) is handled by the existing consolidation module.

## 6. Error Summary

| Criterion | Error | Threshold | Verdict | Evidence grade |
|---|---|---|---|---|
| C1 | 5.85% (surrogate draw) | < 10% | gate reacts as designed | **WITHHELD** |
| C2 | 5.00% (typed ×1.10) | < 15% | gate reacts as designed | **WITHHELD** |
| C3 | 9.09% (typed ×1.15) | < 20% | gate reacts as designed | **WITHHELD** |
| C4 | Trend consistent | — | analytic identity holds | DERIVED |
| C5 | 0.00% | < 2% | bookkeeping identity | DERIVED |
| C6 | 28/28 passed | ≥ 12 | suite green | REAL |

**Overall: framework consistency SATISFIED (C4–C6); measurement validation
WITHHELD for C1–C3.** *Is the error < 3 %?* The question is not applicable to
C1–C3: their "error" is a constant typed into the test, not an agreement with any
physical measurement. No external (experimental or DEM) benchmark for this case
exists in the repository, so no <3 % claim is made — and none should be.

Reproduce:

```bash
PYTHONPATH=src python -m pytest tests/test_seepage_piping.py -q      # 28 passed
PYTHONPATH=src python tools/self_verification_scan.py                 # exit 0
```
