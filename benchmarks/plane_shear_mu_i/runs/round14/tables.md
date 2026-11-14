# CASE-DEM-PLANE-SHEAR-MUIDI — Round 14: Synthesis of 13 Rounds of Investigation

> **Anchor**: `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open)  
> **Repository**: TensorDEM  
> **Mission**: m20590  
> **Kind**: synthesis of R1-R13  
> **Artifacts**: `synthesis_r14.py`, `runs/round14/result.json`, `runs/phi_provenance_audit.json`, `runs/fig_mu_i_synthesis.png`, `runs/fig_phi_provenance.png`

## ① Case Introduction

Dense plane shear of frictional spheres in a constant-stress servo shear cell, confronted with the local rheology `mu(I)`, `phi(I)` of da Cruz et al. (2005, PRE 72:021309) and GDR MiDi (2004, Eur. Phys. J. E 14:341) over the inertial-number range `I = 1e-3 ... 0.3`.

The ledger acceptance criterion is: **<10% error on both mu(I) and phi(I) curves**.

## ② Method Summary

Over 13 rounds, the investigation progressed through:

- R1-R2: Initial sweep, discovery of phi defects (D1: porosity/solid fraction confusion; D2: phi reference sign error)
- R3-R5: Biaxial comparison, constant-stress validation, pressure control gate
- R6-R8: Bulk rheometry, full-cell virial stress estimator with analytic anchor
- R9-R10: Abscissa audit (I_target vs I_vol vs I_bulk)
- R11-R13: Rough wall mode, multi-seed ensemble, full-range comparison

Round 14 consolidates all findings into a single verdict.

## ③ Results

### mu(I) assessment

- **Sweep** (6 points): within 10% = False, worst error = 19.5%
- **Full-cell** (4 points): within 10% = False, worst error = 27.8%
- **Best single point**: I=0.01, mu=0.3751, err=-0.1%

| source | I | mu_meas | mu_ref | err [%] |
|---|---|---|---|---|
| sweep | 0.001007 | 0.3295 | 0.3616 | -8.9 |
| sweep | 0.003092 | 0.3289 | 0.3649 | -9.9 |
| sweep | 0.01085 | 0.3850 | 0.3768 | +2.2 |
| sweep | 0.03455 | 0.4074 | 0.4096 | -0.5 |
| sweep | 0.1179 | 0.4302 | 0.4954 | -13.2 |
| sweep | 0.3283 | 0.4916 | 0.6108 | -19.5 |
| full_cell | 0.01 | 0.3751 | 0.3755 | -0.1 |
| full_cell | 0.03 | 0.4090 | 0.4036 | +1.3 |
| full_cell | 0.1 | 0.3772 | 0.4800 | -21.4 |
| full_cell | 0.3 | 0.4333 | 0.6000 | -27.8 |

### phi(I) assessment

- **Sweep**: FAIL (phi_bed is constant = initial settled height, not steady-state)
- **Full-cell**: FAIL (phi_slab increases with I: compaction artifact from servo wall)
- **Provenance**: PROVENANCE-UNVERIFIED (no in-repo literature anchor; plausible for 3D glass beads)

## ④ Figures

- `runs/fig_mu_i_synthesis.png`: mu(I) comparison (simulation vs reference)
- `runs/fig_phi_provenance.png`: phi(I) comparison with both 3D and 2D references

## ⑤ Root Causes

- 1. Wall slip: smooth walls underestimate shear stress, especially at high I
- 2. Servo compaction: the constant-stress servo pushes the top wall down at high shear rate, artificially increasing phi
- 3. phi_bed is the initial settled height, not the steady-state sheared value (sweep data defect)
- 4. phi_slab estimator includes wall-affected regions, biasing the solid fraction
- 5. The reference phi magnitudes (0.64, 0.58) have no literature anchor
- 6. Particle count (N=294) is too small for proper bulk rheometry at high I

## ⑥ Ledger Verdict

- **mu(I) curve**: HONEST FAIL (sweep: worst -19.5%; full_cell: worst -27.8%; only I=0.01 full_cell is within 10%)
- **phi(I) curve**: WITHHELD (reference magnitudes unverified; measured phi shows wrong trend in both datasets)
- **Overall**: FAIL — the ledger acceptance (<10% on both mu and phi curves) is not met

## ⑦ Recommendations

- R1: Use rough walls (grafted monolayer) to suppress wall slip — already attempted in R13 but servo failed at high I
- R2: Fix phi_bed_steady in sweep data — the steady-state sheared height must be measured, not the initial height
- R3: Diagnose the servo compaction artifact — the top wall displacement should be tracked and corrected
- R4: Increase particle count (N > 1000) for proper bulk statistics
- R5: Anchor the phi reference to a specific 3D experimental source (Boyer et al. 2011, or Chialvo et al. 2012)
- R6: Re-scope the ledger acceptance: mu(I) <10% is achievable at low I; phi(I) needs a verified reference first

## ⑧ Round History

- **R1**: Initial sweep, mu within 10% at low I, phi defect D1 (porosity vs solid fraction)
- **R2**: Phi reference sign fix (D2), phi still unverified
- **R3**: Biaxial shear band comparison
- **R4**: Premise audit
- **R5**: Constant-stress audit, pressure control gate
- **R6**: Bulk rheometry introduction
- **R7**: Defect fixes (D1, D2), phi_reference_audit
- **R8**: Full-cell virial stress estimator, analytic anchor
- **R9**: Not archived in runs/
- **R10**: Abscissa audit (I_target vs I_vol vs I_bulk)
- **R11**: Rough wall mode introduced
- **R12**: Multi-seed ensemble (3 seeds)
- **R13**: Full-range rough vs smooth comparison, servo failures at high I rough
- **R14**: This synthesis — consolidated verdict, phi provenance audit, path forward
