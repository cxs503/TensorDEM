# CASE-TRIAXIAL-CD: drained (CD) triaxial friction angle (TensorDEM)

Engine: `tensordem.triaxial_cd` (driver) on the granular kernel of `tensordem.friction_surface` (Cundall-Strack friction + rolling resistance, Love-Weber stress). Artifacts: `artifacts/case_triaxial_cd/*.json` (one file per point), `artifacts/case_triaxial_cd/fig_case_triaxial_cd.png`.

## 1. Problem and reference

A drained triaxial test on a cohesionless soil measures the Mohr-Coulomb friction angle from the peak stress state, `sin(phi) = (sigma1 - sigma3)/(sigma1 + sigma3)` with `c = 0`. The two boundary conditions that define *drained* are reproduced here: (i) a dry assembly, so no pore pressure can develop, and (ii) a lateral **stress-controlled** (periodic barostat) boundary that holds `p_lat = (sigma_xx + sigma_yy)/2 = sigma3` while the specimen is free to change volume, and (iii) an axial platen at a constant strain rate.

**Reference.** The task fixes the literature sand friction-angle domain **30-40 deg** (mission brief CASE-TRIAXIAL-CD, 2026-11-12; standard handbook domain for sands). The original handbook table page could **not** be re-read inside this sandbox: network egress is closed (`urllib` -> `Errno 99 Cannot assign requested address`) and no local handbook copy is mounted. The comparison therefore uses, as declared *a priori*, the band edges `[30, 40]` and the band midpoint `35.0 deg`; no point value is hand-copied from a source. Criterion: phi_peak must lie inside the band, and its relative deviation from the band midpoint must be <= 10%.

## 2. Method and implementation

1. **Specimen preparation** (`lattice_gap` = 0.15, `jitter` = 0.3): a loose, jittered HCP lattice is compacted **isotropically** to `sigma3` by the lateral barostat plus the axial platen pressure servo. The large volumetric compaction (solid fraction 0.467 -> 0.556) destroys the lattice registry, so the sheared specimen is a disordered jammed packing, not a crystal.
2. **Consolidation gate**: the run only enters shear when `|p_lat - sigma3|/sigma3 <= 0.03` and the same for `sigma_zz`, sustained over consecutive checks.
3. **Shear**: kinematic axial platen at `eps1_dot = 0.50 1/s` up to `eps1 = 0.25`; the barostat keeps running (drained). Stress state from the Love-Weber sum over particle pairs **plus the wall contacts** (the wall branch is `signed*n`, exact for a plane wall); the platen reaction is an independent cross-check.
4. **Reduction**: `phi_peak` at the peak of `q = sigma1 - sigma3` after `eps1 > 0.01`; `phi_crit` as the mean over the last 20% of the shear history; dilatancy from the volumetric strain history.

## 3. Parameters (all dimensionless numbers reported)

| quantity | symbol | value |
|---|---|---|
| interparticle friction | mu_p | 0.50 |
| rolling resistance | mu_r | 0.25 |
| normal stiffness / (E d) scale | kn | 1.00e+06 N/m |
| tangential / normal stiffness | kt/kn | 0.50 |
| particle radius | r | 5.0e-03 m |
| particle density | rho | 2650 kg/m3 |
| restitution | e | 0.30 |
| particles | N | 360 |
| initial solid fraction | phi_s0 | 0.467 |
| solid fraction at peak | phi_s,p | 0.555 |
| timestep | dt | 8.275e-06 s |
| axes ratio H/W | - | 1.42 |
| inertial number | I | 8.14e-04 |

Confining-pressure ladder (all with mu_p = 0.5, mu_r = 0.25):

| tag | sigma3 [kPa] | I [-] | dt [s] | steps | N | phi_s at peak |
|---|---|---|---|---|---|---|
| ladder_s50000 | 50 | 1.2e-03 | 8.27e-06 | 100424 | 360 | 0.555 |
| ladder_s100000 | 100 | 8.1e-04 | 8.27e-06 | 100424 | 360 | 0.555 |
| ladder_s200000 | 200 | 5.8e-04 | 8.27e-06 | 100424 | 360 | 0.564 |

## 4. Reference vs measured (per-point, with pass marks)

| tag | sigma3 [kPa] | phi_peak [deg] | phi_crit [deg] | in band [30,40] | dev. from midpoint | <10% | drained gate | sigma3 drift |
|---|---|---|---|---|---|---|---|---|
| ladder_s50000 | 50 | 34.32 | 29.08 | OK | -1.9% | PASS | OK | 7.2% |
| ladder_s100000 | 100 | 33.42 | 28.79 | OK | -4.5% | PASS | OK | 6.2% |
| ladder_s200000 | 200 | 32.60 | 27.40 | OK | -6.8% | PASS | OK | 5.1% |

**Headline** (ladder_s100000, sigma3 = 100 kPa): phi_peak = **33.42 deg**, phi_crit = 28.79 deg, in band = OK, deviation from band midpoint = -4.5%, drained-gate drift = 6.2%; platen cross-check (sigma1 interior vs platen reaction) = 4.57%.

Sensitivity and convergence points:

| tag | swept value | phi_peak [deg] | phi_crit [deg] | open |
|---|---|---|---|---|
| conv_rate_0p25 | mu_p=0.50, mu_r=0.25, rate=0.25, N=360 | 32.65 | 29.05 | PASS |
| conv_size_n96 | mu_p=0.50, mu_r=0.25, rate=0.50, N=96 | 35.33 | 31.14 | PASS |
| sens_mu_0p30 | mu_p=0.30, mu_r=0.25, rate=0.50, N=360 | 31.36 | 25.06 | FAIL |
| sens_mu_0p70 | mu_p=0.70, mu_r=0.25, rate=0.50, N=360 | 35.95 | 32.48 | PASS |
| sens_mur_0p10 | mu_p=0.50, mu_r=0.10, rate=0.50, N=360 | 28.46 | 22.03 | FAIL |
| sens_mur_0p40 | mu_p=0.50, mu_r=0.40, rate=0.50, N=360 | 35.92 | 29.05 | PASS |

## 5. Convergence and parameter sensitivity

* **Pressure independence (Mohr-Coulomb consistency)**: the ladder fits `sigma1 = a*sigma3 + b` with a = 3.251, b = 18.3 kPa (R^2 = 0.9999), i.e. an envelope friction angle of 31.97 deg for c = 0. The peak friction angle spread across the ladder is 1.72 deg (5.1% of the mean), which is the drained Mohr-Coulomb consistency check.
* **Interparticle friction mu_p**: mu_p=0.30 -> 31.36 deg; mu_p=0.70 -> 35.95 deg (span 4.59 deg).
* **Rolling resistance mu_r** (angularity proxy): mu_r=0.10 -> 28.46 deg; mu_r=0.40 -> 35.92 deg (span 7.47 deg).
* **Strain-rate convergence**: rate=0.25 1/s (I=4.1e-04) -> 32.65 deg; rate=0.50 1/s (I=8.1e-04) -> 35.33 deg.
* **Specimen-size convergence**: N = 96 -> 360 particles changes phi_peak by -1.91 deg (-5.7%).

## 6. Failure modes, limitations and artifacts

* **Sphere-to-sand gap.** With mu_p = 0.5 and mu_r = 0.25 the peak angle (33.42 deg) is in the literature band, but the critical-state angle (28.79 deg) is below it: spherical grains mobilise strength through interlocking/dilatancy at the peak, while the critical state of spheres is intrinsically below angular-sand values. Reporting only the peak would overstate the material's residual strength -- both numbers are reported for this reason.
* **Boundary-layer stress closure (defect found and fixed in this node).** The pair-only Love-Weber sum under-predicted the axial stress by 16% at 6 layers (a boundary layer), which inflated the apparent drained-gate error. The wall branch `signed*n` is now added; the residual platen-vs-interior difference at the peak is 4.57%.
* **Periodic lateral boundary.** The lateral barostat applies an affine rescaling of the cell, the standard cheap barostat; it is not a Parrinello-Rahman deformation, so contact-history re-mapping is approximate at the moment the cell changes size.
* **Specimen size.** 360 particles is small for a strength envelope; the size-convergence point quantifies the resulting bias (see 5.).
* **Artifacts / reproduction**: `bash benchmarks/case_triaxial_cd/run_all.sh artifacts/case_triaxial_cd` runs the whole matrix and aggregates it; every point JSON carries its own `repro_cmd`, the full sample history and the gate records.

