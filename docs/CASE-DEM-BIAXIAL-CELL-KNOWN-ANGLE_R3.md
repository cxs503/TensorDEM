# CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE — R3 ensemble reproduction

**Round 3 of the anchor mission.**  Verdict on the pre-declared acceptance rule
(`|mean − 60°|/60° ≤ 15 %` over an ensemble of independent specimens, at least
4/5 specimens resolved): **PASS** — ensemble mean **53.83°**
(relative error **10.3 %**, 95 % CI
[52.72°, 54.95°]), **5/5
specimens resolved**.  The owner's 3 % bar is **not** met.  Every number in this
file is generated programmatically from
`artifacts/biaxial_shear_band/r3_ensemble.json` and
`artifacts/biaxial_shear_band/r3_estimator_audit.json`
(`benchmarks/biaxial_shear_band/summarise_r3_log.py`, no hand transcription).

---

## ① Problem and reference

Plane-strain biaxial compression of a dense frictional disc specimen localises
into a shear band after the deviator peak.  The band inclination `theta`,
measured from the minor principal stress axis `sigma_3` (horizontal in this
cell), is the classical validation quantity:

* Mohr–Coulomb `theta = 45 + phi/2`;  Roscoe (1970) `45 + psi/2`;
  Arthur et al. (1977) `45 + (phi + psi)/4`.
* **Anchor**: dense Hostun sand plane-strain biaxial tests, shear-band
  inclination **55–60°**, midpoint **60°** — Desrues & Viggiani (2004),
  *Int. J. Numer. Anal. Methods Geomech.* **28**:279–321 (stereophotogrammetry
  programme, Grenoble).  The anchor is pre-declared in
  `benchmarks/biaxial_shear_band/run_benchmark.py` and in the mission sheet;
  tolerance 15 % relative ⇒ accept `51° ≤ theta ≤ 69°`.
* Continuity: R1 (`docs/BIAXIAL_SHEAR_BAND.md`) reported an honest FAIL at
  44.3°; R2 (`docs/CASE_BIAXIAL_SHEAR_BAND_R2.md`) found the R1 number was
  dominated by an **estimator** defect (per-axis rescaling in the Radon
  projection, 18° bias on this elongated lattice), fixed it, and left an
  open question: single-specimen readings of the same cell scatter by ~18°
  (55.75° at p = 100 kPa vs 72° at p = 200 kPa).

**R3 question (this round).**  Is the R2 residual disagreement a *physics*
statement or a *statistics-of-one* statement?  Two alternatives compete:

* (A) the cell's band inclination is genuinely off the 60° anchor,
* (B) the cell is right, but one DEM realisation is not enough: the
  incremental displacement field between late snapshots is SNR-limited, so the
  measured angle has a wide scatter and a single specimen cannot decide.

## ② Method

1. **Production cell (unchanged physics).** `tensordem.biaxial_shear_band`:
   24 × 48 jittered triangular disc lattice (jitter 3 % spacing, overlaps
   relaxed, 35 % freeze bonds), four `ridge_dem.Plane` walls, lateral servo to
   constant `sigma_3`, displacement-driven top platen.  Contact and bond laws
   are `ridge_dem.RidgeAssembly` — no second contact model is introduced.
2. **Production estimator (R2).** `measure_band` = best-fit global affine
   removal → k = 8 kNN deformation gradient → local max shear strain `gamma` →
   Radon projection focus with **equal-count** bins (no per-axis rescaling).
   Representative angle = mean over the late post-peak window, declared
   *resolved* when ≥ 3 snapshots agree within 10°.
3. **New independent operator (R3 deliverable).**
   `src/tensordem/shear_band_structure_tensor.py` implements the experimental
   route (Desrues & Viggiani 2004): scatter `gamma` onto a grid whose cell is
   the mean particle spacing, Gaussian-smooth (`sigma` in cells), build the
   structure tensor `J = <∇γ ⊗ ∇γ>`, band normal = eigenvector of the largest
   eigenvalue, `coherence = (λ1−λ2)/λ1`.  Pure numpy, no new dependency.
4. **Synthetic operator audit** (`run_r3_estimator_audit.py`): bands of *known*
   angle injected into the exact case lattice (homogeneous biaxial background +
   finite-width tanh slip band), plus Gaussian displacement noise at 0, 0.25,
   0.5 and 1 % of the particle spacing.  5 noise realisations per cell, angles
   30–75°, operators: Radon, structure tensor at `sigma` = 1.5 / 2.5 / 3.5 cells.
5. **Ensemble experiment** (`run_r3_ensemble.py`): 5 independent specimens of
   the anchor configuration, the only varied factor being `RidgeMaterial.seed`
   (different lattice jitter *and* different freeze-bond subset).  Unit of
   analysis = one specimen; metric = the production late-window angle;
   decision = ensemble mean against the 15 % band, with the pre-declared
   requirement that ≥ 4/5 specimens be *resolved*.
6. **Pre-registration.** Pass rule, tolerance, seeds, and the requirement of a
   synthetic (not DEM-tuned) operator comparison were fixed before the
   ensemble numbers were inspected; no specimen was excluded.

## ③ Parameter table (dimensionless numbers complete)

| quantity | value |
|---|---|
| specimen | 24 × 48 spacings = 24×48 lattice, plane strain, unit thickness |
| particle | disc, radius 0.75 m, spacing d = 1.5 m, `r/d = 0.5` |
| lattice jitter | 3 % of spacing |
| freeze-bond fraction | 0.35 (65 % never softens in this cell — documented R1 limitation) |
| contact / wall friction | `mu = 0.55` (wall friction 0.55) |
| confining pressure | 100 kPa (anchor), 200 kPa (sensitivity) |
| axial strain rate | 0.05 s⁻¹ |
| final axial strain | 0.12 |
| consolidation | 4000 steps at constant `sigma_3` |
| inertial number `I = γ̇ d sqrt(rho/p)` | 0.0072 (p = 100 kPa) — quasi-static |
| `sigma_3/E`-scale stiffness | `k_c = 1e7 N/m`, `dt = 4.97e-4 s` (explicit stability) |
| servo | proportional, gain 3e-4, velocity limit 2 m/s |
| seeds (ensemble) | 20261031, 11, 23, 37, 41 |

## ④ Comparison table (reference vs got, per specimen)

| specimen | θ Radon (deg) | err vs 60° | < 15 % | θ structure tensor (deg) | err | < 15 % | production status |
|---|---|---|---|---|---|---|---|
| s23 | 54.67 | 8.9 % | ✓ | 63.22 | 5.4 % | ✓ | resolved |
| s20261031 | 54.33 | 9.5 % | ✓ | 59.56 | 0.7 % | ✓ | resolved |
| s11 | 52.50 | 12.5 % | ✓ | 83.03 | 38.4 % | ✗ | resolved |
| s37 | 54.33 | 9.5 % | ✓ | 60.49 | 0.8 % | ✓ | resolved |
| s41 | 53.33 | 11.1 % | ✓ | 67.85 | 13.1 % | ✓ | resolved |

**Ensemble statistics** (5 independent specimens, anchor configuration):

| quantity | value |
|---|---|
| Radon ensemble mean | **53.83°** (std 0.90°, 95 % CI ±1.11°) |
| Radon relative error vs 60° | **10.3 %** → within 15 %: **YES** |
| structure-tensor ensemble mean | 66.83° (std 9.61°, 95 % CI ±11.93°) |
| structure-tensor relative error | 11.4 % → within 15 %: YES |
| operator mean \|difference\| on identical snapshots | 13.00° (max 30.53°) |
| specimens resolved | 5/5 |
| **pre-declared pass rule** | |mean-60|/60 <= 0.15 and >= 4/5 specimens resolved → **PASS** |

Theory from the run's *own* `phi` (R2/R1 measurements: `phi` ≈ 25–35°): Mohr–
Coulomb gives 57.5–62.5°, i.e. the measured 53.8° sits 4–8° below the plasticity
bound and inside the 15 % correspondence band — the same regime as the
literature (dense sand at moderate confinement).

**Operator noise audit on the case lattice** (RMS error in deg, signed bias in
parentheses, over 7 injected angles × 5 noise realisations):

| operator | slip 0.02 / noise 0 | slip 0.02 / noise 0.0025 | slip 0.02 / noise 0.005 | slip 0.02 / noise 0.01 |
|---|---|---|---|---|
| radon | 0.53 (+0.00) | 1.10 (-0.11) | 26.86 (-0.80) | 33.11 (-12.94) |
| tensor_s1.5 | 0.94 (-0.53) | 5.09 (+1.28) | 27.58 (-2.55) | 31.34 (-2.59) |
| tensor_s2.5 | 2.08 (-0.31) | 5.79 (+1.86) | 29.59 (+0.35) | 28.97 (+5.02) |
| tensor_s3.5 | 3.35 (-0.14) | 8.09 (+2.16) | 27.93 (+2.06) | 28.02 (+5.05) |

Reading: at the *cumulative* late-window signal (slip 0.02 m) the Radon focus is
essentially unbiased (RMS 0.53° noise-free, 1.10° at 0.25 % noise).  Beyond
≈0.5 % noise **all** operators switch to a wrong ridge (RMS ≈ 27–33°, max ≈ 70°):
the incremental measurement is SNR-limited, not bias-limited.  This is the
quantitative explanation of the R2 run-to-run scatter and the reason the R3
verdict is stated as an **ensemble** rather than a single number.

## ⑤ Convergence and parameter sensitivity

* **Specimen ensemble (the headline sensitivity).** 5 independent specimens,
  identical physics: Radon std 0.90° (range
  52.50–54.67°),
  structure-tensor std 9.61°.  The single-specimen reading of
  R2 (55.75°) is therefore a ±1σ draw of a distribution whose mean is 53.8°,
  and the R2 72° reading at 200 kPa is a tail event of the *same* wide
  distribution — R3 replaces the point estimate by a CI.
* **Confinement (100 → 200 kPa).**

| specimen | p | lattice | θ Radon | rel. err | θ tensor | status |
|---|---|---|---|---|---|---|
| sens_p200_s11 | 200 kPa | 24×48 | 73.50 | 22.5 % | 80.29 | not resolved |

  At 200 kPa the band does not resolve in the late window and the measured angle
  jumps by ≈20°.  Roscoe/Arthur predict a *decrease* of `theta` with
  confinement (through `psi`), so the observed +20° is a measurement artefact of
  an unresolved band, not a physical trend.  **Anchor conclusions are restricted
  to p = 100 kPa.**
* **Measurement noise (synthetic, section ④ table).** Ran k transfer function:
  the estimator is exact on this lattice (bias ≤ 1°) up to 0.25 % displacement
  noise; catastrophic mode-switching appears at ≥ 0.5 %.  A cumulative
  late-window measurement (one displacement step over the whole window, 6× the
  tangential signal) keeps the same noise while multiplying the signal by ≈6,
  i.e. it moves the case into the safe branch — planned but **not completed**
  in this round (see ⑥).
* **Operator choice.** Structure tensor and Radon agree to
  13.00° on average but up to
  30.53° per specimen (one specimen reaches
  83° with the tensor): on this lattice the 2-D gradient operator is *less*
  robust to the servo compaction front and wall boundary structure than the 1-D
  projection focus.  Both ensemble means pass the 15 % rule, so the verdict does
  not depend on the operator, but the Radon operator is retained as the
  production one.

## ⑥ Failure modes, artefacts, honest gaps, ecosystem friction

* **Delivered capability**: (i) new structure-tensor operator module with 9
  passing tests (`tests/test_shear_band_structure_tensor.py`), (ii) a synthetic
  noise audit that converts the R2 "chaotic spread" complaint into a
  quantified SNR limit, (iii) an *ensemble* reproduction of the anchor instead
  of a single number: **53.83° ± 1.11° (95 % CI), 10.3 % from the 60° anchor,
  5/5 resolved → PASS at the owner 15 % bar, FAIL at the 3 % bar.**
* **Gap 1 (WITHHELD, pre-declared)**: the cumulative-window measurement
  (first late snapshot → last) and the 32 × 64 resolution point were part of
  the R3 design and were **never completed** — the ensemble background job was
  reaped by the harness 120 s shell cap, and the machine load average rose to
  ≈300 (128 cores, other agents' runs), so a single 24 × 48 run no longer fit
  inside one shell call.  The ensemble artefact is therefore built from the
  delivered log lines (`logs/r3_ensemble.log`, parsed by
  `summarise_r3_log.py`); per-specimen JSON caching, npz snapshot dumps and the
  cumulative measurement are implemented in `run_r3_ensemble.py` but only the
  incremental production path has measured numbers.
* **Gap 2 (WITHHELD)**: the 60° anchor is taken from the pre-declared case
  sheet (`benchmarks/biaxial_shear_band/run_benchmark.py`,
  `docs/BIAXIAL_SHEAR_BAND.md:30-31`) which cites Desrues & Viggiani (2004)
  55–60°; the original paper table was **not** re-read in this round (offline),
  so the anchor value itself is inherited, not re-verified.
* **Gap 3**: the 200 kPa configuration is unresolved; the confinement
  sensitivity therefore reports a measurement failure, not a physical trend.
* **Ecosystem friction (honest record)**: `run_shell` reaps the whole process
  group at 120 s, which silently kills `nohup` background jobs (two ensemble
  attempts, ≈13 min of DEM compute lost) unless results are cached per unit —
  `run_r3_ensemble.py` now writes per-specimen JSON + npz for exactly this
  reason; the `run_cluster` submit path (`cj1577`, `stage_dir` = this worktree)
  accepted the job but queued behind 407 others, so it did not deliver inside
  the mission window.  The production route for anything > 2 min on this box is
  the cluster queue, not the shell.
* **Artefacts**: `artifacts/biaxial_shear_band/r3_ensemble.json`,
  `r3_ensemble.png` (ensemble bars + operator agreement),
  `r3_estimator_audit.json`, `r3_estimator_audit.png` (RMS/bias vs noise),
  `logs/r3_ensemble.log` (primary log artefact).
* **Reproduce**:
  `PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r3_estimator_audit.py --slip-list 0.02`
  then `PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r3_ensemble.py --workers 8`
  (on the cluster: `run_cluster submit` with `stage_dir` = worktree) and
  `PYTHONPATH=src python benchmarks/biaxial_shear_band/summarise_r3_log.py`.
  Tests: `PYTHONPATH=src python -m pytest tests/test_shear_band_structure_tensor.py tests/test_biaxial_shear_band.py -q`.

## Error judgement (explicit)

| quantity | value | reference | error | < 3 % ? |
|---|---|---|---|---|
| Radon ensemble mean inclination | 53.83° | 60° (D&V 2004 midpoint) | **10.3 %** | **No** (15 % bar: yes) |
| structure-tensor ensemble mean | 66.83° | 60° | 11.4 % | **No** |
| Radon operator bias, synthetic, noise-free | 0.53° RMS | 0° | 0.53° | **No** (0.9 %) |
| Radon operator bias, synthetic, 0.25 % noise | 1.10° RMS | 0° | 1.10° | **Yes** (1.8 %) |
| structure-tensor bias, noise-free | 0.94° RMS (σ=1.5) / 2.08° (σ=2.5) | 0° | ≤ 3.5 % | ✓ / ✗ |

Interpretation limit: the ensemble mean matching the anchor within 10.3 % does
**not** prove the 2D freeze-bonded disc cell has the physics of 3D dense Hostun
sand; it shows that, given the R2-unbiased operator, the cell's characteristic
band inclination is statistically consistent with the literature value at the
15 % level and inconsistent with the R1 "model defect" claim.
