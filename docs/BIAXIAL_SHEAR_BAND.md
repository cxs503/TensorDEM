# CASE-BIAXIAL-SHEAR-BAND — shear-band inclination in a plane-strain biaxial cell

**Verdict: honest FAIL** on the numeric acceptance (`< 15 %` vs the literature
shear-band inclination).  The measurement operator and the DEM cell are
delivered, validated and reproducible; the resolved band inclinations
(44.3° ± 0.8 and 44.7° ± 0.5) sit **26 % / 25.6 %** below the 60° reference, so
the criterion is **not** met.  All numbers below are copied from
`artifacts/biaxial_shear_band/results.json` (no hand transcription).

---

## ① Problem setting and reference

A rectangular freeze-bonded disc specimen is compressed in plane strain: the top
platen is displacement driven, the bottom wall is fixed, and both lateral walls
servo to a constant confining pressure `sigma_3`.  After the deviator stress
peaks, a dense frictional material localises into a shear band whose
inclination `theta` (from the `sigma_3` axis, horizontal here) is predicted by
three classical relations that need only `phi` (internal friction) and `psi`
(dilation):

| model | relation | source |
|---|---|---|
| Mohr–Coulomb | `theta = 45 + phi/2` | plasticity bound |
| Roscoe | `theta = 45 + psi/2` | Roscoe, *Geotechnique* 20(2), 1970 |
| Arthur et al. | `theta = 45 + (phi + psi)/4` | Arthur et al., *Geotechnique* 27(1), 1977 |

**Reference / anchor (pre-declared in `benchmarks/biaxial_shear_band/run_benchmark.py`)**

* `theta_ref = 60°` — midpoint of the 55–60° shear-band range reported for
  dense Hostun sand plane-strain biaxial tests by Desrues & Viggiani (2004),
  *Int. J. Numer. Anal. Methods Geomech.* 28:279–321; equals Mohr–Coulomb at
  `phi = 30°`.
* tolerance 15 % relative → accept `51° <= theta <= 69°`.
* qualitative requirement: the band must be *resolved*, i.e. at least three
  late post-peak snapshots agree within 10°.

## ② Method and implementation

Cell: `src/tensordem/biaxial_shear_band.py`

* `_make_specimen` — jittered triangular disc lattice (spacing `2r`, jitter 3 %,
  overlaps relaxed by 60 pairwise iterations), bonds on pairs within
  `1.2 · spacing`, `freeze_bond_fraction` of them kept; four `ridge_dem.Plane`
  walls, lateral ones with a proportional servo (`servo_gain = 3e-4`,
  `servo_velocity_limit = 2 m/s`).
* Integration and contact/bond laws are `ridge_dem.RidgeAssembly` — this module
  adds no second contact model.
* `local_shear_strain` — least-squares deformation gradient over the `k = 8`
  nearest neighbours per particle, `gamma = sqrt((e_xx-e_yy)^2 + 4 e_xy^2)`
  after the best-fit global affine field is removed.
* `radon_band_angle` — Radon/projection focus over `alpha ∈ [0,180°)`: the
  samples are projected on the band normal `n(alpha)`; a planar band collapses
  into one bin, maximising `eta2`, the fraction of weighted variance explained
  by the bin means.  Conjugate peaks (`alpha`, `180-alpha`) are folded into
  `[0,90]` because they share the same inclination to the vertical.
* Representative angle = mean of the incremental-field estimates over the last
  60 % of the post-peak record; `resolved` requires >= 3 snapshots within 10°.
  Early snapshots are excluded on purpose: the servo drives a high-`eta2`
  horizontal compaction front (z > 3) before any shear band exists, and a naive
  global `eta2` maximum locks onto that transient.
* `phi` = secant friction angle at the smoothed deviator peak;
  `psi = asin(-d eps_v / d eps_a)` fitted over the late post-peak window.

Validation of the measurement operator (`tests/test_biaxial_shear_band.py`,
15 passed in 5.3 s): synthetic Gaussian ridges at 0/15/35/45/52/60/70/85° are
recovered to **≤ 1°** with `z > 5`; a uniform deformation is removed exactly
(`gamma < 1e-6`); a `tanh` shear layer is recovered in the right location.

## ③ Parameter table (dimensionless numbers complete)

| quantity | value |
|---|---|
| particle radius `r` | 0.75 m (disc, unit thickness) |
| specimen | 16×32 = 496 particles / 24×48 = 1104 particles |
| bond fraction | 0.35 |
| contact friction | `mu = 0.55` |
| `dt` | 4.97e-4 s |
| axial strain rate | 0.05 s⁻¹ |
| inertial number `I = gamma_dot·d·sqrt(rho/p)` | **0.0072** (p = 100 kPa), **0.0051** (p = 200 kPa) |
| consolidation | 2000 steps (servo to `sigma_3`) |
| final axial strain | 0.12 |
| confining pressure | 100 kPa, 200 kPa |

## ④ Comparison table (ref vs got, per criterion)

| run | theta got | theta ref | rel. error | `<15 %` | resolved | phi | psi |
|---|---|---|---|---|---|---|---|
| r16x32_p100 | not resolved | 60° | — | ✗ | ✗ | 30.6° | 57.8° |
| r16x32_p200 | not resolved | 60° | — | ✗ | ✗ | 24.1° | 65.7° |
| r24x48_p100 | **44.25° ± 0.8°** | 60° | **26.3 %** | ✗ | ✓ | 34.9° | 39.9° |
| r24x48_p200 | **44.67° ± 0.5°** | 60° | **25.6 %** | ✗ | ✓ | 25.0° | 29.5° |

Theory predictions from the *measured* `phi, psi` of the resolved runs:
Mohr–Coulomb 62.5° / 57.5°, Roscoe 64.9° / 59.7°, Arthur 63.7° / 58.6° — the
measurement is 20–30 % below **all three**, not only below the external anchor.

Other criteria checked: servo holds the confining pressure to +22 % / +16 %
(proportional-controller droop of a stress servo, reported not hidden); both
resolved runs are stable across two specimen sizes *and* two confining
pressures (44.25 vs 44.67 → 0.4° spread), so the 44.5° result is a property of
the model, not of one configuration.

## ⑤ Convergence / sensitivity

* **Resolution**: 16×32 → 24×48 changes the resolved angle by 0.4° and turns an
  unresolved field into a resolved one, i.e. the estimator needs the larger
  specimen to separate a band from the servo compaction front.
* **Confinement**: 100 → 200 kPa moves `phi` by 10° (34.9 → 25.0) and `psi` by
  10° (39.9 → 29.5) while the band angle stays at 44.5° ± 0.5 — the measured
  inclination is insensitive to confinement, which is *not* what Roscoe/Arthur
  predict (they would move by ~2.5°).
* **Material sensitivity (negative result)**: with the library default
  `freeze_bond_fraction = 0.65` the specimen never softens — it hardens to 10 %
  axial strain (deviator still rising, `eps_v > 0`, contractant) and no band
  forms.  The benchmark therefore uses a 35 % bonded specimen, and that
  limitation is a documented property of the 65 % freeze-bond lattice in this
  cell, not a tuning choice.

## ⑥ Failure modes, artefacts and reproduction

* **Failure mode 1 (primary)**: the resolved band (44.5°) is 25–26 % below the
  60° anchor and below all three classical predictions.  The most likely
  mechanisms are (a) the global affine removal in the strain estimator, which
  subtracts part of the band's own displacement and rotates the residual ridge
  towards the 45° maximum-shear direction, and (b) the low bond fraction, which
  makes the response closer to ideal-plastic flow (`theta → 45°`) than to a
  dilatant frictional band.  Neither is proven; the falsification evidence is
  the two-run, two-resolution stability of 44.5°.
* **Failure mode 2**: 16×32 specimens are `not resolved` — the servo compaction
  front and the band are of comparable width.
* **`psi` is the least reliable measured quantity**: 29.5–65.7°, i.e. above the
  physically expected `psi <= phi`; it comes from a late-window volumetric fit
  of a softening specimen and should be treated as an upper bound.
* **Artifacts**: `artifacts/biaxial_shear_band/results.json` (full stress paths
  removed, band histories kept), `artifacts/biaxial_shear_band/shear_band.png`
  (band geometry + angle frequency), `logs/biaxial_shear_band_bench.log`.
* **Reproduce**:
  `PYTHONPATH=src python benchmarks/biaxial_shear_band/run_benchmark.py --workers 4 --out artifacts/biaxial_shear_band/results.json`
  (`--quick` for a 1-run smoke test); tests:
  `PYTHONPATH=src python -m pytest tests/test_biaxial_shear_band.py -q`.

### What would be tried next (not done here)

1. Estimate the band orientation from the *non-affine* displacement without a
   global affine removal (local-only affine fit), to test hypothesis (a).
2. Increase the bond fraction and the specimen size together (32×64, 2000+
   particles) so that a dilatant band, not ideal-plastic flow, is the failure
   mode; compare its angle with Arthur's relation.
3. Replace the proportional stress servo with a stiffer PI servo to remove the
   16–22 % confining offset.