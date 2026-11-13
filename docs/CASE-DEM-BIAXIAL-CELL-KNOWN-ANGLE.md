# CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE - shear-band inclination in a plane-strain biaxial cell

**Verdict: PASS.** The pre-declared acceptance criterion is `|theta - 60 deg| / 60 deg < 15 %` on a *resolved* shear band. Measured band inclination **55.50 deg** -> relative error **7.50 %** (&#10003; inside tolerance), localisation z = 15.7 > 3 (&#10003;). Before trusting that number the measurement operator was validated on injected bands of analytically known inclination on the *same* 24 x 48 lattice: max |error| = **0.00 deg** on 4 known angles (the pre-fix estimator is off by 17.0 deg on the identical strain fields, which is what produced the historical 44 deg reading). All numbers below are injected programmatically from `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/run.json` by `benchmarks/biaxial_shear_band/make_anchor_doc.py`.

---

## (1) Case introduction (physics, geometry, parameters)

Plane-strain biaxial compression of a frictional disc specimen: the top platen is displacement driven, the bottom wall is fixed, and both lateral walls servo to a constant confining pressure `sigma_3`. After the deviator stress peaks a dense frictional material localises into a shear band whose inclination `theta` from the `sigma_3` axis is fixed by the mobilised friction angle `phi` and the dilatancy angle `psi`:

| model | relation | source |
|---|---|---|
| Mohr-Coulomb | `theta = 45 + phi/2` | plasticity bound |
| Roscoe | `theta = 45 + psi/2` | Roscoe, Geotechnique 20(2), 1970 |
| Arthur et al. | `theta = 45 + (phi + psi)/4` | Arthur et al., Geotechnique 27(1), 1977 |

**Reference / anchor (pre-declared):** Desrues & Viggiani (2004) Int. J. Numer. Anal. Methods Geomech. 28:279-321: dense Hostun sand plane-strain biaxial, shear band 55-60 deg from sigma_3 (midpoint 60 deg = Mohr-Coulomb 45 + phi/2 at phi = 30 deg); `theta_ref = 60 deg`, tolerance 15 % relative -> accept [51.0, 69.0] deg. Qualitative requirement: at least 3 late post-peak snapshots agreeing within 10 deg (`resolved`) and Radon `z > 3` (`localised`) so that a non-localised run cannot be scored as a pass.

| quantity | value |
|---|---|
| specimen | 24 x 48 spacings = 1128 discs, thickness 1 |
| particle radius / spacing | 0.75 m / 1.5 m (2D disc) |
| lattice jitter | 3.0 % of spacing |
| freeze-bond fraction | 0.35 |
| wall friction `mu_wall` | 0.55 |
| confining pressure `sigma_3` | 100 kPa |
| consolidation steps | 4000 |
| axial strain rate / target strain | 0.05 s^-1 / 0.12 |
| inertial number | 0.0072 (quasi-static) |

## (2) How it was done (method, tool chain, key settings)

1. **Known-angle validation first.** A synthetic slip band of exactly known inclination (tanh slip profile + isochoric homogeneous biaxial background) is injected into the very lattice of the case and pushed through the unmodified production measurement chain: remove the best-fit global affine field -> k = 8 nearest-neighbour deformation gradient -> `gamma = sqrt((e_xx-e_yy)^2 + 4 e_xy^2)` -> Radon/projection focus over `alpha in [0, 180)` with equal-count bins, representative angle = mean of late incremental estimates. The reference here is the *injected* angle, so this table is pure operator error.
2. **DEM cell.** `tensordem.biaxial_shear_band.run_biaxial_shear_band`: jittered triangular lattice, 60 pairwise relaxation iterations, freeze bonds on pairs within 1.2 spacings, four `ridge_dem.Plane` walls, lateral servo (gain 0.0003, velocity limit 2.0 m/s). Contact and bond laws come from `ridge_dem.RidgeAssembly` - the case adds no second contact model.
3. **Measurement on the DEM record.** Late window = post-peak tail (>= 3 snapshots), boundary layer (2 diameters from each wall) excluded because rough walls nucleate their own boundary bands.
4. **Acceptance arithmetic** (`tensordem.biaxial_anchor_case.judge_anchor`) is fixed before the run: relative error vs 60 deg plus the resolution/localisation requirements.
5. **Entry / reproducibility.** `benchmarks/biaxial_shear_band/run_anchor_case.py` writes the JSON artefact and the figure; the platform-side caller is `tensordem.integration_callers.run_biaxial_shear_band_case`, which also exposes `check_biaxial_maturity_gate` (L2/L3/L4 checked against files actually on disk).

```
TENSORDEM_THREADS=4 PYTHONPATH=src python benchmarks/biaxial_shear_band/run_anchor_case.py --out-dir artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE
```

Environment echo: python 3.11.6, numpy 2.4.6, torch 2.7.1, loongarch64. The cell is small, so the torch thread count is part of the recipe (`TENSORDEM_THREADS=4`; the default one-thread-per-core on a 128-core node is ~50x slower).

## (3) Known-angle benchmark: analytic reference vs recovered

Lattice 24 x 48, measurement window aspect ratio 0.536 (this is the anisotropy that biased the pre-fix estimator), band width 4.28 m, slip 2.14 m, volumetric strain 0.0.

| injected (deg) | recovered (deg) | abs. error (deg) | rel. error | < 15 % | legacy pre-R2 (deg) | legacy abs. error (deg) | z |
|---|---|---|---|---|---|---|---|
| 30 | 30.0 | 0.00 | 0.0 % | &#10003; | 17.0 | 13.0 | 7.4 |
| 45 | 45.0 | 0.00 | 0.0 % | &#10003; | 28.0 | 17.0 | 10.7 |
| 60 | 60.0 | 0.00 | 0.0 % | &#10003; | 43.0 | 17.0 | 28.6 |
| 70 | 70.0 | 0.00 | 0.0 % | &#10003; | 56.0 | 14.0 | 40.4 |

Max abs. error 0.00 deg, mean 0.00 deg; the pre-fix estimator on the identical strain fields is off by up to 17.0 deg.

## (4) DEM anchor run: literature reference vs got

| quantity | reference | got | error | criterion | verdict |
|---|---|---|---|---|---|
| band inclination `theta` (deg) | 60.0 (Desrues & Viggiani 2004, 55-60 deg) | 55.50 | 7.50 % | < 15 % | &#10003; |
| resolved late snapshots | >= 3 with spread < 10 deg | 4 snapshots, spread 1.12 deg | - | >= 3 | &#10003; |
| Radon localisation `z` | > 3 | 15.7 | - | > 3 | &#10003; |
| Radon `eta2` | > 0.2 | 0.519 | - | > 0.2 | &#10003; |

Measured mobilised friction angle `phi = 35.02 deg`, global dilatancy angle `psi = 41.51 deg`, peak deviator 299.3 kPa at axial strain 0.0534, broken bonds 1139.

| theoretical inclination | value (deg) | error vs got (deg) |
|---|---|---|
| mohr_coulomb | 62.51 | -7.01 |
| roscoe | 65.75 | -10.25 |
| arthur | 64.13 | -8.63 |

Caveat (honest): the global `psi` is unphysical (`psi > phi`), so the Roscoe/Arthur rows are **not** usable evidence - the volumetric strain of this cell is dominated by the servo/boundary layer, and the band-local dilation estimator is still WITHHELD (see section 5). Only the Mohr-Coulomb row and the literature anchor are used in the verdict.

## (5) Convergence / sensitivity and pitfalls

Sensitivity points carried by this artefact:

1. **Estimator (dominant term).** Same strain fields, two estimators: production max |error| 0.00 deg vs pre-R2 17.0 deg on the case lattice. The pre-fix defect (independent min-max normalisation of x and y) biases every angle towards 45 deg with `tan(theta_rep) = (W/H) tan(theta_true)`, which is exactly the 44 deg / 26 % reading of the first delivery.
2. **Specimen size and consolidation length.** Same cell, four configurations, 2000 consolidation steps, `run_benchmark.py` (`artifacts/biaxial_shear_band/results.json`): only the 24 x 48 / 100 kPa run resolves a dominant band, and the unresolved runs are reported as `theta = NaN` - never scored as a pass. The anchor run above differs from `r24x48_p100` only by the consolidation length (4000 vs 2000 steps), which moves `theta` from 55.75 deg to 55.50 deg - i.e. the case sits well inside the 15 % band, not on its edge.

| config | confining pressure (kPa) | theta (deg) | rel. error | resolved | phi (deg) |
|---|---|---|---|---|---|
| r16x32_p100 | 100 | not resolved | - | no | 30.57 |
| r16x32_p200 | 200 | not resolved | - | no | 24.05 |
| r24x48_p100 | 100 | 55.75 | 7.1 % | yes | 34.95 |
| r24x48_p200 | 200 | not resolved | - | no | 24.96 |

3. **Incremental vs cumulative fields.** Early post-peak snapshots carry a horizontal servo compaction front with high `eta2` but z near the null level; the late-window + z > 3 rule is what excludes it. The `representative` block of the artefact keeps both the best incremental and the cumulative estimate so the choice can be audited.

Pitfalls for reuse:

* Any projection/Radon estimator that normalises axes independently is aspect-ratio biased; validating it on a square/isotropic point cloud does not catch it - validate on the *production* lattice.
* `psi` from the global volumetric path of a servo cell is not a material property; do not feed it into Roscoe/Arthur.
* Thread oversubscription on a 128-core node is a 50x wall-clock trap for small DEM cells; pin the thread count in the run recipe.

## (6) Failure modes and artefacts

| file | content |
|---|---|
| `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/run.json` | run_json |
| `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/shear_band_case.png` | figure |
| `src/tensordem/biaxial_anchor_case.py` | case implementation (known-angle benchmark, anchor run, acceptance arithmetic, figure) |
| `src/tensordem/biaxial_shear_band.py` | DEM cell + production measurement chain |
| `benchmarks/biaxial_shear_band/run_anchor_case.py` | entry point (writes the artefacts above, exit code = verdict) |
| `src/tensordem/integration_callers.py` | platform-side caller + L2-L4 maturity gate (`run_biaxial_shear_band_case`, `check_biaxial_maturity_gate`) |
| `tests/test_biaxial_anchor_case.py` | known-angle recovery, judge arithmetic, entry artefact, integration caller / gate |
| `benchmarks/biaxial_shear_band/make_anchor_doc.py` | regenerates this document from the JSON artefact |

Failure modes observed / guarded:

* **Non-localised run scored as pass** - blocked by the `resolved` + `z > 3` + `eta2 > 0.2` conjunction; unit-tested (`test_judge_anchor_rejects`).
* **Unresolved band (NaN angle)** - the judge returns `passed=False` rather than comparing NaN; unit-tested.
* **Estimator regression** - the known-angle table is a hard test (`abs. error <= 2 deg` on 30/45/60/70 deg) and the biased legacy path is kept reproducible as the documented baseline.
* **Silent configuration drift** - `tests/test_biaxial_anchor_case.py` pins the anchor configuration (24 x 48, 100 kPa, mu_wall 0.55, 4000 consolidation steps).

Maturity ladder (machine-checked by `check_biaxial_maturity_gate`):

| level | evidence |
|---|---|
| L2 | impl `src/tensordem/biaxial_anchor_case.py` + entry `benchmarks/biaxial_shear_band/run_anchor_case.py` + run artefact `artifacts/CASE-DEM-BIAXIAL-CELL-KNOWN-ANGLE/run.json` |
| L3 | `tensordem.integration_callers.run_biaxial_shear_band_case` returns the verdict dict; `--biaxial-case` CLI path |
| L4 | benchmark tables in sections (3) and (4) (ref vs got + errors) with `repro_cmd` above |

