# CASE-BIAXIAL-SHEAR-BAND — R2: the 44.5° reading was an estimator defect

**Verdict (honest, mixed).** The estimator that produced the delivered
"44.25° / 44.67° → 25–26 % error → FAIL" result was **biased by 18–20°** on the
elongated specimen of this case: it min-max rescaled `x` and `y` *independently*
before projecting, which is a non-conformal (anisotropic) map, so the reported
angle obeyed `tan(theta_reported) = (W/H) · tan(theta_true)` with `W/H ≈ 0.54`.
After the fix, the criterion `< 15 %` vs the 60° anchor is **met on the primary
resolved run** (`55.75°`, 7.1 %) and on the paired re-measurement of the
100 kPa run (9.4 %), but **not met** on the paired re-measurement of the
200 kPa run (`72.0°`, 20 %). One of two resolved runs is therefore still outside
tolerance → the case stays an **honest FAIL**, but the previously reported
"model defect" explanation is withdrawn: the dominant error term was
measurement, not physics. All numbers below are programmatically injected from
`artifacts/biaxial_shear_band/*.json` (no hand transcription).

---

## ① Case introduction (physics, geometry, parameters)

Plane-strain biaxial compression of a freeze-bonded disc specimen: top platen
displacement driven, bottom wall fixed, both lateral walls servo to a constant
`sigma_3`. A dense frictional material localises into a shear band at/after the
deviator peak; the band inclination from the `sigma_3` axis is a classical
validation quantity (Mohr–Coulomb `45 + phi/2`, Roscoe 1970 `45 + psi/2`,
Arthur et al. 1977 `45 + (phi + psi)/4`).

| quantity | value |
|---|---|
| specimen (primary config) | 24 × 48 spacings = 1128 discs, thickness 1 |
| particle radius / spacing | 0.75 m / 1.5 m (2D disc) |
| lattice jitter | 3 % of spacing (no lattice-aligned failure plane) |
| freeze-bond fraction | 0.35 |
| contact friction `mu` | 0.55 (library default) |
| confining pressure | 100 kPa and 200 kPa |
| axial strain rate / final strain | 0.05 s⁻¹ / 0.12 |
| inertial number | 0.0051–0.0072 (quasi-static) |
| anchor | 60° (Desrues & Viggiani 2004, dense Hostun sand 55–60°), tolerance 15 % |

## ② What was done

1. **Independent re-derivation of the estimator.** A synthetic band of *known*
   inclination was injected into the exact case lattice (`x_now = x + homogeneous
   biaxial background + finite-width tanh slip band`), pushed through the
   unmodified production pipeline (affine removal → kNN strain → Radon focus),
   and compared with the pre-R2 estimator on the identical strain field
   (`benchmarks/biaxial_shear_band/run_r2_estimator_audit.py`,
   `src/tensordem/biaxial_band_diagnostics.py`).
2. **Root cause.** `radon_band_angle` rescaled `x` and `y` each by their own
   min-max range. A physical direction `(cos a, sin a)` then becomes
   `(cos a / W, sin a / H)`, i.e. `tan(theta_reported) = (W/H) tan(theta_true)`.
   Diagnosis was confirmed two ways: the recovered angle matched the analytic
   distortion to ≤ 0.6° (24×48) and ≤ 2.2° (16×32), and a *weight-free* ridge
   (analytic Gaussian, no DEM at all) reproduced the same bias, while a
   PCA of the gamma ridge in physical coordinates recovered the true angle.
3. **Fix** (minimal, in `src/tensordem/biaxial_shear_band.py`):
   * no per-axis rescaling — only the 1-D projection coordinate `s` is
     normalised (monotone, angle preserving);
   * bins are now **equal-count (quantile)** rather than equal-width, so the
     count weighting of `eta2` can no longer encode the point-set anisotropy;
   * `binning="equal_width"` is retained as an option and
     `biaxial_band_diagnostics.legacy_radon_band_angle` is kept purely as the
     audit baseline.
4. **Paired re-measurement.** The two resolved configurations were re-run with
   `record_snapshots=True` (new opt-in keyword, default off) and measured twice
   *on the same snapshots* — once with the R2 estimator, once with the pre-R2
   one — plus a band-local dilation diagnostic
   (`benchmarks/biaxial_shear_band/run_r2_case_rerun.py`).
5. **Case re-run** with the fixed production pipeline
   (`benchmarks/biaxial_shear_band/run_benchmark.py`, unchanged) → new
   `artifacts/biaxial_shear_band/results.json`.

## ③ Results (numbers)

**Estimator audit (synthetic band of known angle, no DEM):**

| lattice | fixed estimator max abs error | pre-R2 max abs error | analytic-prediction residual |
|---|---|---|---|
| 24 × 48 (W/H = 0.537) | **0.00°** | **18.00°** | 0.62° |
| 16 × 32 (W/H = 0.510) | 2.00° | 20.00° | 2.16° |

Per-angle, 24 × 48: injected 60° → pre-R2 **43°**, R2 **60°**; injected 45° →
pre-R2 **28°**, R2 **45°**. The case's own headline numbers are reproduced
exactly by the bias, not by the DEM.

**Archived results re-read through the analytic inversion**
`theta_true = atan(tan(theta_reported) / (W/H))`, `W/H = 0.537`:
`44.25° → 61.1°`, `44.67° → 61.4°` — i.e. the delivered "FAIL" readings
correspond to band inclinations *just above* the 60° anchor.

**Case re-run, fixed pipeline (4 configs, 16×32 still unresolved):**

| run | theta | theta_ref | rel. error | < 15 % | resolved | phi |
|---|---|---|---|---|---|---|
| r16x32_p100 | not resolved | 60° | — | ✗ | ✗ | 24.05° |
| r16x32_p200 | not resolved | 60° | — | ✗ | ✗ | 30.57° |
| **r24x48_p100** | **55.75°** | 60° | **7.08 %** | **✓** | ✓ | 34.95° |
| r24x48_p200 | not resolved | 60° | — | ✗ | ✗ | 24.96° |

**Paired re-measurement (identical snapshots, two estimators):**

| config | theta R2 | theta pre-R2 | rel. error R2 | rel. error pre-R2 |
|---|---|---|---|---|
| r24x48_p100 | 54.33° | 43.33° | 9.4 % ✓ | 27.8 % ✗ |
| r24x48_p200 | 72.00° | 65.00° | 20.0 % ✗ | 8.3 % ✓ |

The two configurations are reordered relative to the delivered report because
the estimator defect is aspect-ratio driven and both aspect ratios are close;
the honest reading is that the *pre-R2* numbers carried a 15–20° systematic
error while the residual run-to-run spread of the corrected measurement is
~18° between the two confining pressures (DEM is chaotic and the band is
measured on 8–9 late snapshots).

## ④ Figures

* `artifacts/biaxial_shear_band/r2_estimator_audit.png` — (a) recovered vs
  injected inclination for both estimators on both lattices, with the analytic
  distortion curve; (b) bias before/after the fix.
* `artifacts/biaxial_shear_band/r2_case_rerun.png` — (a) same-run comparison of
  the two estimators against the ±15 % band around 60°; (b) Roscoe/Arthur laws
  against the measured band-local dilatancy; (c) late-window `gamma` field cloud
  with both fitted band lines drawn.

## ⑤ Pitfalls, limitations, scope

1. **Estimator bug class.** Any projection/Radon estimator that normalises axes
   independently is silently aspect-ratio biased. The existing unit tests did
   *not* catch it because they used an isotropic random point cloud in a square
   box: the operator was validated on a fixture that does not resemble the
   production point set (an elongated lattice). The R2 test suite therefore
   validates on the *case lattice* itself.
2. **Band-local dilation is WITHHELD.** `band_local_dilation` was written to
   replace the global `psi` (which the delivered report already flagged as
   unphysical, `psi > phi`). A synthetic unit test exposed a real defect in it
   (regressing per-pair increments instead of accumulated strains gives a
   zero slope); that is fixed and the synthetic isochoric/dilatant cases pass,
   but the DEM application still returns a **degenerate/clamped** fit
   (`psi_band` pinned at the ±72° guard, `band_r2` low), so the `psi_band` /
   `psi_theory` fields in `artifacts/biaxial_shear_band/r2_case_rerun.json` are
   **not evidence** and are excluded from the verdict. The k = 8 kNN fit also
   attenuates a finite-width band's dilation to ≈ 0.44 of the injected value in
   the synthetic test.
3. **Chaotic spread.** Two confining pressures of the same geometry give 54.3°
   and 72.0°; the 200 kPa run is at the top of the plateau and passes through
   45° during the post-peak window. A single-number case per configuration is
   fragile; the re-measurement reports mean ± std over late snapshots.
4. **16 × 32 is under-resolved** for this estimator (2° residual in the
   synthetic audit, "not resolved" in every DEM run) → only 24 × 48 counts.
5. Scope: 2D discs, no rolling resistance, bonded lattice with 35 % bonds. The
   60° dense-sand anchor belongs to a strongly dilatant 3D material; matching it
   with a 2D isochoric-ish band is *not* guaranteed by construction.

## ⑥ Error judgement (vs the 60° anchor, explicit)

| quantity | value | reference | error | < 3 % ? |
|---|---|---|---|---|
| band inclination, r24x48_p100 (case re-run) | 55.75° | 60° | 7.08 % | **No** |
| band inclination, r24x48_p100 (paired re-measurement) | 54.33° | 60° | 9.4 % | **No** |
| band inclination, r24x48_p200 (paired re-measurement) | 72.00° | 60° | 20.0 % | **No** |
| estimator bias, synthetic band on the case lattice (R2) | 0.00° max | 0° | 0 % | **Yes** |
| estimator bias, pre-R2 (documented baseline) | 18.00° max | 0° | 18° abs | **No** |

No quantity reaches the 3 % bar against the *physical* anchor: the case's
acceptance criterion was 15 %, and only the primary resolved run meets it. The
one thing that does meet 3 % is the claim that the estimator itself is now
unbiased on this lattice, which is a *numerical* identity, not a physics result.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r2_estimator_audit.py
PYTHONPATH=src python benchmarks/biaxial_shear_band/run_r2_case_rerun.py --workers 2
PYTHONPATH=src python benchmarks/biaxial_shear_band/run_benchmark.py --workers 4 \
    --out artifacts/biaxial_shear_band/results.json
PYTHONPATH=src python -m pytest tests/test_biaxial_band_diagnostics.py tests/test_biaxial_shear_band.py -q
```

## Follow-ups (next owner)

1. Replace the Radon focus with a structure-tensor estimator on a gridded
   `gamma` field (the standard Desrues & Viggiani route); compare its angle to
   the fixed Radon on the stored snapshots before spending more DEM hours.
2. Finish the band-local dilation estimator: use a band-aligned strain measure
   (normal/tangential components in the band frame, computed from a wider fit
   window centred on band particles) and re-derive `psi_band` with its own
   uncertainty; then re-test Roscoe/Arthur law-level agreement.
3. Re-run 32 × 64 (≈ 3000 particles) so that band width/noise ratio improves
   and the p200 configuration becomes resolved; report mean ± std over ≥ 8
   snapshots and treat the two-pressure spread as a reported uncertainty.
4. Feed the case-lattice estimator-recovery audit into any other case that uses
   `radon_band_angle` (grep `radon_band_angle` / `measure_band`) — all of them
   were exposed to the same aspect-ratio bias if their point sets are elongated.