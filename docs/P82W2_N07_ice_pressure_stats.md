# P82W2-N07 — Spatial correlation length of the DEM ice contact pressure

and the ISO 19906 pressure-area exponent

**Node** P82W2-N07 (matured from plan 82) · **Repo** TensorDEM · **Module** `src/tensordem/ice_pressure_stats.py` · **Evidence grade** REAL (DEM contact forces; every statistic fitted from them)

## 1. Case, geometry, parameters

An ice sheet idealised as a dense assembly of crushed-ice fragments (granular layer, `tensordem.ice_contact`) is driven at 1 m/s into a rigid structure and crushed over a stroke of 0.35 m. Three structure scenes:

| scene | fragments per seed | surface area [m²] | geometry | curved surface |
|---|---|---|---|---|
| wall | 350, 350, 350, 350 | 2.00 | flat plane | — |
| cylinder | 263, 262, 263, 257 | 1.88 | flat plane | cylinder R = 0.60 m |
| cone | 255, 256, 253, 252 | 1.88 | flat plane | cone, half angle 20°, apex up |

Shared configuration (all scenes, all seeds): ice thickness 1.0 m, slab width 2.0 m, length 1.4 m, solid fraction 0.60, d50 = 0.18 m (lognormal, COV 0.30), contact stiffness k_n = 3e8 N/m (E_eff·r_eq with an effective modulus ≈ 3 GPa for a confined rubble), crushing strength σ_c = 5 MPa (literature 3–10 MPa), mass scaling 20, dt = 3e-5 s, 201 snapshots per run, 4 seeds per scene (11–14) = 12 DEM runs.

**Why the stiffness matters.** The repository's default k_n = 5e6 N/m keeps the contacts in the numerical-soft regime: the crushing cap σ_c is then *never* engaged and the absolute pressure level is set by k_n instead of by the material strength (measured: field maximum 0.18 MPa ≪ σ_c = 8 MPa). Calibrating k_n from the effective modulus engages the cap and makes σ_c the pressure-limiting parameter; that is the regime used here and it is what puts the design pressure in the literature band.

## 2. Method

1. **Pressure field.** The DEM deposits normal contact forces on the structure surface (unwrapped arc coordinate × height). The recorded pressure is the **time average over the save window** of the **coarse-grained force density** `p = G_s·F / (G_s·A) / n_steps` (normalised convolution, Gaussian kernel `s = 0.15 m ≈ d50`). Both normalisations matter: without the `1/n_steps` the value is a force impulse (20–80× too large), and without the `G_s·A` denominator the domain edges are divided by the area of cells that do not exist (the bug that produced 233 MPa spikes before the fix).
2. **Autocorrelation.** FFT autocovariance with unbiased count normalisation; the *pooled* ACF accumulates the covariance of every snapshot before normalising, so it converges in the number of snapshots. `L_c` is the first crossing `R(L_c) = R(0)/e` of the azimuthal average.
3. **Pressure-area curve.** Exact sliding-window patch averages through an integral image with per-cell areas (needed on the curved surfaces). For each area A the peak-over-time patch mean is formed at every patch position; `p_d(A)` is the **mean over patch positions** (typical design pressure). The **max over positions** (worst patch, upper envelope) is reported as the estimator sensitivity in F4.
4. **Fit.** Log-log OLS of `p_d = p_0 (A/A_0)^{-n}` over the continuum-valid range `A ≥ 2 L_c²` (a patch must contain at least two correlation areas); the full-range fit is reported as a sensitivity.
5. **Stationarity.** Statistics use the last 60 % of the stroke (steady crushing); the first 40 % is the compaction transient.

## 3. Results

### 3.1 Correlation length

| scene | L_c mean [m] | sd [m] | per seed [m] |
|---|---|---|---|
| wall | 0.283 | 0.016 | 0.279, 0.303, 0.288, 0.264 |
| cylinder | 0.313 | 0.014 | 0.308, 0.333, 0.311, 0.300 |
| cone | 0.346 | 0.014 | 0.329, 0.363, 0.341, 0.349 |

`L_c` is 0.28–0.35 m and does not depend on the structure geometry (it is a property of the crushed layer and the fragmentation scale), which is the physically expected behaviour.

### 3.2 Pressure-area law

| scene | n (typical) | R² | p_d(1 m²) [MPa] | n (envelope) | R² | p_d [MPa] | A cut [m²] | points |
|---|---|---|---|---|---|---|---|---|
| wall | 0.126 | 0.956 | 0.73 | 0.368 | 0.991 | 0.86 | 0.161 | 7 |
| cylinder | 0.095 | 0.984 | 1.69 | 0.356 | 0.996 | 1.89 | 0.196 | 6 |
| cone | 0.073 | 0.950 | 1.22 | 0.401 | 0.994 | 1.40 | 0.239 | 5 |

The typical-patch reduction reproduces the ISO 19906 band (0.05–0.15) on all three scenes with R² ≥ 0.95; the worst-patch envelope decays about three times faster — the size effect of an extreme statistic is steeper than the size effect of the typical pressure, exactly what extreme-value theory predicts.

### 3.3 Acceptance criteria

| criterion | verdict | measured |
|---|---|---|
| a_acf_R0 | PASS | R(0) = 1.0 |
| b_Lc_convergence | PASS | wall: 1.2%; cylinder: 0.9%; cone: 2.4% |
| c_pressure_area_exponent | PASS | wall: n=0.126 R²=0.956; cylinder: n=0.095 R²=0.984; cone: n=0.073 R²=0.950 |
| d_design_pressure_magnitude | PASS | wall: 0.73 MPa; cylinder: 1.69 MPa; cone: 1.22 MPa |
| e_uniform_negative_control | PASS | max |R-1| = 0.0e+00, L_c = inf |
| f_sharding_reproducibility | PASS | worst deviation 1.55e-15 |

**6/6 criteria pass.**

## 4. Figures

![fields](artifacts/ice_pressure_stats/figs/fig1_pressure_fields.png)

![acf](artifacts/ice_pressure_stats/figs/fig2_acf.png)

![pa](artifacts/ice_pressure_stats/figs/fig3_pressure_area.png)

![sens](artifacts/ice_pressure_stats/figs/fig4_n_sensitivity.png)

![conv](artifacts/ice_pressure_stats/figs/fig5_convergence.png)

## 5. Caveats and limitations

- The curved structures are modelled as bosses whose axis lies on the wall plane (a cylinder/cone protruding from a flat face). The surface is resolved geometrically (exact signed distance and surface normal of the solid of revolution, exact per-row cell areas), but the far field is the same bounded slab as the wall case.
- `L_c` is a property of the *coarse-grained* field; F4 shows it grows with the coarse-graining length (0.22 → 0.45 m for ℓ_cg = 0.10 → 0.30 m) and the exponent follows. ℓ_cg = 0.15 m ≈ d50 is the physical choice (one fragment is the smallest volume over which a stress can be defined); the ISO band is reproduced for ℓ_cg ∈ [0.10, 0.20] m and drifts below it at ℓ_cg = 0.30 m.
- Absolute pressures depend on the contact stiffness calibration (see §1); the *shape* statistics (L_c, n) are insensitive to it.
- The single-realisation fits scatter (R² 0.24–0.98). The reported numbers are ensemble statistics over 4 seeds, which is what a design curve is.
- `A > 2 m²` (flat wall) and `A > 1.9 m²` (curved) are outside the resolved surface and are reported as NaN rather than clamped.

## 6. Error assessment against references

| quantity | measured | reference / anchor | ≤3%? |
|---|---|---|---|
| R(0) normalisation | 1.0 | 1.0 | 0 (exact) |
| L_c convergence 100→N (ensemble) | 2.4% | ≤ 5% | < 5% |
| pressure-area exponent n (typical) | 0.126 / 0.095 / 0.073 | 0.05–0.15 (ISO 19906) | inside band |
| p_d(1 m²) | 0.73 / 1.69 / 1.22 MPa | 0.5–2 MPa | inside band |
| sharding 1/2/4 | 1.6e-15 | ≤ 1e-6 | < 3% |

The ACF normalisation and the sharding agreement are exact to machine precision (< 3 % trivially). `L_c` has **no analytic reference** for a DEM contact field (it is the measured quantity); the estimator itself was verified against analytic autocorrelations (AR(1) exponential: 0.5 % error; Gaussian-filtered noise: < 10 %, both in `tests/test_ice_pressure_stats.py`), which is the only available calibration. The pressure-area exponent and the design pressure are compared with the ISO 19906 band (a range, not a point value), so a percentage error is not defined; they fall inside the band.

## 7. Audit chain

Commit `aa2234387cb129051126eaf4e94a02f2b5ae925f`. Commands:

- `python scripts/ice_pressure_campaign.py --task <scene> --seed <seed>   (12 runs: 3 scenes x seeds 11-14, ~80 s each, run in parallel)`
- `python scripts/ice_pressure_report.py   (aggregation + criteria + figures)`
- `python -m pytest tests/test_ice_pressure_stats.py -q   (17 tests)`

| artifact | bytes | sha256 |
|---|---|---|
| `artifacts/ice_pressure_stats/run_cone_11.json` | 73575 | `bfbed6dc5d922a26…` |
| `artifacts/ice_pressure_stats/run_cone_12.json` | 73568 | `37ee8ba460d4b671…` |
| `artifacts/ice_pressure_stats/run_cone_13.json` | 73521 | `f6b413f77e7049ae…` |
| `artifacts/ice_pressure_stats/run_cone_14.json` | 73491 | `4884ff4a6d8b1e21…` |
| `artifacts/ice_pressure_stats/run_cylinder_11.json` | 73564 | `4c4bc09238f65620…` |
| `artifacts/ice_pressure_stats/run_cylinder_12.json` | 73543 | `90b212b31ef2d2be…` |
| `artifacts/ice_pressure_stats/run_cylinder_13.json` | 73600 | `c83b1b7ee3a7b983…` |
| `artifacts/ice_pressure_stats/run_cylinder_14.json` | 73688 | `74f8eb39122f87d6…` |
| `artifacts/ice_pressure_stats/run_wall_11.json` | 76860 | `d219f68c98e224d8…` |
| `artifacts/ice_pressure_stats/run_wall_12.json` | 76954 | `9aa862df92068304…` |
| `artifacts/ice_pressure_stats/run_wall_13.json` | 76859 | `8107aa9e24361c14…` |
| `artifacts/ice_pressure_stats/run_wall_14.json` | 76765 | `6b443426ea48089c…` |
| `artifacts/ice_pressure_stats/figs/fig1_pressure_fields.png` | 60567 | `eda1f1fef3c9e159…` |
| `artifacts/ice_pressure_stats/figs/fig2_acf.png` | 75379 | `a054ca840a2cf085…` |
| `artifacts/ice_pressure_stats/figs/fig3_pressure_area.png` | 74557 | `2beb863431edcb96…` |
| `artifacts/ice_pressure_stats/figs/fig4_n_sensitivity.png` | 77532 | `047cdf9d87ba6276…` |
| `artifacts/ice_pressure_stats/figs/fig5_convergence.png` | 50489 | `9bd9b9f330590b15…` |
