# Multi-year ice pressure-ridge keel interaction (TensorDEM)

**Campaign node** P33-W2-N04 · **repository** TensorDEM · **chain** keel geometry statistics → plane-strain DEM crushing → extreme-load scaling → ice-belt plate-panel check

## 1. Case, geometry and parameters

A multi-year (MYI) pressure ridge is characterised by a keel of freeze-bonded rubble below the waterline.  The chain starts from the published ridge statistics summarised by Timco & Frederking (1998), Timco & Burden (1997) and Strub-Klein & Sudom (2012): keel depth log-normal with mean 16.0 m and CV 0.35 truncated to sampled range [10.0, 29.4] m, rubble porosity 0.20-0.40, block diameter 2-10 m (log-normal, median 4 m), keel flank angle 22 deg from the vertical (keel depth : width self-similar).

| quantity | value | source |
|---|---|---|
| keel depth mean | 16.0 m (bounds 10-30 m) | Timco & Frederking 1998 parameterisation |
| porosity n | 0.20-0.40 (0.30 nominal) | ridge morphology compilations |
| block diameter | 2-10 m, median 4 m | ridge block size statistics |
| ice strength sigma_ice | 5, 7.5, 10 MPa (swept) | campaign specification |
| hull speed V | 1, 2, 2.5, 5 m/s (swept) | campaign specification |
| freeze-bond cohesion | 0.10 sigma_ice (0.75 MPa at 7.5 MPa) | calibration input |
| loaded hull length L_c | 1.5 m | contact patch assumption |
| plate advance | 0.8 m | DEM control |
| ice-belt steel | EH36: E = 206 GPa, nu = 0.3, sigma_y = 355 MPa, FS = 1.15 | structural check |

## 2. Method

**Ridge statistics** (`tensordem.myi_ridge.geometry`): truncated log-normal sampling with moment matching (the location parameter is solved so the *truncated* law keeps the target mean); plane-strain keel sections are built by random sequential adsorption of large blocks (block-size distribution) followed by an adaptive fragment lattice, and the achieved porosity and keel depth are measured, never assumed.

**DEM crushing** (`tensordem.myi_ridge.crushing`): bonded disc assembly (monodisperse, plane strain, unit thickness) thinned from a jitter-free triangular lattice so the initial state is stress free; penalty contact with normal damping, regularised Coulomb friction, and freeze bonds carrying an axial spring with a tensile cut-off plus a tangential spring bounded by a cohesion that softens with plastic slip.  The hull ice belt is a rigid plate spanning the keel depth that advances at the prescribed speed; crushed rubble is extruded upward over the plate.  Explicit integration with quasi-static local damping; every storing and dissipating term is accumulated so the ledger `E_input = dKE + dU + E_cohesion + W_friction + W_damping` closes.

**Load model** (`tensordem.myi_ridge.scaling`): the DEM supplies the plate force per unit width f; the extreme load is that force over the loaded hull length, `F_max = f_max * L_c = p * h_keel * L_c`, and the exponents of `F_max ~ sigma_ice^a h_keel^b` are fitted in log space (two regressors, standard errors reported).

**Structural check** (`tensordem.myi_ridge.panel`): Bogner-Fox-Schmit bicubic Hermite plate element (16 dof), 3x3 Gauss integration, clamped ice-belt panel between frames, peak bending stress from the recovered moments compared with sigma_y/1.15.

## 3. Results

### 3.1 Geometry statistics (criterion a)

- sampled keel-depth mean 16.144 m vs target 16.0 m → error +0.90 % (limit 5 %); sampled std 4.24 m
- block diameter mean 4.45 m vs truncated-law mean 4.44 m → error +0.28 %
- section reconstruction over 9 (h, n) combinations: max porosity error +0.0055 (limit 0.02), max keel-depth error +0.00 % (limit 5 %)

### 3.2 DEM crushing and the energy ledger (criterion b)

| case | sigma [MPa] | h_keel [m] | V [m/s] | n | bodies | p_peak [MPa] | p_steady [MPa] | F_max peak [MN] | F_max steady [MN] | ledger closure |
|---|---|---|---|---|---|---|---|---|---|
| depth_10 | 7.50 | 10 | 2.00 | 0.30 | 107 | 11.00 | 9.19 | 164.95 | 137.92 | 0.35 % |
| depth_30 | 7.50 | 30 | 2.00 | 0.30 | 963 | 8.73 | 6.60 | 392.77 | 297.21 | 11.14 % |
| porosity_0.40 | 7.50 | 20 | 2.00 | 0.40 | 367 | 3.99 | 3.05 | 119.72 | 91.51 | 1.78 % |
| sigma_10.00 | 10.00 | 20 | 2.00 | 0.30 | 428 | 9.02 | 6.46 | 270.69 | 193.94 | 0.98 % |
| sigma_5.00 | 5.00 | 20 | 2.00 | 0.30 | 428 | 7.74 | 5.73 | 232.11 | 171.89 | 0.74 % |
| sigma_7.50 | 7.50 | 20 | 2.00 | 0.30 | 428 | 8.39 | 6.11 | 251.78 | 183.36 | 0.92 % |
| speed_1.00 | 7.50 | 20 | 1.00 | 0.30 | 428 | 6.40 | 4.60 | 191.99 | 137.91 | 0.59 % |
| speed_2.50 | 7.50 | 20 | 2.50 | 0.30 | 428 | 9.37 | 6.78 | 281.13 | 203.26 | 1.44 % |
| speed_5.00 | 7.50 | 20 | 5.00 | 0.30 | 428 | 13.79 | 9.67 | 413.64 | 290.09 | 3.23 % |

Energy-ledger closure (residual over the exchanged energy) is at most 11.14 % over the campaign.

Time-step convergence of the ledger (base case):

| dt | steps | p_steady [MPa] | closure |
|---|---|---|---|
| 4.001e-05 | 9999 | 6.112 | 0.92 % |
| 2.000e-05 | 19998 | 8.425 | 2.09 % |

The residual does not fall when dt is halved, and the steady pressure changes by 38 % between the two time steps.  Two things follow, both reported as measured: the ledger closure is not monotonically improved by refining dt (the crushing history is chaotic, so the two runs are not the same trajectory and the residual is a per-run property), and **the measured plate pressure is not time-step converged** (it varies by tens of percent with dt).  The scaling exponents are therefore exponents at one fixed dt, comparable between cases but not converged physical values.

### 3.3 Extreme-load scaling (criterion c/e)

- fitted exponents (peak force): a = 0.209 +/- 0.786, b = 0.726 +/- 0.489, R^2 = 0.275
- fitted exponents (steady force): a = 0.169, b = 0.608, R^2 = 0.219
- case count 9 (>= 5 required); F_max envelope 119.7 - 413.6 MN; pressure range 3.99 - 13.79 MPa
- monotonicity (one-factor sweeps): sigma_ice increasing, h_keel increasing, hull_velocity increasing

### 3.4 Ice-belt panel check (criterion d)

| frame spacing | p [MPa] | t [mm] | sigma_max [MPa] | allowable [MPa] | utilisation | verdict |
|---|---|---|---|---|---|---|
| 2.4m | 9.67 | 30 | 18634.9 | 308.7 | 60.37 | FAIL |
| 3.0m | 9.67 | 30 | 29117.1 | 308.7 | 94.32 | FAIL |
| 3.6m | 9.67 | 30 | 41928.6 | 308.7 | 135.82 | FAIL |

Thickness required to satisfy sigma_max < sigma_y/1.15 at a 2.4 m frame spacing under the campaign design pressure 9.67 MPa: **233.1 mm** (Kirchhoff scaling sigma ~ 1/t^2, verified by a second solve in the test suite).

## 4. Criterion verdict

| criterion | verdict | evidence |
|---|---|---|
| (a) keel statistics, mean error < 5 %, n error < 0.02 | PASS | depth 0.90 %, porosity 0.0055 |
| (b) energy-closure < 5 % | FAIL | max closure 11.14 % |
| (c) a, b in [0.8, 1.2] | FAIL | a = 0.21, b = 0.73 |
| (d) sigma_max < sigma_y/1.15 | FAIL | worst utilisation 135.82 at 3.6m spacing |
| (e) >= 5 cases, monotone in sigma, h, V | PASS | 9 cases |
| (f) pytest >= 15 green | PASS | tests/test_myi_ridge_*.py, 36 tests |
| (g) reproducible | PASS | fixed seeds, identical peaks on repeat runs |

## 5. Interpretation, limitations and honest failure report

**What is established.** The geometry end of the chain is verified against its own targets (criterion a).  The DEM crushing model is stable and deterministic, and its energy ledger closes to a few percent in eight of the nine cases (worst case depth_30: 11.1 %, so criterion (b) is failed as measured).  The extreme load is monotone in ice strength, keel depth and speed over the swept range (criterion e).  The ice-belt plate-panel solver reproduces the classical clamped and simply supported plate benchmarks to better than 3 % (deflection 1.0042x Timoshenko at 12x12).

**What fails, and why.** Criterion (c) is not met: the fitted strength exponent is a = 0.21 (target 0.8-1.2) and the depth exponent b = 0.73.  The measured plate pressure is almost independent of the ice strength (p_peak 7.7 MPa at sigma = 5 MPa and 9.0 MPa at sigma = 10 MPa) because the plane-strain configuration in which the rubble is pushed against a rigid far-field wall and floor is *compaction dominated*: the resistance is set by the elastic/geometric constraint of squeezing a nearly incompressible disc assembly, not by the freeze-bond strength.  The same mechanism makes the absolute pressure (5-14 MPa, i.e. 1-2 sigma_ice) far above the 1-3 MPa local ice pressures used in ice-belt design, so this pressure must not be read as a design pressure.

The structural check therefore fails by a large margin at the DEM pressure; the report quantifies instead the thickness that would be required.  Reporting the filtered number instead of the measured one would be exactly the failure mode the campaign discipline forbids, so it is reported as measured.

**Scope limits.** Two-dimensional plane strain with unit thickness (no out-of-plane escape of crushed ice); monodisperse rigid discs without block communition; buoyancy switched off by default because the DEM runs in effective stress (submerged ice is nearly neutrally buoyant and the ridge is held 
by the surrounding sheet); freeze-bond cohesion is a calibration input; the hull is a rigid plate rather than a hull surface; L_c is an assumption.

**What the result implies for the campaign.** A confined 2D rubble cell is not a crushing-pressure meter.  The next iteration should either (i) give the crushed ice an escape path with a servoed confining stress (a proper biaxial cell at a prescribed ridge confining pressure), or (ii) let the ice blocks break (comminution) so that a steady crushing pressure can form, or (iii) calibrate the pressure level against ridge indentation data and keep the DEM only for the pressure *ratio* between cases.  Until then the DEM pressure supplies a monotone load ordering, not a design pressure.

## 6. Error assessment against references

| quantity | reference | model | error | < 3 %? |
|---|---|---|---|---|
| clamped square plate w_centre (12x12) | 0.00126 q a^4/D (Timoshenko) | 2.484181e-05 m | 0.42 % | yes |
| clamped square plate sigma_max (12x12) | 6 x 0.0513 q a^2/t^2 | 3.345738e+06 Pa | 2.2 % | yes |
| simply supported w_centre (12x12) | 0.00406 q a^4/D | 7.975775e-05 m | 0.06 % | yes |
| specimen porosity (DEM) | requested n | measured | <= 0.03 | yes |
| keel-depth mean (sampler) | 16.0 m | 16.144 m | 0.90 % | yes |
| DEM pressure vs 1-3 MPa ridge local pressures | 1-3 MPa | 9.7 MPa | 383 % | **no — honest FAIL** |
| strength exponent a vs 0.8-1.2 | 0.8-1.2 | 0.21 | 79 % below 1 | **no — honest FAIL** |

No reference value exists for the MYI ridge crushing pressure in this configuration; the literature range (1-3 MPa local ridge pressure) is quoted only to show that the model output lies outside it.  The DEM has no analytic reference, so its verification is internal: contact law = k x overlap, buoyant free acceleration = (rho_w - rho_i) g / rho_i, plasticity bounds, and the energy ledger.

## 7. Reproduction

```
python scripts/run_myi_ridge_campaign.py --all --jobs 9
python scripts/run_myi_ridge_campaign.py --aggregate
python scripts/myi_ridge_ledger_convergence.py
python scripts/make_myi_ridge_figures.py
python scripts/make_myi_ridge_report.py
python -m pytest tests/test_myi_ridge_geometry.py tests/test_myi_ridge_crushing.py tests/test_myi_ridge_scaling.py tests/test_myi_ridge_panel.py -q
```

Artifacts: `runs/myi_ridge/*.json` (per case), `artifacts/myi_ridge/myi_ridge_results.json` (aggregate), `artifacts/myi_ridge/ledger_convergence.json`, figures in `docs/myi-ridge-keel-interaction/figures/`.
