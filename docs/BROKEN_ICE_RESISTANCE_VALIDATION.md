# DEM broken-ice direct-navigation resistance: validation surface

**Mission** p247w1-N07 · repository `TensorDEM` · module `tensordem.broken_ice_resistance` · campaign ledger `c3dd639f8b81f978`

## 1. The case (physics, geometry, parameters)

A ship transits a broken first-year ice field. The field is a discrete pack of rigid circular floes (polydisperse radii, jittered-triangular placement at the requested areal concentration, no initial overlap) floating in water and driven only by hull contacts, floe-floe contacts and quadratic water drag. The hull is a plan-view convex outline: a parallel body plus a wedge bow of the stated entry angle.

| quantity | value | source |
|---|---|---|
| ice_density_kg_m3 | 917 | MT Uikku sheet III (measured) |
| water_density_kg_m3 | 1025 | MT Uikku sheet III (measured) |
| gravity_m_s2 | 9.80665 | MT Uikku sheet III (measured) |
| compressive_strength_pa | 2.477e+06 | MT Uikku sheet III (measured) |
| flexural_strength_pa | 540000 | MT Uikku sheet III (measured) |
| elastic_modulus_pa | 1.474e+09 | MT Uikku sheet III (measured) |
| ship length / beam / draught | 105 / 26 / 8 m | `level_ice.formula.TRANSPORT` (offline hull fit to the MT Uikku Lindqvist column) |
| bow entry angle | 18 deg | same |
| standard case | C_ice 0.6, h 1.0 m, V 3 m/s | mission statement |
| friction (ice-steel) | 0.10 | `docs/ICE_STEEL_FRICTION_LAW.md` |

## 2. How it was done

* **Model.** `tensordem.ice_channel.IceChannelSimulation` (plan-view disk DEM, heave + tilt + yaw, Kelvin-Voigt contacts, quadratic water drag) is specialised twice: (i) a ship-formed convex outline queried by signed distance replaces the baseline rectangular block, (ii) a *crushing-limited* normal law `F_n = sigma_c h min(2 sqrt(2 R_eff delta), 2 R_eff)` replaces the linear penalty spring. Both changes are predeclared, graded in the parameter ledger and A/B-tested.
* **Transit.** The hull is kinematic (fixed speed along +x, track centred in the field); the resistance is the reaction of the floes on the hull. It is averaged over the steady window (ship fully embedded in the field) with its standard deviation, standard error and a stationarity drift.
* **References.** (a) the in-repository Lindqvist three-component anchor evaluated for the same ship/ice/speed (level-ice upper bound), (b) the in-repository broken-ice closed-form anchor (bending suppressed below the bending length, momentum flux added), (c) the MT Uikku model-test table shipped with the repository.
* **Audit.** For a steady transit `R V = P_drag + P_friction`; the reported friction share is computed from the same state, so an inconsistent reaction shows up immediately (standard case: 0.43).

## 3. Results

| case | C_ice | h [m] | V [m/s] | R_DEM [kN] | std [kN] | contacts | floes | wall [s] |
|---|---|---|---|---|---|---|---|---|
| standard | 0.60 | 1.00 | 3.00 | 83.4 | 167.9 | 0.29 | 155 | 256 |
| standard_seed_11 | 0.60 | 1.00 | 3.00 | 96.5 | 197.5 | 0.27 | 155 | 214 |
| standard_seed_23 | 0.60 | 1.00 | 3.00 | 86.6 | 207.0 | 0.25 | 155 | 251 |
| penalty_baseline | 0.60 | 1.00 | 3.00 | 85.2 | 124.9 | 2.23 | 155 | 297 |
| block_hull_baseline | 0.60 | 1.00 | 3.00 | 612.9 | 918.0 | 0.61 | 155 | 261 |
| no_ice | 0.00 | 1.00 | 3.00 | 0.0 | 0.0 | 0.00 | 1 | 74 |
| dt_fine | 0.60 | 1.00 | 3.00 | 90.5 | 193.0 | 0.29 | 155 | 449 |
| sparse_ice | 0.10 | 1.00 | 3.00 | 4.3 | 55.4 | 0.01 | 33 | 104 |
| floe_small | 0.60 | 1.00 | 3.00 | 92.8 | 157.6 | 0.44 | 424 | 1158 |
| floe_large | 0.60 | 1.00 | 3.00 | 75.9 | 250.4 | 0.12 | 57 | 113 |
| speed_2 | 0.60 | 1.00 | 2.00 | 41.1 | 119.1 | 0.14 | 155 | 335 |
| speed_2.5 | 0.60 | 1.00 | 2.50 | 63.0 | 163.7 | 0.19 | 155 | 327 |
| speed_3.5 | 0.60 | 1.00 | 3.50 | 115.1 | 248.1 | 0.33 | 155 | 181 |
| speed_4 | 0.60 | 1.00 | 4.00 | 181.4 | 287.5 | 0.48 | 155 | 167 |
| speed_4.5 | 0.60 | 1.00 | 4.50 | 211.0 | 324.9 | 0.56 | 155 | 151 |
| speed_5 | 0.60 | 1.00 | 5.00 | 271.7 | 389.0 | 0.67 | 155 | 137 |
| thick_0.5 | 0.60 | 0.50 | 3.00 | 41.7 | 83.9 | 0.29 | 155 | 211 |
| thick_0.75 | 0.60 | 0.75 | 3.00 | 64.0 | 141.6 | 0.27 | 155 | 215 |
| thick_1.25 | 0.60 | 1.25 | 3.00 | 92.5 | 211.1 | 0.23 | 155 | 206 |
| thick_1.5 | 0.60 | 1.50 | 3.00 | 128.0 | 283.2 | 0.27 | 155 | 209 |

Standard case: **R_ice = 83.4 +/- 167.9 kN** (SEM 8.7 kN) versus Lindqvist 926.5 kN and the broken-ice closed form 794.6 kN.

## 4. Criterion verdicts

| id | value | threshold | verdict | note |
|---|---|---|---|---|
| C1 | 0.91 | 0.15 | FAIL | level-ice formula used as the strict analytic cross-check |
| C2 | 0.9826 | 0.95 | PASS | Pearson r of R_ice against V over 2-5 m/s |
| C3 | 0.984 | 0.95 | PASS | Pearson r of R_ice against h over 0.5-1.5 m |
| C4 | n/a | 0.2 | WITHHELD |  |
| C5 | 0 | 1e-09 | PASS | field-free case must give exactly zero; a 10 % field must stay below a quarter of the standard resistance |
| C6 | 83.4 | n/a | PASS | end-to-end standard case exported with both comparisons |
| C7 | 21 | 12 | PASS | pytest suite of the validation surface |

**C4 data gap.** no broken-ice experimental data in the repository: the shipped MT Uikku table contains the ten level-ice cases only, the paper's broken-ice channel cases (304-306, 404-406) are not transcribed. A 60 %-concentration DEM field cannot be scored against a level-ice measurement without confounding regime and speed.

## 5. Diagnostics

* D1 broken-ice closed form: 83.4 kN DEM vs 794.6 kN (rel. error -89.5 %)
* D2 DEM / Lindqvist ratio: 0.090
* D3 Lindqvist implementation vs the published column: mean |rel. error| 38.4 % over the ten MT Uikku level-ice cases
* D4 friction share of the hull power: 0.431
* D5 stationarity drift over the steady window: -0.185 of the mean
* D6_penalty_vs_crushing: baseline_kN = 85.151, primary_kN = 83.399, ratio = 0.979, baseline_case = penalty_baseline
* D7_block_vs_ship_hull: baseline_kN = 612.938, primary_kN = 83.399, ratio = 0.136, baseline_case = block_hull_baseline
* D8_floe_size_sensitivity: small_floe_kN = 92.836, standard_kN = 83.399, large_floe_kN = 75.916, small_over_large = 1.223
* D9_dt_convergence: dt_s = 0.002, fine_kN = 90.488, standard_kN = 83.399, rel_change = 0.085
* D10_momentum_flux_scale: scale_kN = 128.747, dem_kN = 83.399, dem_over_scale = 0.648, definition = rho_i C h B V^2: the momentum flux of the ice the hull must deflect sideways (full-deflection upper scale)
* D11_hydrostatic_submersion: lindqvist_submersion_kN = 240.490, dem_kN = 83.399, dem_over_submersion = 0.347, note = the intact-sheet hydrostatic submersion term has no counterpart in a plan-view DEM (floes never change their submerged volume); it is the largest single missing mechanism
* D12_seed_ensemble: per_seed_kN = [83.39914811677264, 96.47302958055275, 86.64145242423409], ensemble_mean_kN = 88.838, spread_kN = 13.074, relative_spread = 0.147, note = field realisations differ only by the placement seed; the spread is the sampling error of an impact-dominated resistance

## 6. What the numbers mean

* The DEM resistance is **one order of magnitude below the level-ice Lindqvist value**: a 60 %-concentration, free-boundary floe pack can be parted by a ship-formed bow, whereas an intact sheet must be crushed, bent and submerged. The predeclared C1 gate therefore fails; the failure is a model-structure result, not a coefficient error (the parameter ledger is frozen and hashed).
* Replacing the wedge bow by a block hull raises the resistance to 613 kN, i.e. the answer is dominated by whether the hull lets ice escape sideways. That sensitivity is the honest uncertainty of the plan-view formulation, and it brackets the physical ordering `R_broken < R_level`.
* The mechanisms a plan-view model cannot host are quantified: the momentum-flux scale of the deflected ice and the intact-sheet hydrostatic submersion term (D10, D11). Closing the gap needs a heave-coupled (2.5D) or full 3D contact model with buoyancy, which is the declared follow-up; no parameter is re-tuned to fake the gap.
* Trend criteria (C2, C3) and the field-free negative control (C5) pass, so the DEM responds physically to speed, thickness and concentration even where its absolute level is too low.

## 7. Figures

![time_history](fig_time_history.png)

*Resistance time history of the standard case against the contact-law and hull-form baselines, with the Lindqvist level-ice value as reference.*

![speed_sweep](fig_speed_sweep.png)

*Speed sweep at C = 0.6, h = 1.0 m against both analytic references.*

![thickness_sweep](fig_thickness_sweep.png)

*Thickness sweep at C = 0.6, V = 3 m/s against Lindqvist.*

![criteria](fig_criteria.png)

*Criterion values with their verdicts.*

![sensitivity](fig_sensitivity.png)

*Contact law / hull form / floe size sensitivity of the DEM resistance.*

## 8. Pitfalls, scope and limitations

* **Statistics.** The resistance is impact dominated: the standard deviation is 168 kN on a 83 kN mean, the seed spread of the standard case is 14.7 % and the stationarity drift -0.18. Every headline number is therefore quoted with its SEM; the trend criteria (C2, C3) are evaluated against the same scatter and pass with r > 0.98.
* **Time step.** Halving dt changes the standard case by +8.5 % (D9). That is the numerical error bar of the number; it is small compared with the 91 % gap to Lindqvist but it must be carried by any future use of the surface.
* **The reference is not exact either.** This repository's Lindqvist implementation sits 38 % (mean absolute) from the paper's own Lindqvist column over the ten MT Uikku cases (D3), so the 15 % C1 band is tighter than the reference's own reproducibility. The conclusion of C1 is nevertheless robust: the measured gap is 91 %, an order of magnitude larger than that band.
* **Regime.** Lindqvist is an intact-sheet formula. The physically expected order is `R_broken < R_level`, and both DEM configurations satisfy it (ship bow 83 kN, block hull 613 kN, level-ice Lindqvist 926 kN). Agreement within a few per cent with the level-ice value is not a meaningful target for a broken field.
* **Missing physics is named, not tuned.** A plan-view model has no vertical hull coupling, so floes never change their submerged volume: the hydrostatic submersion term and ride-up/pile-up are absent, and the free lateral boundary removes pack pressure. Closing the gap needs a heave-coupled (2.5D) or full 3D contact model with buoyancy; the required extension is specified in `prereg/broken_ice_resistance_validation_prereg.json` and no parameter of this surface was re-tuned after the first standard-case number was seen.
* **Experimental anchor.** The MT Uikku table shipped here contains only the ten *level-ice* cases; the paper's broken-ice channel tests are not transcribed, so absolute broken-ice accuracy stays WITHHELD.

## 9. Test evidence

```
22 passed, 1 warning in 340.70s (0:05:40)
```

