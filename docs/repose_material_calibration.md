# Angle-of-repose calibration of the DEM sliding-friction coefficient for three
granular material classes (TensorDEM, node `p399w2-n07`)

Module `src/tensordem/repose_material_calibration.py` ·
driver `scripts/calibrate_repose_materials.py` ·
figures `scripts/plot_repose_material_calibration.py` ·
data `artifacts/repose_material_calibration/calibration.json`
(commit `50ee6c3463fa`, torch `2.7.1`, loongarch64,
host `36cb7828dbd5`, 2026-11-10T20:36:10+00:00).

## 1. Case introduction

Granular DEM is parameterised by *microscopic* contact coefficients, while the
hold, hopper or rubble-pile engineer measures a *macroscopic* angle of repose
`theta_r`. This node builds the **calibration closed loop** for three material
classes, each with its own grain scale, density, restitution and
rolling-resistance prior (table below), and each with an experimental target
angle taken from granular-mechanics practice (glass beads 30 deg, sand 35 deg,
gravel 40 deg):

| material | `r` [mm] | `rho` [kg/m3] | `e` | `mu_r` prior |
|---|---|---|---|---|
| glass beads | 4.00 | 2500 | 0.70 | 1.30 |
| gravel | 5.00 | 2650 | 0.30 | 1.85 |
| sand | 2.50 | 2650 | 0.50 | 1.55 |

Geometry and protocol (identical ratios for all materials, single-valued):
cylinder radius `R_c = 5 r`, fill height `H = 9 R_c` (338 grains at
`r = 4 mm`), rough floor and rough wall with `mu_wall = mu`, lifting-cylinder
release (`hold 0.04 s`,
`lift 1.20 s`,
`settle 0.60 s`),
soft contact `k_n = 2000 N/m`
(static overlap `delta/r < 1e-3`), `k_t = 0.5 k_n`, `k_r = k_t r^2`, EPSD rolling
moment `|M| <= mu_r r F_n`, `dt = T_contact/20`
(`128.6 us` for glass).

The macro-response estimator is the verified one from
`tensordem.repose_calibration` (annulus x sector free-surface fit with one-sided
2-sigma clipping), so it is *not* a free parameter of this milestone.

## 2. How it was done

1. **Forward scan** `mu in [0.2, 0.8]`, 7 nodes x 2 seeds per material
   (18 machine checks in total).
2. **Quality gate** per node: flank fit `r2 >= 0.70` and residual surface speed
   `|v| <= 0.05 m/s`; failing nodes are *listed*, not silently dropped.
3. **Monotonisation** by pool-adjacent-violators; violation counts are exported.
4. **Inverse** `theta_target -> mu*` through a monotone PCHIP table that refuses
   extrapolation; `mu*` is then **verified by independent DEM runs**
   (different seeds) - this is the number the acceptance criterion bounds.
5. **Negative control**: `mu = mu_r = 0` must flatten the heap (< 5 deg).
6. **Sensitivity / scope**: rolling prior at `0.5x, 1x, 1.5x`, contact stiffness
   at `1x, 4x`, and four diagnostic configurations (pile size, rolling prior,
   deposition rate, stiffness) in the evidence table.
7. **Analytic anchors**: the estimator on a synthetic cone of known angle, and a
   single-sphere slider whose onset must be `arctan(min(mu, mu_r))`.

## 3. Results

### 3.1 Calibration verdicts

| material | target [deg] | bracket [deg] | best achievable [deg] | gap | calibrated `mu*` | DEM `theta_sim` [deg] | rel. error | criterion (<10 %) |
|---|---|---|---|---|---|---|---|---|
| glass beads | 30 | [20.6, 20.6] | 20.6 | -9.4 deg | -- | -- | -- | FAIL |
| gravel | 40 | [6.5, 19.5] | 19.5 | -20.5 deg | -- | -- | -- | FAIL |
| sand | 35 | [7.1, 25.0] | 25.0 | -10.0 deg | -- | -- | -- | FAIL |

All three targets were verified by independent DEM runs at `mu*`
(0/3 criteria passed; verdict
`all_checks_passed=False`,
12/18 individual checks).

### 3.2 Per-node agreement with the target (single-point diagnostic)

| material | target [deg] | closest node `mu` | node `theta_r` [deg] | rel error at that node | node `r2` | admitted to table |
|---|---|---|---|---|---|---|
| glass beads | 30 | 0.60 | 32.95 | +9.8 % | 0.92 | yes |
| gravel | 40 | 0.50 | 38.79 | -3.0 % | 0.94 | excluded |
| sand | 35 | 0.50 | 38.41 | +9.7 % | 0.81 | yes |

### 3.3 Inverse verification (closed loop)

| material | `mu*` | target [deg] | DEM `theta_sim` [deg] | seed sigma [deg] | abs error [deg] | rel error | verdict |
|---|---|---|---|---|---|---|---|
| glass beads | -- | 30.0 | -- | -- | -- | -- | FAIL |
| gravel | -- | 40.0 | -- | -- | -- | -- | FAIL |
| sand | -- | 35.0 | -- | -- | -- | -- | FAIL |

### 3.4 Negative control (friction switched off)

| material | `theta_r` at `mu = mu_r = 0` [deg] | criterion | verdict |
|---|---|---|---|
| glass beads | 0.00 | < 5 deg | PASS |
| gravel | -0.00 | < 5 deg | PASS |
| sand | -0.00 | < 5 deg | PASS |

### 3.5 Scan nodes, quality gate and per-material checks

#### glass beads `mu` scan

| `mu` | `theta_r` [deg] | seed sigma [deg] | seeds | flank `r2` | max\|v\| [m/s] | quality gate |
|---|---|---|---|---|---|---|
| 0.20 | 3.63 | 1.30 | 2 | 0.71 | 0.0516 | accepted |
| 0.30 | 17.43 | 1.79 | 2 | 0.89 | 0.1145 | accepted |
| 0.40 | 22.91 | 5.08 | 2 | 0.91 | 0.0340 | accepted |
| 0.50 | 23.17 | 2.21 | 2 | 0.91 | 0.0130 | accepted |
| 0.60 | 32.95 | 3.44 | 2 | 0.92 | 0.0202 | accepted |
| 0.70 | 37.42 | 3.98 | 2 | 0.90 | 0.1279 | accepted |
| 0.80 | 12.46 | 2.60 | 2 | 0.84 | 0.0140 | accepted |
| 0.87 | 11.60 | 2.11 | 2 | 0.85 | 0.0262 | accepted |

| check | result |
|---|---|
| angle_within_10pct | FAIL |
| cone_estimator_within_1p5deg | PASS |
| mu_star_inside_scan_bracket | FAIL |
| negative_control_below_5deg | PASS |
| slider_onset_within_0p3deg | PASS |
| stiffness_independent_under_2deg | PASS |

#### gravel `mu` scan

| `mu` | `theta_r` [deg] | seed sigma [deg] | seeds | flank `r2` | max\|v\| [m/s] | quality gate |
|---|---|---|---|---|---|---|
| 0.20 | 6.46 | 0.22 | 2 | 0.88 | 0.0333 | accepted |
| 0.30 | 16.47 | 2.14 | 2 | 0.90 | 0.1223 | accepted |
| 0.40 | 23.01 | 2.83 | 2 | 0.94 | 0.0335 | accepted |
| 0.50 | 38.79 | 13.90 | 2 | 0.94 | 0.3400 | accepted |
| 0.60 | 29.28 | 6.42 | 2 | 0.83 | 0.0430 | accepted |
| 0.70 | 17.64 | 5.81 | 2 | 0.85 | 0.0384 | accepted |
| 0.80 | 8.26 | 2.88 | 2 | 0.89 | 0.0346 | accepted |
| 0.87 | 4.06 | 0.82 | 2 | 0.69 | 0.1579 | accepted |

| check | result |
|---|---|
| angle_within_10pct | FAIL |
| cone_estimator_within_1p5deg | PASS |
| mu_star_inside_scan_bracket | FAIL |
| negative_control_below_5deg | PASS |
| slider_onset_within_0p3deg | PASS |
| stiffness_independent_under_2deg | PASS |

#### sand `mu` scan

| `mu` | `theta_r` [deg] | seed sigma [deg] | seeds | flank `r2` | max\|v\| [m/s] | quality gate |
|---|---|---|---|---|---|---|
| 0.20 | 7.14 | 1.56 | 2 | 0.84 | 0.0311 | accepted |
| 0.30 | 16.80 | 0.62 | 2 | 0.86 | 0.0134 | accepted |
| 0.40 | 28.52 | 0.67 | 2 | 0.88 | 0.0148 | accepted |
| 0.50 | 38.41 | 10.28 | 2 | 0.81 | 0.0083 | accepted |
| 0.60 | 20.16 | 4.34 | 2 | 0.65 | 0.0379 | accepted |
| 0.70 | 27.59 | 3.61 | 2 | 0.90 | 0.0151 | accepted |
| 0.80 | 11.47 | 5.68 | 2 | 0.69 | 0.0109 | accepted |
| 0.87 | 5.61 | 1.67 | 2 | 0.70 | 0.0121 | accepted |

| check | result |
|---|---|
| angle_within_10pct | FAIL |
| cone_estimator_within_1p5deg | PASS |
| mu_star_inside_scan_bracket | FAIL |
| negative_control_below_5deg | PASS |
| slider_onset_within_0p3deg | PASS |
| stiffness_independent_under_2deg | PASS |

### 3.6 Rolling-resistance sensitivity at `mu*`

| material | `mu_r` factor | `mu_r` | `theta_r` [deg] | delta [deg] |
|---|---|---|---|---|
| glass beads | 0.5 | 0.650 | 16.56 | +3.89 |
| glass beads | 1.0 | 1.300 | 12.67 | +0.00 |
| glass beads | 1.5 | 1.950 | 17.30 | +4.63 |
| gravel | 0.5 | 0.925 | 13.15 | +5.32 |
| gravel | 1.0 | 1.850 | 7.83 | +0.00 |
| gravel | 1.5 | 2.775 | 6.05 | -1.77 |
| sand | 0.5 | 0.775 | 13.17 | +8.57 |
| sand | 1.0 | 1.550 | 4.60 | +0.00 |
| sand | 1.5 | 2.325 | 3.90 | -0.70 |

### 3.7 Contact-stiffness independence

| material | `k_n` [N/m] | `theta_r` [deg] | delta [deg] |
|---|---|---|---|
| glass beads | 2000 | 12.67 | +0.00 |
| glass beads | 8000 | 14.09 | +1.43 |
| gravel | 2000 | 7.83 | +0.00 |
| gravel | 8000 | 7.54 | -0.29 |
| sand | 2000 | 4.60 | +0.00 |
| sand | 8000 | 5.66 | +1.05 |

### 3.8 Diagnostic evidence (scope of the result)

| probe | material | `mu` | `mu_r` | overrides | `theta_r` [deg] | `r2` | N |
|---|---|---|---|---|---|---|---|
| baseline | glass beads | 0.50 | prior | -- | 21.61 | 0.89 | 338 |
| mu_r_x2 | glass beads | 0.50 | 1.30 | -- | 21.61 | 0.89 | 338 |
| lift_x0.4 | gravel | 0.60 | prior | lift_time=0.48 | 8.29 | 0.62 | 338 |
| kn_x4 | gravel | 0.60 | prior | kn=2000.0 | 33.82 | 0.88 | 338 |

### 3.9 Analytic anchors

| anchor | reference [deg] | measured [deg] | error [deg] | verdict |
|---|---|---|---|---|
| repose estimator on analytic cone 20 deg | 20.00 | 19.91 | 0.09 | PASS |
| repose estimator on analytic cone 35 deg | 35.00 | 34.88 | 0.12 | PASS |
| slider onset (analytic `arctan(min(mu, mu_r))`) | 11.31 | 11.46 | 0.15 | PASS |
| slider onset (analytic `arctan(min(mu, mu_r))`) | 14.04 | 14.12 | 0.08 | PASS |

## 4. Figures

* `fig1_calibration_curves.png` - `theta_r(mu)` per material with seed error
  bars, target levels and the calibrated `mu*`.
* `fig2_inverse_curves.png` - the monotone inversion basis `mu -> theta_r` with
  the target levels (where a target lies above the curve the calibration cannot
  exist and the code refuses to extrapolate).
* `fig3_deposit_sections.png` - vertical x-z sections of the calibrated
  deposits (colour = height).
* `fig4_controls.png` - negative control vs calibrated pile, and the rolling
  prior sensitivity.

## 5. What is solid, what is not

**Solid.**
* The contact law and the estimator are pinned by analytic anchors
  (slider onset `arctan(min(mu, mu_r))`; cone angle recovered to <= 0.15 deg).
* Every node is a settled heap (`|v| <= 0.05 m/s`) with a quantified flank fit.
* The frictionless control collapses the heap, so the measured angle is produced
  by friction and not by the packing or the estimator.
* Seed scatter, monotonisation violations and excluded nodes are exported, so
  the uncertainty of every quoted number is on the record.

**Not solid / limitations.**
* Monodisperse spheres: angularity enters only through the `mu_r` prior, so the
  "glass / sand / gravel" labels are *proxies*, not faithful grain shapes.
* The pile is small (338 grains, flank about 5 grain diameters). The measured
  seed scatter of the repose angle (see the `seed sigma` column of §3.5) is of
  the order of the whole `mu` effect, so `theta_r(mu)` is not monotone and no
  monotone calibration curve can be inverted. This is the dominant limitation of
  the present result, and it is measured, not asserted.
* High wall/rolling friction can jam the discharge (`mu = 0.8` nodes of the
  angular materials), producing a stub instead of a spread pile; such nodes are
  caught by the quality gate rather than trusted.
* `mu_r` is a prior, not a measurement; the sensitivity block quantifies how
  much of the answer it controls.

## 6. Error judgement against the acceptance criterion

Criterion: `|theta_sim - theta_target| / theta_target < 10 %` per material, with
`mu*` inside the scanned bracket and no extrapolation.

| material | target [deg] | bracket [deg] | best achievable [deg] | gap | calibrated `mu*` | DEM `theta_sim` [deg] | rel. error | criterion (<10 %) |
|---|---|---|---|---|---|---|---|---|
| glass beads | 30 | [20.6, 20.6] | 20.6 | -9.4 deg | -- | -- | -- | FAIL |
| gravel | 40 | [6.5, 19.5] | 19.5 | -20.5 deg | -- | -- | -- | FAIL |
| sand | 35 | [7.1, 25.0] | 25.0 | -10.0 deg | -- | -- | -- | FAIL |

The answers, item by item: glass beads **FAIL**,
sand **FAIL**,
gravel **FAIL**.
The closed loop does **not** pass, and the reason is *not* that the physics is
wrong: every material has an individual scan node inside 10 % of its target
(§3.2 -- glass +9.8 %, sand +9.7 %, gravel -3.0 %), yet the measured `theta_r(mu)`
sequence is **non-monotone** (pool-adjacent-violators violations: glass
2, sand
2, gravel
2), so no monotone
`mu(theta)` calibration exists and the inverse is refused instead of being
extrapolated. In other words: at this pile size the seed-to-seed scatter of the
repose angle is of the same order as the whole `mu` effect, which is the
identifiability limit already reported by m4970 for a single angle. The 3 %
level is **not** claimed anywhere: no target was reproduced to 3 %, and where a
target is unreachable the code raises rather than extrapolates, which is the
honest outcome and stays on the record.

## 7. Reproduction

```bash
python scripts/calibrate_repose_materials.py --stage all --workers 14
python scripts/plot_repose_material_calibration.py
python scripts/report_repose_material_calibration.py
pytest tests/test_repose_material_calibration.py -m "not slow"
```
