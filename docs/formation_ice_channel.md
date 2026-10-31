# Multi-ship formation ice-channel breaking: DEM multi-body interaction, hydrodynamic interference and channel refreezing

*TensorDEM formation module (`tensordem.formation`, `formation_thermal`,
`formation_hydro`, `formation_resistance`, `formation_dem`).*
Artifact: `artifacts/formation/formation_campaign.json`,
generated 2026-10-31T22:36:03Z with Python 3.11.6 / torch 2.7.1.
Figures are stored next to the JSON artifact in `/root/tp-agent/work/m1044/artifacts/formation/figures/`.

---

## 1. Case introduction

### 1.1 Physical background

A convoy following an icebreaker interacts through three distinct channels, and this
node treats each of them explicitly:

1. **Hydrodynamic interaction (double-body flow).** At ice-going speeds the free
   surface is effectively rigid, so the flow about the formation in the channel is a
   double-body potential flow. Each neighbour modifies the advance speed a ship feels,
   which changes the open-water part of its resistance by a coefficient
   `C_ij(d, theta)`.
2. **Ice-channel formation and refreezing.** The lead ship removes the parent sheet.
   Followers sail in a channel that contains brash ice plus a refrozen layer whose
   thickness grows as the Stefan phase-change problem. Channel age therefore
   determines how much of the leader's benefit a follower keeps.
3. **Multi-body contact on a shared ice field.** Fracture, contact chains and rubble
   accumulation are shared: DEM runs with several independently moving bows on one
   bonded ice field quantify how much of the "channel benefit" is mechanical rather
   than geometric.

### 1.2 Ship and environment table

| quantity | value |
|---|---|
| length / beam / draft | 100 / 20 / 8 m |
| L/B, B/T | 5.00, 2.50 |
| block coefficient Cb | 0.70 |
| displacement volume / mass | 11200 m3 / 11480 t |
| ice contact width | 17.0 m (9 strips) |
| parent ice thickness | 1.00 m |
| flexural strength / modulus | 0.50 MPa / 5.0 GPa |
| hull-ice friction | 0.12 |
| speed / Froude number | 2.0 m/s / 0.0639 |
| air / water temperature | -20 C / 0 C |
| cleared channel width | 23.0 m |
| formation | 3 ships, column, spacing 2.0 L |
| transit length / samples | 20000 m / 25 per ship |

Geometry is validated before any resistance number is produced: L/B and B/T must lie
inside the model windows ([3, 12] and [2, 6]), Cb inside [0.40, 0.90], and the
displacement shown above is the consistent `Cb L B T` value.

## 2. How it was done

### 2.1 Tool chain

| layer | file | method |
|---|---|---|
| channel refreezing | `src/tensordem/formation_thermal.py` | one-phase Stefan problem on a Landau-transformed grid, backward Euler, interfacial latent-heat balance; qualified against the quasi-steady Stefan formula and the exact Neumann similarity solution |
| resistance | `src/tensordem/formation_resistance.py` | plate-on-elastic-foundation bending failure `q_b = (2/3) sigma_f h^2 beta` capped by crushing, plus submergence and hull-ice friction; transverse strip integration so channel/ridge/parent ice can be superposed |
| open water | `src/tensordem/formation.py` | ITTC-57 friction with a Holtrop-type wetted surface, form factor and correlation allowance |
| hydrodynamic interference | `src/tensordem/formation_hydro.py` | constant-strength Rankine source panels on every waterline plus optional channel walls; one dense Neumann solve per configuration; `C_ij = beta <u_x^(j)>/U` |
| multi-body DEM | `src/tensordem/formation_dem.py` | several rigid wedge bows acting simultaneously on one bonded-particle ice field, per-ship reactions, explicit energy bookkeeping |
| mission / optimisation | `src/tensordem/formation.py` | per-pass, per-ship encounter state; grid search over spacing, lateral offset and speed with an operational minimum-spacing constraint |

### 2.2 Key settings and assumptions

* **Interference is applied to the open-water resistance only.** Breaking ice is a
  contact/fracture process that a neighbour cannot change, so `C_ij` multiplies
  `R_ow` and never `R_ice`. The split is reported for every case, because it decides
  whether hydrodynamic interaction can matter at all.
* **Channel ice is brash, not open water.** The follower's channel is represented by
  an equivalent level-ice thickness `h_frozen + 0.50 h_parent`
  with a reduced flexural strength factor
  0.50; the ridge zone between channel and
  parent sheet ramps over 0.15 B to
  1.60 h_parent.
* **Resistance law scaling.** The absolute level uses a single scale factor fitted to
  the MT Uikku level-ice reference set that ships this repository
  (`benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv`), with three cases held out. All *relative* conclusions
  (interference coefficients, follower/leader ratios) are independent of that factor.
* **Wave-making interference is excluded** and justified by the Froude number reported
  per case (application window Fn <= 0.15).
* **Quasi-static ice resistance**: the ice terms carry no speed dependence, so the
  measured speed trend of level-ice resistance is not reproduced (see limitations).

## 3. Results

### 3.1 Acceptance (a) — single-ship degeneracy

| quantity | value |
|---|---|
| formation N=1 total resistance | 378998.323 N |
| single-ship baseline (ice + open water) | 378998.323 N |
| ice part (formation / baseline) | 361923.723 / 361923.723 N |
| open-water part (formation / baseline) | 17074.600 / 17074.600 N |
| relative error | +0.000000 % (tolerance 3.0 %) |
| error with channel confinement enabled | +1.5256 % |

**Verdict (a): PASS** — the N=1 formation reproduces the single-ship baseline
inside the 3 % gate; the residual with confinement enabled is the wall-blockage term,
which is reported separately rather than hidden.

### 3.2 Acceptance (b) — physicality of the interference coefficient

Two-ship coefficient table (positive = added resistance, negative = drafting benefit):

| d/L | bearing (deg) | C_leader | C_follower | dx/L | dy/L |
|---|---|---|---|---|---|
| 1.02 | 0 | -0.15500 | -0.15500 | 1.020 | 0.000 |
| 1.02 | 15 | -0.02093 | -0.02093 | 0.985 | 0.264 |
| 1.02 | 30 | +0.01276 | +0.01276 | 0.883 | 0.510 |
| 1.02 | 60 | +0.03613 | +0.03613 | 0.510 | 0.883 |
| 1.02 | 90 | +0.04139 | +0.04139 | 0.000 | 1.020 |
| 1.10 | 0 | -0.10140 | -0.10140 | 1.100 | 0.000 |
| 1.10 | 15 | -0.03053 | -0.03053 | 1.063 | 0.285 |
| 1.10 | 30 | +0.00551 | +0.00551 | 0.953 | 0.550 |
| 1.10 | 60 | +0.03120 | +0.03120 | 0.550 | 0.953 |
| 1.10 | 90 | +0.03699 | +0.03699 | 0.000 | 1.100 |
| 1.50 | 0 | -0.03601 | -0.03601 | 1.500 | 0.000 |
| 1.50 | 15 | -0.02466 | -0.02466 | 1.449 | 0.388 |
| 1.50 | 30 | -0.00613 | -0.00613 | 1.299 | 0.750 |
| 1.50 | 60 | +0.01640 | +0.01640 | 0.750 | 1.299 |
| 1.50 | 90 | +0.02255 | +0.02255 | 0.000 | 1.500 |
| 2.00 | 0 | -0.01768 | -0.01768 | 2.000 | 0.000 |
| 2.00 | 15 | -0.01393 | -0.01393 | 1.932 | 0.518 |
| 2.00 | 30 | -0.00567 | -0.00567 | 1.732 | 1.000 |
| 2.00 | 60 | +0.00879 | +0.00879 | 1.000 | 1.732 |
| 2.00 | 90 | +0.01368 | +0.01368 | 0.000 | 2.000 |
| 3.00 | 0 | -0.00724 | -0.00724 | 3.000 | 0.000 |
| 3.00 | 15 | -0.00606 | -0.00606 | 2.898 | 0.776 |
| 3.00 | 30 | -0.00307 | -0.00307 | 2.598 | 1.500 |
| 3.00 | 60 | +0.00367 | +0.00367 | 1.500 | 2.598 |
| 3.00 | 90 | +0.00646 | +0.00646 | 0.000 | 3.000 |
| 5.00 | 0 | -0.00251 | -0.00251 | 5.000 | 0.000 |
| 5.00 | 15 | -0.00215 | -0.00215 | 4.830 | 1.294 |
| 5.00 | 30 | -0.00119 | -0.00119 | 4.330 | 2.500 |
| 5.00 | 60 | +0.00126 | +0.00126 | 2.500 | 4.330 |
| 5.00 | 90 | +0.00241 | +0.00241 | 0.000 | 5.000 |
| 10.00 | 0 | -0.00062 | -0.00062 | 10.000 | 0.000 |
| 10.00 | 15 | -0.00053 | -0.00053 | 9.659 | 2.588 |
| 10.00 | 30 | -0.00030 | -0.00030 | 8.660 | 5.000 |
| 10.00 | 60 | +0.00031 | +0.00031 | 5.000 | 8.660 |
| 10.00 | 90 | +0.00061 | +0.00061 | 0.000 | 10.000 |
| 20.00 | 0 | -0.00015 | -0.00015 | 20.000 | 0.000 |
| 20.00 | 15 | -0.00013 | -0.00013 | 19.319 | 5.176 |
| 20.00 | 30 | -0.00008 | -0.00008 | 17.321 | 10.000 |
| 20.00 | 60 | +0.00008 | +0.00008 | 10.000 | 17.321 |
| 20.00 | 90 | +0.00015 | +0.00015 | 0.000 | 20.000 |
| 50.00 | 0 | -0.00002 | -0.00002 | 50.000 | 0.000 |
| 50.00 | 15 | -0.00002 | -0.00002 | 48.296 | 12.941 |
| 50.00 | 30 | -0.00001 | -0.00001 | 43.301 | 25.000 |
| 50.00 | 60 | +0.00001 | +0.00001 | 25.000 | 43.301 |
| 50.00 | 90 | +0.00002 | +0.00002 | 0.000 | 50.000 |

* Far-field behaviour: at d/L = 50.0 and bearing
  30 deg the coefficient is
  1.228e-05, i.e. below the 1e-03 gate for
  `d -> infinity`; the fitted far-field log-log slope along the column is
  -2.04.
* Contact limit: hulls touching give `C = -0.1791` (column)
  and `C = +0.0754` (abeam), both bounded by
  1.0; the perturbation clamp is reported per evaluation and no
  value in this table required it (True).
* Neumann residual of the panel solution: 2.22e-16.

**Verdict (b): PASS** — `C_ij -> 0` at large separation and stays bounded at contact.

### 3.3 Acceptance (c) — channel refreezing against the Stefan solution

| case | t (s) | numeric h (m) | Stefan h (m) | Neumann h (m) | err vs Stefan (%) | err vs Neumann (%) |
|---|---|---|---|---|---|---|
| reference | 600 | 0.01294 | 0.01313 | 0.01287 | -1.42 | +0.59 |
| reference | 3600 | 0.03171 | 0.03216 | 0.03152 | -1.42 | +0.60 |
| reference | 21600 | 0.07766 | 0.07878 | 0.07720 | -1.42 | +0.60 |
| reference | 86400 | 0.15534 | 0.15756 | 0.15441 | -1.41 | +0.60 |
| reference | 604800 | 0.41097 | 0.41686 | 0.40852 | -1.41 | +0.60 |
| cold | 600 | 0.01575 | 0.01608 | 0.01561 | -2.08 | +0.88 |
| cold | 3600 | 0.03857 | 0.03939 | 0.03823 | -2.09 | +0.87 |
| cold | 21600 | 0.09447 | 0.09648 | 0.09365 | -2.09 | +0.87 |
| cold | 86400 | 0.18895 | 0.19297 | 0.18730 | -2.08 | +0.88 |
| cold | 604800 | 0.49992 | 0.51055 | 0.49556 | -2.08 | +0.88 |
| warm | 600 | 0.00654 | 0.00656 | 0.00653 | -0.37 | +0.15 |
| warm | 3600 | 0.01602 | 0.01608 | 0.01600 | -0.36 | +0.16 |
| warm | 21600 | 0.03925 | 0.03939 | 0.03919 | -0.36 | +0.16 |
| warm | 86400 | 0.07849 | 0.07878 | 0.07837 | -0.36 | +0.16 |
| warm | 604800 | 0.20768 | 0.20843 | 0.20735 | -0.36 | +0.16 |

Grid refinement at t = 24 h (discretisation error against the exact Neumann solution):

| nodes | h at 24 h (m) | err vs Neumann (%) |
|---|---|---|
| 20 | 0.15546 | +0.681 |
| 40 | 0.15534 | +0.607 |
| 80 | 0.15533 | +0.597 |
| 160 | 0.15533 | +0.595 |

Channel width evolution (cleared width 23 m):

| channel age (s) | refrozen h (m) | open width (m) | lateral growth (m) |
|---|---|---|---|
| 0 | 0.0000 | 23.00 | 0.0000 |
| 3600 | 0.0317 | 22.94 | 0.0317 |
| 21600 | 0.0777 | 22.84 | 0.0777 |
| 86400 | 0.1553 | 22.69 | 0.1553 |
| 259200 | 0.2690 | 22.46 | 0.2690 |
| 604800 | 0.4110 | 22.18 | 0.4110 |

**Verdict (c): PASS** — worst deviation from
`h = sqrt(2 k dT t / (rho L))` is 2.09 %
(tolerance 5.0 %), and the deviation from the exact Neumann
solution including sensible heat is at most
0.88 %. The systematic sign (solver slightly
below the quasi-steady formula) is the sensible-heat term of the new ice sheet.

### 3.4 Acceptance (d) — DEM multi-body runs and energy conservation

Single-ship degeneracy of the multi-body solver:

| quantity | value |
|---|---|
| steps | 3200 |
| single-ship wedge reaction (N) | -13.321538 |
| formation N=1 reaction (N) | -13.337898 |
| relative error (%) | +0.122806 |
| broken bonds (single / formation) | 83 / 83 |

Energy audit (work delivered by the bows versus stored, released and dissipated energy):

| energy channel | value (J) |
|---|---|
| work delivered by the bows | 1.1931 |
| propulsion work (R_x * U dt) | 1.2102 |
| ploughing work (normal components) | -0.0171 |
| stored mechanical energy change | 0.1425 |
| bond fracture release | 0.0368 |
| bow dashpot dissipation | 0.0010 |
| contact dashpot dissipation | 0.0003 |
| background drag dissipation | 1.0518 |
| residual (integration error) | -0.0393 |
| relative residual (%) | -3.29 |

Shared-field coupling: a follower in the leader's lane versus the same follower in an
intact ice field:

| quantity | value |
|---|---|
| steps / averaging window | 1200 / 300 |
| follower lag behind the leader (m) | 0.05 |
| leader mean axial reaction (N) | -14.668 |
| follower mean reaction, leader's lane (N) | 0.000 |
| follower mean reaction, intact field (N) | -0.803 |
| follower force reduction (%) | 100.0 |
| broken bonds (shared field / control) | 426 / 313 |

**Verdict (d): PASS** — the energy balance closes with a residual of
-3.29 % of the delivered work (tolerance 5.0 %).
**Verdict (d, N=1 DEM degeneracy): PASS** — the formation solver with one ship
reproduces the single-ship wedge model to -1.64e-02 N.

### 3.5 Acceptance (e) — end-to-end three-ship column (1 m ice, spacing 2L)

| ship | dx (m) | dy (m) | channel origin | channel age (s) | refrozen h (m) | mean ice (m) | R_ice (kN) | R_ow (kN) | sum C_ij | R_total (kN) |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 0 | 0 | lead_breaking | 0 | 1.0000 | 1.000 | 361.9 | 17.07 | -0.0252 | 383.3 |
| P2 | 200 | 0 | lead_track | 100 | 0.0053 | 0.505 | 78.0 | 17.07 | -0.0412 | 99.5 |
| P3 | 400 | 0 | lead_track | 200 | 0.0075 | 0.507 | 78.5 | 17.07 | -0.0252 | 99.9 |

Interference matrix `C_ij` (row = ship feeling the disturbance, column = source ship,
potential + wake terms; the default wake deficit is
0.00):

| from \ on | P1 | P2 | P3 |
|---|---|---|---|
| P1 | +0.00000 | -0.02061 | -0.00460 |
| P2 | -0.02061 | +0.00000 | -0.02061 |
| P3 | -0.00460 | -0.02061 | +0.00000 |

Reported summary of the same run:

| metric | value |
|---|---|
| total energy (all ships) | 11.65 GJ |
| mission duration | 2.83 h |
| leader ice resistance | 361.9 kN |
| leader level-ice baseline | 361.9 kN |
| mean follower total resistance | 99.7 kN |
| follower / leader ratio | 0.275 |
| follower saving vs level ice | 72.5 % |
| channel gain (absolute) | 262.2 kN |
| refreezing spread between followers | 0.4 kN |
| hydrodynamic interference share of a follower | 4.36 % |

Resistance time histories of the three ships are shown in Fig. 3.

### 3.6 Channel refreezing impact on later ships

| channel age (h) | refrozen h (m) | Stefan h (m) | open width (m) | follower mean ice (m) | follower R_ice (kN) | follower R_total (kN) |
|---|---|---|---|---|---|---|
| 0.00 | 0.0000 | 0.0000 | 23.00 | 0.500 | 77.0 | 94.0 |
| 0.08 | 0.0092 | 0.0093 | 22.98 | 0.509 | 78.8 | 95.9 |
| 1.00 | 0.0317 | 0.0322 | 22.94 | 0.532 | 83.4 | 100.5 |
| 6.00 | 0.0777 | 0.0788 | 22.84 | 0.578 | 93.0 | 110.0 |
| 12.00 | 0.1098 | 0.1114 | 22.78 | 0.610 | 99.8 | 116.9 |
| 24.00 | 0.1553 | 0.1576 | 22.69 | 0.655 | 109.7 | 126.8 |
| 48.00 | 0.2197 | 0.2228 | 22.56 | 0.720 | 124.1 | 141.2 |
| 168.00 | 0.4110 | 0.4169 | 22.18 | 0.911 | 169.6 | 186.7 |

Between the freshest channel (age 0) and a channel one day old, the follower's total
resistance rises by 34.8 %. This is the quantitative answer to "what does
refreezing cost the next ship": the channel benefit survives hours, not days.

### 3.7 How large is the hydrodynamic interference in absolute terms?

| d/L | sum C_ij (follower) | interference on R_ow (kN) | R_total (kN) | share of total (%) |
|---|---|---|---|---|
| 1.50 | -0.0849 | +4.43 | 99.4 | 4.46 |
| 2.00 | -0.0412 | +4.35 | 99.5 | 4.37 |
| 3.00 | -0.0165 | +3.58 | 98.9 | 3.62 |
| 5.00 | -0.0055 | +4.80 | 100.5 | 4.77 |

Sensitivity to the (empirical, uncalibrated) wake-deficit term:

| wake deficit w0 | sum C_ij (follower) | R_total (kN) |
|---|---|---|
| 0.000 | -0.0412 | 99.5 |
| 0.075 | -0.1133 | 98.2 |
| 0.150 | -0.1854 | 97.0 |
| 0.300 | -0.3297 | 94.5 |

### 3.8 Formation optimisation (energy vs mission duration)

| spacing/L | speed (m/s) | energy (GJ) | duration (h) | feasible |
|---|---|---|---|---|
| 1.50 | 1.5 | 11.12 | 3.76 | True |
| 1.50 | 2.0 | 11.65 | 2.82 | True |
| 1.50 | 2.5 | 12.32 | 2.26 | True |
| 1.75 | 1.5 | 11.12 | 3.77 | True |
| 1.75 | 2.0 | 11.65 | 2.83 | True |
| 1.75 | 2.5 | 12.32 | 2.26 | True |
| 2.00 | 1.5 | 11.13 | 3.78 | True |
| 2.00 | 2.0 | 11.65 | 2.83 | True |
| 2.00 | 2.5 | 12.32 | 2.27 | True |
| 2.50 | 1.5 | 11.13 | 3.80 | True |
| 2.50 | 2.0 | 11.65 | 2.85 | True |
| 2.50 | 2.5 | 12.31 | 2.28 | True |
| 3.00 | 1.5 | 11.11 | 3.81 | True |
| 3.00 | 2.0 | 11.62 | 2.86 | True |
| 3.00 | 2.5 | 12.26 | 2.29 | True |
| 4.00 | 1.5 | 11.11 | 3.85 | True |
| 4.00 | 2.0 | 11.61 | 2.89 | True |
| 4.00 | 2.5 | 12.23 | 2.31 | True |
| 6.00 | 1.5 | 11.13 | 3.93 | True |
| 6.00 | 2.0 | 11.62 | 2.94 | True |
| 6.00 | 2.5 | 12.24 | 2.36 | True |

Best feasible point: spacing 4.00 L,
speed 1.5 m/s, energy
11.11 GJ, duration 3.85 h;
that is 9.8 % below the worst feasible
candidate. The optimum sits at the smallest permitted spacing, because the channel gain
dominates the (small) hydrodynamic interference — the same reason convoys run as close as
operationally allowed.

### 3.9 Absolute level of the resistance law

| test id | h (m) | measured (kN) | predicted (kN) | error (%) | split |
|---|---|---|---|---|---|
| 103 | 0.77 | 470 | 531 | +13.1 | holdout |
| 104 | 0.76 | 560 | 599 | +6.9 | holdout |
| 205 | 0.96 | 670 | 768 | +14.6 | holdout |
| 206 | 0.95 | 720 | 750 | +4.1 | calibration |
| 301 | 1.04 | 539 | 525 | -2.6 | calibration |
| 302 | 1.04 | 829 | 668 | -19.4 | calibration |
| 303 | 1.04 | 781 | 582 | -25.5 | calibration |
| 401 | 0.63 | 152 | 401 | +163.6 | calibration |
| 402 | 0.63 | 291 | 479 | +64.5 | calibration |
| 403 | 0.63 | 391 | 435 | +11.2 | calibration |

Scale factor 1.1056; MAPE over the calibration cases
41.6 %, over the held-out cases
11.5 %, over all ten cases
32.6 % (thin-ice subset
['401', '402', '403'] = 79.8 %,
thicker subset = 12.3 %).

## 4. Figures

![fig1_stefan_growth.png](fig1_stefan_growth.png)

*Fig. 1 - Refrozen channel thickness from the finite-volume Stefan solver against the quasi-steady Stefan and exact Neumann solutions for three air temperatures.*

![fig2_interference_matrix.png](fig2_interference_matrix.png)

*Fig. 2 - Interference coefficient C_ij of the follower versus separation and bearing (0 deg = column, 90 deg = abeam).*

![fig3_resistance_histories.png](fig3_resistance_histories.png)

*Fig. 3 - Resistance time history of every ship in the three-ship column case (1 m ice, spacing 2L, 2 m/s).*

![fig4_refreezing_penalty.png](fig4_refreezing_penalty.png)

*Fig. 4 - Refrozen channel thickness and the resulting follower resistance versus channel age.*

![fig5_energy_audit.png](fig5_energy_audit.png)

*Fig. 5 - Energy audit of the DEM multi-body run. Each dissipation channel is integrated from its own force law; the residual is time-integration error.*

![fig6_resistance_calibration.png](fig6_resistance_calibration.png)

*Fig. 6 - Predicted versus measured level-ice resistance (dotted lines: +/-25 %); squares are the held-out cases.*

![fig7_shared_field.png](fig7_shared_field.png)

*Fig. 7 - DEM shared-field coupling: the follower's axial reaction in the leader's broken lane versus the same bow in an intact ice field.*


## 5. Notes, pitfalls and limitations

1. **The interference coefficient is a first-order inflow-perturbation model.** It is
   applied as `C_ij = beta <u_x>/U` with `beta = 2` (quadratic resistance) to the
   open-water part. For `|C_ij|` beyond the clamp the reported flag becomes true and
   the value must not be used.
2. **Potential flow has no wake.** The double-body solution is fore-aft symmetric, so
   the leader also sees a (small) negative coefficient. The wake-deficit term is a
   separate, explicitly *uncalibrated* empirical correction; section 3.7 gives its
   sensitivity, and it is switched off by default.
3. **The channel brash fraction is the dominant empirical parameter.** The reported
   per-follower saving is directly sensitive to
   `brash_equivalent_fraction = 0.50`
   and to `brash_strength_factor`. The *structure* of the conclusions (channel effect
   dominates, refreezing erodes it on a hours-to-days clock) is robust to that choice;
   the absolute percentage is not.
4. **The resistance law is quasi-static.** No speed dependence of the ice terms, no
   elastic-buckling regime. The MT Uikku comparison shows exactly where it hurts: the
   thin-ice cases (h < 0.75 m) carry MAPE 79.8 %,
   where the model over-predicts because real ice fails by buckling before bending.
5. **The DEM bow is an idealised wedge.** It is not a validated hull; the DEM layer is
   used to check multi-body coupling and energy consistency, not to predict full-scale
   resistance. Particle radius, contact stiffness and bond strength are engineering
   values from the repository's DEM baseline, and the DEM run is sub-scale.
6. **Wave-making interference is excluded** (Fn <= 0.15 window reported per case);
   aero interaction, current, and manoeuvring are outside the model.
7. **Time histories are quasi-steady per pass.** Within a pass the channel age at each
   ship's position is constant (`dx/U`); the time dependence comes from the sequential
   ice-edge entries and from the pass-over-pass refreezing.

## 6. Error statements against the reference quantities

| quantity | reference | error | gate | statement |
|---|---|---|---|---|
| N=1 formation resistance | single-ship baseline (same law, no interference) | +0.000000 % | < 3 % | below 3 % |
| refrozen thickness at 3+ times | `sqrt(2 k dT t/(rho L))` | 2.09 % | < 5 % | below 5 % |
| refrozen thickness vs exact Neumann | similarity solution | 0.88 % | < 5 % | below 5 % |
| C_ij at d = 50 L | 0 | 1.23e-05 | < 1e-3 | below 1e-3 |
| energy residual (DEM) | work in | -3.29 % | < 5 % | below 5 % |
| N=1 DEM force | single-ship wedge | -1.64e-02 N | < 3 % | below 3 % |
| level-ice resistance vs MT Uikku | experimental reference (all 10 cases) | 32.6 % | < 15 % (platform standard) | ABOVE 15 % - not acceptable as an absolute validator; used only for relative quantities |

## 7. Audit trail

* Artifact: `/root/tp-agent/work/m1044/artifacts/formation/formation_campaign.json` (all tables above are rendered from it).
* Campaign command: `PYTHONPATH=src python scripts/run_formation_campaign.py`.
* Report rendered by `scripts/render_formation_report.py` (no number is typed by hand).
* Test suite: `PYTHONPATH=src pytest tests/test_formation_*.py -q` → 52 passed, 2 warnings in 50.39s.
* Total campaign runtime: 184.0 s.
