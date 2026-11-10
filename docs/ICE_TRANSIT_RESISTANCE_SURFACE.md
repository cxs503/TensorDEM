# Steady-state broken-ice direct-navigation resistance (p156w2-n01)

**Scope.** This document is the deliverable of campaign node *p156w2-n01* (TensorDEM, broken-ice-field straight-ahead ice resistance).  The sibling node *p82w2-n05* built the plan-view transit surface and reported an honest FAIL of the 15 % Lindqvist gate together with two named structural causes (a finite domain whose ice plug grows instead of reaching a steady state, and a free lateral boundary that removes the pack pressure), plus one energy-inconsistent contact regularisation.  This node repairs all three, closes the energy ledger at the contact level, and turns the resistance question into a measured three-parameter sensitivity surface.

## 1. The decision this campaign supports

Does a plan-view floe pack with a *zero-offset crushing law*, a *recycling inlet* (true steady state over a route of 10 ship lengths) and a *compliant pack boundary* reproduce the Lindqvist (1989) level-ice resistance within 15 % in the consolidated limit, and does its energy ledger close?  The two competing explanations are separated by the concentration sweep and by the mechanism ledger:

* **H1 (measurement-limited):** the earlier shortfall was an artefact of the finite domain and the free lateral boundary; once repaired, R must approach Lindqvist as C -> 0.907.
* **H2 (structure-limited):** a plan-view disk pack has no out-of-plane bending/submersion mechanics, so it must remain a *bounded fraction* of the level-ice anchor at every concentration.

## 2. Method

**Field and hull.**  A jittered triangular floe field fills a conveyor window of 80 m ahead of the stem and 110 m abaft, inside a pack of width 100 m.  Floe radii are drawn from a unilateral spread that is *reduced automatically* at high concentration (one disk per lattice site must clear the largest disk): the geometric limit C(j) = (pi/2 sqrt3)(1+j^2/3)/(1+j)^2 reaches 0.9069 only for a monodisperse field, so a 0.90 field is nearly monodisperse and the realised spread is reported with each case.  A relaxation pass removes any placement overlap, so no elastic energy is injected at t = 0.  The hull is the MT Uikku equivalent transport hull of `tensordem.level_ice.formula.TRANSPORT` (L = 105 m, B = 26 m, 18 deg entry angle) as a convex outline.

**Contact law (new).**  The crushing chord `w = 2 sqrt(2 R delta)` is evaluated with the offset removed, `F = sigma_c h [2 sqrt(2 R (delta + eps)) - 2 sqrt(2 R eps)]`, capped at the floe diameter, with `eps = sigma_c R / E` (the elastic penetration at which an ice edge reaches its crushing strength: 8.4 mm for a 5 m floe).  `F(0) = 0` exactly and the closed-form potential `U(delta) = int_0^delta F ds` satisfies `dU/ddelta = F` (unit test).  The legacy surface used the same chord *without* subtracting the offset, which loads every touching floe with 1.1 MN at zero penetration.

**Hull geometry defect fixed.**  `ShipForm.signed_distance` returns the distance to the nearest *face plane*.  Near the 18 deg bow apex that is a poor lower bound of the true distance (a floe 7 m ahead of the stem is reported at 2.2 m), so floes were put in contact — and, under a strength-limited law, loaded with a ~24 MJ spurious contact — before they touched the hull.  `tensordem.ice_transit.laws.polygon_distance` returns the true point-to-polygon distance and is used throughout.

**Recycling inlet.**  A floe that falls behind the stern re-enters ahead of the stem at rest with equilibrium heave; the kinetic and potential energy it carries out is booked on the `removed` side of the ledger.  This replaces the growing plug of the finite domain by a genuine steady state over 10 ship lengths (1050 m, 350 s).

**Energy ledger (new).**  Every contact reports its elastic potential and its dashpot/Coulomb powers, so the audit

```
W_thrust + W_numerical + W_injected - W_removed - W_diss = d(KE + PE_heave + U_contact + U_wall)
```

is an identity, not an assumption.  `W_thrust = int R V dt` (not the floe-side work: with a kinematic boundary the two differ by the relative-frame work the contacts dissipate), and `W_numerical` is the exact `(1/2) F^2/m dt^2` term injected by the semi-implicit update.  The physical channels are water drag, hull dashpot, hull sliding, floe-floe dashpot, floe-floe sliding, pack-wall dashpot and heave damping.

**Parameters and grades.**  Material properties are the measured MT Uikku sheet III values shipped with the repository; friction is the cold steel-on-ice value pinned by `docs/ICE_STEEL_FRICTION_LAW.md`; literature agreement of the whole set is **WITHHELD** (the broken-ice channel tests 304-306/404-406 of the source paper are not transcribed offline).  The graded ledger hashes to `3410fb6fd4e1baf7...`.

## 3. Results

### 3.1 Standard case

| quantity | value |
|---|---|
| ice concentration / realised | 0.900 / 0.900 |
| ice thickness | 1.00 m |
| ship speed | 3.00 m/s |
| floe count / mean radius | 63 / 5.32 m |
| route | 1050 m (10.0 L) |
| steady window | 120-350 s |
| **R_ice (mean +/- SEM)** | **59.1 +/- 6.9 kN** (std 294.5 kN, peak 4116.5 kN) |
| Lindqvist (same ice, hull, speed) | 858.7 kN |
| late-window mean (last 40 % of the window, t >= 258 s) | 45.0 +/- 13.2 kN |
| relative error vs Lindqvist | 93.1% |
| momentum flux rho_i C h B V^2 | 193.1 kN |
| in-repository broken-ice anchor | 794.6 kN |
| energy-ledger residual | 5.34% of the thrust work |

### 3.2 Energy ledger of the standard case

| channel | energy [kJ] | share of thrust work |
|---|---|---|
| floe_normal_damping | 10287.3 | 14.7% |
| floe_sliding | 3094.4 | 4.4% |
| heave_damping | 0.0 | 0.0% |
| hull_normal_damping | 7538.1 | 10.8% |
| hull_sliding | 5745.3 | 8.2% |
| pack_wall | 1122.7 | 1.6% |
| water_drag | 40651.4 | 58.3% |
| stored: contact | 0.4 | 0.0% |
| stored: heave | 0.0 | 0.0% |
| stored: kinetic | 648.8 | 0.9% |
| stored: pack_wall | 0.0 | 0.0% |
| numerical injection (1/2 F^2/m dt^2) | 4433.5 | 6.4% |
| recycled out | 1392.0 | 2.0% |
| **thrust work R V T** | **69774.0** | 100 % |
| residual | 3727.1 | 5.34% |

### 3.3 Concentration sweep

| C | R_DEM [kN] | SEM [kN] | R_DEM/R_Lindqvist |
|---|---|---|---|
| 0.50 | 23.8 | 4.6 | 0.028 |
| 0.60 | 39.1 | 6.2 | 0.045 |
| 0.70 | 43.5 | 6.0 | 0.051 |
| 0.80 | 65.9 | 7.9 | 0.077 |
| 0.90 | 55.7 | 6.9 | 0.065 |

### 3.4 Thickness and strength sensitivity

| h [m] | R [kN] | SEM [kN] |
|---|---|---|
| 0.50 | 27.0 | 3.4 |
| 0.75 | 42.6 | 5.4 |
| 1.00 | 59.1 | 6.9 |
| 1.25 | 66.1 | 7.6 |
| 1.50 | 101.7 | 11.4 |

| sigma_c [MPa] | R [kN] | SEM [kN] |
|---|---|---|
| 1.00 | 50.3 | 5.0 |
| 1.50 | 50.2 | 5.6 |
| 2.48 | 59.1 | 6.9 |
| 3.70 | 55.2 | 7.7 |

### 3.5 Sensitivity to the declared model choices

| case | R [kN] | SEM [kN] | ledger residual |
|---|---|---|---|
| block_hull | 319.6 | 13.9 | 2.99% |
| dt_fine | 63.0 | 5.4 | 2.64% |
| floe_large | 98.3 | 15.0 | 2.30% |
| no_ice | 0.0 | 0.0 | n/a (degenerate: no thrust work) |
| no_pack_boundary | 59.2 | 7.4 | 5.29% |
| pack_narrow | 130.5 | 7.3 | 3.60% |
| seed_11 | 65.6 | 7.8 | 5.41% |
| seed_23 | 58.7 | 7.2 | 5.17% |
| standard | 59.1 | 6.9 | 5.34% |

## 4. Criterion verdicts

| id | statement | value | threshold | verdict |
|---|---|---|---|---|
| E1 | consolidated broken-ice field (C=0.90): |R_DEM - R_Lindqvist| / R_Lindqvist | 0.931 | 0.150 | **FAIL** |
| E2 | energy-ledger residual |W_thrust + W_num + W_inj - W_rem - W_diss - dE| / W_thrust | 0.053 | 0.100 | **PASS** |
| E3 | field-free control: resistance of a transit with no ice in the window | 0.000 | 1.000 | **PASS** |
| E4 | resistance increases with ice concentration (Pearson r) | 0.891 | 0.900 | **FAIL** |
| E5 | resistance increases with ice thickness; R ~ h^m with m in [1, 3] | 0.972 | 0.950 | **PASS** |
| E6 | resistance increases with crushing strength (Pearson r) | 0.668 | 0.900 | **FAIL** |
| E7 | steady state: |linear drift over the averaging window| / mean | 0.352 | 0.150 | **FAIL** |
| E8 | seed ensemble: (max - min) / mean of the steady resistance | 0.113 | 0.300 | **PASS** |
| E9 | pytest coverage of the new surface | 24.000 | 12.000 | **PASS** |
| E10 | transit route covers at least 10 ship lengths | 10.000 | 10.000 | **PASS** |

6 of 10 criteria pass; the headline gate E1 is **FAIL**.

## 5. Figures

![fig_history.png](fig_history.png)

*resistance history with the steady window and the Lindqvist anchor (sha256 `91b1d2a60698...`)*

![fig_concentration.png](fig_concentration.png)

*concentration sweep and the recovered fraction of Lindqvist (sha256 `cd66c6e8069b...`)*

![fig_thickness_strength.png](fig_thickness_strength.png)

*thickness and crushing-strength sensitivity of the steady resistance (sha256 `cd3b60095f49...`)*

![fig_ledger.png](fig_ledger.png)

*where the thrust work goes (contact-level energy ledger) (sha256 `251d88fb7e42...`)*

![fig_mechanisms.png](fig_mechanisms.png)

*Lindqvist component decomposition against the measured DEM resistance (sha256 `970b56286def...`)*

![fig_model_choices.png](fig_model_choices.png)

*resistance sensitivity to pack boundary, floe size, hull form and time step (sha256 `cd99ec19490a...`)*

## 6. Pitfalls, scope and limitations

* **Stationarity.** The predeclared averaging window starts at 120 s; the late-window mean over its last 40 % is 45.0 kN against 59.1 kN for the whole window (drift statistic -0.35).  The conveyor needs roughly ``(window_upstream + window_downstream) / V`` = 63 s to be populated, so a residual drift of this size is the honest uncertainty of the mean, not a converged number.
* **Statistical scatter.** The resistance is impact dominated: the standard-case standard deviation is 295 kN on a 59 kN mean (SEM 7 kN, 12%).  Every headline number is quoted with its SEM and the sweep criteria are evaluated on trends, not on single-case differences.
* **Numerical injection.** The semi-implicit update injects 6.4% of the thrust work (booked explicitly).  A halved time step (`dt_fine`) keeps the resistance outside the seed spread, i.e. the surface is time-step converged to the accuracy it is quoted at.
* **What the model still cannot host.** A plan-view disk pack has no out-of-plane degree of freedom for the ice, hence no bending failure and no buoyancy-driven submersion of a *continuous* sheet — the two Lindqvist mechanisms that carry most of the level-ice resistance.  The mechanism figure shows the gap component by component.
* **Concentration limit.** Non-overlapping disks cap the realisable coverage at 0.9069; a request above that is refused rather than clamped.  Fields above C ~ 0.75 are nearly monodisperse by geometry, which is a real difference from a natural floe field.
* **Cases not measured.** floe_small, pack_wide are in the predeclared case list but are *not* reported here: their floe count (315 and 112 floes, i.e. 25x and 3x the pair count of the standard case) put them outside the compute budget of this pass.  No number for them is quoted, and the floe-size / pack-width sensitivity is carried by `floe_large` and `pack_narrow` respectively.
* **Reference status.** Lindqvist's coefficient values are WITHHELD offline; the comparison tests *structure and order of magnitude*, never coefficient fidelity.  The MT Uikku broken-ice channel tests that would settle the broken-field question are not transcribed in this repository.
* **Route.** The recycling inlet is a declared model choice: it removes the growing plug of a finite domain but it also re-injects ice at rest, which is only exact for a *stationary* far field.

## 7. Reproduction

```bash
python scripts/run_ice_transit_campaign.py --workers 24   # all cases
python scripts/run_ice_transit_campaign.py --report        # artefacts + this file
python -m pytest tests/test_ice_transit.py -q      # 24 tests
```

## 8. Error statement (explicit)

* **R_ice vs Lindqvist:** 93.1% relative error (59.1 kN vs 858.7 kN) — **NOT < 3 %**; the predeclared 15 % gate is **FAIL**.
* **Energy ledger:** residual 5.34% of the thrust work — **NOT < 3 %**, gate (10%) **PASS**.
* **Contact law:** `dU/ddelta = F` to <0.02 % and `F(0) = 0` exactly (unit tests), i.e. the contact energy is consistent by construction.
* No experimental broken-ice resistance data are available offline, so the absolute level of the DEM resistance is **WITHHELD** against full-scale measurements; only the analytic anchors are quoted.

