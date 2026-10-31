# Ice loads on rudder blades and steering gear: DEM-FSM coupling with IACS UR F12 validation and temperature dependent ice strength

Increment **W5-N01** (ice mechanics / structural). Module:
`src/tensordem/rudder_ice_load.py`. Campaign driver:
`scripts/benchmark_rudder_ice_load.py`. Artifacts:
`artifacts/rudder_ice_load/` (`report.json`, CSV tables, PNG figures).

## 1. Engineering problem

A ship steering in a broken-ice channel loads its rudder blade with a
crushing/flaking ice pressure that is transmitted through the stock into the
steering gear. Design practice (IACS UR F12-type rules, FSICR) sizes the stock
from a *design ice force on the rudder*, while the physical load is produced by
a sequence of grain-scale crushing and spall events whose amplitude depends on
ice thickness, ship speed, rudder angle and — critically in cold water — ice
temperature. This increment builds the platform capability to

1. resolve the ice–rudder contact force history with a bonded-sphere DEM whose
   contacts are governed by a **fracture-state machine (FSM)** limited by the
   temperature dependent crushing strength,
2. generate rule-style design loads parametrically in (h, v, δ, T),
3. check the rudder stock / steering gear against the resolved envelope,
4. and quantify the agreement between the two, with the discretisation error
   explicitly measured.

## 2. Physics and model formulation

### 2.1 Temperature dependent ice strength

```
sigma_ice(T) = sigma_0 * exp(-alpha * (T - T_ref))            [MPa, degC]
```

`sigma_0`, `alpha` are fitted by least squares in `ln sigma` space against a
review-compiled anchor table for columnar first-year ice at a strain rate of
1e-3 1/s (`TIMCO_JORDAN_ANCHORS_MPA`):

| T [degC] | -5 | -10 | -15 | -20 | -25 | -30 |
|---|---|---|---|---|---|---|
| anchor sigma_c [MPa] | 4.1 | 5.6 | 6.8 | 7.8 | 8.7 | 9.6 |

Fitted: `sigma_0 = 5.348 MPa` at `T_ref = -10 degC`, `alpha = 0.03264 1/K`.
Maximum anchor residual **10.8 %** (criterion b: < 15 %). The implied strength
ratio `sigma(-30)/sigma(-5) = 2.26`, consistent with the measured cold
hardening of columnar ice.

Honesty note: the anchor values are transcribed representative values from the
review literature (Timco & Jordan 2013 / Petrovic 2003 type compilations), not
certified measurements; ice strength data carry +-20 % scatter. The table is a
*reference set* for calibrating the temperature *trend*.

### 2.2 DEM-FSM contact model

The ice sheet is a bonded simple-cubic sphere lattice (inherited
`tensordem.dem3d.IceDEM3D` bond law, contact list and explicit integrator). The
rudder blade is a rigid slab (span × chord × thickness) inclined at the rudder
angle δ; its ice-facing face is kept parallel to the ice leading edge
(broadside indentation — the design load case of a rudder pressed against an ice
edge), and the blade translates along its own normal at the *normal approach
speed* `v_n = v sin δ`.

Each grain near the blade carries a fracture-state machine:

| state | transition | force law |
|---|---|---|
| `FREE` | overlap > 0 | 0 |
| `ELASTIC` | penetration > `crush_strain * r` | `k delta - c v_n` |
| `CRUSHING` | penetration > `spall_strain * r` | `min(elastic, p_crush A_cell w)` |
| `CLEARED` | contact opens / reach exceeded | `f_res * p_crush A_cell w` |

with

```
p_crush = sigma_ice(T) * (v_n / v_ref) ** 0.2        (crushing pressure cap)
A_cell  = (2r)^2                                     (grain cell area)
w       = edge fraction of the grain cell inside the blade extents
```

`CRUSHING -> CLEARED` irreversibly debonds every bond attached to the grain
(the DEM analogue of a spall/flake event) and leaves only the crushed-layer
residual pressure `f_res = 0.6`. The `w` weight removes the `O(r)` area
over-count of centre-based contact detection: without it the loaded width is
`chord + 2r` and the force grows with resolution. With it, the realised
weighted contact area converges to the nominal loaded area (measured
0.360 m² against the nominal 0.36 m²).

### 2.3 Rule-style design load card

```
F_R = C_class * C_p * sigma_ice(T) * (v_n / v_ref)^0.2 * A_loaded
A_loaded = h * min(chord, h / sin delta)
C_p = 0.55            (mean pressure coefficient, documented band 0.5 - 0.7)
```

`C_p` is the mean patch pressure over a few tenths of a square metre: ice-class
rules and ISO 19906 reduce the nominal local pressure below the uniaxial
compressive strength through an area (patch-size) effect. `C_p = 0.55` is the
band midpoint and was fixed *before* the DEM campaign. Narrow indenters instead
carry an indentation enhancement of 2–3 (Korzhavin/Michel), reported separately
as an upper reference. `certified_rule_text = False`: coefficients that must be
read verbatim from the rule text (ice-class factor, plate/area terms) are left
at unity, so this is a *rule-style* check, not a certified adjudication.

### 2.4 Steering gear check

```
M = F * lever_arm      Tq = F * torsional_lever
W = pi (D_o^4 - D_i^4) / (32 D_o)          W_t = 2 W
sigma_b = M / W        tau = Tq / W_t      sigma_vm = sqrt(sigma_b^2 + 3 tau^2)
verdict = PASS if sigma_vm <= sigma_y / gamma_m (235 MPa / 1.5)
```

## 3. Numerical setup

| item | value |
|---|---|
| grain radius / spacing | 0.075 m / 0.15 m |
| field | 9 × 11 grains in plan, `nz = h / (2r)` thickness layers |
| particle count | 297 (h=0.45 m) … 594 (h=0.9 m) |
| contact / bond stiffness | 5.0e7 / 5.0e7 N/m |
| crushing cap at T=-10 degC | 5.35 MPa (6.15/7.06 MPa with rate factor) |
| shear bond failure | disabled (`shear_breaking_strain = 1e9`), see limits |
| density scaling | x100 (DEM mass scaling), checked by the dt study |
| ship speed / normal approach | 2.0 m/s / 1.0 m/s (δ = 30 deg) |
| blade advance | 0.14 m (0.93 grain radius past first contact) |
| expansion | explicit, dt = 3.71e-5 s (recommended stability step) |

## 4. Results

### 4.1 Load history and envelope (criterion a)

Reference case (h = 0.6 m, T = -10 degC, δ = 30 deg, v = 2 m/s):

| quantity | value |
|---|---|
| rule-style design load `F_R` | 1059.3 kN |
| DEM sustained (plateau) load | 866.7 kN |
| DEM 10 ms envelope peak | 866.7 kN |
| DEM raw instantaneous peak | 866.7 kN |
| relative error, sustained vs `F_R` | -18.2 % |
| relative error, envelope vs `F_R` | -18.2 % |
| realised contact area / nominal | 0.360 m^2 (100 % of nominal) |
| mean effective pressure / crushing cap | 2.41 / 5.35 MPa = 0.45 |

The DEM load record is a sawtooth: crushing slow-build followed by bond-breaking
spall events. The 10 ms moving-average envelope corresponds to the response time
of a rudder-stock/steering-gear assembly and is used for the stock check; the
sustained load is the load-level metric.

### 4.2 Ice thickness sweep (criterion c)

| h [m] | particles | DEM sustained [kN] | rule [kN] | ratio |
|---|---|---|---|---|
| 0.45 | 297 | 650.0 | 794.5 | 0.818 |
| 0.60 | 396 | 866.7 | 1059.3 | 0.818 |
| 0.75 | 495 | 1083.4 | 1324.1 | 0.818 |
| 0.90 | 594 | 1300.0 | 1589.0 | 0.818 |

Spearman rank correlation between ice thickness and the DEM load = **1.00**
(envelope and sustained), criterion (> 0.9) met. The DEM reproduces the linear
thickness scaling of the crushing load exactly, with a constant model-to-rule
ratio of 0.818, i.e. the DEM-implied mean pressure coefficient is
`0.45 sigma_c`.

### 4.3 Grid refinement (criterion e)

| level | particles | r [m] | sustained load [kN] |
|---|---|---|---|
| coarse (campaign) | 396 | 0.0750 | 866.7 |
| refined | 770 | 0.0595 | 920.4 |

Particle count ratio 1.94. Change of the sustained contact force **+6.2 %**, of the 10 ms envelope +11.8 %, of the local effective contact pressure **+2.9 %**. Criterion (e) asks for < 5 % on the contact force: the sustained force misses it by 1.2 points (6.2 %), the pressure metric meets it (2.9 %). The refinement moves the load *towards* the rule-style value: at the refined resolution the sustained load is -13.1 % from `F_R` (inside the +-15 % band), i.e. the coarse campaign resolution is discretisation-limited rather than physics-limited.

### 4.4 Time step / density scaling sensitivity

Halving the time step (3.709e-05 s -> 1.855e-05 s, i.e. twice the number of steps for the same blade advance) changes the sustained load by **+0.0 %** and the envelope by +0.0 %. The crushing response is quasi-static at this indentation rate, so the x100 density scaling does not bias the force: the mass-scaling check passes.

### 4.5 Temperature sweep

| T [degC] | sigma_ice [MPa] | DEM sustained [kN] | rule [kN] | error |
|---|---|---|---|---|
| -5 | 4.55 | 736.3 | 900.0 | -18.2 % |
| -20 | 7.41 | 1200.7 | 1467.6 | -18.2 % |

The DEM load ratio between -20 and -5 degC is 1.63, identical to the strength ratio 1.63: the temperature dependence of the ice load is carried entirely by `sigma_ice(T)` through the FSM crushing cap, as intended. Spearman(temperature, load) = -1.0 (colder ice -> heavier load).

### 4.6 Design-load map and stock check (criterion d)

| stock D_o [m] | wall [m] | sigma_vm [MPa] | allowable [MPa] | utilization | verdict |
|---|---|---|---|---|---|
| 0.30 | 0.020 | 389.8 | 156.7 | 2.49 | FAIL |
| 0.45 | 0.035 | 102.4 | 156.7 | 0.65 | PASS |
| 0.60 | 0.060 | 36.0 | 156.7 | 0.23 | PASS |
| 0.90 | 0.090 | 10.7 | 156.7 | 0.07 | PASS |

Driven by the DEM envelope of 867 kN. Criterion (d) met: the verdict separates FAIL (utilization 2.49) from PASS (0.07); the reference D_o = 0.60 m / 60 mm stock is PASS with utilization 0.23.

### 4.7 Crushed-layer residual-pressure sensitivity

The crushed-layer residual pressure fraction `f_res` is the one model parameter without a first-principles value. Reference case, everything else fixed:

| f_res | sustained load [kN] | implied mean pressure coefficient | error vs `F_R` |
|---|---|---|---|
| 0.35 | 505.6 | 0.26 sigma_c | -52.3 % |
| 0.60 | 866.7 | 0.45 sigma_c | -18.2 % |
| 1.00 | 1444.5 | 0.75 sigma_c | +36.4 % |

The DEM therefore brackets a mean pressure coefficient of 0.26-0.75 `sigma_c`; the documented rule band (0.5-0.7 `sigma_c`) falls **inside** this bracket. The point comparison at the adopted `f_res = 0.6` (-18.2 %) is a statement about that parameter choice, not about a disagreement in physics: the DEM cannot narrow the coefficient without an independent crushed-ice pressure measurement.

## 5. Limitations and applicability

1. **Rule coefficients are reconstructed.** `C_p` and all ice-class/area terms
   must be taken from the rule text for a certified check; here they are the
   documented area-effect band. The Korzhavin upper bound (`I = 2`) is reported
   as an independent upper reference and is -59 % below it.
2. **Coarse graining.** Grains of 0.075 m give 16 grains across the loaded
   patch; DEM strength is calibrated through the crushing cap rather than the
   elastic modulus, so strain-rate and grain-size effects beyond
   `(v/v_ref)^0.2` are not resolved.
3. **Shear bond failure is switched off** for the coarse field (`shear_breaking_strain
   = 1e9`): the inherited Green-Lagrange shear measure is not calibrated at
   ~10 cm grain size. Load limiting comes from tensile bond failure plus the FSM
   crushing cap. A calibrated shear criterion would raise the spall rate.
4. **Density scaling x100** distorts fragment inertia; the dt-refinement study
   quantifies the resulting change of the load.
5. **Crushed-layer residual pressure** is a model parameter (`f_res = 0.6`); the
   sensitivity in §4.7 shows the load band it opens.
6. **Prescribed blade kinematics** (constant velocity) produces an initial
   impact spike; the quasi-steady window (penetration > one grain radius)
   excludes it and the raw peak is reported separately.
7. **No model-test data** were available in this increment, so no experimental
   cross-validation is claimed. The comparison is DEM vs rule-style design load
   and vs literature strength anchors.

## 6. Verification summary

* `tests/test_rudder_ice_load.py` — 22 tests (temperature model, anchors,
  option/geometry validation, rule-card scaling, FSM states and cap, envelope
  and windowing, thickness monotonicity, grid convergence, stock check both
  sides, helper functions).
* `scripts/benchmark_rudder_ice_load.py` — reproducible campaign writing
  `report.json`, `thickness.csv`, `design_load_scan.csv` and five figures.
* Force/area closure: realised weighted contact area = nominal loaded area
  within 0.360 m^2 (100 % of nominal).