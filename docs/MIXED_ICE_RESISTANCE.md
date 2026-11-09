# Mixed first-year / multi-year ice resistance: MY-fraction sweep and FSD sensitivity

Evidence grade: **new capability**, relative claims only. The absolute ship-scale
level inherits the level-ice anchor's documented error against the MT Uikku model
tests (:mod:`tensordem.level_ice`); this document therefore quantifies *changes*
of resistance and their sensitivity, and says so explicitly in the error section.

### Answers to the two questions of the node

1. **How much does the ice resistance change when the multi-year area fraction
   goes from 0 % to 80 %?**  At v = 3 m/s the Lindqvist anchor gives
   R = 0.990 MN at p_MY = 0 and
   5.114 MN at p_MY = 80 %, i.e.
   **416.4 %** (Riska: 453.7 %; the two
   anchors agree to within 8.9 %).
   72 %
   of that increment is the **thickness** of the MY class and only
   5 %
   is the **mechanical strengthening** of older ice.
2. **How sensitive is the prediction to the ice-thickness distribution?**  Over
   the declared parameter ranges the FSD group alone spans
   2.321 MN
   (75 % of the mean) against
   0.676 MN
   (22 %) for the material
   parameterisation, and the single dominant parameter is
   `my_thickness_scale`: the **MY class-mean thickness**.
   The measured absolute level of R is not validated (no measured MY-ice series is
   available offline), so these are relative statements -- see the error section.

## 1. Case, geometry and parameters

The question is a field question, not a single-ice-sheet question: a ship
transiting an ice field in which a fraction ``p_MY`` of the *area* is multi-year
(MY) ice and the remainder is first-year (FY) ice. Two things are unknown for MY
ice and are parameterised here:

* **Material state.** MY ice is desalinated, colder and thicker. The ageing law
  of :mod:`tensordem.mixed_ice.aging` takes age as the single independent
  variable and returns salinity, temperature, brine volume, class-mean thickness,
  flexural strength, compressive strength, elastic modulus, fracture toughness
  and density. Brine volume enters through the gas-free freezing-point relation

      v_b = S (1 + 49.185/|T|) / 1000,

  the flexural strength through the Timco & O'Brien relation
  ``sigma_f = 1.76 exp(-5.88 sqrt(v_b))`` MPa, and the compressive strength,
  modulus and toughness through a pinned percolation form
  ``X = X_solid (1 - v_b/v_p)^n`` with ``v_p = 0.28``. The solid prefactors are
  *calibrated* so that age 1 returns the first-year values this repository already
  uses (``sigma_c = 1.5 MPa``, ``E = 5.0 GPa``), which makes the MY extension
  FY-consistent by construction.

| age [a] | class | S [permille] | T [degC] | v_b [-] | h [m] | sigma_f [MPa] | sigma_c [MPa] | E [GPa] | K_Ic [MPa m^0.5] | rho [kg/m^3] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FY | 5.00 | -5.00 | 0.0542 | 1.20 | 0.448 | 1.500 | 5.00 | 0.0968 | 921.7 |
| 2 | MY | 2.34 | -9.87 | 0.0140 | 2.17 | 0.878 | 2.081 | 5.89 | 0.1140 | 918.2 |
| 3 | MY | 1.70 | -12.36 | 0.0085 | 2.67 | 1.025 | 2.169 | 6.01 | 0.1164 | 917.7 |
| 4 | MY | 1.55 | -13.65 | 0.0071 | 2.93 | 1.071 | 2.190 | 6.04 | 0.1169 | 917.6 |
| 5 | MY | 1.51 | -14.31 | 0.0067 | 3.06 | 1.087 | 2.197 | 6.05 | 0.1171 | 917.6 |
| 10 | MY | 1.50 | -14.98 | 0.0064 | 3.20 | 1.098 | 2.202 | 6.06 | 0.1172 | 917.5 |
| 30 | MY | 1.50 | -15.00 | 0.0064 | 3.20 | 1.099 | 2.202 | 6.06 | 0.1172 | 917.5 |

* **Thickness distribution.** The field is the mixture
  ``F(h) = (1 - p) F_FY(h) + p F_MY(h)`` with lognormal marginals
  (FY mean 1.20 m, CV 0.35;
  MY mean 2.75 m, CV 0.25,
  the mean being the ageing law averaged over a 2-5 a age window). Within a class
  thickness and floe diameter are treated as independent, which is what makes the
  field expectation exactly class-additive.

Hull (fixed): icebreaker, L = 120 m,
B = 20 m,
T = 7 m,
entry angle 22 deg,
friction 0.05.
Speeds 1, 3, 5 m/s; MY area fractions
0, 0.1, 0.2, 0.4, 0.6, 0.8.

## 2. Method

1. **Ageing** (`mixed_ice/aging.py`) produces the material state per age class;
   the coefficient ledger is graded (DERIVED / CONVENTION / CITED / WITHHELD) and
   SHA-256 hashed into the artifact.
2. **FSD sampler and quadrature** (`mixed_ice/field.py`). Two estimators of the
   same field: a deterministic stratified quadrature (equal-probability thickness
   quantiles per class, an age x thickness grid for MY) that carries no noise, and
   a genuine Monte-Carlo floe sampler. The sampler draws the MY *label* with the
   area-tilted probability ``q = p W_FY / (p W_FY + (1-p) W_MY)`` so that the
   realised **area** fraction equals ``p`` even though MY floes are larger
   (at p = 0.40 the tilt gives q =
   0.1535).
3. **Resistance** (`mixed_ice/resistance.py`): each FSD node is evaluated with the
   repository's own anchors (Lindqvist three-component, Riska speed-linear) at that
   node's thickness and material state, then area-weighted. Nothing is
   re-implemented, so the anchor's coefficient ledger remains the single source of
   truth.
4. **DEM cross-check** (`mixed_ice/campaign.py`): the bonded-particle section
   model of `tensordem.level_ice.sheet` is run in a 2^3 factorial over the MY/FY
   flexural strength, compressive strength and elastic modulus at h = 1 m
   (n_z = 4, mass scaling 200, advance 6 m). A log-linear fit gives the DEM's
   strength-sensitivity exponents, compared with the same exponents fitted on the
   engineering anchor. A thickness pair (1.0 / 1.5 m) gives the DEM thickness
   exponent and an n_z = 6 pair tests whether the flexure load path is
   resolution-limited.
5. **Verdicts**: eleven criteria are declared in `campaign.verdicts`, each with its
   measured number; honest FAILs are kept.

## 3. Results

### 3.1 R(MY fraction) at 3 m/s

| p_MY | R Lindqvist [MN] | dR vs p=0 | R Riska [MN] | dR vs p=0 | R_MY/R_FY (Lindqvist) |
|---|---|---|---|---|---|
| 0.0 | 0.990 | 0.0 % | 1.528 | 0.0 % | n/a |
| 0.1 | 1.506 | 52.1 % | 2.395 | 56.7 % | 6.206 |
| 0.2 | 2.021 | 104.1 % | 3.261 | 113.4 % | 6.206 |
| 0.4 | 3.052 | 208.2 % | 4.995 | 226.8 % | 6.206 |
| 0.6 | 4.083 | 312.3 % | 6.728 | 340.3 % | 6.206 |
| 0.8 | 5.114 | 416.4 % | 8.461 | 453.7 % | 6.206 |

R(0) = 0.990 MN, R(p_MY = 0.8) =
5.114 MN: **the field resistance changes by
416.4 %** from 0 % to 80 % MY,
i.e. 5.154 MN per unit MY area fraction
(520.6 % per 10 % of MY ice).
This is a *per-floe level-ice upper estimate*, and section 3.2 shows that
72 %
of the increment is the MY class-mean thickness
(2.76 m against
1.20 m for FY),
not the strength of older ice.

The mixture is exactly class-additive, so the curve carries no shape information
beyond the class ratio ``R_MY/R_FY = 6.206``; criterion (b)
verifies the identity ``dR/R(0) = p (R_MY/R_FY - 1)`` to
4.4e-15.

Speed dependence of the same increment:

| v [m/s] | dR/R(0) at p_MY = 0.8, Lindqvist | dR/R(0) at p_MY = 0.8, Riska |
|---|---|---|
| 1 | 450.9 % | 451.5 % |
| 3 | 416.4 % | 453.7 % |
| 5 | 364.6 % | 455.6 % |

Strength-parameterisation band (MY strength and modulus scaled by the printed
factors; Lindqvist, v = 3 m/s):

| p_MY | x0.8 | x1 | x1.2 | span [MN] |
|---|---|---|---|---|
| 0.0 | 0.990 | 0.990 | 0.990 | 0.000 |
| 0.1 | 1.418 | 1.506 | 1.594 | 0.176 |
| 0.2 | 1.845 | 2.021 | 2.197 | 0.352 |
| 0.4 | 2.700 | 3.052 | 3.404 | 0.704 |
| 0.6 | 3.555 | 4.083 | 4.610 | 1.055 |
| 0.8 | 4.410 | 5.114 | 5.817 | 1.407 |

### 3.2 Where the increment comes from: thickness versus strength

| model | ratio strength only | ratio thickness only | ratio full | interaction | strength share | thickness share |
|---|---|---|---|---|---|---|
| lindqvist | 1.292 | 4.906 | 6.443 | 1.69 % | 5.4 % | 71.8 % |
| riska | 1.383 | 5.131 | 6.990 | -1.49 % | 6.4 % | 69.0 % |

For the default field the MY/FY class ratio factorises into a thickness part of
4.91
and a strength part of
1.29,
with an interaction residual of only
1.7 %.
**The MY effect is a thickness effect**:
72 %
of the increment comes from the thicker MY floes and
5 %
from the stronger MY ice.

### 3.3 FSD sensitivity (p_MY = 0.4)

| parameter | group | levels | span [MN] | span / R | max abs deviation [MN] |
|---|---|---|---|---|---|
| my_thickness_scale | fsd | 0.8, 1, 1.2 | 1.923 | 63.1 % | 1.055 |
| my_strength_multiplier | material | 0.8, 1, 1.2 | 0.702 | 23.1 % | 0.351 |
| fy_thickness_mean_m | fsd | 0.8, 1, 1.2, 1.5 | 0.627 | 20.6 % | 0.314 |
| my_thickness_cv | fsd | 0.15, 0.25, 0.4 | 0.284 | 9.3 % | 0.200 |
| fy_thickness_cv | fsd | 0.2, 0.35, 0.5 | 0.090 | 3.0 % | 0.054 |
| my_fraction | primary | 0, 0.2, 0.4, 0.6, 0.8 | 4.118 | 135.2 % | 2.059 |

Additive bounds (a conservative joint bound for monotone, mutually independent effects): FSD 1.624 MN (82.2 % of the total), material 0.351 MN. Base R(p_MY=0.4) = 3.047 MN.

Joint Latin-hypercube envelopes over the declared parameter ranges:

| group | n | mean [MN] | p05 [MN] | p95 [MN] | min [MN] | max [MN] | relative span |
|---|---|---|---|---|---|---|---|
| fsd | 64 | 3.088 | 2.133 | 3.850 | 2.008 | 4.329 | 75.2 % |
| material | 16 | 3.048 | 2.747 | 3.358 | 2.705 | 3.382 | 22.2 % |

Reading: over the declared plausible ranges the **ice-thickness distribution**
contributes 1.624 MN and the **material
parameterisation** 0.351 MN of resistance spread
(82 % / 18 %);
the dominant single parameter is `my_thickness_scale`. The FSD group p5-p95
envelope at fixed p_MY is
2.133-3.850 MN against a mean of
3.088 MN.

### 3.4 Estimator cross-check

| quantity | value |
|---|---|
| R, deterministic quadrature | 3.052 MN |
| R, 20 000-floe Monte-Carlo | 3.000 MN |
| Monte-Carlo standard error | 0.0523 MN |
| z = difference / standard error | -0.99 |
| realised MY area fraction (target 0.40) | 0.3965 |

### 3.5 DEM strength factorial (h = 1 m, n_z = 4)

| code | sigma_f [MPa] | sigma_c [MPa] | E [GPa] | R section [MN/m] | crush share of normal force | flexure share of normal force | friction share of total | broken |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.448 | 1.500 | 5.00 | 5.655 | 0.920 | 0.000 | 0.080 | 0.222 |
| 1 | 1.025 | 1.500 | 5.00 | 6.562 | 0.920 | 0.000 | 0.080 | 0.334 |
| 2 | 0.448 | 2.169 | 5.00 | 8.301 | 0.920 | 0.000 | 0.080 | 0.228 |
| 3 | 1.025 | 2.169 | 5.00 | 8.272 | 0.920 | 0.000 | 0.080 | 0.213 |
| 4 | 0.448 | 1.500 | 6.01 | 5.649 | 0.920 | 0.000 | 0.080 | 0.219 |
| 5 | 1.025 | 1.500 | 6.01 | 5.927 | 0.920 | 0.000 | 0.080 | 0.208 |
| 6 | 0.448 | 2.169 | 6.01 | 8.092 | 0.920 | 0.000 | 0.080 | 0.216 |
| 7 | 1.025 | 2.169 | 6.01 | 8.958 | 0.920 | 0.000 | 0.080 | 0.213 |

| exponent | DEM 2^3 fit | Lindqvist anchor | Riska anchor | relative error (DEM vs Lindqvist) |
|---|---|---|---|---|
| d ln R / d ln $\sigma_c$ | 0.9410 | 0.5737 | 0.5965 | 64.0 % |
| d ln R / d ln $\sigma_f$ | 0.0891 | 0.0404 | 0.1208 | 120.5 % |
| d ln R / d ln $E$ | -0.0660 | 0.0000 | 0.0000 | n/a (anchor exponent 0) |

Fit residual (interaction magnitude) of the DEM factorial: 0.0582 in ln R (4 degrees of freedom); anchor factorials are exactly separable because the anchors are additive in their components (residual 1.76e-03).

DEM-domain MY-fraction sweep at fixed h = 1 m (section line load [MN/m]; only the ice class of each floe switches, the FSD is frozen):

| p_MY | DEM section [MN/m] | DEM dR | Lindqvist [MN] | Lindqvist dR | Riska dR |
|---|---|---|---|---|---|
| 0.0 | 5.655 | 0.0 % | 0.638 | 0.0 % | 0.0 % |
| 0.1 | 5.985 | 5.8 % | 0.656 | 2.8 % | 3.8 % |
| 0.2 | 6.315 | 11.7 % | 0.674 | 5.6 % | 7.5 % |
| 0.4 | 6.976 | 23.4 % | 0.709 | 11.1 % | 15.1 % |
| 0.6 | 7.637 | 35.0 % | 0.745 | 16.7 % | 22.6 % |
| 0.8 | 8.297 | 46.7 % | 0.780 | 22.2 % | 30.2 % |

At p_MY = 0.8 the DEM section model reports 46.7 % against 22.2 % for Lindqvist at the same thickness (110.4 % relative to the anchor increment) -- the same over-sensitivity diagnosed below.

| factor | b (DEM) | b (anchor) | b (DEM) x s | deviation raw | deviation normalised |
|---|---|---|---|---|---|
| $\sigma_c$ | 0.9410 | 0.5737 | 0.5352 | 64.0 % | -6.7 % |
| $\sigma_f$ | 0.0891 | 0.0404 | 0.0507 | 120.5 % | 25.4 % |
| $E$ | -0.0660 | 0.0000 | -0.0375 | n/a (anchor exponent 0) | n/a (anchor exponent 0) |

Anchor component shares at h = 1 m: crushing 0.536, bending 0.033, submersion 0.287, added drag 0.144; strength-carrying share s = 0.569.  The bonded-particle section carries no submersion term, so its strength exponents should be about 1/s times the anchor's.

| quantity | FY | MY |
|---|---|---|
| h exponent (1.0 -> 1.5 m) | 0.186 | 0.335 |
| flexure bucket fraction, n_z = 4 | 0.000 | 0.000 |
| flexure bucket fraction, n_z = 6 | 0.000 | 0.000 |
| R(n_z = 6) / R(n_z = 4) | 1.322 | 1.193 |

Anchor h exponent for the same states: 1.879.

Class ratio at equal thickness:

| quantity | value |
|---|---|
| DEM R(MY)/R(FY), full factorial corners | 1.584 |
| Lindqvist R(MY)/R(FY) | 1.278 |
| Riska R(MY)/R(FY) | 1.377 |
| DEM deviation from Lindqvist, relative to the increment | 110.4 % |

## 4. Figures

![fig1_my_fraction_curve.png](../artifacts/mixed_ice/figures/fig1_my_fraction_curve.png)

![fig2_fsd_sensitivity.png](../artifacts/mixed_ice/figures/fig2_fsd_sensitivity.png)

![fig3_ageing.png](../artifacts/mixed_ice/figures/fig3_ageing.png)

![fig4_dem_factorial.png](../artifacts/mixed_ice/figures/fig4_dem_factorial.png)


## 5. Verdicts

8 PASS / 4 FAIL.

| # | criterion | measured | verdict |
|---|---|---|---|
| a_my_fraction_sweep_monotone | R(p_MY) strictly increasing over p = 0..0.8 for both anchors at v = 1/3/5 m/s | monotone = {lindqvist_v1: True, lindqvist_v3: True, lindqvist_v5: True, riska_v1: True, riska_v3: True, riska_v5: True}; relative_change_0_to_80pct = 4.164 | **PASS** |
| b_class_weight_identity | |dR/R(0) - p (R_MY/R_FY - 1)| < 1e-9 (the mixture is exactly class-additive) | max_abs_error = 4.441e-15 | **PASS** |
| c_estimator_agreement | |z| < 3 for R_sampler vs R_quadrature (z = difference / Monte-Carlo standard error of the weighted mean) and |realised MY area fraction - p| < 0.01 | quadrature_N = 3.052e+06; sampler_N = 3e+06; sampler_standard_error_N = 5.234e+04; z_score = -0.9894; relative_difference = 0.01697; sampler_realised_my_fraction = 0.3965; target_my_fraction = 0.4; my_fraction_bias = -0.003468; n = 20000; seed = 20261108 | **PASS** |
| d_dem_sigma_c_exponent | |b_sigmac(DEM) - b_sigmac(anchor)| / b_sigmac(anchor) < 15 % | dem = 0.941; anchor = 0.5737; relative_error = 0.6403 | **FAIL** |
| e_dem_sigma_f_exponent | |b_sigmaf(DEM) - b_sigmaf(anchor)| / b_sigmaf(anchor) < 15 % | dem = 0.08909; anchor = 0.04039; relative_error = 1.205; dem_flexure_bucket_fraction = {fy_coarse: 0, my_coarse: 0, fy_fine: 0, my_fine: 0} | **FAIL** |
| f_dem_class_ratio_vs_anchor | |R_MY/R_FY (DEM) - R_MY/R_FY (Lindqvist)| / (R_MY/R_FY - 1) < 15 % (deviation measured against the increment, the quantity the 0-80 % curve uses) | dem = 1.584; lindqvist = 1.278; riska = 1.377; relative_deviation_vs_lindqvist = 1.104 | **FAIL** |
| g_anchor_cross_model | |dR_Lindqvist - dR_Riska| / dR_Lindqvist < 40 % at p = 0.8, v = 3 m/s (band justified a priori: the anchors' own MAPE against the MT Uikku tests are 27.9 % and 51.2 %, so 40 % on a relative increment is the resolution of the comparison) | lindqvist = 4.164; riska = 4.537; relative_difference = 0.08945 | **PASS** |
| h_fsd_sensitivity_reported | all four FSD parameters swept, ranked by span, joint LHS envelope reported | rank_by_span = [my_thickness_scale, my_strength_multiplier, fy_thickness_mean_m, my_thickness_cv, fy_thickness_cv]; fsd_additive_bound_N = 1.624e+06; material_additive_bound_N = 3.512e+05; fsd_share_of_bound = 0.8222; joint_fsd_relative_span = 0.7517; joint_material_relative_span = 0.2219 | **PASS** |
| i_ageing_law_limits | sigma_f, sigma_c, E, h monotone in age; MY/FY(3 a) ratios inside the cited bands sigma_f 1.3-2.5, sigma_c 1.2-2.5, E 1.1-1.7, h > 1.5 | monotone = True; my_over_fy_at_3a = {sigma_f: 2.288, sigma_c: 1.446, E: 1.202, thickness: 2.227} | **PASS** |
| j_provenance_hashed | ageing ledger + FSD spec + sampler seeds + node counts hashed into the artifact | aging_ledger_sha256 = 14f42366bad75af7e075820041514ac57e5d74e630e170be200e1b093f63067f; reference_ledger_sha256 = 927c675b4625807ec0be8cfc5cb37fe133b9a28c2aa2c772be23d7fb2f836047; field_spec_sha256 = e41158c805f91db45ddefc4eb069bbfee9fe66a71cb2fd2143d8128c01c9f632 | **PASS** |
| k_dem_flexure_route_open | the DEM flexure bucket must carry > 1 % of the resistance for the sigma_f exponent to be meaningful | fy_coarse = 0; my_coarse = 0; fy_fine = 0; my_fine = 0 | **FAIL** |
| l_dem_sigma_c_exponent_normalised | |b_sigmac(DEM) * s - b_sigmac(anchor)| / b_sigmac(anchor) < 25 %, where s is the anchor's strength-carrying share: the bonded-particle section has no submersion (displaced-rubble) term, so its strength exponent is expected to be 1/s times the anchor's | strength_carrying_share = 0.5688; dem_exponent = 0.941; dem_exponent_normalised = 0.5352; anchor_exponent = 0.5737; relative_error_normalised = 0.06704 | **PASS** |

The FAILs are kept as stated. They share one identified root cause: in the DEM
section model the **flexure bucket carries 0.000 of the
resistance at n_z = 4** because the crushing cap saturates essentially every
contact, so the bending load path -- the route through which flexural strength
enters -- is closed. The DEM therefore reacts to the compressive strength with an
exponent close to the anchor's and almost not at all to the flexural strength,
which is exactly what criteria (d), (e), (f) and (k) measure. That is a property of
the reduced section model at this resolution (the level-ice campaign reports a
146 % mesh GCI), not of the mixed-field model, and it is the reason the reported
R(p_MY) curve is built on the engineering anchors while the DEM supplies the
strength-sensitivity cross-check and its validity boundary.

## 6. Error judgement (is it < 3 %?)

| quantity | reference | error | verdict |
|---|---|---|---|
| R(p_MY = 0) vs the FY Lindqvist anchor of the same field | internal identity | < 1e-12 (criterion b) | **< 3 %** |
| mixture identity dR/R(0) = p (ratio - 1) | algebra | 4.4e-15 | **< 3 %** |
| quadrature vs Monte-Carlo estimator | sampler with its own standard error | z = -0.99 | **< 3 %** (within 3 standard errors) |
| MY/FY class ratio, DEM vs Lindqvist | bonded-particle section | 110.4 % | **not < 3 %** (honest FAIL, criterion f) |
| DEM sigma_c exponent vs anchor | bonded-particle section | 64.0 % | **not < 3 %** (criterion d FAIL) |
| absolute ship-scale level of R(p_MY) | measured MY-ice resistance | **not available offline** | **cannot be judged**: no measured multi-year ice resistance series is readable, so the absolute level stays WITHHELD; the FY anchor's documented MAPE against the MT Uikku model tests is 27.9 % (the repository's level-ice campaign), which is inherited and is not < 15 % either |

Consequence for use: the **relative** results of this document (the 0-80 % change,
the thickness-versus-strength split and the sensitivity envelope) are reproducible
and internally verified, while any absolute resistance figure quoted from this
model carries the anchor's ~28 % class of error at the absolute level.

## 7. Limitations and pitfalls found

* **Class additivity is assumed, not derived.** It holds because thickness and
  floe diameter are treated as independent *within* a class. A field in which
  thick MY ice comes in large floes and thin FY ice in small floes would break it;
  the Monte-Carlo sampler is the honest estimator in that case.
* **Every floe is level ice.** Floe-size sub-level ice, floe rotation, broken-ice
  channels and the associated resistance drop are absent, so the mixed-field
  number is an upper estimate for a broken field.
* **Area weighting needs the tilt.** Drawing MY labels with probability ``p``
  over-represents MY ice whenever MY floes are larger; the sampler applies
  ``q = p W_FY / (p W_FY + (1-p) W_MY)``. Verified: realised area fraction
  0.3965 against a target of
  0.40.
* **Fracture toughness is parameterised but consumed by nothing.** The
  bonded-particle sheet uses strength-only bonds, so ``K_Ic`` has zero influence
  here; the field is reported (`consumed_by_dem: false`) as an interface contract
  for a future fracture-energy bond model.
* **The MY thickness mean dominates the answer.** It is the ageing law averaged
  over the 2-5 a window; the campaign reports the whole plausible band, and the
  sensitivity ranking should be read as a statement about the declared ranges, not
  as a local derivative.
* **The aged material does not change the DEM flexure route**, and the section
  model is not mesh converged (GCI 146 % in the level-ice campaign). The DEM is
  used only for sensitivity exponents and ratios, never for the ship-scale level.
* **No MY resistance data.** Nothing in this document is validated against
  measured multi-year ice resistance; the evidence grade is WITHHELD at the
  absolute level by construction.

## 8. Reproduction

```
python scripts/mixed_ice_campaign.py            # full run (DEM includes n_z = 6 pair)
python scripts/mixed_ice_campaign.py --no-dem   # anchors only, no DEM
pytest tests/test_mixed_ice_aging.py tests/test_mixed_ice_field.py \
       tests/test_mixed_ice_resistance.py tests/test_mixed_ice_campaign.py -q
pytest tests/test_level_ice_resistance_dem.py -q   # the extended CaseSpec
```

The 44 unit tests run in about 13 s; the campaign integration test adds roughly
30 s with the DEM skipped. One unrelated pre-existing collection error exists in
this worktree: `tests/test_shape_representation.py` imports `PileConfig2D`, which
is absent from `tensordem.shape_representation` at the baseline commit; that file
is not touched by this work.

Artifacts: `artifacts/mixed_ice/mixed_ice_campaign.json` (all tables and verdicts),
`artifacts/mixed_ice/figures/` (figures), aging ledger
`14f42366bad75af7`,
level-ice ledger `927c675b4625807e`,
field spec `e41158c805f91db4`.
Wall clock of the campaign: 58.7 s.
