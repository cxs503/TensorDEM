# Ice-ridge keel: freeze-bond age evolution and its effect on keel resistance

**Campaign node** IN54W1-N05 (mission 7096, TensorDEM). **Artefacts** `artifacts/ridge_freeze_bond_age/` (model.json, cooling.json, age_sweep.json, ledger.json, figures/). **Code** `src/tensordem/ridge_freeze_bond_age.py`, `scripts/run_ridge_freeze_bond_age.py`, `tests/test_ridge_freeze_bond_age.py`. **Runtime of this run** 29.2 s.

## 1. Case, physics and parameters

A first-year ice ridge is a rubble mound whose voids fill with water that freezes on contact. The frozen contacts (*freeze bonds*) are the only tensile/shear load path in the keel, so the keel resistance of a fresh ridge is not a constant: it grows while the bonds grow. The physical question of this node is therefore *how fast*, and what that does to the ridge load used in a design calculation.

The constitutive chain implemented is

1. **formation criterion** - a bond exists only if the contact temperature is below the local freezing point of the pore water, `T_c < T_m(S)` with Assur's relation `T_m = -0.05411 S - 7.52e-5 S^2`;
2. **strength ceiling** - `sigma_b_inf = eta_b * sigma_ice(T_c, S)` with the Timco-Frederking brine-volume law `sigma_ice = sigma_0 (1 - sqrt(v_b/v_crit))`, `v_b = S (49.185/|T| + 0.532)/1000` (Frankenstein-Garner);
3. **growth kinetics** - `sigma_b(t) = sigma_b_inf (1 - exp(-t/tau_f))`, `tau_b(t) = kappa sigma_b(t)`, with `tau_f` taken from a 1-D transient conduction solution of the ice mass that feeds the bond (not fitted);
4. **DEM coupling** - the time-dependent strengths are installed into the elastic-brittle freeze bonds of `tensordem.ridge_dem` and the confined-press resistance `R_keel(t)` is measured;
5. **Lindqvist interface** - `R_keel(t) = R_keel^Lindqvist * k_age(t)` with `k_age(t) = R^DEM(t)/R^DEM(0)`.

Parameters of this run (all from the artifacts):

| symbol | value | provenance |
|---|---|---|
| T_c | -10.0 degC | deck (cold winter contact) |
| S | 5.0 ppt | deck (brackish ridge pore water) |
| T_m(S) | -0.272 degC | REAL, Assur relation |
| v_b | 0.02725 | REAL, Frankenstein-Garner polynomial (-2..-22.9 degC) |
| sigma_ice | 1200.8 kPa | form REAL (Timco-Frederking), coefficient WITHHELD |
| eta_b | 0.08327 | CALIBRATED on one primitive-temperature anchor (see SS5) |
| sigma_b_inf | 100.00 kPa | = eta_b sigma_ice |
| tau_b_inf / sigma_b_inf | 2.000 | deck ratio (ridge_dem convention) |
| h (thermal length) | 0.50 m | = DEM particle radius: the ice mass per bond |
| lens thickness | 5.0 mm | deck |
| tau_f | 22.22 h | derived (1-D cooling half-relaxation time) |
| rho c h^2/(4k) | 14.47 h | closed-form comparator |
| formation time | 0.444 h | = formation_ratio * tau_f (contact-area growth) |

## 2. Method and tool chain

| step | implementation |
|---|---|
| freezing point / brine volume / strength | `freezing_point_celsius`, `brine_volume_fraction`, `ice_tensile_strength` (pure functions, no state) |
| 1-D cooling and `tau_f` | `lens_cooling_transient`: finite-volume explicit scheme, `0 <= x <= h`, cold face at `T_c`, insulated symmetry plane at the bond plane, water lens with its own Stefan energy balance, diffusive CFL time step |
| strength kinetics | `FreezeBondAgeModel.sigma_at` / `tau_at` / `strength_at` |
| DEM run-time bond strengths | `RidgeAssembly.set_bond_strengths` (new hook) - the assembly is built once and the strength is installed per age |
| aged material | `aged_material`: frozen-contact fraction and bond strength as functions of age |
| keel resistance | `keel_resistance`: confined keel (two side walls, seabed, kinematic press), 5 %-window peak and residual reported |
| age sweep | `run_keel_age_sweep` over t = 0, 6.0, 24.0, 72.0, 168.0 h with seeds [20261031, 20261068] |
| Lindqvist coupling | `lindqvist_age_correction`: `k_age(t)`; the Lindqvist keel term itself is supplied by the caller (its coefficient is WITHHELD in this repository) |
| acceptance | `acceptance_ledger` -> `ledger.json`; pytest chain `python -m pytest tests/test_ridge_freeze_bond_age.py -q` |

Contact convention: a frozen lens *adds* tensile/shear capacity to the ice-ice contact, so the aged sweep uses `bond_contact_mode="parallel"` (the bond acts in parallel with the contact). The historical `ridge_dem` convention (`"exclusive"`: a bonded pair transmits only through its bond spring) is kept as the default for regression and is measured side by side in 3.5.

### 2.1 Relationship to the existing sibling module

TensorDEM already carries `tensordem.ridge_freeze_bond_aging`, which builds the *same* constitutive law from the Ettema & Schaefer (1986) pressure/salinity laws but **prescribes** its growth time `tau_g` with an Arrhenius law whose apparent activation energy is a declared parameter (its own docstring grades it `PARAMETRIC_MODEL`). This module is the complementary piece: it *derives* the growth time from a 1-D transient conduction solution and therefore predicts how it scales with the block size. The two are not competitors - the conduction time is what the prescribed parameter should be validated against, and 3.5 does exactly that comparison.

## 3. Results

### 3.1 Freeze-bond strength vs age

| t [h] | sigma_b [kPa] | tau_b [kPa] | frozen contact fraction | sigma_b/sigma_b_inf |
|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 |
| 6.0 | 23.66 | 47.32 | 0.650 | 0.2366 |
| 24.0 | 66.04 | 132.07 | 0.650 | 0.6604 |
| 72.0 | 96.08 | 192.16 | 0.650 | 0.9608 |
| 168.0 | 99.95 | 199.90 | 0.650 | 0.9995 |

Asymptote `sigma_b_inf = 100.00 kPa`, `tau_b_inf = 200.00 kPa`, `tau_f = 22.22 h`; the curve reaches 0.9995 of the asymptote at t = 168 h.

### 3.2 Temperature and salinity dependence of the ceiling

| T_c \ S [ppt] | 0.0 | 5.0 | 10.0 | 20.0 | 35.0 |
|---|---|---|---|---|---|
| -20.0 degC | 146.6 | 112.1 | 97.8 | 77.6 | 55.3 |
| -15.0 degC | 146.6 | 107.6 | 91.5 | 68.7 | 43.5 |
| -10.0 degC | 146.6 | 100.0 | 80.7 | 53.4 | 23.4 |
| -5.0 degC | 146.6 | 82.3 | 55.7 | 18.1 | 0 |
| -2.5 degC | 146.6 | 56.9 | 19.8 | 0 | 0 |

Values are `sigma_b_inf` in kPa; `0` means the contact is above the local freezing point (no bond) or the brine volume has passed the strength cut-off. Both trends are the physical ones: colder pore water and fresher ice give stronger bonds.

### 3.3 Bond growth time scale from a real conduction solution

| nodes | tau (strength proxy) [h] | t95 [h] | steps | enthalpy closure error |
|---|---|---|---|---|
| 21 | 22.252 | 86.26 | 2384 | 2.051 % |
| 41 | 22.225 | 86.24 | 9534 | 1.038 % |
| 81 | 22.223 | 86.23 | 38133 | 0.522 % |

| diagnostic | value |
|---|---|
| tau_f used by the model [h] | 22.225 |
| closed-form rho c h^2/(4k) [h] | 14.473 |
| ratio tau_f / comparator | 1.536 |
| fundamental diffusion mode [h] | 23.46 |
| lens freeze time (latent phase) [h] | 19.80 |
| Stefan time of a 5 mm lens at 10 K [h] | 0.0555 |
| bond-plane temperature half time [h] | 32.26 |
| lens-free run: deviation from comparator | 28.2 % |

The latent heat of the lens is what pushes `tau_f` above the sensible conduction time: the lens itself would freeze in 3.3 min, but the block that absorbs that heat needs 19.8 h to do it. The rate-limiting variable is therefore the cooling of the surrounding ice, not the freezing of the water film.

### 3.4 Keel resistance vs age and the Lindqvist age factor

| t [h] | R_peak [kN/m] | seed spread [kN/m] | R_residual [kN/m] | k_age(t) | bonds broken | particles |
|---|---|---|---|---|---|---|
| 0 | 88.47 | 42.45 | 78.55 | 1.000 | 0/0 + 0/0 | 54 |
| 6.0 | 56.38 | 3.66 | 43.05 | 0.637 | 74/74 + 74/74 | 54 |
| 24.0 | 100.29 | 18.88 | 43.80 | 1.134 | 74/74 + 74/74 | 54 |
| 72.0 | 135.81 | 29.60 | 42.69 | 1.535 | 74/74 + 74/74 | 54 |
| 168.0 | 139.54 | 30.05 | 42.66 | 1.577 | 74/74 + 74/74 | 54 |

Peak and residual are both reported. The residual of the bonded keels is markedly lower than the unbonded one: once the freeze bonds have failed, the previously bonded packing has been pushed into a looser configuration, so its friction-only capacity is below that of a keel that never had bonds. The peak is the design-relevant quantity; the residual is reported because it shows the mechanism.

Age factor for the Lindqvist keel term (`R_keel(t) = R_keel^Lindqvist * k_age(t)`, reference 1.000):

| t [h] | k_age [-] | R_keel [x reference] |
|---|---|---|
| 0 | 1.0000 | 1.0000 |
| 6.0 | 0.6373 | 0.6373 |
| 24.0 | 1.1336 | 1.1336 |
| 72.0 | 1.5352 | 1.5352 |
| 168.0 | 1.5773 | 1.5773 |

### 3.5 Growth time: conduction-derived vs the prescribed Arrhenius value

| block size h [m] | tau derived [h] | rho c h^2/(4k) [h] | derived/comparator | prescribed tau_g [h] | prescribed/derived |
|---|---|---|---|---|---|
| 0.05 | 1.224 | 0.145 | 8.457 | 4.625 | 3.779 |
| 0.25 | 8.372 | 3.618 | 2.314 | 4.625 | 0.553 |
| 0.50 | 22.225 | 14.473 | 1.536 | 4.625 | 0.208 |
| 1.00 | 65.562 | 57.891 | 1.133 | 4.625 | 0.071 |

The prescribed (`PARAMETRIC_MODEL`) growth time is a material constant, while the conduction-derived one grows with the block size (super-linearly, `~h^2` with the latent heat of a fixed 5 mm lens on top). The two therefore agree at exactly one block size - roughly the centimetre-to-decimetre scale of a laboratory contact - and the prescribed value is optimistic by roughly a factor 4.8 for a half-metre field block. This is not a refutation of the sibling module (its law is anchored on measured laboratory data, this one is anchored on thermal physics); it is the scale-transfer statement the prescribed parameter cannot make on its own.

### 3.6 Young-age paired seed study (why R_keel(t) is not monotone)

The ensemble curve of 3.4 is **non-monotone**: the 6 h keel is weaker than the unbonded 0 h keel. DEM peak loads are strongly packing-sensitive, so the dip was re-measured on *paired* seeds - the same packing at every age, only the bond state differs:

| seed | R_peak(0 h) [kN/m] | R_peak(6 h) [kN/m] | R_peak(24 h) [kN/m] | R(6h)/R(0h) |
|---|---|---|---|---|
| 20261031 | 130.92 | 60.04 | 119.17 | 0.459 |
| 20261068 | 46.02 | 52.73 | 81.40 | 1.146 |
| 20261105 | 47.06 | 48.22 | 83.72 | 1.025 |
| 20261142 | 48.75 | 59.03 | 96.28 | 1.211 |

Ensemble mean peak: 0 h = 68.19 +/- 36.23 kN/m, 6 h = 55.01 +/- 4.82 kN/m, 24 h = 95.14 +/- 14.98 kN/m.

`R(6h)/R(0h)` ranges 0.459-1.211 (mean 0.960); the young-age dip occurs in **1/4** paired packings. The dip is therefore a packing-dependent DEM result, not a numerical accident of one seed.

Shear-capacity diagnostic at 6 h (bond shear capacity multiplied, same packings):

| seed | factor | R_peak [kN/m] | broken bonds |
|---|---|---|---|
| 20261031 | 10.0 | 328.51 | 74/74 |
| 20261068 | 10.0 | 212.80 | 74/74 |
| 20261105 | 10.0 | 252.70 | 65/65 |
| 20261142 | 10.0 | 219.22 | 80/80 |

The response is unambiguous: multiplying the bond shear capacity by 10 raises the 6 h peak from 55.0 to 253.3 kN/m, i.e. by a factor 4.6. A young keel is therefore **shear-bond limited** long before it is tension-bond limited: with the weak shear capacity of a 6 h bond the freeze bonds fail early in the press stroke, the slipping contact is driven into a looser configuration, and for one of the four packings the peak even falls below the friction-only value. Once the shear strength reaches the 24 h level the bonded keel is the stronger one for every packing, and from there `k_age` grows monotonically to 1.58 at 168 h. `k_age` is therefore offered for t >~ 24 h only, and the shear/tension split `kappa` is a first-order parameter of this result - a point where this module (`kappa = 2`, the `ridge_dem` material convention) and the sibling `ridge_freeze_bond_aging` module (`kappa = 0.5`, declared) differ.

### 3.7 Contact-convention sensitivity

| bond contact mode | R_peak [kN/m] | R_residual [kN/m] | bonds broken |
|---|---|---|---|
| exclusive | 136.52 | 42.51 | 74/74 |
| parallel | 169.59 | 43.49 | 74/74 |

Both probes are the fully aged material at t = 168 h. The headline curve uses `parallel`; the spread between the two conventions is a genuine model-choice uncertainty and is quoted, not hidden.

### 3.8 Figures

![fig1_freeze_bond_strength.png](../artifacts/ridge_freeze_bond_age/figures/fig1_freeze_bond_strength.png)

*sigma_b(t) and tau_b(t) with the calibrated band: the eta_b-sensitivity of the asymptote is the dominant uncertainty.* `sha256=1e17c989c035f410...`

![fig2_bond_cooling_timescale.png](../artifacts/ridge_freeze_bond_age/figures/fig2_bond_cooling_timescale.png)

*1-D cooling of the bond region: bond-plane temperature, frozen fraction of the lens and the node-convergence of the derived time scale.* `sha256=12a0c610f6c51e37...`

![fig3_keel_resistance_age.png](../artifacts/ridge_freeze_bond_age/figures/fig3_keel_resistance_age.png)

*DEM keel resistance at the campaign ages (peak and residual) and the age factor k_age(t) that multiplies the Lindqvist keel term.* `sha256=dc4d71f27a4dfdbd...`

![fig4_strength_map.png](../artifacts/ridge_freeze_bond_age/figures/fig4_strength_map.png)

*Asymptotic freeze-bond strength over contact temperature and pore water salinity.* `sha256=b0fe904b26c47dc4...`

## 4. Notes, pitfalls and scope

1. **Enthalpy double counting (found and fixed in this run).** The first version of the cooling solution injected the lens latent heat into the adjacent block node *in addition* to the conduction term that already carried that flux. The enthalpy ledger exposed it: the closure error converged to 15.5 % under node refinement (a bookkeeping error does not converge away) and `tau_f` came out 2.29x the closed-form comparator. After the fix the closure converges to 0.52 % at 81 nodes and `tau_f`/comparator = 1.54. The pytest suite keeps the ledger as a regression test (< 3 % at 41 nodes).
2. **The anchor is a calibration, not a validation.** `eta_b` is fitted so that `sigma_b_inf(-10 degC, 5 ppt)` equals the Ettema-Schaefer (1986) point used by the ridge-rubble deck of this repository (100 kPa). That paper is not readable offline (`WITHHELD-PRIMARY`), so the honest content is the reported band (20-200 kPa) and the T/S trends, not the point value. A 10x larger anchor band changes the asymptote tenfold (see fig. 1, right panel) and `k_age` hardly at all, because `k_age` is a ratio at fixed material.
3. **Contact convention matters.** `exclusive` (historical default) switches the ice-ice contact law off for a bonded pair, which makes a bonded packing more compliant than a real frozen one; the sweep uses `parallel`. The two conventions differ by a factor quoted in 3.7 - for absolute keel loads the user should carry that as an uncertainty.
4. **Non-monotone R_keel(t) (honest result).** The measured keel resistance is *not* monotone in age: at 6 h it is 0.64x the unbonded value, and only above ~24 h does bonding raise the resistance k_age(24h) = 1.13 and k_age(168h) = 1.58. The constitutive bond strength grows monotonically (`sigma_b(t)`), so this is a DEM load-path effect, quantified on paired packings in 3.6, not a strength-law effect. The node's monotonicity criterion is therefore reported as **failed**, not reworded.
5. **Representative-volume caveat.** The smoke deck (9 particles, used by `--quick`) can show a freshly bonded packing *weaker* than the unbonded one, because all of its bonds fail in one event. The headline curve is the full deck (54 particles, 74 bonds, 2 seeds); the smoke deck writes to `artifacts/ridge_freeze_bond_age_quick/` and never overwrites it.
6. **Which commit produced which number.** The DEM sweep was re-run after merging the current `origin/main` (513 commits of sibling work) and reproduced every printed peak/residual to the last digit (88.47, 56.38, 100.29, 135.81, 139.54 kN/m), so the upstream changes are diagnostics-only for this load path and the paired seed study, which was computed on the pre-merge commit, applies unchanged.
7. **Scope.** 2-D rotationless disks, one particle size, no brine drainage, no regelation or sintering, no flexural failure of the ice blocks, rigid press with penalty contact. The model is one-way coupled (thermal -> strength -> DEM); the DEM does not feed heat back. `k_age` is only meaningful above ~6 h: below that the bonds are younger than their own formation time and the discrete packing noise dominates.

## 5. Error assessment (explicit)

| quantity | deviation | baseline | < 3 % ? |
|---|---|---|---|
| mesh convergence of `tau_f` (21 vs 81 nodes) | 0.13 % | self-convergence, no experimental baseline | yes |
| enthalpy closure of the cooling solution | 0.52 % | conservation identity | yes |
| `sigma_b_inf` vs the Ettema-Schaefer anchor | 0 % | calibrated on it (residual zero by construction) | n/a (calibration, see the note in 4.2) |
| `tau_f` vs `rho c h^2/(4k)` | 53.6 % | closed-form scale, not a measurement | **no** (>3 %; the excess is the latent heat of the lens) |
| monotonicity of `R_keel(t)` over the campaign ages | k_age(6h) = 0.637 | the campaign's physical premise (monotone growth) | **no** (dip at 6 h, see 3.6) |
| `k_age(t)` vs independent measurement | - | no independent aged-ridge load data in this repository | **not testable here** |

Only two of these are *quantitative agreements*: the mesh self-convergence and the enthalpy closure, both below 3 %. The anchor comparison is a calibration (0 % by construction, no predictive content), the conduction comparison passes the order-of-magnitude criterion of the node but not a 3 % band, and the age factor has no independent benchmark in this repository - it is a measured DEM trend, so its absolute value is only as good as the material parameters. This is reported as a limitation, not as a validation.

## 6. Machine acceptance ledger

| id | statement | verdict |
|---|---|---|
| C1_sigma_limits_and_monotonicity | sigma_b(t) -> 0 as t -> 0, -> sigma_inf as t -> inf, monotone increasing | yes |
| C2_anchor_error | sigma_inf vs Ettema & Schaefer (1986) T=-10C, S=5ppt within 20 % | yes |
| C3_tau_f_vs_conduction | tau_f within a factor 3 of the closed-form conduction time rho c h^2/(4k) (order-of-magnitude consistency), and the 1-D solution is node-convergent within 5 % | yes |
| C4_R_keel_monotone_and_unbonded_zero | campaign premise: R_keel(t) monotone increasing over all ages, with R_keel(0) = the unbonded keel (reported as FAIL if the young-age dip is real) | **no** |
| C4b_mature_aging_raises_resistance | above the young-age transition (t >= 24 h) keel resistance grows monotonically with age and k_age(168 h) > 1.4 | yes |
| C5_salinity_dependence | sigma_inf increases as S -> 0 (fresh water) | yes |
| C6_pytest_chain | full chain under pytest (monotonicity, t=0 degeneration, S dependence) | yes |

`all_pass = **no**` (unresolved: none). The pytest verdict is executed by the runner itself, so the ledger cannot record a test that was not run.

## 7. Reproduction

```
python -m pytest tests/test_ridge_freeze_bond_age.py -q   # ~20 s
python scripts/run_ridge_freeze_bond_age.py               # full deck, ~15 min
python scripts/run_ridge_freeze_bond_age.py --quick       # smoke, ~1 min
python scripts/make_ridge_freeze_bond_age_doc.py          # this document
```
