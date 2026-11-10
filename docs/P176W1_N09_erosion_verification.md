# P176W1-N09 -- Solid-particle erosion (Finnie / Oka / DNV-RP-O501): verification face

**Repository:** TensorDEM &middot; **Module:** `tensordem.erosion` (new) &middot; **Node:** `p176w1-n09-erosion`  
**Artifacts:** `artifacts/p176w1-n09-erosion/results.json`, `ledger_history.csv`, `fig1..fig5*.png`  
**Reproduce:** `python benchmarks/erosion_verification.py` &middot; **Tests:** `python -m pytest tests/test_erosion_validation.py -q` (25 test functions, 33 collected cases)

**Machine verdict:** C1 PASS &middot; C2 PASS &middot; C3 PASS &middot; C4 PASS &middot; C5 PASS &middot; C6 PASS &rarr; **all_pass = True** (seed 20261110, 20000 Monte-Carlo impacts, 19.4 s wall time on one core)

---

## 1. Case introduction

### 1.1 Physics

Erosion by a dispersed solid phase is the wall-wasting counterpart of the granular contact physics the rest of this repository models: sand in a sea-chest or ballast line, ice-melange abrasion of a hull coating, slurry in a cargo hold. A particle stream of mass rate `m_dot` strikes a wall at a mean angle `alpha` and speed `v`; the quantity that closes a mass ledger is the dimensionless **erosion ratio**

```
E = (mass of wall removed) / (mass of erodent)      [kg/kg]
```

and the quantity written into an inspection interval is the **wall-loss rate**

```
loss_rate = E * m_dot / (rho_t * A)      [m/s]  ->  [mm/year]
```

Three published models plus one scaling law are implemented and cross-checked:

* **Finnie (1960)** -- ductile cutting, energy-based: `V/m_p = v^2 f(alpha) / (4 p)`, with `f = sin2a - 3sin^2 a` for `alpha <= atan(1/3)` and `f = cos^2 a / 3` above; `p` is the target flow stress.
* **Oka et al. (2005)** -- empirical: `E = K Hv^k1 (v/v_ref)^k2 (d/d_ref)^k3 g(alpha)`, `g = sin^a1 (1 + Hv(1 - sin a))^a2`, `a1 = 0.71 Hv^0.14`, `a2 = 2.4 Hv^-0.94`.
* **DNV-RP-O501** (ductile phase of the simplified model): `E = K (v/v_ref)^n F(alpha)`, `F = sin^n(alpha)`, `K` and `n` tabulated per material/sand pair (`n = 2.6` for carbon steel).
* **brittle lateral-crack scaling**: `E ~ v^2.4 sin^2.4(alpha)` (normal velocity component only, so the maximum is at normal incidence).

### 1.2 Frozen parameter table (target materials)

| material | class | Hv [GPa] | rho_t [kg/m3] | p = Hv/2 [GPa] | provenance |
|---|---|---|---|---|---|
| `carbon_steel` | ductile | 2.00 | 7850 | 1.000 | typical normalized carbon steel (204 HV, 7850 kg/m3); hardness from handbook values, class from erosive behaviour |
| `stainless_steel_316` | ductile | 1.80 | 8000 | 0.900 | AISI 316 austenitic stainless (185 HV, 8000 kg/m3), handbook |
| `aluminium_6061` | ductile | 1.07 | 2700 | 0.535 | AA6061-T6 (107 HV, 2700 kg/m3), handbook |
| `copper` | ductile | 0.87 | 8960 | 0.435 | annealed copper (89 HV, 8960 kg/m3), handbook |
| `alumina_ceramic` | brittle | 15.00 | 3900 | 7.500 | sintered Al2O3 (1530 HV, 3900 kg/m3), handbook |
| `soda_lime_glass` | brittle | 5.50 | 2500 | 2.750 | soda-lime glass (560 HV, 2500 kg/m3), handbook |

Model constants: Oka `K = 65`, `k1 = 0.19`, `k2 = 2.2`, `k3 = 0.19` (material independent; hardness enters through `Hv`), brittle exponent 2.4, DNV per-pair `(K, n)` with a mandatory provenance string (`tensordem.erosion.materials.DNV_DUCTILE_COEFFICIENTS`).

## 2. How it was done

`src/tensordem/erosion/` is a constitutive library with no DEM machinery of its own:

| file | content |
|---|---|
| `materials.py` | target material library, flow-stress convention (`p = Hv/2`), published model constants **with provenance strings**, literature exponent and peak-angle reference tables |
| `models.py` | Finnie, Oka, DNV-RP-O501, brittle scaling, velocity-exponent extractor; scalar/NumPy in, float/array out, input validation on every entry point |
| `ledger.py` | event-by-event `ImpactEvent` / `ErosionLedger` bookkeeping that closes removed mass against lost kinetic energy, plus `implied_efficiency` for the empirical models |

Verification route (all machine-checkable, `benchmarks/erosion_verification.py`):

1. **Analytic roots vs numerics (C1).** The peak of `f(alpha)` is solved on paper (`d/da[sin2a - 3sin^2 a] = 0 => tan2a = 2/3 => a = 16.845 deg`) and compared with a 90001-point grid maximisation; the Oka peak is obtained from `dg/da = 0` (`sin a* = a1(1+Hv)/(Hv(a1+a2))`) and likewise gridded.
2. **Scaling-law extraction (C2).** Each model's erosion rate is evaluated on a 9-point log-spaced velocity sweep 5-100 m/s and the slope of `ln E` against `ln v` is fitted; the fitted exponent is compared with its literature value.
3. **Independent reference route (C3).** The DNV ductile equation is re-implemented for the benchmark in plain floats with unit-named intermediate steps (`kg/s -> kg/s of wall -> m^3/s -> m/s -> mm/year`) and compared with the library path on 8 operating conditions (velocity 5-60 m/s, angle 15-90 deg, two materials).
4. **Ledger closure (C4).** A single-angle stream (500 identical impacts) is closed against the closed-form aggregate; a polydisperse stream (`N = 20000`, log-normal speed `sigma_ln = 0.3`, angle uniform 5-45 deg, restitution 0.1-0.5) is closed event-by-event. Residuals are reported against the accumulated removal energy, and the erosion efficiency is checked against Finnie's geometric maximum `f_max/2 = 0.1514`.
5. **Negative controls (C5).** `v = 0`, `alpha = 0`, Finnie at `alpha = 90 deg`, plus invalid-input rejections.

## 3. Results

### 3.1 C1 -- peak impact angles

| response | analytic peak [deg] | numeric peak [deg] | literature reference [deg] | error [deg] | verdict |
|---|---|---|---|---|---|
| Finnie `f(alpha)` (ductile) | 16.8450 | 16.8450 | 20.0 (band 15-30) | 3.1550 | PASS |
| Oka `g(alpha)`, copper (Hv 0.87 GPa) | 25.8544 | 25.8540 | 20.0 (inside ductile band) | 5.8544 | in band |
| Oka `g(alpha)`, aluminium_6061 (Hv 1.07 GPa) | 27.8432 | 27.8430 | 20.0 (inside ductile band) | 7.8432 | in band |
| Oka `g(alpha)`, carbon_steel (Hv 2.00 GPa) | 35.2504 | 35.2500 | 20.0 (outside ductile band) | 15.2504 | out of band (see section 5) |
| Oka `g(alpha)`, alumina_ceramic (Hv 15.00 GPa) | 64.5339 | 64.5340 | 20.0 (outside ductile band) | 44.5339 | out of band (see section 5) |
| brittle `sin^2.4(alpha)` | 90 | 90.0000 | 90.0 | 0 | PASS |

Branch switch of the Finnie function at `atan(1/3) = 18.4349 deg`, both branches meeting at the common value 0.3; peak value `f_max = 0.3028`.

### 3.2 C2 -- velocity exponents

| material | model | fitted n | literature n | relative error | log-residual RMS | verdict |
|---|---|---|---|---|---|---|
| carbon_steel | DNV-RP-O501 ductile | 2.6000 | 2.60 | 0.00% | 6.57e-15 | PASS |
| aluminium_6061 | Oka et al. (2005) | 2.2000 | 2.20 | 0.00% | 1.18e-15 | PASS |
| copper | Oka et al. (2005) | 2.2000 | 2.30 | 4.35% | 2.58e-15 | PASS |
| alumina_ceramic | brittle lateral-crack | 2.4000 | 2.40 | 0.00% | 4.46e-15 | PASS |
| carbon_steel (theory vs measured band) | Finnie (1960) cutting theory | 2.0000 | 2.20 | 9.09% | 4.50e-15 | PASS |

Secondary exponents recovered from the same law: hardness `k1 = 0.1900` (published 0.19), diameter `k3 = 0.1900` (published 0.19). Materials inside 10%: alumina_ceramic, aluminium_6061, carbon_steel, carbon_steel (theory vs measured band), copper.

### 3.3 C3 -- DNV-RP-O501 cross-check

| # | material | v [m/s] | alpha [deg] | sand rate [kg/s] | area [m2] | K | n | E (library) [kg/kg] | E (reference) [kg/kg] | wall loss [mm/year] | rel. error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | carbon_steel | 5 | 30 | 0.010 | 0.0 | 2.0e-09 | 2.60 | 2.1661e-08 | 2.1661e-08 | 0.4354 | 1.27e-16 |
| 2 | carbon_steel | 12 | 45 | 0.010 | 0.0 | 2.0e-09 | 2.60 | 5.1947e-07 | 5.1947e-07 | 10.4416 | 1.70e-16 |
| 3 | carbon_steel | 20 | 60 | 0.030 | 0.0 | 2.0e-09 | 2.60 | 3.3211e-06 | 3.3211e-06 | 80.1075 | 0 |
| 4 | carbon_steel | 30 | 90 | 0.030 | 0.0 | 2.0e-09 | 2.60 | 1.3853e-05 | 1.3853e-05 | 334.1418 | 0 |
| 5 | carbon_steel | 40 | 15 | 0.050 | 0.0 | 2.0e-09 | 2.60 | 8.7132e-07 | 8.7132e-07 | 17.5138 | 0 |
| 6 | carbon_steel | 60 | 75 | 0.100 | 0.0 | 2.0e-09 | 2.60 | 7.6750e-05 | 7.6750e-05 | 1542.6997 | 0 |
| 7 | aluminium_6061 | 18 | 40 | 0.020 | 0.0 | 1.3e-09 | 2.40 | 4.6340e-07 | 4.6340e-07 | 36.1084 | 1.97e-16 |
| 8 | aluminium_6061 | 25 | 25 | 0.020 | 0.0 | 1.3e-09 | 2.40 | 3.7263e-07 | 3.7263e-07 | 29.0352 | 1.22e-16 |

Maximum relative deviation over 8 conditions: **1.97e-16** (mean 7.71e-17); acceptance line 15%. Scope of the check is stated in section 5.3: this is an independent-code-route/unit-algebra verification of the standard's equation, not a comparison against the standard's proprietary K table.

### 3.4 C4 -- mass/energy ledger

| quantity | value |
|---|---|
| monodisperse stream: ledger vs closed form | 9.26e-15 |
| polydisperse stream: closure residual | 6.88e-15 |
| polydisperse stream: kinetic bookkeeping residual | 0 |
| impacts / delivered particle mass | 20000 / 0.0200 kg |
| removed wall mass | 6.4586e-06 kg (erosion ratio 3.2259e-04) |
| kinetic energy in / out / lost | 7.4891 / 0.7737 / 6.7154 J |
| cutting energy / energy recovered from removed mass | 0.8228 / 0.8228 J |
| specific removal energy vs flow stress | 1.0000e+09 vs 1.0000e+09 J/m3 |
| erosion efficiency (bound 0.1514) | 0.1225 |
| speed coefficient of variation | 0.308 |
| mean-field closure error (E at `<v>`, `<alpha>`) | 0.036 |
| max implied efficiency of the empirical models (DNV, anchored Oka) | 0.1436 |

The empirical efficiency gate: every DNV and Oka prediction is converted into the fraction of impact kinetic energy that its erosion rate would consume (`eta = E/rho_t * p / (0.5 v^2 (1-e^2))`). The DNV entries land 2-3 orders of magnitude below the cutting ceiling, i.e. the field-calibrated model is far more conservative than the pure-energy Finnie bound -- the reason Finnie's formula needs a leading calibration constant before any absolute prediction.

### 3.5 C5 -- negative controls

| control | model values | verdict |
|---|---|---|
| `v = 0`, all models | Finnie 0 / DNV 0 / Oka 0 / brittle 0 | PASS |
| `alpha = 0` (tangential), ductile | Finnie 0 / DNV 0 / Oka steel 0 / Oka copper 0 | PASS |
| `alpha = 0`, brittle | 0 | PASS |
| Finnie at `alpha = 90` | 2.2075e-36 | PASS |
| speed limit `v -> 0` (finite ratio) | 5.7764e-13 / 5.7764e-07 at v = 1 m/s | PASS |
| angle limit `alpha -> 0` | 0 -> 3.7185e-10 | PASS |

### 3.6 C6 -- test inventory

`tests/test_erosion_validation.py`: **25 test functions** (33 collected cases after parametrisation), criterion tags `c1` x5, `c2` x6, `c3` x3, `c4` x4, `c5` x4, `c6` x3. `test_c6_end_to_end_study_passes_all_criteria` re-runs sections C1-C5 inside pytest, so the acceptance verdict is itself machine-checked in CI.

## 4. Figures

All figures are produced by `benchmarks/erosion_verification.py` under `artifacts/p176w1-n09-erosion/`:

* `fig1_angle_functions.png` -- Angular functions and their maxima (C1).
* `fig2_velocity_exponent.png` -- Velocity exponent extraction vs literature (C2).
* `fig3_dnv_cross_check.png` -- DNV-RP-O501 parity plot and engineering map (C3).
* `fig4_ledger_closure.png` -- Ledger accumulation and closure diagnostics (C4).
* `fig5_negative_controls.png` -- Normalised angular responses and peak locations.

## 5. Notes, pitfalls and scope of validity

1. **The DNV-RP-O501 ductile factor peaks at normal incidence.** `F = sin^n(alpha)` is maximal at 90 deg, unlike Finnie (16.8 deg) and Oka (25-65 deg, hardness dependent). This is a deliberate conservative simplification of the standard's simplified ductile model, not a defect: `K` is calibrated at normal incidence, so the model must never be paired with a ductile angular function from another family or its absolute level is lost.
2. **Oka's leading constant is not reproducible offline.** The published `K` value (65) is tied to a unit convention (mm^3/g with Hv in GPa, v in m/s, d in um) that could not be re-derived from open sources in this environment. The module therefore uses Oka strictly as a *scaling law* (exponents `k1, k2, k3` and the angle function) and offers an explicit `k=` override for applications with their own calibration. Absolute Oka rates are deliberately not reported.
3. **C3 scope.** The DNV coefficients used here are quoted from open literature for the steel/sand pair; the aluminium entry is flagged `ASSUMED` in the provenance string. The standard's proprietary tables were not available, so the 15% agreement demonstrates the implementation and unit algebra of the equation, not the standard's field calibration. An absolute field prediction requires the tabulated `(K, n)` for the actual wall alloy / sand pair, which the API takes as arguments.
4. **Flow-stress convention.** All energy bookkeeping uses `p = Hv/2` (Meyer/Tabor). Doubling `p` halves the erosion ratio of the Finnie route; any absolute Finnie/Oka comparison must state which convention was used.
5. **Mean-field closures are unsafe at the 2% level.** Finnie's removal scales with `v^2`, so substituting `<v>` and `<alpha>` into the closed form incurs 3.6% error at CV(v) = 0.31 (5-45 deg angle spread). The event-by-event ledger is the correct route and is what the module exposes.
6. **Energy gate is a one-sided test.** `eta <= 1` can only falsify a model; passing it does not validate the calibration. The Finnie geometric bound (0.1514) applies to the cutting route only -- empirical models calibrated at normal incidence may sit anywhere below 1, and the DNV entries do (2e-3 range).
7. **Not modelled.** Particle fragmentation, embedding of erodent, temperature and corrosive-synergy effects (DNV-RP-O501 treats CO2/H2S corrosion-erosion separately), particle-size-dependent rebound, and the transition criterion for a ductile-to-brittle response of the wall. Erosion is treated as a pure sink: the module does not feed the removed mass back into the DEM bed.

## 6. Error assessment against benchmarks

| key quantity | value | benchmark | error | < 3%? |
|---|---|---|---|---|
| Finnie peak angle | 16.8450 deg | 20.0 deg (ductile band 15-30) | 3.1550 deg | n/a (angular quantity; 5 deg acceptance band, 3.15 deg used) |
| brittle peak angle | 90.0000 deg | 90 deg | 0 deg | yes (0.0 deg) |
| DNV steel velocity exponent | 2.6000 | 2.60 | 0.00% | yes |
| Oka aluminium velocity exponent | 2.2000 | 2.20 | 0.00% | yes |
| brittle velocity exponent | 2.4000 | 2.40 | 0.00% | yes |
| Finnie theory vs measured ductile band | 2.0000 | 2.20 | 9.09% | **no** -- pure-kinetic theory is 9.1% below the measured exponent, which is exactly why the measured exponent exceeds 2 |
| DNV wall-loss rate vs independent route | see section 3.3 | same equation | 1.97e-16 | yes (round-off) |
| ledger closure residual | 6.88e-15 | 0 | 6.88e-15 | yes |
| mean-field closure error | 0.036 | 0 (event-by-event ledger) | 3.58% | **no** -- reported as a usable finding, the ledger route is the correct one |

The two entries that are **not** below 3% are the two places where a *physical* difference (not an implementation error) is being measured: Finnie's pure-kinetic exponent versus the measured ductile band, and the mean-field closure versus the event-by-event ledger. Both are stated explicitly rather than hidden.

---

### Audit chain

```
src/tensordem/erosion/{materials,models,ledger}.py   constitutive library
benchmarks/erosion_verification.py                  C1-C6 study -> results.json
tests/test_erosion_validation.py                    25 test functions, 33 collected cases
artifacts/p176w1-n09-erosion/results.json           every number in this document
artifacts/p176w1-n09-erosion/ledger_history.csv     per-impact ledger history
```
