# Mixed first-year / multi-year ice field in the DEM section: is the linear mixture rule true?

Evidence grade of this node: **capability extension** (temperature-resolved MY
strength law) + **new capability** (direct heterogeneous-field test of the
mixture rule). All claims are *relative*: the absolute ship-scale level inherits
the documented error of the level-ice anchor against the MT Uikku model tests
(`docs/LEVEL_ICE_RESISTANCE_DEM.md`).

### Answers to the node's questions

1. **How much does the total ice resistance change when the multi-year area
   fraction goes 0 % → 60 %?**
   * **Ship scale** (anchor quadrature with class-mean thickness and the
     extended ageing law, v = 3 m/s, Lindqvist):
     R = 0.990 MN at p = 0 →
     **5.044 MN** at p = 0.6, i.e.
     **+409.4 %**
     (Riska +442.8 %). Of the class ratio
     9.306, **54 %** of the increment is the MY class
     thickness, 8 % the material state and 38 % their interaction.
   * **DEM section scale** (material only, geometry fixed, 10 balanced /
     16 random field realisations per fraction):
     R = 5.340 MN/m at p = 0 →
     **7.790 MN/m** at the measured
     p = 0.600, i.e.
     **+45.9 %** (fitted line at p = 0.60:
     +54.9 %).
2. **Is the linear mixture rule true?**
   * **11 of 12** measured configurations are inside the ±15 %
     band, and the ensemble bias is **systematically negative** — the
     heterogeneous field carries *less* than the area average, by roughly 10 % on
     average (measured −7.0 … −13.3 % over the interior fractions of the
     balanced layouts).
   * The single configuration outside the band is the nearly-pure MY field
     (traversed p = 0.88, random layout): R is **-22.7 %** below the linear rule —
     inserting a thin FY seam into an otherwise MY field costs about a quarter of
     the resistance (**weakest-link** effect: the compliant seam fractures first,
     the MY sheet then clears without being crushed).
   * Because of that, criterion (b) as a *universal* statement is
     **FAIL** (worst bias 22.7 % > 15 %);
     as a ~15 %-accuracy statement over 0–80 % MY it holds in
     11/12 configurations. Fitting the whole ensemble
     gives a slope of 5.310 MN/m per unit p against an ideal
     6.012 MN/m, i.e. 51 % of the ideal slope.
   * The harmonic (series) mixture is off by up to
     18 % and is decisively rejected: the section
     shares load in parallel, not in series.
3. **Is the MY/FY strength ratio physical?** The extended ageing law gives
   σ_c^MY/σ_c^FY = **2.075** at t = 4 a, inside the literature band
   [1.5, 3.0]. The pre-extension law (porosity only) gave
   1.460 — **below** the band — and the other parameterisation already in
   the repository (`level_ice.multiyear_ice` defaults) gives 5.00 —
   **above** it. Neither existing parameterisation satisfied the criterion; the
   extended law does.

## 1. Case, geometry and parameters

**Physical setting.** A ship transiting a field in which a fraction ``p`` of the
*area* is multi-year ice, the rest first-year ice. MY ice is thicker, colder,
less saline, stronger and stiffer. The *material* change at fixed geometry is
measured with the DEM; the *class-mean thickness* enters through the anchor
quadrature (`tensordem.mixed_ice`, physics unchanged).

**Age-dependent material law (the extension).** Age is the single independent
variable; salinity, mean temperature and class-mean thickness relax exponentially
to observed MY asymptotes; brine volume follows the gas-free freezing-point
relation, and

    v_b = S (1 + 49.185/|T|) / 1000,
    sigma_f = 1.76 exp(-5.88 sqrt(v_b)) MPa                      (Timco & O'Brien)
    sigma_c = sigma_solid (|T|/|T_FY|)^m (1 - v_b/v_p)^2         (extended here)
    E       = E_solid (1 - v_b/v_p),      v_p = 0.28

The only change to the pre-existing law is the temperature amplification
``(|T|/|T_FY|)**m`` with m = 0.35 (band
0.2–0.5). It equals 1.0 at the first-year reference temperature, so
the age-1 state — and every first-year number in the repository — is unchanged to
machine precision; it saturates because the mean temperature saturates.

| quantity | FY (t = 1 a) | MY (t = 4 a) | MY/FY |
|---|---|---|---|
| thickness [m] | 1.20 | 2.93 | 2.44 |
| salinity [‰] | 5.00 | 1.55 | 0.31 |
| mean temperature [°C] | -5.00 | -13.65 | 2.73 |
| brine volume [-] | 0.0542 | 0.0071 | 0.13 |
| σ_f [MPa] | 0.448 | 1.071 | 2.392 |
| σ_c [MPa] | 1.500 | 3.113 | 2.075 |
| E [GPa] | 5.000 | 6.042 | 1.208 |

**DEM case.** 2-D bonded-particle plane section, h = 1 m,
particle size h/4 = 0.25 m, field length 24 m,
hull advance 6 m, bow 30°, speed 3 m/s,
mass scaling 200, buoyant foundation on, evaluation window
(0.55, 0.98), deterministic step
3.0008e-04 s.

**Heterogeneous field construction.** Each particle carries a class label (FY/MY)
and per-particle multipliers of σ_f, E and σ_c equal to the *ratios* of that
class' material state to the first-year reference (table above); a bond
straddling a boundary takes the arithmetic mean of its two end particles. Two
layouts are simulated:

* **balanced (primary).** The class pattern is periodic with a period equal to
  the traversed window (20 columns = 5 m), so **every**
  transit is exposed to exactly the nominal MY fraction and the only remaining
  randomness is the fracture path. Nominal patch lengths 2 m and 4 m; the
  realised MY floe length inside the transit is reported per row (it is
  ``min(patch, p·5 m)``).
* **random (field-realistic scatter).** Contiguous 4 m MY patches at seeded
  random positions in the 24 m field: what a single transit through a real mixed
  field samples.

``p = 0`` gives unit multipliers everywhere, so the run *is* the existing
homogeneous model.

## 2. How it was done

1. Extended the ageing law with the temperature amplification (one convention
   parameter, swept), keeping the FY state exact.
2. Added an optional per-particle ``material_field`` to
   `tensordem.level_ice.sheet.LevelIceSheet` (σ_f, E and crushing-cap
   multipliers). The homogeneous path is untouched — with unit multipliers the
   legacy branch runs verbatim, so no existing number changes.
3. Made the stable step deterministic (the ARPACK start vector is pinned).
   Before the fix two *identical* runs gave 5.261 vs 5.340 MN/m —
   1.5 % — purely from the random ``dt``.
4. Swept ``p`` in the balanced layout at two nominal patch lengths, ten field
   realisations per interior fraction (sixteen in the random layout), and
   compared with (i) the linear mixture
   rule using **the same model's** p = 0 and p = 1 resistances and (ii) the
   harmonic (Reuss) mixture.
5. Repeated the sweep with the random layout to measure the single-transit
   scatter, and cross-checked the DEM class ratio against the anchor at the same
   geometry.
6. Produced the ship-scale 0 → 60 % answer with the anchor quadrature including
   the class-mean thicknesses, and decomposed the increment into thickness,
   material and interaction parts.

## 3. Results

### 3.1 Criterion (b): mixture rule vs direct DEM (balanced layout, 5 m transit)

| nominal patch [m] | p (traversed) | MY patches in transit | MY floe length [m] | seeds | DEM R [MN/m] | s.d. | linear (1) | err [%] ±CI95 | series err [%] |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.00 | 0.0 | 0.00 | 1 | 5.340 | 0.000 | 5.340 | +0.0 ± 0.0 | +0 |
| 2 | 0.20 | 1.3 | 0.85 | 10 | 5.951 | 0.582 | 6.542 | -9.0 ± 7.0 | -0 |
| 2 | 0.40 | 1.7 | 1.30 | 10 | 6.914 | 0.675 | 7.745 | -10.7 ± 7.0 | +2 |
| 2 | 0.60 | 2.5 | 1.25 | 10 | 7.790 | 0.818 | 8.947 | -12.9 ± 7.5 | -0 |
| 2 | 0.80 | 2.7 | 1.53 | 10 | 8.799 | 1.116 | 10.149 | -13.3 ± 9.1 | -5 |
| 2 | 1.00 | 1.0 | 5.00 | 1 | 11.352 | 0.000 | 11.352 | +0.0 ± 0.0 | +0 |
| 4 | 0.00 | 0.0 | 0.00 | 1 | 5.340 | 0.000 | 5.340 | +0.0 ± 0.0 | +0 |
| 4 | 0.20 | 1.3 | 0.85 | 10 | 5.951 | 0.582 | 6.542 | -9.0 ± 7.0 | -0 |
| 4 | 0.40 | 1.7 | 1.30 | 10 | 6.914 | 0.675 | 7.745 | -10.7 ± 7.0 | +2 |
| 4 | 0.60 | 1.9 | 1.65 | 10 | 7.979 | 1.232 | 8.947 | -10.8 ± 11.0 | +2 |
| 4 | 0.80 | 1.9 | 2.20 | 10 | 9.218 | 1.150 | 10.149 | -9.2 ± 8.9 | -1 |
| 4 | 1.00 | 1.0 | 5.00 | 1 | 11.352 | 0.000 | 11.352 | +0.0 ± 0.0 | +0 |

Criterion (b), **strict form** (every measured configuration, including its 95 %
confidence half-width, inside ±15 %): worst bias
22.7 % ± 11.0 % CI →
**FAIL**; 11 of
12 configurations are individually inside the band.
The outlier is the random layout at traversed p = 0.88 (worst-case arrangement:
a single thin FY seam inside an otherwise MY transit). Grouping the
configurations by the length of the MY floe crossed does not separate the
result — coarse floes (≥ 1.5 m) reach
22.7 %, fine floes 12.9 % — so the deviation is driven by
the *arrangement* of the MY/FY interface, not by the floe-size resolution.

Ensemble regression over all interior runs: slope
3.036 ± 0.431 MN/m per unit p
(95 % CI 2.191 … 3.881) against an ideal
6.012 MN/m, i.e. 51 % of the ideal slope
(119 runs, R² = 0.297,
residual scatter 15 %). The slope is significantly below the
ideal one: the mixture rule over-states the sensitivity of the section.

DEM class ratio at fixed geometry: ``R_MY/R_FY = 2.126``;
Lindqvist anchor at the same geometry 1.619
(porosity-only ageing law 1.288), so the DEM is more sensitive
to the material state than the anchor's strength scaling by
24 %.

### 3.2 Single-transit scatter (random layout, 4 m MY patches)

| p nominal | p traversed | MY patches in transit | DEM R [MN/m] | s.d. | single-run CV [%] | err vs linear [%] ±CI95 |
|---|---|---|---|---|---|---|
| 0.00 | 0.00 | 0.0 | 5.340 | 0.000 | 0 | +0.0 ± 0.0 |
| 0.20 | 0.07 | 0.2 | 6.193 | 0.688 | 11 | +8.0 ± 5.9 |
| 0.40 | 0.35 | 0.7 | 6.908 | 0.935 | 14 | -7.0 ± 7.2 |
| 0.60 | 0.60 | 1.1 | 7.883 | 1.463 | 19 | -11.7 ± 9.9 |
| 0.80 | 0.88 | 1.1 | 8.236 | 1.420 | 17 | -22.7 ± 9.2 |
| 1.00 | 1.00 | 1.0 | 11.352 | 0.000 | 0 | +0.0 ± 0.0 |

The single-transit coefficient of variation is 11–19 % (and reached 35 % in the
first six-seed pass, which is why the ensembles were enlarged): a *single* 5 m
transit through a mixed field is a poor estimator of the field-average
resistance, while the ensemble mean is stable to the CI in the last column. This
is a property of the measurement (chaotic fracture paths and which patches a
transit meets), not of the mixture rule. The p = 0.20 row has a traversed
fraction of only 0.07 because most random layouts place their MY patch outside
the 5 m transit at that fraction; its +8.0 % is therefore a near-pure-FY case.

### 3.3 Criterion (a): MY/FY compressive-strength ratio

σ_c^MY/σ_c^FY = **2.075** ∈ [1.5, 3.0] →
**PASS**. Porosity-only value
1.460; `level_ice.multiyear_ice` defaults 5.00;
FY degeneration of the law exact: True.

| temperature exponent m | σ_c^MY [MPa] | ratio MY/FY | inside band |
|---|---|---|---|
| 0.20 | 2.677 | 1.785 | yes |
| 0.35 | 3.113 | 2.075 | yes |
| 0.50 | 3.619 | 2.412 | yes |

### 3.4 Criterion (c): monotone and saturating σ_c(t)

Strictly increasing: **True** (smallest increment
3.341e-03 Pa over the tabulated ages, i.e. positive but tiny between
20 a and 50 a), concave: True; saturation
σ_c(∞) = 3.2341 MPa (ratio 2.1561); the residual between
t = 50 a and t → ∞ is 0.0000 % — the exponential
relaxation has reached its analytic asymptote, which is what "saturating" means
here → **PASS**.

### 3.5 Criterion (d): first-year degeneration

| quantity | value |
|---|---|
| existing model (CaseSpec default, deterministic dt) | 5.340057 MN/m |
| heterogeneous implementation at p = 0 | 5.340057 MN/m |
| relative difference | 0.0e+00 |
| material field homogeneous at p = 0 | True |

→ **PASS** (bit-for-bit degeneration).

### 3.6 Time-step sensitivity of the reference case

| dt factor | dt [s] | R [MN/m] | error vs nominal [%] |
|---|---|---|---|
| 0.80 | 2.4006e-04 | 5.3423 | +0.04 |
| 1.00 | 3.0008e-04 | 5.3401 | +0.00 |
| 1.25 | 3.7510e-04 | 5.5261 | +3.48 |

The solver's own dt sensitivity is ±3.5 %, which is the
reproducibility floor of any single DEM comparison.

### 3.7 Ship scale: 0 % → 60 % MY ice

| model | p | R [MN] | R_FY class [MN] | R_MY class [MN] | class ratio | ΔR/R(p=0) [%] |
|---|---|---|---|---|---|---|
| lindqvist | 0.0 | 0.990 | 0.990 | — | — | +0.0 |
| lindqvist | 0.2 | 2.342 | 0.990 | 7.747 | 7.824 | +136.5 |
| lindqvist | 0.4 | 3.693 | 0.990 | 7.747 | 7.824 | +273.0 |
| lindqvist | 0.6 | 5.044 | 0.990 | 7.747 | 7.824 | +409.4 |
| lindqvist | 0.8 | 6.396 | 0.990 | 7.747 | 7.824 | +545.9 |
| lindqvist | 1.0 | 7.747 | — | 7.747 | — | +682.4 |
| riska | 0.0 | 1.528 | 1.528 | — | — | +0.0 |
| riska | 0.2 | 3.784 | 1.528 | 12.807 | 8.380 | +147.6 |
| riska | 0.4 | 6.040 | 1.528 | 12.807 | 8.380 | +295.2 |
| riska | 0.6 | 8.295 | 1.528 | 12.807 | 8.380 | +442.8 |
| riska | 0.8 | 10.551 | 1.528 | 12.807 | 8.380 | +590.4 |
| riska | 1.0 | 12.807 | — | 12.807 | — | +738.0 |

Decomposition of the class ratio 9.306 at the class means:
thickness +4.059 MN (54.5 %),
material +0.569 MN (7.6 %),
interaction +2.822 MN (37.9 %).

### 3.8 What the strength-law extension does at ship scale

| quantity | porosity-only law (pre-extension) | extended law (this node) | change |
|---|---|---|---|
| σ_c^MY [MPa] | 2.190 | 3.113 | +42.1 % |
| class-mean ratio R_MY/R_FY at the class means | 7.285 | 9.306 | +27.7 % |
| quadrature R at p = 0.6 [MN] | 4.083 | 5.044 | +23.5 % |
| ΔR/R(0) at p = 0.6 (quadrature) | +312.3 % | +409.4 % | +97.1 pp |

So the age-dependent strength law raises the ship-scale 0 → 60 % answer by about
97 percentage points on the quadrature. The
thickness of the MY class remains the dominant contributor to the increment.

## 4. Figures

* `fig_mixture_rule.png` — DEM resistance vs the linear and series
  mixture rules, and the mixture-rule error with 95 % CI;
* `fig_age_strength.png` — age-dependent strengths (monotone,
  saturating) against the criterion (a) band, and the exponent sensitivity;
* `fig_ship_scale.png` — ship-scale resistance vs MY fraction and the
  increment decomposition;
* `fig_class_field.png` — a balanced class map with the hull transit
  marked, and the deviation from the linear rule against the length of the MY
  floe crossed by one transit (all layouts).

## 5. Notes, limitations, applicability

* **L1 — thickness heterogeneity.** The plane-section lattice has one thickness,
  so a thickness step-up (MY keel next to a thin FY sheet) cannot be resolved.
  The material mixture is measured directly; the thickness part is carried by the
  anchor quadrature as area weighting of *independent* floe resistances (declared
  assumption of `mixed_ice.field`) and reported as a decomposition.
* **L2 — transit length.** The 5 m transit is five thicknesses long. The tested
  field-realistic regime is therefore "one MY floe edge per transit"; longer
  transits would tighten the statistics but leave the intact-sheet regime (the
  broken-bond fraction reaches 0.7 after 18 m of advance).
* **L3 — interface convention.** Boundary bonds take the mean of the two end
  particles' multipliers. The measured sub-linearity in the intimately mixed
  regime is partly this seam compliance; a different convention would change the
  size, not the sign, of the effect.
* **L4 — crushing-dominated regime.** Crushing carries 92 % of the
  section resistance, so the mixture test mainly probes the crushing term; the
  bending term is nearly absent at h = 1 m, bow 30°.
* **L5 — single MY age.** The DEM uses one MY state (t = 4 a). The real MY
  population spans 2–5+ a; the anchor quadrature covers that window, the DEM does
  not.
* **L6 — the temperature exponent** m = 0.35 is a convention with a
  swept band, normalised away at the FY state; it shapes the MY-to-∞ trend only.
* **L7 — time step.** Before this node the stable step came from an ARPACK call
  with a random start vector, so identical runs differed by ~1.5 %; it is now
  deterministic (criterion (d) is bit-for-bit because of that fix).
* **L8 — absolute level.** Nothing here validates the absolute ship resistance.

## 6. Error assessment (explicit)

| quantity | reference | value here | error | < 3 % ? |
|---|---|---|---|---|
| p = 0 field run | existing homogeneous DEM model | 5.340057 vs 5.340057 MN/m | 0.0e+00 | **yes** |
| linear mixture, 11 of 12 configurations | direct heterogeneous DEM | bias −7.0 … −13.3 % (always low) | — | **no** as a 3 % test, but inside the ±15 % criterion band → met |
| linear mixture, worst configuration | direct heterogeneous DEM | bias 22.7 % ± 11.0 % (random layout, p = 0.88) | — | **no** (outside the ±15 % band → strict criterion not met) |
| σ_c^MY/σ_c^FY | literature band [1.5, 3.0] | 2.075 | inside band | n/a (band, not a point) |
| σ_c(t) monotone/saturation | own asymptote | residual 0.000 % of σ_c(∞) | 0.0 % | **no** as a 3 % test, but the criterion is < 2 % → pass |
| DEM class ratio | Lindqvist anchor ratio | 2.126 vs 1.619 | +31.3 % | **no** — declared; the two models answer at different scales |
| run-to-run reproducibility | same configuration | dt sensitivity ±3.5 % | — | **no** |

Honest statements. (i) The linear mixture rule stays inside ±15 % in
11 of 12 measured configurations, with an ensemble bias of
−7 % … −13 % that is **always low**; the single exception is the worst-case
arrangement (a thin FY seam inside an otherwise MY transit,
22.7 % ± 11.0 %). Criterion (b) is therefore
**FAIL as a universal ±15 % statement** and would only be
met if the claim were restricted to the ordinary arrangements — that restriction
is *not* made here, so the verdict is FAIL.
(ii) Because the bias is systematically low, the rule must not be used as a
10 %-accurate model; the ensemble regression slope is only
51 % of the ideal one.
(iii) The mixture test is material-only — thickness mixing is exact only under
the independent-floe assumption.
(iv) The ship-scale 0 → 60 % number is RELATIVE and inherits the anchor's
absolute-level limitation.

## 7. Reproduction

```bash
python scripts/run_mixed_ice_dem_field.py          # ≈ 4 min, 8 workers
python -m pytest tests/test_mixed_ice_dem_field.py -q
python -m pytest tests/test_mixed_ice_aging.py tests/test_mixed_ice_field.py \
    tests/test_mixed_ice_resistance.py tests/test_level_ice_resistance_dem.py -q
```

Artifact: `artifacts/mixed_ice_dem_field/campaign.json`; the digest
`d397f29f9f336eb02609a1e2bd492e1785a345a644666f03a93a32db03bd1617` is the SHA-256 of the measurement payload (every
field of that file except `artifact_sha256` and `doc_sha256`, serialised with
sorted keys), so it can be recomputed from the file itself; it includes the
wall-clock `runtime_s`, which is the only field that changes between two runs of
the same code. The ageing-law coefficient ledger is
`2a9af9291596cf6d6c91fb7bdebce42bcfb91930c1b3086d4bc1e7b4c997597d`.
