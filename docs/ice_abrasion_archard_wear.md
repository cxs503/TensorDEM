# Ice-hull waterline-band abrasion wear: DEM contact statistics x Archard law,
# wear depth field and coating life (W10-N06)

Module `src/tensordem/ice_abrasion.py` · tests `tests/test_ice_abrasion.py` ·
driver `scripts/benchmark_ice_abrasion.py` · artifacts `artifacts/ice_abrasion/`.

**Evidence level.**  The numerics and physics (Hertzian contact with a crushing
cap, Cundall-Strack tangential saturation, the Archard law, the Finnie angle
function, volume-conserving projection, Roache GCI, the energy ledger) are
**REAL** and are unit-tested against closed-form values.  The *absolute* wear
level is **CALIBRATED**: one parameter, the specific wear rate
`k = K/H`, is fitted to a single field anchor (Baltic 1A ice-belt coating loss).
Nothing in this document should be read as a validated wear predictor: the
literature anchor is a weak reference and the calibration is explicit.

## 1. Engineering problem

An ice-class hull loses coating and steel in the **waterline band (ice belt)**
because broken ice slides along the shell with local contact pressures of order
1-3 MPa while the ship advances.  Design consequences: coating service life and
recoating interval, steel wastage allowance for the ice belt, and the friction
work that has to be accounted for in the ice-resistance budget.  TensorDEM could
already resolve ice-hull *contact forces* (`hull_contact`, `hull_mesh_contact`,
`threefield.hull_ice`) and ice *resistance* (`threefield.resistance`), but had no
material-removal law, so an ice-class coating had no life model.  This increment
adds the missing loop:

```
DEM contact statistics  ->  Archard / Finnie wear law  ->  waterline-band depth
field  ->  coating life  ->  energy ledger consistent with the resistance module
```

## 2. Case definition

| quantity | value | audit |
|---|---|---|
| L x B x T [m] | 185.00 x 32.260 x 11.50 | prescribed |
| Cb target / mesh | 0.8200 / 0.82002 | rel. err 2.26e-05 |
| B target / mesh [m] | 32.260 / 32.2600 | rel. err 2.56e-07 |
| Cp / Cm | 0.8283 / 0.990 | m = 0.3692, n = 0.0101 |
| geometry gate (<1 %) | yes | Cb/L/B re-integrated |
| ice-belt band (one side) [m^2] | 407.00 | 9 x 740 cells at 0.25 m |
| band below / above waterline [m] | 1.20 / 1.00 | FSICR-like ice belt |

Voyage profile (Gulf of Bothnia round trip, one direction listed; the return legs
repeat the same conditions in ballast):

| leg | distance [nm] | speed [kn] | ice thickness [m] | concentration |
|---|---|---|---|---|
| light brash ice | 120 | 9.0 | 0.25 | 0.40 |
| medium level ice | 110 | 6.0 | 0.45 | 0.70 |
| heavy channel | 60 | 4.0 | 0.60 | 0.90 |

Seasonal model: 10 round voyages per year,
belt energy fraction 0.60 (share of the
ice friction work dissipated in the ice belt rather than along the deeper hull),
friction coefficient mu = 0.10 fed to the
resistance baselines, coating 350 um DFT epoxy (H = 200 MPa), bare AH36 steel
(H = 2.0 GPa).

## 3. Method

### 3.1 DEM contact statistics (the discretised `f(p_c, v_s)`)

Each fragment of size `l_f` (lognormal, median 0.35 m) carries the normal load
`F = p_nom * kappa_A * l_f^2` with the nominal ice pressure `p_nom` lognormal
(median 0.7 MPa) over the load-bearing fraction `kappa_A = 0.02` of its footprint.
The normal contact is Hertzian, `p_H = F/(pi a^2)` with
`a = (3 F R^* / 4 E^*)^(1/3)`, **capped by the local crushing capacity**
`p_crush` (lognormal, median 1.5 MPa); the crushed patch radius then follows from
`F = p_c pi a^2`.  Tangential motion is a Cundall-Strack spring
(`k_t = 8 G^* sqrt(R^* delta)`, Mindlin) saturated at `mu F_n`, integrated over the
contact duration `L/v_s`, and projected back onto the friction cone after every
step.  Only sliding increments count as wear-relevant slip, so the ensemble
returns `(p_c, v_s, L_slip, N, a)`.

| statistic | value |
|---|---|
| sampled contacts per leg | 8192 |
| contact pressure median [MPa] | 1.518 |
| contact pressure P95 [MPa] | 3.481 |
| crushing-limited contacts | 100.0 % |
| slip speed median [m/s] | 1.505 |
| sliding fraction (Cundall-Strack) | 1.000 |
| crushed patch radius median [mm] | 18.91 |
| normal load median [N] | 1685 |
| timesteps per contact | 96 |

The elastic branch is fully crushing-limited
(100.0 % of contacts): for a 0.35 m fragment
the Hertz pressure would be tens of MPa, which ice cannot sustain, so the contact
pressure is governed by the crushing-strength scatter.  This is a physical result
of the model, not a numerical artefact.

### 3.2 Wear law

Archard, written per contact as a pit depth,

```
d = K p_c L_slip / H        (equivalently d = k p_c L_slip, k = K/H)
```

with `k` in m^3/(N m) - the form ice-abrasion data are reported in.  The high
impact-angle correction is Finnie (1960),

```
V = C_F (m v_n^2 / sigma_f) f(alpha),  sigma_f = H/2.5,
f(alpha) = sin(2a) - 3 sin^2(a)  (a <= atan(1/3)),  cos^2(a)/3  (a > atan(1/3))
```

applied only where the ice approach is oblique (obliquity decays aft from the
stem) and with only the corner mass (`0.02`-type
small fraction, 0.3 %) and the normal approach speed (3 % of the slip speed)
participating.  The impact branch contributes
4.0 % of the removed volume, i.e. it is a
bow-localised correction, not the main mechanism.

### 3.3 Depth field

Every contact deposits its pit volume `d * pi a^2` at a position drawn from the
ice-belt activity distribution (peak at the stem and fore shoulder, vertical peak
just below the waterline) and the volume is split over the touched band cells by
footprint overlap.  Volume is conserved exactly, so the cell-averaged depth is
grid independent in the dilute-contact limit.

### 3.4 Energy ledger and the friction-module interface

The friction work of each leg is taken from the `threefield.resistance`
baselines as the mu-dependent part of the resistance,

```
W_f = (R(mu) - R(mu=0)) * distance
```

and the belt share is `belt_energy_fraction * concentration * W_f`.  The event
population is then scaled so that `sum(N L_slip) = W_f,belt / mu` exactly, which
makes the Archard branch satisfy `V H / W_f = K / mu`.  Two consequences are
worth stating:

* the circulation of mass is *not* fitted (only `k` is), so the belt-mean depth is
  a genuine prediction of the friction work and of the contact statistics;
* because both baselines are linear in mu, `W_f/mu` is mu independent and so is
  the predicted wear depth - the mu input re-partitions the energy ledger but not
  the wear magnitude (see the sensitivity table).

### 3.5 Life prediction

Life to local breach = coating thickness / annual depth rate at the chosen
statistic, with a Monte Carlo (20000 samples) over lognormal uncertainty in the
specific wear rate (sigma_ln = 0.5), the friction coefficient (0.2) and the ice
resistance (0.15).  Three statistics are reported: the worst 0.25 m cell, the P95
survey patch (the calibration statistic) and the belt mean.

### 3.6 Geometry gate

No wear number is produced unless the parametric hull reproduces its own
principal particulars from the mesh quadrature (Cb, L, B) within 1 %.

## 4. Results

### 4.1 Geometry audit

| quantity | value | audit |
|---|---|---|
| L x B x T [m] | 185.00 x 32.260 x 11.50 | prescribed |
| Cb target / mesh | 0.8200 / 0.82002 | rel. err 2.26e-05 |
| B target / mesh [m] | 32.260 / 32.2600 | rel. err 2.56e-07 |
| Cp / Cm | 0.8283 / 0.990 | m = 0.3692, n = 0.0101 |
| geometry gate (<1 %) | yes | Cb/L/B re-integrated |
| ice-belt band (one side) [m^2] | 407.00 | 9 x 740 cells at 0.25 m |
| band below / above waterline [m] | 1.20 / 1.00 | FSICR-like ice belt |

### 4.2 Wear depth field (calibrated season)

| material | per voyage mean / P95 / max [um] | per season mean / P95 / max [um] | volume per season, one side [m^3] |
|---|---|---|---|
| coating | 1.31 / 3.50 / 11.11 | 13.07 / 35.00 / 111.07 | 5.321e-03 |
| bare AH36 steel | 1.26 / 3.33 / 10.46 | 12.61 / 33.33 / 104.62 | 5.133e-03 |

`k` fitted to the anchor: **8.190e-15 m^3/(N m)**
(within the literature range 1e-15 ..
1e-13,
yes); the implied Archard coefficient
is `K = k H = 1.638e-06`, i.e.
`K/mu = 1.638e-05`, well inside
(0, 1) as required.

### 4.3 Coating and steel life

| material | thickness used | statistic | rate [mm/year] | life P10 / P50 / P90 [years] |
|---|---|---|---|---|
| coating | 350 um DFT | worst_survey_patch | 0.1111 | 1.55 / 3.17 / 6.42 |
| coating | 350 um DFT | p95_survey_patch | 0.0350 | 4.90 / 10.06 / 20.37 |
| coating | 350 um DFT | mean_field | 0.0131 | 13.13 / 26.93 / 54.54 |
| AH36 steel | 5.0 mm allowance | worst_survey_patch | 0.1046 | 23.44 / 48.08 / 97.38 |
| AH36 steel | 5.0 mm allowance | p95_survey_patch | 0.0333 | 73.58 / 150.90 / 305.65 |
| AH36 steel | 5.0 mm allowance | mean_field | 0.0126 | 194.42 / 398.76 / 807.70 |

The worst 0.25 m cell is a Poisson-dominated extreme-value statistic (see 4.5) and
is reported as a lower bound; the P95 survey patch is the design value used for
the recoating interval; the belt mean corresponds to a full-belt renewal.

### 4.4 Energy ledger

| ledger quantity | value |
|---|---|
| friction work per season, both sides [J] | 1.2472e+11 |
| friction work per season, one side [J] | 6.2359e+10 |
| friction work recovered from the event population [J] | 6.2359e+10 |
| friction-interface deviation [%] | 1.22e-14 |
| Archard wear volume, one side [m^3/season] | 5.1073e-03 |
| Finnie (impact) volume fraction | 4.0 % |
| wear energy V H [J/season, one side] | 1.0643e+06 |
| V H / W_f (measured) | 1.6380e-05 |
| K / mu (set) | 1.6380e-05 |
| energy-identity deviation [%] | 1.03e-13 |
| implied load-bearing contact duty factor | 1.183e-02 |
| Archard coefficient K = k H [-] | 1.638e-06 |

The recovered event friction work matches the resistance-module friction work to
1.2e-14 %, and the
energy ratio reproduces `K/mu` to
1.0e-13 %.  The implied
duty factor `1.183e-02` is the physical
interpretation of the calibrated magnitude: about
1.18 % of the fragment-passage
opportunities at a fixed hull point actually carry a load-bearing sliding contact
(the rest are water-lubricated or non-load-bearing).

### 4.5 Convergence

| study | levels | observed order | GCI(fine) [%] | criterion <5 % |
|---|---|---|---|---|
| DEM particle count | 4000 / 8000 / 16000 | 1.00 | 0.388 | yes |
| contact timestep | 48 / 96 / 192 | machine precision | 0.000 | yes |
| band grid, belt mean | 1 / 0.5 / 0.25 | 0.53 | 0.162 | yes |
| band grid, P95 depth | 1 / 0.5 / 0.25 | 1.00 | 19.780 | no |

The DEM sample-count levels are *nested* (prefixes of one ensemble) and the metric
is the mean pit depth that the season integration consumes; the timestep affects
the slip distance and is converged to machine precision already at 48 steps.  The
belt-mean depth is grid independent to
0.16 %, but the P95
statistic changes by
19.8 % between
1.0 m and 0.25 m cells: in the dilute-contact limit the extreme-value statistics of
a shot-noise field are resolution dependent even when the mean is not.  This is a
documented limitation, not a converged quantity.

### 4.6 Sensitivity

| parameter | value | P95 depth [um/season] | belt mean [um/season] |
|---|---|---|---|
| spec_rate | 1e-15 | 6.321 | 2.135 |
| spec_rate | 1e-14 | 43.003 | 15.925 |
| spec_rate | 1e-13 | 411.508 | 153.820 |
| spec_rate_archard_only | 8.19e-15 | 33.604 | 12.549 |
| mu | 0.05 | 35.571 | 13.152 |
| mu | 0.08 | 35.571 | 13.152 |
| mu | 0.12 | 35.571 | 13.152 |

The Archard branch is exactly linear in `k` (log-log slope 1); the flat `mu` rows
are a consequence of the linear resistance baselines discussed in 3.4.

### 4.7 Held-out validation

The specific wear rate fitted on the coating anchor is applied unchanged to a
harsher, independent profile (1A Super traffic, 20 round trips per season, 0.7-1.0 m
ice) on bare steel:

| quantity | model | literature band |
|---|---|---|
| belt-mean wastage | 0.092 mm/yr | 0.05 - 0.15 mm/yr |
| P95 survey patch | 0.246 mm/yr | 0.05 - 0.15 mm/yr |
| worst 0.25 m cell | 0.691 mm/yr | - |

The belt mean lands -7.7 % from the band
midpoint (inside the band), while the P95 patch exceeds the band
(+146 %).  The honest reading: the model
reproduces the *representative* wastage rate at which a class survey quotes ice-belt
steel loss, and over-predicts the single worst patch.

## 5. Acceptance criteria

| criterion | verdict | key number |
|---|---|---|
| (a) analytic Archard degradation | yes | max rel. deviation 0.0e+00 (direct), 5.4e-16 (pit chain), tolerance 1e-10 |
| (b) literature / survey benchmark (weak) | yes | held-out belt mean 0.092 mm/yr = -7.7 % of the 0.10 mm/yr midpoint; P95 patch 0.246 mm/yr = +146 % (outside band) |
| (c) energy consistency | yes | V H / W_f = 1.6380e-05 vs K/mu = 1.6380e-05, deviation 1.03e-13 % |
| (d) DEM / timestep convergence | yes | sample GCI 0.388 %, timestep GCI 0.000 % |
| (e) friction-module interface | yes | deviation 1.22e-14 % at equal mu |
| (f) pytest + full case | yes | 39 tests, status "39 passed" |

### 5.1 Error judgements (explicit)

| key quantity | benchmark | error | < 3 % ? |
|---|---|---|---|
| Archard volume `V = K W L / H` | closed form | 0.0e+00 (rel.) | yes |
| pit chain `d pi a^2` vs `K W L / H` | closed form | 5.4e-16 (rel.) | yes |
| `V H / W_f` vs `K/mu` | energy identity | 1.0e-13 % | yes |
| friction work vs resistance module | same mu input | 1.2e-14 % | yes |
| DEM particle-count GCI | r = 2 self-consistency | 0.388 % | yes |
| contact timestep GCI | r = 2 self-consistency | 0.000 % | yes |
| belt-mean depth vs grid | 1.0 m vs 0.25 m cells | 0.16 % | yes |
| P95 depth vs grid | 1.0 m vs 0.25 m cells | 19.8 % | **no** |
| coating P95 vs anchor A | 35 um/season midpoint | 2.0e-14 % (calibrated) | yes by construction |
| held-out steel belt mean vs anchor B | 0.10 mm/yr midpoint | -7.7 % | **no** (inside band) |
| held-out steel P95 vs anchor B | 0.10 mm/yr midpoint | +146 % | **no** |

The calibration criterion is met only because it is a one-parameter fit; the
genuine test is the held-out line and it does **not** meet a 3 % (or even 25 %)
tolerance on the patch statistic.  Both are stated rather than hidden.

## 6. Attention points and limitations

1. **One-parameter calibration.** `k` is fitted to a single literature anchor
   (Baltic 1A ice-belt coating loss, 20-50 um/season).  The anchor itself scatters
   by a factor of ~2-3, so the calibration inherits that.  The fitted value
   (8.19e-15 m^3/(N m)) is, however, inside the
   published ice-abrasion range, and it is below dry-sliding values, which is
   consistent with water-lubricated, intermittent contact.
2. **Belt energy fraction.** The split of friction work between the ice belt and
   the deeper hull (`0.6`) is an engineering assumption motivated by the waterline
   pressure peak; a range of 0.3-0.8 would move the predicted wear by a factor of
   ~2.  It is the second largest uncertainty after `k`.
3. **The P95 / max statistics are not converged.** Shot-noise extreme-value
   statistics are resolution dependent (19.8 % between
   1.0 m and 0.25 m cells).  The calibration uses the 0.25 m P95; using a different
   cell size shifts `k` accordingly.
4. **Friction-work source.** Lindqvist and Riska friction works differ by
   101 % on the
   light-ice leg (the design value is their mean), so the absolute wear level
   inherits that baseline spread.
5. **mu insensitivity.** With linear friction baselines the predicted depth does
   not depend on mu; a nonlinear friction model would break that and is the natural
   next increment.
6. **Model simplifications.** No coating delamination/chipping, no corrosion, no
   temperature or ice-salinity dependence of the wear coefficient, no ridge keel
   impact, no hull flare above the waterline, single ice-belt band on each side,
   fragment shape described by one size statistic, and the surface-length error of
   the band cells (deadrise neglected) is a few percent.
7. **Not a validated predictor.** The model is a physically structured wear loop
   with verified numerics; only a meaured thickness survey of a real ship could
   make it a predictive tool.

## 7. Reproduction

```bash
python -m pytest tests/test_ice_abrasion.py -q          # 39 tests
python scripts/benchmark_ice_abrasion.py --stage cases
python scripts/benchmark_ice_abrasion.py --stage convergence
python scripts/benchmark_ice_abrasion.py --stage sensitivity_rates
python scripts/benchmark_ice_abrasion.py --stage sensitivity_mu
python scripts/benchmark_ice_abrasion.py --stage report
python scripts/build_ice_abrasion_doc.py                # regenerate this document
```

## 8. Audit chain

| link | path |
|---|---|
| model | `src/tensordem/ice_abrasion.py` |
| reused engines | `src/tensordem/dem3d.py` (Hertz), `src/tensordem/hertz_mindlin.py` (Mindlin), `src/tensordem/contact_tangential.py` (Cundall-Strack), `src/tensordem/threefield/resistance.py` (friction work) |
| tests | `tests/test_ice_abrasion.py` (39 tests) |
| driver | `scripts/benchmark_ice_abrasion.py` |
| numeric record | `artifacts/ice_abrasion/report.json` |
| fields | `artifacts/ice_abrasion/wear_field_coating.csv`, `wear_field_steel.csv`, `profile.csv` |
| life / sensitivity | `artifacts/ice_abrasion/life_table.csv`, `sensitivity.csv`, `convergence.json` |
| figures | `artifacts/ice_abrasion/figs/*.png` |
| API contract | `ice_abrasion_calculation` -> `POST /api/v1/calculations/structure/ice_abrasion` |

**API deployment status.** The request/response contract is implemented and tested
inside this repository (`ice_abrasion_calculation`; unknown material or missing
principal particulars raise, which a FastAPI route maps to 422).  The route itself
must be registered by the ship-performance service, which lives in a different
repository; as of this run `POST :8002/api/v1/calculations/structure/ice_abrasion`
still answers 404, so the platform capability is *not yet exposed* - the contract
and its tests are ready for the wiring.

Figures: `figs/wear_field.png` (depth field, coating/steel), `figs/wear_profiles.png`
(longitudinal/vertical profiles vs contact activity), `figs/life_distribution.png`
(Monte Carlo life), `figs/convergence.png` (DEM and grid GCI),
`figs/sensitivity.png` (k and mu), `figs/energy_ledger.png` (criterion c).
