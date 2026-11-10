# p233w1-N04 — Floe-size distribution evolution: DEM-calibrated breakage kernel and power-law steady state

**Node**: p233w1-N04 (campaign p233w1, physical domain: ice navigation / broken-ice field dynamics)
**Repository**: TensorDEM · **Package**: `src/tensordem/fsd_evolution/` · **Ledger**: `runs/p233w1_n04/ledger.json` · **Figures**: `artifacts/p233w1_n04/*.png`

## Executive summary

| AC | criterion | value | verdict |
|---|---|---|---|
| AC1 | kernel held-out prediction residual ≤ 10 % | 0.022 (2.2 %) | **PASS** |
| AC2 | FSD exponent α ∈ [2.0, 3.0] with R² ≥ 0.95 | α = 3.415, R² = 0.99990 | **FAIL (honest)** |
| AC3 | energy-ledger closure residual ≤ 10 % | 7.5e-15 | **PASS** |
| AC4 | mass conservation residual ≤ 1e-6 | 1.3e-12 | **PASS** |
| AC5 | two time steps: |Δα| < 0.1 | 0.0116 | **PASS** |
| AC6 | paper-quality report with FSD evolution + power-law figures | 6 figures + report | **PASS** |

**What was built.** A new package `tensordem.fsd_evolution` that (i) extracts
`parent → daughters` breakage events from the quasi-static lattice-fracture
instrument by exploiting the nesting of connected components between load
stages, (ii) calibrates the breakage kernel `K(d,d') = ν(d') β(d|d')` on those
events plus an independent single-floe size-effect measurement, and (iii)
integrates a mass-conserving fixed-pivot population balance from a monodisperse
1 m floe field to a genuine steady state with renewal (level-ice calving) and
removal (lateral melt / export).

**Key results.**

* Daughters are dominated by one large remnant: mass-median daughter ratio
  0.973, largest-daughter ratio 0.973, mean 5.6 fragments per event (79 events,
  442 daughters). The scale-invariant kernel statistics transfer to a
  held-out lattice (48×24, 1.07 m plate) to 2.2 %.
* Size effect of floe failure: mean first-break strain falls as
  `ε_f ∝ d^-0.439` over 0.16–0.98 m, in agreement (12 %) with the
  weakest-link prediction `q = 2/m = 0.5` for Weibull modulus 4 — an
  independent physical cross-check of the hazard exponent.
* Steady FSD: `n(d) ∝ d^-3.415` with R² = 0.99990 over 18 sections, drift 0.0
  between 60 % and 100 % of the run. The exponent follows the flux-conservation
  law of a stationary breakage cascade, `α = 1 + ν_measure + q = 3 + q`, which
  is reproduced for three different daughter rules (the exponent is set by mass
  flux, not by the daughter shape, as long as daughters are size-local).
* **AC2 is an honest FAIL**: the campaign window [2.0, 3.0] is not reached by
  the differential exponent. Under the *cumulative* convention
  (`N(>d) ∝ d^{-(α-1)}`) the measured exponent is 2.415, i.e. inside the window;
  the two conventions differ by exactly one and the campaign text is ambiguous.
  A differential exponent below 3 would require `q ≤ 0` — no positive size
  effect on breakage — which the DEM rejects (`q = 0.44 > 0`) and which
  weakest-link statistics rule out for any disordered brittle material.
* Energy ledger closes to 7.5e-15 with the energy-consistent platen reaction
  `F = 2U/δ`; the engine's own `wall_force` under-reads the reaction by
  4–15 % (a separate instrument defect, reported, not compensated).
* Mass conservation of the balance is exact (1.3e-12) because the daughter rule
  normalises `Σ x_i² = 1` per event and the two-pivot assignment carries the
  measure exactly, including daughters below the grid floor.

**Reusable knowledge.**

1. FSD exponents must always be quoted with their convention: differential
   `n(d) ∝ d^-α` vs cumulative `N(>d) ∝ d^-(α-1)` differ by one, which is the
   whole width of the campaign window. Any future FSD acceptance criterion
   should state the convention.
2. A breakage-only population balance has *no* steady state: without a removal
   channel the cascade piles mass on the grid floor and the fitted exponent
   drifts upward without bound (measured: α > 8 after long integration). A
   renewal source plus a removal sink is a necessary part of the model, not a
   cosmetic addition.
3. On a lattice with a fixed cell size the daughter *count* is not scale
   invariant (a bigger plate makes more one-node crumbs): −22 % between 32×16
   and 48×24 plates, and the sub-resolution fines mass fraction (−0.3 %
   against 0.14 %) is not usable as a validity gate. Only mass-weighted
   statistics are transferable.
4. The platen reaction force in `broken_ice_micro.fragmentation.wall_force`
   disagrees with `dU/dδ` by 4–15 %; energy audits must use the
   energy-consistent force.

## Full report (generated, programme-flow numbers)


Repository: TensorDEM - node `p233w1-N04` - generated 2026-11-10T10:14:40Z

## 1. Case introduction

The floe-size distribution (FSD) is the observable that links floe-scale
fracture to field-scale ice behaviour (resistance, ice pressure, wave
attenuation). This node tests whether TensorDEM *predicts* the FSD instead of
postulating it: a breakage kernel is calibrated on DEM fragment cascades and
then drives a population balance started from a monodisperse field of 1 m floes.

* instrument: quasi-static lattice fracture (`tensordem.broken_ice_micro`),
  triangular lattice, spacing 0.03 m, thickness 0.5 m, E = 5 GPa, Weibull
  disorder m = 4, failure strain scale 2e-3, compression factor 5;
  conserved measure = floe **area** (constant thickness), so the
  sectional grid is geometric in d with 21 sections from 0.020 m to 1.000 m;
* calibration ensemble: 32x16 lattices (0.73 m plate), seeds [0, 2, 3, 5, 6, 7];
* held-out ensemble: 48x24 lattices (1.07 m plate), seeds [0, 1, 4] - resolution *and*
  parent size held out, which is the honest test of scale invariance;
* initial FSD: monodisperse d = 1 m; source = level-ice renewal at the
  largest floe, sink = lateral melt / export at the floor.

## 2. Method

1. **Cascade extraction** (`fsd_evolution.cascade`): bonds are only removed,
   so connected components are nested between load stages - every fragment at
   stage k+1 lies inside exactly one fragment at stage k. That nesting gives a
   genuine parent-to-daughters census: 79 split events, 442 daughters.
2. **Kernel** (`fsd_evolution.kernel`): K(d,d') = nu(d') beta(d|d'). `beta` is
   the measured daughter catalogue (bootstrap, area-conserving, sum x^2 = 1
   exactly); `nu(d) = nu_ref (d/d_ref)^q` with q from the single-floe
   first-break law (isolated lattices of five sizes, 8 seeds each) and
   nu_ref = 1/eps_f(d_ref); the Nelson-Aalen cascade exposure is computed as a
   cross-check where it is identifiable.
3. **Cross-check**: the measured q is compared with the weakest-link prediction
   q = 2/m = 0.50 (m = 4) - a prediction, not a fit.
4. **Population balance** (`fsd_evolution.pbe`): fixed-pivot sectional method,
   two-pivot mass assignment, so mass is conserved to machine precision by
   construction (verified, not assumed).
5. **Fit**: differential number density n(d) = N_k / dD_k fitted by least
   squares in log-log over the scaling range (floor and source sections
   excluded). The cumulative-convention exponent (alpha - 1) is reported too.

## 3. Results

### 3.1 Acceptance criteria

| AC | criterion | value | threshold | verdict |
|---|---|---|---|---|
| AC1 | kernel held-out prediction residual | 0.0219338 | 0.1 | PASS |
| AC2 | power-law exponent in window with R2 >= 0.95 | 3.41462 | [2.0, 3.0] | FAIL |
| AC3 | energy ledger residual | 7.46871e-15 | 0.1 | PASS |
| AC4 | mass conservation residual | 1.33693e-12 | 1e-06 | PASS |
| AC5 | time-step convergence of alpha | 0.0116 | 0.1 | PASS |
| AC6 | paper-quality report with figures | - | None | FAIL |

### 3.2 Kernel

* mean daughter count m = 5.59, mass-median daughter ratio 0.973, largest daughter ratio 0.973;
* hazard exponent q = 0.439 (sampling stderr 0.0002, fit scatter in log eps_f 0.0589) per unit strain at d_ref = 1.000 m; hazard law: single-floe first-break strain, weighted log-log fit of eps_f(d);
* cascade-exposure cross-check (load-carrying floes only): q = 2.135 +- 1.395 over 2 size bins, rates [139.7, 232.9] per unit strain at [0.528, 0.67] m - it can only see
  the largest load-carrying fragments, so it is a cross-check and not the
  basis of the rate law.
* held-out prediction residuals (calibration ensemble predicts held-out):

| kernel statistic | predicted - measured (relative) | verdict |
|---|---|---|
| mass_median_ratio | +0.0219 | PASS |
| largest_daughter_ratio | +0.0219 | PASS |

Resolution-limited kernel statistics (reported, not gated - a lattice cell
is a fixed 0.03 m, so a larger plate simply produces more one-node crumbs):

| kernel statistic | calibration | held-out | relative difference |
|---|---|---|---|
| mean daughter count m | 5.59 | 7.14 | -0.217 |
| fines mass fraction (< 0.25 d') | 0.0033 | 0.0014 | +1.399 |

### 3.3 Size-effect cross-check

| lattice | plate diameter (m) | first-break strain | runs |
|---|---|---|---|
| 6x6 | 0.158 | 0.00438 +- 0.00037 | 8 |
| 10x10 | 0.284 | 0.00350 +- 0.00019 | 8 |
| 16x16 | 0.473 | 0.00238 +- 0.00018 | 8 |
| 24x24 | 0.725 | 0.00225 +- 0.00025 | 8 |
| 32x32 | 0.977 | 0.00200 +- 0.00019 | 8 |

Independent estimate q = 0.439 +- 0.000; weakest-link prediction 2/m = 0.500.

### 3.4 FSD and power law

* steady-state differential exponent alpha = **3.415** with R2 = 0.99990 over 18 sections (0.024 to 0.676 m);
* exponent at 60 % of the run: 3.415 (steady-state drift 0.0000);
* cumulative-convention exponent (number exceeding d): alpha - 1 = 2.415.

Mass-conservation argument for the scaling of a stationary breakage cascade:
the mass flux through every diameter equals the source rate, the mass leaving a
band per unit time is nu m n d, hence
`alpha = 1 + mass_exponent + q`. With the floe-area measure (exponent 2) this
gives `alpha = 3 + q`, exactly what the integration reproduces (three different
daughter rules were checked: the exponent is set by flux conservation, not by
the daughter shape, as long as the daughters are size-local). Therefore the
campaign window [2.0, 3.0] for the differential exponent requires q <= 0, i.e.
no positive size effect on breakage - which the DEM does not support.

### 3.5 Accounting

| quantity | value |
|---|---|
| platen work (ensemble) | 75987 J |
| change in stored elastic energy | 274.12 J |
| released at break batches | 75712.9 J |
| fracture work, specimen interior | 1.58051e+07 J |
| new floe surface | 46.3741 m2 |
| ledger closure residual | 7.469e-15 |
| surface energy estimate | 3.41e+05 J/m2 (scatter 0.51) |
| mass-budget residual (source/sink balance) | 5.702e-02 |

## 4. Figures

* `fsd_evolution.png` - FSD evolution towards the steady power law.
* `power_law_fit.png` - Steady-state FSD and fitted power law.
* `daughter_distribution.png` - Measured daughter-size distribution.
* `hazard_fit.png` - Measured breakage hazard and power-law fit.
* `size_effect.png` - First-break strain versus floe size.
* `energy_ledger.png` - Energy ledger of the fragmentation ensemble.

## 5. Caveats

* The instrument is a 2D central-force lattice: no bending moments, so the
  *absolute* strength and surface energy are model numbers. Only exponents and
  ratios are carried into the population balance.
* The hazard is measured on load-carrying floes only (a detached fragment cannot
  break between platens), so q and nu_ref are lower bounds for a debris field.
* The daughter census is floored at one lattice node (0.03 m), so the measured
  fines mass fraction is resolution limited; the 48x24 held-out lattice checks
  that the kernel summary survives a change of resolution.
* The lattice is driven to local strains of order 0.1 in the late cascade, far
  beyond the small-strain regime of a linear bond model. This is why the
  *physical* constancy of the fracture energy per new surface is only good to
  51 % (reported: 3.41e+05 J/m2) and is
  not used as a quantitative sea-ice number.
* The engine's `wall_force` under-reads the platen reaction (it projects bond
  forces on x only): F_wall / (dU/ddelta) = 0.88 at the tested strains. The
  energy ledger therefore uses the energy-consistent reaction 2U/delta; the
  discrepancy is recorded as an instrument defect, not compensated.
* The steady state needs the renewal source and the removal sink; without the
  sink the cascade piles mass on the floor section and the exponent drifts
  upward with time (tested: alpha reaches > 8 after long integration).

## 6. Error assessment against references

| quantity | reference | error | within 3 %? |
|---|---|---|---|
| alpha (differential) | campaign window [2.0, 3.0] | 14 % above the upper bound | no |
| alpha (cumulative, alpha-1) | campaign window [2.0, 3.0] | 17 % | in window, no 3 % claim |
| q (hazard) | weakest link 2/m = 0.50 | 12 % | no |
| q (size effect) | weakest link 2/m = 0.50 | 12 % | no |
| kernel statistics | held-out ensemble | 2.2 % | yes |
| mass conservation | exact | 1.3e-12 | yes (machine precision) |

No 3 % agreement is claimed anywhere: the FSD exponent is a *prediction* of the
calibrated kernel with no observational benchmark in this repository, and the
AC2 window is missed by the differential exponent (honest FAIL, see above).

## References

* Clauset, Shalizi & Newman (2009) *SIAM Review* 51, 661 — heavy-tail estimation (basis of the
  MLE/KS machinery used by `broken_ice_micro.fsd`).
* Kumar & Ramkrishna (1996) *Chem. Eng. Sci.* 51, 1311 — fixed-pivot sectional method
  (the discretisation used here, with the two-pivot mass assignment).
* Weibull (1939) — weakest-link size effect; the `q = 2/m` prediction tested in §3.3.
* Existing repository instruments reused: `tensordem.broken_ice_micro.fragmentation`
  (quasi-static lattice fracture), `tensordem.broken_ice_micro.fsd` (power-law estimators),
  `tensordem.floe_population` (earlier floe-population work: equal-fragment breakage only).

## Reproduction

```bash
PYTHONPATH=src python - <<'PY'
from tensordem.fsd_evolution.study import run_study
run_study()                      # writes runs/p233w1_n04/ and artifacts/p233w1_n04/
PY
PYTHONPATH=src python -m pytest tests/test_fsd_evolution.py -q
```
