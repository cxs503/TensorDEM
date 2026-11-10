# What the 4x4x3 opposed-platen specimen actually measured (re-audit of m6464)

**Status: the m6464 specimen campaign is re-attributed.**  Its honest FAIL was
not caused by inertia.  It was caused by two measurable bookkeeping errors: the
rigid-plate reference bound counted **16** of the
**100** bonds that cross the gauge plane (a 6.25x undercount),
and the dependent variable (the maximum axial stress of the run) is a
**post-failure** debris-compaction peak in **12/12**
runs.  With the pre-failure gate applied, the purely elastic negative control
becomes rate independent (exponent **-0.035**,
10.3 % spread) instead of the spurious
**+0.117** (38 % spread) the campaign
reported.  The specimen-level ductile exponent, however, is still **not**
`1/(3)`: the gated values are **+0.438** (+31.4 %) and
**+0.461** (+38.4 %), i.e. an honest FAIL that is now
attributed to the specimen rather than to the solver.

## 1. Why this re-audit exists

`docs/ice_bond_specimen_rate_dependence.md` (m6464) reported, for a 4x4x3
opposed-platen specimen:

| claim as recorded | value |
|---|---|
| fitted strength exponent, linear (negative control) | +0.117 |
| fitted strength exponent, linear + creep | +0.244 |
| fitted strength exponent, Maxwell-Wiechert + creep | +0.198 |
| analytic ductile envelope `1/(m+1)`, `m = 2` | 0.333 |
| measured strength vs "rigid-plate bond bound" | 7-9x |
| attribution of the discrepancy | "inertia-dominated" |

Three of those could not survive a check, and the report contradicts itself:
its own limitation 4 says the fastest rate is "3 % of the lattice wave speed, so
inertia is present but small".  Nothing in the campaign measured *where* the
peak occurs, *how many* bonds cross the gauge section, or whether the reported
stress is carried by bonds at all.  This document measures all three, using the
same `TriaxialRunner` and the same constants, at the same cost as the original
campaign (`scripts/verify_ice_bond_specimen_attribution.py`, one thread, ~4 min).

## 2. How it was done

Three independent parts, each a pure measurement:

**Part A — load-path integrity and the exact bound.**  Drive one specimen step by
step and compare the platen reaction with the axial force carried by the bonds
that cross the gauge plane (`z = 1.5 R`, between layers 1 and 2).  A bonded
specimen has no other load path while the bonds are intact, so the ratio must be
1.  The crossing bonds and their axial projections also give the exact
rigid-plate bound through `tensordem.specimen_attribution.bond_path_capacity`: a
bond of rest length `L` and axial projection `dz` carries `k_u eps dz` axially at
its own cleavage strain, so the bound is `sum(k_u eps_b |dz|) / A_gross`.

**Part B — peak attribution.**  Re-run the 4 arms x 3 rates of the original
campaign with the recorded `broken_bonds` counter attached to every stress
sample, and split the history into a *pre-failure service stress* (maximum stress
while no bond has broken) and the *jamming peak* (maximum over the whole run).

**Part C — the two inertia numbers.**  Report the local single-platen-speed ratio
that the original report quoted **and** the specimen-scale wave-transit ratio
`Dyn = (H/c)/(eps_b/edot)`, which is the criterion that decides whether the test
is quasi-static.

Geometry and material (identical to m6464): `nx x ny x nz = 4x4x3`,
`R = 0.025 m`, `k_u = 2000 N/m`,
`eps_b = 0.015`, `rho = 917 kg/m^3`,
gross gauge area `A = 0.0400 m^2`, `dt` from the campaign's own hint
`3.4e-05 s`.

## 3. Results

### 3.1 The gauge section has 100 bonds, not 16

| quantity | this re-audit | as recorded by m6464 |
|---|---|---|
| bonds in the specimen | 326 (nearest, face and body diagonals) | — |
| bonds crossing the gauge plane | **100** | 16 (`nx*ny`) |
| axial projection of every crossing bond | 0.05 m | `L0` assumed |
| rigid-plate bond bound | **3750 Pa** | 600 Pa |
| undercount factor | **6.25x** | 1 (by definition) |
| platen reaction / bond axial force while intact | mean 1.0006, range [0.930, 1.113] (n=55, take-up transient discarded) | not measured |

The specimen's bond list is not the simple-cubic nearest-neighbour set: at
`R = 0.025 m` the pairing cutoff also bonds face and body diagonals
(rest lengths `0.05`, `0.0707`, `0.0866` m), so **100** bonds cross the
gauge plane and all of them have the same axial projection `2R`.  The
load-path integrity ratio is 1 within 0.06 %, so the reported
stress *is* the bond force: there is no hidden contact/platen load path
(`dem3d.py` masks pair contacts on intact bonds).  Consequently the
"7-9x the rigid-plate bound" of m6464 is a **bound error**: against the correct
bound the same peaks are
0.81-1.60x.

### 3.2 Every recorded peak is a post-failure peak

| arm | peak sigma (Pa) at edot = 0.25 / 1 / 4 | bonds broken at the peak | peak / correct bound | pre-failure sigma (Pa) | pre-failure / correct bound | m6464 exponent | gated exponent | gated spread |
|---|---|---|---|---|---|---|---|---|
| linear | 3932 / 4966 / 5445 | 200 / 326 / 326 | 1.05 / 1.32 / 1.45 | 3024 / 2828 / 2743 | 0.81 / 0.75 / 0.73 | +0.117 | -0.035 | 10.3 % |
| linear+creep | 3043 / 4338 / 5986 | 326 / 326 / 326 | 0.81 / 1.16 / 1.60 | 726 / 1165 / 2445 | 0.19 / 0.31 / 0.65 | +0.244 | +0.438 | 236.9 % |
| maxwell_wiechert | 3660 / 5134 / 5588 | 290 / 326 / 326 | 0.98 / 1.37 / 1.49 | 1367 / 1839 / 2418 | 0.36 / 0.49 / 0.64 | +0.153 | +0.206 | 76.9 % |
| maxwell_wiechert+creep | 3342 / 4809 / 5792 | 326 / 326 / 326 | 0.89 / 1.28 / 1.54 | 653 / 1077 / 2346 | 0.17 / 0.29 / 0.63 | +0.198 | +0.461 | 259.2 % |

`broken_bonds_at_peak` is the whole network (326 bonds) in most runs: the
"strength" of m6464 is the stress at which a column of debris is being compacted
by the platens.  That is why the same electron-elastic negative control, which
cannot be rate dependent by construction, produced a +0.117 slope: post-failure
compaction is rate sensitive through the contact damping, and the bond law is not
in that number at all.  Gated at the first bond break the control is flat
(-0.035, 10.3 % over two
decades of rate), which is the check the original campaign was missing.

### 3.3 The inertia attribution was based on the wrong number

| edot (1/s) | bond wave speed c (m/s) | bond period 1/omega (ms) | single-platen speed / c | t_load / (1/omega) | Dyn = t_transit / t_load | quasi-static? |
|---|---|---|---|---|---|---|
| 0.25 | 9.127 | 5.48 | 0.21 % | 10.95 | 0.27 | yes |
| 1 | 9.127 | 5.48 | 0.82 % | 2.74 | 1.10 | **no** |
| 4 | 9.127 | 5.48 | 3.29 % | 0.68 | 4.38 | **no** |

`m = rho 4/3 pi R^3 = 0.0600 kg`, `c = L0 sqrt(k_u/m)`, specimen height
`H = 0.150 m`.  The report's "3 % of the lattice wave speed" is the
*local* ratio at 4 1/s and it is correct — but it is not the
quasi-staticity criterion.  The specimen needs 4.4 wave transits at
4 1/s (dynamic) and 0.27 at 0.25 1/s (quasi-static).  So
"inertia-dominated at 0.25-4 1/s" is wrong at the slow end: the slow runs are
quasi-static, and their post-failure peaks are still above the bound.  Inertia is
real only at the fastest rate and is *not* what corrupted the exponent.

## 4. Error judgement against references

| quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| bond count crossing the gauge plane | engine bond list (exact) | 100 | exact by construction | yes (identity) |
| rigid-plate bound | `sum(k_u eps_b |dz|)/A` | 3750.0 Pa | analytic | yes (identity) |
| platen reaction vs bond load path | ratio = 1 | 1.0006 | 0.06 % (max single-sample dev 11.3 % on one sample) | yes |
| negative control rate independence | exponent 0 | -0.035 (10.3 % spread) | 3.5 % of the spread explained | borderline: slope is |0.035| < 0.1 x the signal slope, spread 10 % over two decades |
| specimen ductile exponent, linear+creep | 1/3 = 0.3333 | 0.4381 | **31.4 %** | **no — honest FAIL** |
| specimen ductile exponent, MW+creep | 1/3 = 0.3333 | 0.4612 | **38.4 %** | **no — honest FAIL** |
| single-bond constitutive exponent (m6543) | 1/3 | 0.3336 | 0.075 % | yes (unchanged; the constitutive law is fine) |

So the constitutive bond law is verified at the single-bond level, the
specimen-level envelope is **not** verified, and the reason is now localised:
the gated specimen strength is an extreme-value statistic of the bond network
(first bond to reach `eps_b`), so its rate dependence mixes the creep law with
the *spread* of bond strains in the loaded lattice rather than sampling the
constitutive envelope.  Making it sample the envelope needs a load path in which
every load-bearing bond sees the same strain history (a single chain, or a
servo-controlled specimen) — not a faster solver.

## 5. What changed in the record

* `src/tensordem/specimen_attribution.py` (new): the exact bond-path bound, both
  inertia numbers, the pre-failure/jamming split and the load-path integrity
  ratio, as pure functions.
* `scripts/verify_ice_bond_specimen_attribution.py` (new): the campaign re-audit.
* `tests/test_specimen_attribution.py` (new, 9 cases): the pure functions against
  hand values, the exact 100-bond gauge section, the integrity ratio in
  the quasi-static window, and a regression test that the m6464 dependent
  variable is post-failure.
* `docs/ice_bond_specimen_rate_dependence.md`: a correction notice pointing here
  (the original numbers and the honest-FAIL status are kept, unedited).
* `docs/dem_calibration_gap_ledger.md`: write-back with the corrected
  attribution.

## 6. Limitations of this re-audit

1. The pre-failure gate ("no bond broken yet") is conservative and
   resolution-limited: the first break is located to within one recorded row, so
   the gated stress carries a discretisation uncertainty of about the spacing of
   the axial rows.
2. The gated stress is a *first-bond-failure* stress of a homogeneous-strength
   network, not a continuum strength, and the specimen is the smallest one that
   has an interior bond.  Weibull heterogeneity is available in the engine
   (`DEM3DConfig.bond_strength_weibull_modulus`) but was **not** enabled here, so
   the size effect of the specimen remains open.
3. Load-path integrity was probed at 0.25 1/s only, where the test is
   quasi-static; at 4 1/s the instantaneous ratio swings with the
   inertia and only its average is meaningful (measured range at that rate:
   [0.84, 5.84] on three samples, reported in the JSON).
4. Only the tensile cleavage criterion is exercised; shear failure, attrition of
   fragments and the post-failure compaction law are measured but not modelled.
5. The bond spectrum (`0.35 / 0.45 / 0.20`, `tau = 5 ms, 50 ms`) is still a
   modelling choice, so laboratory calibration remains **WITHHELD**.

## 7. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_specimen_attribution.py -q
PYTHONPATH=src:. python scripts/verify_ice_bond_specimen_attribution.py
PYTHONPATH=src:. python scripts/verify_ice_bond_specimen_attribution.py --quick
```

Figures (generated by the script, not by hand):
`artifacts/ice_bond_specimen_attribution/load_path_integrity.png`,
`stress_strain_bounds.png`, `broken_bonds_at_peak.png`, `peak_vs_prefailure.png`.
Numbers: `artifacts/ice_bond_specimen_attribution/verification.json`.
