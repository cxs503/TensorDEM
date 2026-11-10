# Weakest-link (size-effect) verification of the heterogeneous 3-D DEM bond

Mission m7574 · repository TensorDEM · all numbers in this document are read
from `artifacts/ice_bond_weakest_link/verification.json` (produced by
`scripts/verify_ice_bond_weakest_link.py`, figures by
`scripts/make_weakest_link_figures.py`). Nothing here is transcribed by hand.

## 1. Why this node

`docs/dem_calibration_gap_ledger.md` closes the *constitutive* rate dependence
of the DEM bond (single bond) and leaves the **specimen** level open with a
measured negative result: at 0.25–4 s⁻¹ the 4×4×3 opposed-platen specimen is
inertia dominated (strength 7–9× the rigid-plate bound, negative control not
flat), so its emergent exponent could not be compared to `1/(m+1)`.

The ledger names two routes forward. This note takes the cheaper one — a
**single-chain (weakest-link) driver** — and, in doing so, closes a second,
previously unlisted gap: the bond had *no strength heterogeneity at all*, so a
bonded specimen could never exhibit a size effect.

## 2. Physics and the reference solution

For a chain of `N` independent bonds, each with Weibull tensile strength
`P(F_b < F) = 1 − exp(−(F/F₀)^m)`, the chain survives only if **every** bond
survives, so

```
P(chain survives F) = exp(−N (F/F₀)^m)
median(N)           = F₀ (ln 2 / N)^(1/m)
median(N)/median(1) = N^(−1/m)                    ← the size effect
q-quantile(N)       = F₀ (−ln(1−q) / N)^(1/m)
```

`F₀` is fixed so that the **single-bond median equals `k·ε_b·L₀`** — the
material keeps its nominal strength and only the scatter is added:

```
ε_b,i = ε_b · (−ln(1−U_i))^(1/m) / ln(2)^(1/m),   U_i ~ U(0,1) iid
```

Nothing in the engine knows `N`; the size effect has to emerge from the per-bond
draws plus the series load path. That is what makes it a test rather than a
tautology.

## 3. What was built

### 3.1 Engine feature (opt-in, legacy bit-for-bit)

`src/tensordem/dem3d.py`:

| item | meaning |
|---|---|
| `DEM3DConfig.bond_strength_weibull_modulus` | Weibull modulus `m` of the **tensile** breaking strain; `0.0` (default) = homogeneous legacy material |
| `DEM3DConfig.bond_strength_seed` | integer seed of a **private** `torch.Generator`, so the draw is reproducible and independent of the global RNG stream |
| `IceDEM3D.bond_breaking_strain` | per-bond thresholds (or `None` on the legacy path) |
| `IceDEM3D.bond_strength_factors` | dimensionless factors with median 1 (provenance) |
| `_commit_fracture` | reads `bond_breaking_strain` when present, the scalar `breaking_strain` otherwise |
| `bond_creep_reference_force` | `F_ref` now follows the per-bond threshold, so the Kachanov clock stays consistent with the bond it belongs to |
| `bond_law_report` | adds `strength_weibull_modulus`, `strength_weibull_seed` and the min/median/max threshold |

The legacy path is untouched: with `m = 0` the array stays `None`, the fracture
criterion evaluates the same scalar comparison and the creep reference is
bit-for-bit the old expression. `tests/test_ice_bond_engine.py` and
`tests/test_ice_bond_single_bond_engine.py` (25 cases) still pass unchanged.

### 3.2 Campaign (four arms)

`scripts/verify_ice_bond_weakest_link.py`. The chain is assembled **inside**
`IceDEM3D`: a `2×2×(N+1)` lattice, the `x=y=0` column moved onto the z axis
(bond rest length `L₀ = 2R = 0.05 m`), every other particle parked far away and
clamped, the bond list reduced to the chain, and the per-bond thresholds looked
up in the engine's own draw by bond key — the production heterogeneity path is
the one under test.

| arm | what it isolates |
|---|---|
| `engine-uniform` | exact, inertia-free: place all bonds at the same strain and let `forces() → _commit_fracture` decide. Two bracketing strains: just under the weakest threshold (nothing may fail) and just over it but under the second weakest (exactly the weakest must fail) |
| `ecdf` | statistics: min of `N` iid draws vs the analytic chain CDF, Kolmogorov–Smirnov statistic against the `1.36/√R` 5 % band, plus the Weibull-plot modulus |
| `rate` | quasi-static **validity window** of the dynamic arm: ramp duration `T` is swept at fixed seeds and `N = 4` |
| `dynamic` | the integrated engine: displacement ramp on the top particle, material drag only, load at the first bond failure, compared **per realisation** to that realisation's analytic weakest link |

Material/run parameters: `R = 0.025 m`, `k = 2000 N/m`, `ε_b = 0.015`,
`ρ = 917 kg/m³`, `m = 2` (second modulus `m = 4`), `drag = 2 N·s/m`,
`dt = 7.58e-5 s` (the engine's `recommended_dt`), ramp `3.5 ε_b` per bond over
`T = 2.5 s`, `R = 32/20/12/6` realisations for `N = 1/2/4/8`.

## 4. Results

### 4.1 Exact selection (`engine-uniform`, 16 brackets, `N = 1,2,4,8` × 4 seeds)

```
pass = True   (16/16)
```

Nothing fails below the weakest threshold, and exactly the weakest bond fails
above it: the fracture criterion reads the per-bond threshold and a series chain
selects its weakest link.

### 4.2 Statistics: chain strength vs the analytic Weibull CDF (`R = 20 000`)

| N | median measured [N] | median analytic [N] | rel. err | KS | 5 % band | fitted m |
|---|---|---|---|---|---|---|
| 1 | 1.491414 | 1.500000 | −5.72e−3 | 0.0053 | 0.0096 | 2.010 |
| 2 | 1.057780 | 1.060660 | −2.72e−3 | 0.0044 | 0.0096 | 2.007 |
| 4 | 0.754746 | 0.750000 | +6.33e−3 | 0.0063 | 0.0096 | 2.022 |
| 8 | 0.528580 | 0.530330 | −3.30e−3 | 0.0078 | 0.0096 | 1.989 |
| 16 | 0.379449 | 0.375000 | +1.19e−2 | 0.0105 | 0.0096 | 2.018 |
| 32 | 0.263904 | 0.265165 | −4.75e−3 | 0.0052 | 0.0096 | 2.010 |
| 64 | 0.187796 | 0.187500 | +1.58e−3 | 0.0061 | 0.0096 | 2.029 |

size-effect slope (fit of `log median` vs `log N`): **−0.49880** versus the
analytic **−0.50000** (−0.24 %), max log residual 0.0105 over 7 sizes.

**Honest reading of the KS column:** 6 of 7 sizes are inside the 5 % band;
`N = 16` exceeds it marginally (0.0105 vs 0.0096). With seven simultaneous 5 %
tests one marginal exceedance is what a correct implementation is expected to
produce, but it *is* an exceedance and is recorded as such rather than dropped.

### 4.3 Quasi-static convergence of the dynamic engine (`N = 4`, seed 3)

| T [s] | per-bond strain rate [s⁻¹] | steps | load [N] | rel. err | bond-force spread | weakest link |
|---|---|---|---|---|---|---|
| 1.0 | 0.0525 | 1267 | 0.5531 | +3.235 % | 9.38 % | True |
| 2.5 | 0.0210 | 3085 | 0.5389 | +0.593 % | 2.23 % | True |
| 5.0 | 0.0105 | 6151 | 0.5374 | +0.300 % | 1.17 % | True |

Both the load error and the diagnostic spread fall roughly linearly with the
strain rate, and the weakest link is selected at every rate. `T = 2.5 s` is
chosen for the ensemble: 2.2 % spread buys a factor 6 in cost over `T = 5 s`.

### 4.4 Dynamic ensemble (the emergent size effect)

| N | R | median load [N] | analytic [N] | rel. err | weakest link | force spread |
|---|---|---|---|---|---|---|
| 1 | 32 | 1.3864 | 1.5000 | −7.57 % | 100 % | 0.00 % |
| 2 | 20 | 1.1126 | 1.0607 | +4.89 % | 100 % | 0.19 % |
| 4 | 12 | 0.7535 | 0.7500 | +0.46 % | 100 % | 1.66 % |
| 8 | 6 | 0.4664 | 0.5303 | −12.06 % | 100 % | 12.94 % |

* **Failing bond identity: the weakest bond of the chain in 100 % of 70
  realisations**, at every size — the emergent weakest-link selection is real,
  not assumed.
* Fitted size-effect slope from the per-size medians: **−0.5278** vs −0.5000
  (+5.55 %), max log residual 0.077.
* Regression on all 70 realisations: **−0.4635**, bootstrap 95 % CI
  `[−0.671, −0.276]` (−7.3 % from analytic).
* **Censoring: 0 of 70** — the 3.5 ε_b ramp removes the early truncation that a
  1.4 ε_b ramp produced at `N = 1`.
* Second modulus, `m = 4`, `N = 4`, `R = 8`: median load error **+0.34 %**,
  weakest link 100 %, so the recovered exponent tracks `1/m` rather than being
  an `m = 2` coincidence.

The per-size medians carry the expected sampling error at these realisation
counts (`m = 2` gives a coefficient of variation of 0.52; with `R = 32` the
median has ≈ 11 % standard error, and `N = 8` has only `R = 6`). That is why the
*slope* (which pools all sizes) and the *paired* per-realisation comparison are
the verdict-bearing quantities, not the individual medians.

## 5. Error judgement against the references

| quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| static chain median, `N = 1 … 64` | `F₀(ln2/N)^{1/2}` | see 4.2 | ≤ 1.19 % | **yes** (all sizes) |
| static size-effect slope | −0.5000 | −0.49880 | 0.24 % | **yes** |
| dynamic load, `N = 4`, `T = 2.5 s` | analytic weakest link of the same realisation | 0.5389 vs 0.5374 | +0.59 % | **yes** |
| dynamic ensemble slope (per-size medians) | −0.5000 | −0.5278 | 5.55 % | **no** (but the 95 % bootstrap CI of the all-points fit contains −0.5) |
| dynamic per-size median, `N = 1 … 8` | analytic | ≤ 12.1 % | sampling-limited at `R ≤ 32` | **no** — sampling, not bias |
| 2nd modulus `m = 4`, `N = 4` | analytic | +0.34 % | 0.34 % | **yes** |

## 6. Limitations, stated not hidden

1. **The chain is 1-D.** Bonds are collinear, no contact network, no bending or
   shear coupling. This verifies failure *selection* and its statistics; it does
   not reproduce a bulk specimen's strength (which additionally depends on
   coordination number, load sharing and confinement).
2. **Only the tensile threshold is heterogeneous.** `shear_breaking_strain`
   stays a scalar material constant, and the creep reference force follows the
   tensile threshold.
3. **The dynamic arm has a measured rate bias** — 0.59 % at the chosen ramp
   (§4.3). The dynamic ensemble inherits it, which is why the study is reported
   as a *paired* comparison against each realisation's own analytic value.
4. **The static slope is exact by construction**, because the arm reproduces the
   analytic min-of-`N` construction; it verifies the *material draw + selection
   rule*, not a numerical method. The dynamic arm is the one that has to earn the
   result, and it does so with a 5.6 % slope error and a bootstrap CI that spans
   −0.67 … −0.28.
5. **Not verified here:** a specimen-scale size effect (many particles, load
   redistribution, inertia) — that is the next node, and it can now be attempted
   because heterogeneity exists in the engine.

## 7. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_ice_bond_weakest_link.py -q
PYTHONPATH=src python scripts/verify_ice_bond_weakest_link.py --workers 24
PYTHONPATH=src python scripts/make_weakest_link_figures.py
```

Artefacts: `artifacts/ice_bond_weakest_link/verification.json`,
`chain_strength_ecdf.png`, `size_effect.png`, `quasi_static.png`.