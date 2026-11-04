# Fragmentation / Weibull statistics: independent verification (m3475)

Protocol: `tensordem-fragmentation-statistics-verification-v1`
Scope: `src/tensordem/threefield/fragmentation.py` (+ `coupling.fragmentation_stage`,
which consumes it), as published in `docs/threefield/README.md` and
`docs/threefield/FORMULAS.md`.
Artifacts: `artifacts/threefield_fragmentation/` (`verification.json`,
`error_table.csv`, `ks_bias.csv`, `figures/fragmentation_statistics_verification.png`).
Reproduce: `python scripts/verify_fragmentation_statistics.py` and
`python -m pytest tests/test_threefield_fragmentation_statistics.py -q` (52 tests).

## 1. The debt: statistics graded against themselves

The three-field report (`docs/threefield/README.md`, limitation 3) grades the
DEM floe-size census against the experimental reference law
`Weibull(k = 1.8, lambda = 0.12)`. The two estimators that produce that gate
(`fit_weibull`, `fit_weibull_cdf`), the CDF itself (`weibull_cdf`) and the
goodness-of-fit statistic (`kolmogorov_smirnov`) had **no independent
reference**: no test, script or doc ever compared them with anything outside
`fragmentation.py`. Evidence grade in practice: `WITHHELD` — the quoted shape
and scale could have been produced by any estimator, right or wrong.

Two defects surfaced while building the reference comparison:

1. **Density leak in the cascade** (`fragment()`, hard-coded `rho_i = 917`,
   ignoring `BreakupProperties.ice_density_kg_m3`). The cascade turns a mass into
   a plate length `L = sqrt(m / (rho_i h))` and back; with `rho_i != 917` the
   returned masses were wrong by `917/rho_i - 1` **and** the coupled mass ledger
   of `fragmentation_stage` silently created ice: at `rho_i = 900 kg/m3`,
   `8.5 kg` appeared out of `450 kg` of inventory (**1.889 %**, against the
   module's own `1e-6` acceptance gate). Fixed by taking the density from the
   property set.
2. **Mis-named KS statistic**. `kolmogorov_smirnov()` returned
   `max |F_plot(x_i) - F(x_i)|` on Weibull plotting positions `(i+1)/(n+1)` —
   a CDF-plot deviation, not `sup_x |F_n(x) - F(x)|`. It was unused outside the
   module, so the name is now attached to the correct step-CDF statistic and the
   smooth figure quantity is kept separately as
   `kolmogorov_smirnov_plotting_position()`.

## 2. Method

Every assertion uses a reference that is *not* the module:

| anchor | independent reference |
|---|---|
| `bending_stress_pa` | free-ended beam closed form `sigma_b = 3 rho_w C_d u'^2 L^2 / (8 h)` |
| `critical_floe_length` | bisection root of `sigma_b(L) = sigma_f` on the closed form (no module code) |
| `weibull_cdf` | `scipy.stats.weibull_min.cdf` |
| `fit_weibull`, `fit_weibull_cdf` | `scipy.stats.weibull_min.fit` (maximum likelihood, `floc=0`) on samples drawn from the reference law |
| `kolmogorov_smirnov` | `scipy.stats.kstest` with the same reference CDF |
| cascade mass | external `sum(rho_i L^2 h)` over the returned floe list |

Negative controls: a calm race (`u' = 0`) must not split; the size-distribution
gate is re-run on **normalised** lengths `L / L_max` to test whether its failure
is a units artefact; and the three regression tests were re-run against the
pre-fix source (`git show HEAD:...`, density line reverted) to confirm they fail
there.

## 3. Results

Worst relative error per anchor (full table: `artifacts/threefield_fragmentation/error_table.csv`):

| anchor | rows | worst rel. error | verdict |
|---|---|---|---|
| beam stress vs closed form | 4 | 2.0e-16 | identity |
| `L_crit` vs bisection root | 4 | 2.1e-16 | identity |
| `weibull_cdf` vs SciPy | 15 | 5.6e-09 | identity |
| corrected KS vs `scipy.kstest` | 6 | 2.1e-15 | identity |
| cascade mass vs external sum (rho = 917/900/950) | 3 | 2.8e-16 | conserved |
| coupled ledger residual (rho = 917/900/950) | 3 | 0.0 | closed |
| Weibull **scale**, CDF-LS vs MLE | 3 | 2.8e-02 | < 3 % |
| Weibull **scale**, log-LS vs MLE | 3 | 1.8e-02 | < 3 % |
| Weibull **shape**, CDF-LS vs MLE | 3 | 8.7e-02 | > 3 % |
| Weibull **shape**, log-LS vs MLE | 3 | 1.3e-01 | > 3 % |

The five algebraic anchors agree to round-off, i.e. the code implements the
stated laws. The estimator rows are the informative ones.

### 3.1 Estimator accuracy against the maximum-likelihood reference

Samples drawn from `Weibull(1.8, 0.12)`; SciPy MLE is the reference.

| n | MLE (k, lambda) | log-LS (k, lambda) | k error | CDF-LS (k, lambda) | k error |
|---|---|---|---|---|---|
| 60 | 2.012, 0.1280 | 1.748, 0.1301 | -13.1 % | 1.837, 0.1314 | -8.7 % |
| 150 | 1.862, 0.1240 | 1.768, 0.1241 | -5.0 % | 1.725, 0.1254 | -7.4 % |
| 400 | 1.782, 0.1170 | 1.646, 0.1189 | -7.6 % | 1.738, 0.1190 | -2.4 % |

Scale is reproduced to 1.8–2.8 % in every case; the shape carries a one-sided
(negative) small-sample bias of 2–13 %, largest for the double-log transform.
So the fitted **scale** is a 3 %-grade quantity and the fitted **shape** is not:
it must be quoted with an uncertainty band, never as a converged value.

### 3.2 KS estimator

The corrected statistic matches `scipy.stats.kstest` to 2.1e-15 at
n = 20…800. The plotting-position quantity used for the figure curve sits
**below** the true KS distance, by 15.1 % (n = 20), 11.0 % (n = 50), 5.1 %
(n = 100), 1.5 % (n = 200), 4.9 % (n = 400), 1.5 % (n = 800): the deficit is
sample-dependent, not a constant factor, so it cannot be corrected by a scaling
constant. Anything used as a *test statistic* must use `kolmogorov_smirnov`.

### 3.3 The size-distribution gate (honest FAIL, confirmed)

Cascade census (`u' = 2 m/s`, h = 0.10 m, 5 seeds fixed, 53 survivors):

| quantity | value |
|---|---|
| survivors | 53 |
| CDF-LS fit | k = 1.804, lambda = 3.565 m, R² = 0.941, KS = 0.152 |
| log-LS fit | k = 0.697, lambda = 3.867 m, R² = 0.793, KS = 0.299 |
| SciPy MLE | k = 1.339, lambda = 3.033 m |
| gate vs reference (raw lengths, m) | R² = **-2.26** |
| gate vs reference (normalised `L/L_max`) | R² = **-1.41** |

Two findings, both negative for the report's criterion (c):

* The gate fails **under both conventions**, so the failure is a
  model-versus-reference mismatch (the split kernel `L = L_parent sqrt(w)`,
  `w ~ U^3`, does not reproduce an experimental Weibull fragment law), not an
  artefact of comparing metre-scale lengths with a `0.12 m` scale parameter.
  The report's "criterion (c) fails" stands; the reason is the kernel, not the
  census size alone.
* At census size n ≈ 50 the two estimators disagree on the shape by
  `|1.804 - 0.697| / 0.697 = 159 %`. A shape quoted from a ~50-floe census is
  therefore an estimator artefact as much as a physical result; `fit_weibull_cdf`
  is the better-conditioned of the two (R² 0.941 vs 0.793) and should be the one
  quoted.

## 4. Note on the coupled stage (not changed)

`coupling.follower_ship_impact` still divides by a literal `917.0` when it
converts surviving mass into ice area. It has no `ThermalProperties` argument,
so the constant cannot leak into a ledger, but it is the same hard-coding
pattern as defect 1 and should be parameterised if the impact estimate is ever
run with a non-default ice density.

## 5. Evidence grade

| capability | before | after |
|---|---|---|
| turbulent bending criterion (`bending_stress_pa`, `critical_floe_length`) | REAL (untested against closed form) | REAL, identity-verified (2e-16) |
| cascade mass conservation | REAL (asserted) | REAL, verified 2.8e-16 at three densities; density leak fixed |
| Weibull scale estimator | WITHHELD | REAL within 3 % (vs SciPy MLE) |
| Weibull shape estimator | WITHHELD | WITHHELD as a converged quantity — 2–13 % one-sided bias, 159 % estimator spread at n ≈ 50 |
| KS statistic | WITHHELD (wrong definition) | REAL, matches `scipy.kstest` to 2e-15 |
| size-distribution gate | honest FAIL (reason: census size) | honest FAIL (reason: split kernel; fails raw **and** normalised) |

## 6. Reproduction

```bash
python scripts/verify_fragmentation_statistics.py
python -m pytest tests/test_threefield_fragmentation_statistics.py -q   # 52 passed
```