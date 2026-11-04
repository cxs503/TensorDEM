# Fragmentation statistics verification (threefield)

Verification record for the *statistics layer* of `tensordem.threefield.fragmentation`
and for the mass ledger that `threefield.coupling` builds on top of it.

Scope statement, up front: this document verifies **instrumentation and bookkeeping** —
the Weibull estimators, the Kolmogorov-Smirnov statistic, the cascade mass ledger and
the bending-stress coefficient — against references that are independent of the
implementation. It does **not** validate the breakup criterion against ice-tank or
full-scale data, and it does not validate the fragmentation kernel as a physical model
of propeller-race breakup (see §6, where that claim stays WITHHELD).

Harness: `benchmarks/fragmentation_statistics.py` (schema
`tensordem.fragmentation-statistics/1`), figures/tables:
`benchmarks/fragmentation_statistics_figures.py`, regression guard:
`tests/test_fragmentation_statistics.py`. Every number in the tables is written by
those scripts into `artifacts/fragmentation_statistics/validation_report.json` and
`docs/fragmentation_statistics_tables.md`; nothing below is transcribed by hand.

## 1. Why this work exists

`docs/THREEFIELD_ICE_HULL_PROPELLER.md` §3.3 already reports the coupling honestly
(zero split events for the DEM census, R² = 0.030 for its Weibull fit). What no
document covered is whether the *instruments* used to report those numbers are right,
or whether the ledger they feed is closed. Reading the module against its own
docstrings found two defects, both invisible with the default parameters:

| # | defect | where | grade before | effect |
|---|---|---|---|---|
| 1 | `fragment()` rebuilt every fragment mass with a hard-coded `rho_i = 917.0` while `coupling.fragmentation_stage` converts inventory mass to floe length with `props.ice_density_kg_m3`. The two disagreed whenever the density was not 917. | `fragmentation.fragment` | REAL algorithm, **silently wrong parameter** | coupled mass ledger drifts by `917/rho_i - 1`: **+1.89 %** at 900 kg/m³, **+3.47 %** at 950, **+14.62 %** at 800 — larger than the 3 % accuracy bar for a ship-ice mass budget |
| 2 | `kolmogorov_smirnov()` returned `max |(i+1)/(n+1) - F(x_i)|` — Weibull plotting positions on one side of the empirical CDF — not the KS statistic, although the name and the 5 % critical values quoted around it belong to the classical test | `fragmentation.kolmogorov_smirnov` | MOCK-grade instrument (right family, wrong statistic) | statistic biased low by **22.9 %** at n = 10 and **4.5 %** at n = 200, so any threshold gate on it was under-powered (see §4) |

Neither was reachable by the existing suite: `test_fragmentation_conserves_mass` used
the default density, and nothing called `kolmogorov_smirnov` at all — it was dead,
ungraded API. Both are fixed in this branch; the fixes are the only two source changes.

## 2. Method

| quantity | independent reference | why it counts as independent |
|---|---|---|
| bending-stress coefficient | finite-difference solution of `EI w'''' = q` with `w = w'' = 0` at both ends, plus a sampled-statics (lumped point load) model | neither route uses the module's algebraic coefficient; the coefficient is recovered as `6 M_max / h^2` |
| `critical_floe_length` | bisection on the module's own `bending_stress_pa`, plus log-log scaling exponents | a root-find against a different code path than the closed-form `sqrt(...)` |
| cascade mass ledger | `rho_i L^2 h` recomputed outside the module, swept over 800–950 kg/m³, plus the coupled stage's inventory conversion | the ledger is closed from the physics side, not from the module's own `mass_error_kg` |
| KS statistic | Kolmogorov asymptotic `1.358/sqrt(n)` and finite-`n` `kstwo.ppf(0.95, n)` critical values, `scipy.stats.kstest` as an oracle, Monte-Carlo null and alternative | three independent oracles; the statistic itself is compared to machine precision |
| Weibull estimators | `scipy.stats.weibull_min.fit` (MLE) on identically seeded samples | a different estimator on the same data |

Common settings: SI, `float64`, seeded `numpy.random.default_rng`, 4000 Monte-Carlo
trials per sample size for the KS study, 200 trials for the estimator bias (15 for the
expensive CDF least-squares fit), truth `Weibull(shape 1.8, scale 1.0)` for the KS
study and `Weibull(1.8, 0.40 m)` for the estimator bias. Reference case for the
criterion: `u' = 0.75 m/s`, `h = 0.10 m`, `sigma_f = 0.5 MPa`, sea water 1025 kg/m³,
`C_d = 1.2`.

Runtime: about 80 s on one core, no solver sweeps — deliberately sized to be safe on a
loaded control host (`load15` was 122 at the time of writing, `nproc = 128`).

## 3. Results

All eight automated verdicts in the JSON are PASS. Headline numbers:

- **Bending-stress coefficient.** Recovered from `bending_stress_pa` as
  `0.750000000000` (relative error `0.0` against 3/4). The finite-difference route
  reproduces `M_max/(qL²) = 0.125` to `6.0e-07` at the finest mesh (the error *grows*
  with refinement — it is the `h^-4` conditioning of the matrix, not discretisation),
  and the sampled-statics route is exact (`0.0`) on every symmetric mesh tried.
  See §5 for why no grid-convergence index is claimed.
- **`critical_floe_length`.** `13.882112 m` for the reference case; bisection
  reproduces it to `0.0`, and the measured exponents are `-1.000000` (velocity),
  `0.500000` (thickness), `0.500000` (strength) against `-1, 0.5, 0.5`.
- **Mass ledger.** After the fix, initial and final masses match `rho_i L^2 h` to
  `<= 3.4e-16` relative for 800/900/917/940/950 kg/m³ (initial mass error exactly
  `0.0`), and the calm (no-breakup) branch is exact. The coupled stage closes its
  ledger to `<= 3.1e-16` in all four densities **while actually splitting** (4–8 split
  events per density; with the previous test masses every floe sat below `L_crit` and
  no density dependence was exercised at all).
- **KS statistic.** The module function now equals the exact two-sided statistic to
  `0.0` and equals `scipy.stats.kstest` to `0.0`. It is never below the old
  plotting-position proxy. Calibration: exact-statistic false-positive rate at the
  nominal 5 % level is 3.5–5.4 % (finite-`n` critical values: 4.6–5.8 %) over
  n = 10…500.
- **Gate power.** Using the exact statistic instead of the proxy raises the power to
  detect a shape change 1.8 → 2.4 from `0.07 → 0.12` at n = 50 and `0.54 → 0.62` at
  n = 200; at n = 500 both saturate at 0.99.
- **Weibull estimators.** `fit_weibull` (the double-log least-squares fit the coupled
  report quotes) is biased low in shape by `-9.6 %` at n = 30, `-4.4 %` at n = 100 and
  `-2.3 %` at n = 300. The CDF least-squares fit is `-1.9 %, -3.8 %, +1.4 %`; the scipy
  MLE `+5.2 %, +1.7 %, +0.6 %`. The bias decays with n but is not negligible for the
  ~20–200-fragment censuses this pipeline actually produces.
- **Cascade post-conditions.** 60 floes → 137 fragments, 51 split events, largest
  survivor `13.607 m <= L_crit = 13.882 m`, mass relative error `3.4e-16`,
  bit-identical replay. Split events grow monotonically with the race velocity
  (0, 7, 51, 270, 1281 at `u' = 0.2 … 3.0 m/s`).
- **Support-condition sensitivity.** The module's `3/4` is the simply supported value;
  a cantilever gives `L_crit = 6.941 m`, clamped-clamped `17.002 m`. The end condition
  alone spans `2.4495×` in `L_crit` (6× in the stress coefficient).

Full tables: [`fragmentation_statistics_tables.md`](fragmentation_statistics_tables.md)
(regenerate with the two benchmark scripts; the file is machine-written).

## 4. Figures

`artifacts/fragmentation_statistics/fragmentation_statistics.png`, four panels:

1. **Support-condition sensitivity** — `L_crit` for cantilever / simply supported /
   clamped with the module value marked; the numerically verified coefficient is
   annotated on each bar.
2. **KS null distribution** — exact statistic (colour) vs the pre-fix proxy (grey) at
   n = 20 and n = 200 with the 5 % critical values; the proxy's whole distribution sits
   left of the exact one.
3. **Power vs n** — exact vs proxy, with the measured false-positive rate annotated at
   each point; the fix buys power in the 10 ≤ n ≤ 300 range that this pipeline works in.
4. **Mass-ledger error vs ice density** — the pre-fix `917/rho_i - 1` drift against the
   post-fix error floor (`~1e-16`), log scale.

## 5. Caveats and limitations

- **No mesh-convergence study on the beam, on purpose.** The exact solution of a
  uniformly loaded simply supported beam is a quartic, and the five-point stencil is
  exact for polynomials of degree ≤ 4, so every consistent discretisation reproduces
  `M_max = qL²/8` exactly. A convergence index here would be an artefact; what the
  mesh series does show is conditioning (the `h^-4` matrix makes the round-off floor
  grow to `6e-07` at 640 cells). The coefficient is therefore verified *algebraically
  and to machine precision*, not asymptotically — a weaker statement than a converging
  discretisation, and it is stated as such in the JSON (`no_gci_claim`).
- **The KS calibration is Monte-Carlo.** With 4000 trials the standard error of a 5 %
  rate is 0.34 pp; the reported Wilson intervals (in the JSON) do not all contain 0.05
  at the small-n end because the asymptotic critical value is slightly anti-conservative
  there, which is why the finite-`n` column is also reported (4.6–5.8 %).
- **`fit_weibull`'s bias is reported, not removed.** Changing the estimator would
  change numbers already published in the threefield report; the fit choice is a
  modelling decision, not a defect. Anyone gating on a shape parameter from a
  ~50-fragment census should subtract ~5 % of it, or use `fit_weibull_cdf`.
- **The reference law is not commensurate with a field census.** `run_coupled_pass`
  grades the census against `Weibull(shape 1.8, scale 0.12 m)`. For the DEM laboratory
  census (fragments of 0.04 m scale) that is plausible. For the field-scale outlook
  population (0.5–40 m floes, `L_crit = 13.9 m`) the harness measures a fitted scale of
  `6.51 m` — a reference/census scale ratio of `0.018` — and the gate returns
  `R² = -2.67` with max CDF deviation `0.947`. That number says the *units are not
  commensurate*, not that the cascade is wrong; it is recorded rather than "fixed"
  because choosing the reference law is a physics decision, not a verification one.
- **Not covered here:** the thermal (Ranz–Marshall) and LBM wake stages, the DEM
  bond-fracture kernel upstream of the census, and any experimental comparison.

## 6. Error verdict

| key quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| stress coefficient (from `bending_stress_pa`) | 0.75 | 0.75 | `0.0` | yes (exact) |
| `M_max/(qL²)`, finite-difference route | 0.125 | 0.125 | `6.0e-07` worst mesh | yes |
| `M_max/(qL²)`, sampled statics | 0.125 | 0.125 | `0.0` | yes |
| `L_crit` vs bisection on the stress law | 13.882112 m | 13.882112 m | `0.0` | yes |
| scaling exponent, `u'` / `h` / `sigma_f` | −1 / 0.5 / 0.5 | −1.000000 / 0.500000 / 0.500000 | `<= 2e-15` | yes |
| cascade mass ledger (5 densities) | `rho_i L^2 h` | module value | `<= 3.4e-16` | yes |
| coupled ledger (4 densities, splitting) | inventory mass | stage `final_mass_kg` | `<= 3.1e-16` | yes |
| KS statistic vs exact / vs scipy | exact | module | `0.0` | yes |
| KS false-positive rate at 5 % | 0.05 | 0.035–0.054 | `<= 1.5 pp` | yes |
| Weibull shape, `fit_weibull`, n = 100 | 1.8 | `1.8 × (1 − 0.044)` | `−4.4 %` | **no** (reported, §5) |
| Weibull shape, `fit_weibull_cdf`, n = 100 | 1.8 | `1.8 × (1 − 0.038)` | `−3.8 %` | **no** (reported, §5) |
| cascade shape vs experimental law | 1.8 | 0.9465 | `−47 %` | no — kernel closure stays **WITHHELD** |

The two rows that fail the 3 % bar are estimator/physics properties, not implementation
errors; they are stated as found and are not papered over. The kernel's shape mismatch
(`0.95` measured against the `1.8` reference) is the reason the Weibull agreement in the
coupled report remains a *reported statistic* rather than an acceptance gate.

## 7. Reproduction

```
python benchmarks/fragmentation_statistics.py          # ~80 s, writes the JSON (exit != 0 on any FAIL)
python benchmarks/fragmentation_statistics_figures.py  # figure + generated tables
python -m pytest tests/test_fragmentation_statistics.py -q
```

## 8. Files touched

- `src/tensordem/threefield/fragmentation.py` — `fragment()` uses
  `props.ice_density_kg_m3`; `kolmogorov_smirnov()` returns the exact two-sided
  statistic (both documented in-place with the pre-fix behaviour).
- `benchmarks/fragmentation_statistics.py`, `benchmarks/fragmentation_statistics_figures.py`
  — harness and figure/table producer.
- `tests/test_fragmentation_statistics.py` — 16 regression tests (density sweep,
  coupled ledger closure with real splitting, KS vs exact vs scipy, KS gate power,
  beam coefficient, `L_crit` self-consistency, cascade post-conditions, reference-law
  commensurability).
- `artifacts/fragmentation_statistics/` (JSON + PNG), `docs/fragmentation_statistics_tables.md`
  (generated).