# Pre-registration — Beverloo orifice discharge, powered ensemble + bulk-density convention (m22885)

Written **before** any of this mission's DEM runs were read. The estimator that
computes the verdicts lives in `analyse.py` and is pinned on synthetic
ground-truth data by `tests/test_m22885_beverloo_powered.py` (PASS / FAIL /
NOT_DECIDABLE / NOT_RUN branches, Student-t quantiles, CI arithmetic, drift
regression recovery, counting-noise budget). Ledger anchor:
`CASE-DEM-SILO-BEVERLOO` (status `open` at checkout — verified, not redone).

## 0. Why this run exists (what earlier rounds left undone)

Ten earlier rounds of this campaign (`m20055 m20585 m20707 m21033 m21119
m21274 m22124 m22628 m22657 m22853`) produced *at most one seed per
configuration*, and the last three (m22628 `cj4266`, m22657 `cj4437`, m22853)
were submitted but never returned data (`NOT_RUN`). Two things were therefore
never established:

1. **the per-seed scatter** of the measured discharge rate, hence whether a
   single-seed number has any bearing on the ledger's 10 % tolerance at all;
2. **the bulk-density convention**: `run_discharge` passed the *nominal* fill
   height to `bulk_density()`, while the settled bed surface sits below it
   (deposition compaction). `rho_b` enters `W_ref` linearly, so this is a
   systematic (not statistical) bias on the verdict.

This mission fixes the measurement reporting (per-window particle count and bed
surface height), measures the scatter with a 12-seed ensemble, and states the
verdict with an interval instead of a point.

## 1. Law and reference (not refitted)

```
W = C · rho_b · sqrt(g) · (D − k·d)**(5/2)
```

| quantity | value | provenance |
|---|---|---|
| `C` | 0.58 (band 0.55–0.60) | Beverloo, Leniger & van de Velde (1961), *Chem. Eng. Sci.* **15**, 260–269; machine-readable `benchmarks/m20055_beverloo/ref_beverloo_1961.json` |
| `k` | 1.4 (band 1.2–1.6) | idem |
| `g` | 9.81 m/s² | — |
| tolerance | `abs(rel) < 10 %` | `feature_ledger.CASE-DEM-SILO-BEVERLOO.acceptance` |

`C` and `k` are **literature inputs**; no amplitude or exponent is refitted
here. A refitted `C` would be an inversion result, not a validation result.

## 2. Frozen configuration (identical to the campaign's primary case)

| parameter | value |
|---|---|
| particle diameter `d` | 3.0 mm (density 2500 kg/m³) |
| orifice `D` | 15.0 mm (`D/d = 5.00`) |
| silo radius | 24.0 mm (`silo_D/D = 3.20`) |
| fill height `H` | 36.0 mm → `H/D = 2.40` |
| wall friction | 0 (flat bottom plate; Beverloo smooth-wall condition) |
| inter-particle `mu` / `mu_rolling` / restitution | 0.50 / 0.05 / 0.30 |
| `kn` / `kt/kn` | 5.0e3 N/m / 0.5 |
| `g` | 9.81 m/s² |
| settle (plug closed) / discard | 0.06 s / 0.12 s |
| measurement | 8 windows × 0.05 s = 0.40 s |
| seeds | **1..12** (12 independent disordered packings) |
| kernel | `tensordem.silo_flow.study beverloo` (Cundall–Strack, unchanged physics) |

Secondary axis for the H-independence statement: `H = 48.0 mm`
(`H/D = 3.20`, same `D/d`, 4 seeds 1..4). Beverloo's law says the *steady* rate
is independent of the filling height for `H/D >= 1.5`.

Domain checks (computed in `run_case.dimensionless`, printed into every JSON):
`D/d = 5.00 >= 4`, `silo_D/D = 3.20 >= 2.5`, `H/D = 2.40 >= 1.5`.

## 3. Estimators (frozen)

Per run (JSON fields produced by `beverloo.run_discharge`, extended this
mission with `window_particles`, `window_surface_height_m`,
`rho_bulk_measured`, `surface_height_m`):

```
W_i        = window mass / window time                     (kg/s, per window)
W_s        = mean(last 50 % of windows)                    (campaign convention, kept)
N_exit     = sum(window_particles)
noise      = 1 / sqrt(N_exit)                              (counting floor, analytic)
rho_b_nom  = bed_mass / volume_below(nominal fill height)   (old convention)
rho_b_meas = bed_mass / volume_below(measured surface)      (new convention)
W_ref      = C_lit · rho_b · sqrt(g) · (D − k_lit·d)**2.5
rel_i      = W_s,i / W_ref,i − 1
```

Ensemble (n seeds):

```
rel_bar = mean_i rel_i,   sd = sd_i rel_i,   CI95 = rel_bar ± t(0.975, n−1)·sd/sqrt(n)
CV      = sd / |mean_i W_s,i|
```

Drift: ordinary least squares `W_i = a + b·h_i` over the 8 windows, `h_i` =
mid-window bed surface height; `drift_rel = |b|·(h_1 − h_8) / W_s` (fraction of
the rate lost/gained across the measurement span).

## 4. Verdicts (frozen, decided before the data)

| id | statement | PASS | FAIL | otherwise |
|---|---|---|---|---|
| **V1** ledger | `rel_bar` within ±10 % | CI entirely inside (−10 %, +10 %) | CI entirely outside | CI straddles a bound → `NOT_DECIDABLE` |
| **V2** reproducibility | seed scatter | `CV < 10 %` | `CV >= 10 %` | — (`n < 2` → `NOT_RUN`) |
| **V3** steady state | window drift | `drift_rel < 10 %` on ≥ 11/12 seeds | median `>= 10 %` | `NOT_DECIDABLE` |
| **V4** rho_b control | convention | `rho_b_meas/rho_b_nom` spread over seeds `< 3 %` **and** the two conventions differ by more than 1 % (else the fix is moot) | spread `>= 3 %` | — |
| **V5** H-independence | filling height | CI of `W(H/D=3.2)/W(H/D=2.4) − 1` inside ±10 % | outside | straddles → `NOT_DECIDABLE` |
| **V6** noise floor | counting statistics | `noise <= 0.05` so the 10 % band is ≥ 2σ decidable per seed | `noise > 0.05` → single-seed verdicts are noise-limited | — |

V1 uses `rho_b_meas` (physics-correct); the number with `rho_b_nom` is reported
alongside so the two conventions cannot be confused, but it does **not** decide
the ledger test.

## 5. Predictions (stated before the runs)

- **P1** `noise ≈ 5 %` per seed (≈ 340 particles discharged over 0.40 s at
  `W ≈ 0.030 kg/s`), so V6 fails and the 12-seed CI half-width is ≈ 2–3 %
  (t=2.20, `sd ≈ 5–8 %` → `SE ≈ 1.5–2.3 %`). Consequence: V1 is decidable
  unless the systematic bias sits inside ≈ 6.5–13.5 %.
- **P2** `rho_b_meas > rho_b_nom` by 1–4 % (settled surface 0.5–1.5 mm below the
  nominal fill line), i.e. the old convention biased `W_ref` low and every
  historical `W/W_ref` **high** by the same amount.
- **P3** `CV ∈ [5 %, 12 %]` — the seed-to-seed scatter is of the same order as
  the counting floor, so `V2` is marginal, not comfortable.
- **P4** `drift_rel < 10 %` (only ≈ 15 % of the bed drains; V3 passes) but the
  sign is negative (rate falls as the bed drops).
- **P5** V5 (H-independence): ratio within 10 %, because a flat-bottom silo with
  frictionless walls carries almost no Janssen wall stress.

## 6. Honest-FAIL policy

If V1's interval excludes the tolerance band, the ledger anchor keeps an honest
FAIL and this report states the measured deviation with its CI. No seed is
dropped, no window is re-chosen, no `k`/`C` is refitted. A `NOT_RUN` result (job
scheduled but not returned) is reported as `NOT_RUN` with no number attached.