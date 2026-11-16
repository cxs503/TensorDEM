# Pre-registration — seed ensemble + uncertainty budget for the Beverloo ledger quantity (m22657)

Written **before** any of this unit's DEM runs were submitted or read. The
statistics that turn the runs into verdicts are frozen first and pinned by
`tests/test_m22657_beverloo_ensemble.py`.

## 1. The gap this closes

`CASE-DEM-SILO-BEVERLOO` (acceptance: `|W − W_ref| / W_ref < 10 %`, ref
Beverloo et al. 1961) has been adjudicated eight times (m6296 … m22628), and
**every verdict so far rests on one seed per configuration**: m20707 reported
`D5 −6.9 %`, `D6 +5.9 %`, m21033/m21274 reported `+20 %` at `D/d = 4` for a
different kernel, m22628 pre-registered one seed repeat (`D50_s2`) but its runs
were still queued when that unit closed, so it published `NOT_RUN`.

No round has ever reported **the reproducibility scatter of `W` itself**, so
nobody knows whether the ledger's 10 % test is being decided by the model or by
the pseudo-random packing. This unit measures that scatter and adjudicates the
anchor with an interval instead of a point.

## 2. Two things are separated, as in m22628

| quantity | meaning | how it is used |
|---|---|---|
| **single-seed error** | `|W_s/W_ref − 1|` for seed `s` | reported, never used to pass the anchor |
| **ensemble error** | `|mean_s(W_s)/W_ref − 1|` with a 95 % CI | **this is the ledger verdict** |

Passing the anchor on the best of six seeds would be selection on noise; the
ensemble mean with a stated CI cannot do that.

## 3. Frozen configuration (`d = 3 mm`, one geometry, six packings)

Chosen to sit inside every domain constraint of the law and to keep the bed
from draining during the measurement:

| quantity | value | domain check |
|---|---|---|
| particle diameter `d` | 3.0 mm | — |
| orifice `D` | 15.0 mm (`D/d = 5`) | `D/d ≥ 4` ✓ |
| silo radius | 24.0 mm (`silo_D/D = 3.2`) | `≥ 2.5` ✓ |
| fill height `H` | 36.0 mm (`H/D = 2.4`) | `≥ 1.5` ✓ |
| `mu` / `mu_rolling` / `e` | 0.50 / 0.05 / 0.30 | glass-bead high-friction (m20707) |
| `kn` / `kt/kn` | 5.0e3 N/m / 0.5 | campaign convention |
| wall friction | 0 (flat bottom plate) | Beverloo smooth-wall condition |
| `g` | 9.81 m/s² | — |
| settle / discard | 0.06 s / 0.12 s | plug closed during settling |
| measurement | **8 windows × 0.05 s = 0.40 s** | drains ≈ 14 % of the bed |
| seeds | **1, 2, 3, 4, 5, 6** (disordered packing, `build_bed`) | — |
| kernel | `tensordem.silo_flow.study beverloo` (Cundall–Strack) | — |

`d`, `mu`, `kn`, `wall_mu` and the settle/discard protocol are the campaign's
frozen values; the silo is scaled down (24/36 mm instead of 30/45 mm) purely to
keep one case ≈ 2.5 k particles so six independent packings fit in the queue.
Nothing in the verdict depends on the absolute size: `W_ref` is evaluated with
the same `rho_b` that the case itself measures.

Sub-window length 0.05 s (not 0.15 s) is deliberate: it gives 8 points per seed
for the steadiness test at the same total measurement time.

## 4. Estimator (frozen in `analyse.py`, no free choices at verdict time)

```
W_s        = mean(rate of the last 50 % of windows)      # campaign convention
W_ref      = C_lit * rho_b * sqrt(g) * (D − k_lit d)^(5/2)   # C=0.58, k=1.4, literature
rel_s      = W_s / W_ref − 1
W_ens      = mean_s(W_s);  sd = std_s(W_s, ddof=1);  se = sd / sqrt(n_seeds)
CI95       = W_ens ± t(0.975, n−1) · se
```

No coefficient is refitted: `C = 0.58`, `k = 1.4` are read from
`benchmarks/m20055_beverloo/ref_beverloo_1961.json`. A refitted `C` is an
inversion result and is reported separately (`C_recovered` per seed) — it is
**not** allowed to rescue the ledger tolerance (m22628 §1 S5).

## 5. Pre-declared predictions

- **P1** Seed scatter of `W` is ≥ 3 %: the mass that leaves during the counted
  windows is ≈ 340 particles, so counting noise alone is `1/sqrt(340) ≈ 5.4 %`
  on the mean. The repacking adds more.
- **P2** `W_ens` lands **below** `W_ref` by 3–10 % (the sign m20707 measured at
  `D/d = 5` with this kernel, `−6.9 %`), i.e. the kernel under-predicts rather
  than the `+20 %` over-prediction of `silo_discharge`.
- **P3** The counted windows are still accelerating: the earliest counted window
  rate is below the last by 5–25 % (start-up transient survives the 0.12 s
  discard), so the "steady" label is only approximately true and the direction of
  the resulting bias is a **low** `W`.
- **P4** `rho_b` varies by < 3 % across the six packings, so the ensemble mean is
  not confounded by bulk density.
- **P5** With six seeds the 95 % CI half-width is ≤ 6 %, i.e. the ledger's 10 %
  test becomes decidable; if it is not, the unit says so.

## 6. Verdict rules (fixed before the runs)

| id | statement | PASS criterion | honest alternative |
|---|---|---|---|
| **V1** | ledger `S5` at the ensemble mean | `abs(W_ens/W_ref − 1) < 10 %` **and** the 95 % CI excludes the ±10 % boundary | `FAIL` if the point estimate is out; `NOT_DECIDABLE` if the CI straddles the boundary |
| **V2** | reproducibility | `sd/W_ens < 10 %` | `FAIL` — the measurement, not the law, is then the limiting factor |
| **V3** | steady state | `abs(rel drift over the counted windows) < 10 %` | `FAIL` (reported, does not gate V1) |
| **V4** | bulk-density control | `(max rho_b − min rho_b)/mean rho_b < 3 %` | `FAIL` (reported) |

`V1` is the anchor. If the runs do not come back, every DEM verdict is written
as `NOT_RUN` and no accuracy claim is made — the noise budget of §5-P1 is still
reported because it is analytic.

## 7. Falsification policy

- If `V2` fails, the conclusion is *not* "the law is wrong" but "this case
  cannot resolve 10 % with one seed"; the anchor stays open and the report says
  the tolerance needs a different bed size, not a re-tune.
- No case is added, dropped, lengthened or re-seeded after looking at the rates.
- A seed whose packing fails to build or whose bed jams (zero discharge) is
  reported as-is, not resampled.