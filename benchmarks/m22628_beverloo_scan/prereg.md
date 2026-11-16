# Pre-registration — Beverloo orifice scan + filling-height independence (m22628)

Written **before** any of the scan's DEM runs were read. The estimator that
computes the verdicts is already frozen and pinned by
`tests/test_m22628_beverloo_scan.py` (recovery of a known `(C, k, p)`, detection
of a wrong exponent, detection of a wrong offset, standard errors that grow with
noise, verdict arithmetic incl. an honest-FAIL case).

## 1. What is being tested

Beverloo et al. (1961), Eq. (4):

```
W = C · rho_b · sqrt(g) · (D − k·d)^p ,   p_law = 5/2,  C = 0.58,  k = 1.4
```

Three statements, each separately falsifiable, all of which the earlier rounds
of this campaign collapsed into one amplitude number:

| id | statement | acceptance | source |
|----|-----------|-----------|--------|
| **S1** | shape: the exponent is 5/2 | `abs(p_hat − 2.5) ≤ 0.2` (m22628 / P489-N04 acceptance) | campaign convention |
| **S2** | offset: the finite-size intercept is `k·d` with `k ∈ [1.2, 1.6]` | fitted `k` inside that interval | `ref_beverloo_1961.json` |
| **S3** | amplitude: `C ∈ [0.55, 0.60]` once `k` is fitted | fitted `C` inside that interval | idem |
| **S4** | H-independence: `W` does not depend on the filling height (for `H/D ≥ 1.5`) | `max abs rel dev < 10 %` over `H/D ∈ {2.0, 2.5, 3.0}` | Beverloo 1961 |
| **S5** | ledger: `abs(W_dem / W_literature − 1) < 10 %` at the primary case `D/d = 5`, **literature coefficients, not refitted** | `< 10 %` | `feature_ledger` CASE-DEM-SILO-BEVERLOO |

**S5 is deliberately kept separate from S1–S4.** A re-fitted amplitude is an
*inversion* result, not a *validation* result; letting a fitted `C` rescue the
ledger test would be exactly the circular reasoning the campaign has to avoid.

## 2. Predictions (stated before the runs)

- **P1** `p_hat = 2.5 ± se` with `se ≈ 0.05–0.15` at 5 points. Basis: m20055
  measured 2.357 (−5.7 %) with the *biased* `silo_discharge` kernel; the
  `silo_flow` kernel used here removed that bias at two points (m20707).
- **P2** `k_hat < 1.4`. Basis: the campaign's remaining discrepancy is a
  `D`-dependent over-prediction at small `D` (+20 % at `D/d = 4`) that decays to
  ≈ 0 at `D/d ≥ 5` — the signature of a too-large `k` in the *prediction*
  (forcing `k` above its true value shrinks `(D − k d)` and inflates the
  compensating `C`; pinned by the estimator test). Consequence: `C_hat > 0.58`
  under the pinned fit.
- **P3** `rho_b` varies by < 3 % across the scan, so the fit is not confounded
  by bulk density.
- **P4** `D/d = 4` is the case most likely to fail S5 (close to the jamming
  limit `D/d ≈ 3`; the earlier `D4` pilot arched once).
- **P5** S4 (H-independence) holds to < 5 %, because a flat-bottom silo with
  smooth walls carries little Janssen wall friction and the discharge is
  controlled by the free-fall jet through the aperture.

## 3. Falsification and honest-FAIL policy

- If `abs(p_hat − 2.5) > 0.2`, S1 fails and the law's *shape* is not reproduced
  by this kernel; the campaign says so and does not re-tune `k` until the
  exponent "looks right" (that would be fitting a 2-parameter model to 5 points
  and reporting `R²`).
- If S5 fails, the ledger anchor keeps its honest FAIL — the scan's job is to
  explain *why*, not to launder the number.
- Any of `D50`/`D50_s2` disagreement > 10 % makes the measurement noise-limited
  and is reported before any verdict is claimed.
- Fixed configuration, single seed per point except the declared seed repeat.
  No case is added or dropped after the fact.

## 4. Configuration (frozen)

`d = 3 mm`, `R_silo = 30 mm`, `H = 45 mm` (H-scan: 30 / 37.5 / 45 mm),
`mu = 0.50`, `mu_rolling = 0.05`, `e = 0.30`, `kn = 5e3 N/m`, `kt/kn = 0.5`,
`wall_mu = 0`, `g = 9.81`. Protocol: settle 0.06 s, discard 0.12 s,
5 windows × 0.15 s. Domain check per case: `D/d ≥ 4`, `silo_D/D ≥ 2.5`,
`H/D ≥ 1.5` — all 8 cases pass (`scan.py --list`, pinned by a test).

Cases: `D40 D45 D50 D55 D60` (orifice scan) · `D50_H25 D50_H20` (H-independence)
· `D50_s2` (seed repeat).