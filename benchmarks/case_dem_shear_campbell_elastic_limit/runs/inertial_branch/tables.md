# CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT — R4 (inertial-branch reach + elastic/inertial stress partition)

## Points (this round)

| tag | k_n | gdot | kappa | tau* | I | mean ovl | max ovl | kinetic share | gates |
|---|---|---|---|---|---|---|---|---|---|
| kn1e4_g6.0 | 1e+04 | 6 | 160.8 | 3.847 | 0.3649 | 0.0589 | 0.233 | 5.44e-02 | PASS |
| kn1e4_g8.0 | 1e+04 | 8 | 90.42 | 2.364 | 0.4630 | 0.0663 | 0.274 | 6.62e-02 | PASS |
| kn1e4_g12.0 | 1e+04 | 12 | 40.19 | 1.178 | 0.6460 | 0.0812 | 0.317 | 7.70e-02 | PASS |
| kn1e4_g16.0 | 1e+04 | 16 | 22.61 | 0.7335 | 0.8197 | 0.0932 | 0.378 | 7.92e-02 | PASS |
| kn2.5e3_g8.0 | 2.5e+03 | 8 | 22.61 | 0.7335 | 0.8197 | 0.0932 | 0.378 | 7.92e-02 | PASS |
| kn1e3_g6.0 | 1e+03 | 6 | 16.08 | 0.5506 | 0.9438 | 0.1017 | 0.372 | 7.64e-02 | PASS |
| kn1e4_g24.0 | 1e+04 | 24 | 10.05 | 0.363 | 1.1478 | 0.1145 | 0.398 | 7.42e-02 | PASS |
| kn1e3_g8.0 | 1e+03 | 8 | 9.042 | 0.3304 | 1.1993 | 0.1177 | 0.408 | 7.37e-02 | PASS |
| kn1e4_g32.0 | 1e+04 | 32 | 5.651 | 0.217 | 1.4637 | 0.1330 | 0.451 | 7.45e-02 | FAIL |

## Gate wall (rejected points, reported not dropped)

| tag | kappa | failing gates | mean ovl | max ovl | energy defect |
|---|---|---|---|---|---|
| kn1e4_g32.0 | 5.651 | overlap | 0.1330 | 0.451 | 0.126 |

## Local exponents (R4 pairs, descending kappa)

| pair | sqrt(k_lo k_hi) | beta_local | n_local | sigma_n |
|---|---|---|---|---|
| kn1e3_g8.0->kn1e4_g24.0 | 9.531 | 0.892 | 0.217 | 2.006 |
| kn1e4_g24.0->kn1e3_g6.0 | 12.71 | 0.886 | 0.227 | 0.450 |
| kn1e3_g6.0->kn1e4_g16.0 | 19.06 | 0.841 | 0.317 | 0.620 |
| kn2.5e3_g8.0->kn1e4_g12.0 | 30.14 | 0.823 | 0.354 | 0.367 |
| kn1e4_g12.0->kn1e4_g8.0 | 60.28 | 0.859 | 0.282 | 0.261 |
| kn1e4_g8.0->kn1e4_g6.0 | 120.6 | 0.846 | 0.307 | 0.367 |

## Elastic + inertial partition  tau* = A kappa + B

- A = 0.016886 (68% CI [0.016446215694837858, 0.017366887965592884])
- B = 0.25203
- crossover kappa_x = B/A = 14.93 (68% CI [13.515065708147974, 16.26408746460913])
- single power law: beta = 0.8935, log-RSS = 0.1386, AIC = -55.03
- two-term:           log-RSS = 0.5116, AIC = -38.06
- delta AIC (two-term - power law) = 16.98 -> preferred: **single_power_law**

| kappa | tau* measured | tau* model | model beta | model inertial frac | measured kinetic share | ratio |
|---|---|---|---|---|---|---|
| 5.787e+05 | 7853 | 9772 | 1.000 | 0.0000 | 2.28e-04 | 8.849 |
| 5.787e+04 | 748.6 | 977.5 | 1.000 | 0.0003 | 1.30e-03 | 5.040 |
| 2.315e+04 | 316.6 | 391.1 | 0.999 | 0.0006 | 2.59e-03 | 4.027 |
| 5787 | 85.35 | 97.97 | 0.997 | 0.0026 | 5.61e-03 | 2.181 |
| 1447 | 25.14 | 24.68 | 0.990 | 0.0102 | 1.44e-02 | 1.412 |
| 361.7 | 7.712 | 6.36 | 0.960 | 0.0396 | 3.60e-02 | 0.909 |
| 160.8 | 3.847 | 2.967 | 0.915 | 0.0850 | 5.44e-02 | 0.641 |
| 90.42 | 2.364 | 1.779 | 0.858 | 0.1417 | 6.62e-02 | 0.467 |
| 40.19 | 1.178 | 0.9306 | 0.729 | 0.2708 | 7.70e-02 | 0.284 |
| 22.61 | 0.7335 | 0.6338 | 0.602 | 0.3977 | 7.92e-02 | 0.199 |
| 16.08 | 0.5506 | 0.5235 | 0.519 | 0.4815 | 7.64e-02 | 0.159 |
| 10.05 | 0.363 | 0.4217 | 0.402 | 0.5977 | 7.42e-02 | 0.124 |
| 9.042 | 0.3304 | 0.4047 | 0.377 | 0.6227 | 7.37e-02 | 0.118 |

## Verdicts

| prediction | verdict |
|---|---|
| P1_verdict | **PASS** |
| P2_verdict | **NOT_SUPPORTED** |
| P3_verdict | **NOT_SUPPORTED** |
| P4_verdict | **NOT_SUPPORTED** |
| P5_verdict | **kinetic_consistent** |
