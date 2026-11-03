# Friction / rolling verification - programmatic tables

## Contact patch (analytic normal force)

| quantity | value |
|---|---|
| radius R (m) | 0.025 |
| contact stiffness k_n (N/m) | 5000 |
| overlap (m) | 0.005 |
| normal force F_n (N) | 25 |
| time step dt (s) | 1.000000e-04 |

## Elastic (sticking) regime

| quantity | measured | reference | rel. error |
|---|---|---|---|
| tangential force (N) | -0.1 | -0.1 | 2.776e-16 |
| stored slip (m) | 2.000000e-05 | 2.000000e-05 | 1.694e-16 |
| stick ratio slip/a_stick | 0.01333 | < 1 | - |

## Sliding (saturated) regime

| mu | F_limit (N) | measured |F_t| (N) | rel. error | stored slip (m) | a_stick (m) | stored/a_stick | driven slip (m) |
|---|---|---|---|---|---|---|---|
| 0.05 | 1.25 | 1.25 | 7.105e-16 | 2.500000e-04 | 2.500000e-04 | 1 | 0.01 |
| 0.1 | 2.5 | 2.5 | 7.105e-16 | 5.000000e-04 | 5.000000e-04 | 1 | 0.01 |
| 0.2 | 5 | 5 | 7.105e-16 | 0.001 | 0.001 | 1 | 0.01 |
| 0.3 | 7.5 | 7.5 | 8.290e-16 | 0.0015 | 0.0015 | 1 | 0.01 |

## Friction-force reversal (history independence)

Reference: reversing a saturated force needs one ``a_stick`` = 5.000000e-04 m of reversed slip.

| sliding duration (s) | force evaluations | slip before reversal (m) | steps to flip | reversal slip (m) | excess (quanta) |
|---|---|---|---|---|---|
| 0.05 | 500 | 5.000000e-04 | 6 | 6.000000e-04 | 1 |
| 0.1 | 1000 | 5.000000e-04 | 6 | 6.000000e-04 | 1 |
| 0.2 | 2000 | 5.000000e-04 | 6 | 6.000000e-04 | 1 |

Spread across histories: 0 m (history independent: True).

## Contact-normal rotation

| quantity | value |
|---|---|
| normal force measured (N) | 25 |
| normal force reference (N) | 25 |
| spurious normal component of friction (N) | 1.776e-14 |
| as fraction of the friction limit | 3.553e-15 |

## Separation memory

| quantity | value |
|---|---|
| stored slip before separation (m) | 0.001 |
| pairs retained after separation | 0 |
| tangential force on re-contact (N) | 0 |

## Rolling resistance

| quantity | measured | reference | rel. error |
|---|---|---|---|
| tangential force (N) | -1.25 | -1.25 | 7.105e-16 |
| lateral force (N) | 0 | 0 | - |
| dissipation rate (W) | -1.25 | -1.25 | 7.105e-16 |
| control mu_r = 0 force (N) | 0 | 0 | - |

## Sliding work

| quantity | measured | reference | rel. error |
|---|---|---|---|
| total work (J) | 0.04775 | 0.0475 | 0.00526 |
| dissipated (J) | 0.04525 | 0.045 | 0.00556 |

## Time-step convergence of the sliding work

| dt (s) | steps | slip (m) | work measured (J) | work reference (J) | rel. error |
|---|---|---|---|---|---|
| 1.000000e-04 | 100 | 0.01 | 0.04775 | 0.0475 | 0.00526 |
| 5.000000e-05 | 200 | 0.01 | 0.047625 | 0.0475 | 0.00263 |
| 2.500000e-05 | 400 | 0.01 | 0.0475625 | 0.0475 | 0.00132 |

Error ratio first/last: 4 (first order expected: 4).
