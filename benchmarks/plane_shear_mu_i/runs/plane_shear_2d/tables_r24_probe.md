# Round 24 tables - 2-D disk plane shear vs the archived law `mu = 0.22 + 1.00 I`, `phi = 0.81 - 0.30 I` (I <= 0.2)

## Table A - mu(I) and phi(I), seed ensemble per inertial number
| I_target | seeds | I_measured | mu_wall | sd | CV % | 95% CI | mu_law | err % | phi_gap | phi_law | err % | judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 1 | 0.0261 | 0.0963 | nan | nan | [nan, nan] | 0.2461 | -60.9 | 0.873 | 0.802 | +8.8 | yes |
| 0.05 | 1 | 0.0631 | 0.1003 | nan | nan | [nan, nan] | 0.2831 | -64.6 | 0.875 | 0.791 | +10.7 | yes |
| 0.1 | 1 | 0.1393 | 0.1238 | nan | nan | [nan, nan] | 0.3593 | -65.5 | 0.874 | 0.768 | +13.8 | yes |
| 0.2 | 1 | 0.2464 | 0.1294 | nan | nan | [nan, nan] | 0.4664 | -72.3 | 0.875 | 0.736 | +18.8 | NO (extrapolation) |

## Table B - estimator calibration (P1): the servo and the virial
| I_target | p_wall | p_bulk | sigma_yy/P | p_wall/P | ke_settle | drift_z(mu_wall) |
|---|---|---|---|---|---|---|
| 0.02 | 9695 | 7487 | 1.039 | 0.969 | 4.73e-05 | -6.94 |
| 0.05 | 9937 | 7559 | 1.066 | 0.994 | 4.73e-05 | -13.18 |
| 0.1 | 9093 | 6739 | 0.973 | 0.909 | 4.73e-05 | -2.81 |
| 0.2 | 10104 | 8107 | 1.082 | 1.010 | 4.73e-05 | +3.25 |

## Table C - 2-D device against the 3-D device (P3): dimensionality
| I (2-D) | I (3-D, nearest) | mu 2-D | mu 3-D | 2-D/3-D - 1 % |
|---|---|---|---|---|
| 0.0261 | 0.0203 | 0.0963 | 0.3231 | -70.2 |
| 0.0631 | 0.0703 | 0.1003 | 0.3430 | -70.8 |
| 0.1393 | 0.0703 | 0.1238 | 0.3430 | -63.9 |
| 0.2464 | 0.0703 | 0.1294 | 0.3430 | -62.3 |

## Verdict

* judged points: 3 (I <= 0.2)
* P1 calibration (|p_wall/P-1| <= 10 %): True
* P2 mu inside 10 %: False (0/3), max |err| 65.5 %
* P4 phi inside 10 %: False (1/3), max |err| 13.8 %
* P5 stationarity (|drift_z| < 2): False
