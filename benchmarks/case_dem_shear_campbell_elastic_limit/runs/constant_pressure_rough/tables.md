# R5 constant-pressure ladder - tables

| tag | role | kn | I | gdot | kappa | tau* | mu_bulk | phi | Z | ovl proxy | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I0.020_rough | ladder | 6.25e+04 | 0.0501 | 55.9 | 8003 | 159.7 | 0.400 | 0.7745 | 6.10 | 0.0166 | FAIL(pressure_control,shear_rate,profile_r2) |
| I0.040_rough | ladder | 6.25e+04 | 0.1266 | 140.5 | 1266 | 24.4 | 0.407 | 0.9193 | 7.31 | 0.0164 | FAIL(pressure_control,shear_rate) |
| I0.100_rough | ladder | 6.25e+04 | 0.4327 | 407.2 | 150.8 | 1.851 | 0.339 | 1.1670 | 9.93 | 0.0118 | FAIL(pressure_control,shear_rate) |
| I0.200_rough | ladder | 6.25e+04 | 1.2213 | 803.2 | 38.76 | 0.275 | 0.359 | 1.1659 | 10.48 | 0.0058 | FAIL(pressure_control,shear_rate,profile_r2) |
| I0.400_rough | ladder | - | - | - | - | - | - | - | - | - | missing |
| I0.100_kn2.5e5_rough | kn | 2.5e+05 | 0.2370 | 258.0 | 1502 | 7.393 | 0.415 | 0.6658 | 5.52 | 0.0040 | FAIL(pressure_control,shear_rate) |
| I0.100_kn1.0e6_rough | kn | 1e+06 | 0.1591 | 201.9 | 9812 | 16.06 | 0.406 | 0.5025 | 3.15 | 0.0013 | FAIL(pressure_control,shear_rate,phi_min) |
| I0.200_s2_rough | seed | 6.25e+04 | 1.2213 | 803.2 | 38.76 | 0.275 | 0.359 | 1.1659 | 10.48 | 0.0058 | FAIL(pressure_control,shear_rate,profile_r2) |
| I0.200_dt2_rough | dt | 6.25e+04 | 0.6764 | 716.4 | 48.7 | 1.028 | 0.475 | 0.9724 | 8.08 | 0.0150 | FAIL(pressure_control,shear_rate) |

## verdicts

| id | verdict |
|---|---|
| P1 | NOT_SUPPORTED |
| P2 | NOT_SUPPORTED |
| P3 | NOT_SUPPORTED |
| P4 | None |
| P5 | NOT_SUPPORTED |
| P6 | NOT_SUPPORTED |
