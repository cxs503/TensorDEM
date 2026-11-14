# R5 constant-pressure ladder - tables

| tag | role | kn | I | gdot | kappa | tau* | mu_bulk | phi | Z | ovl proxy | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|
| I0.010 | ladder | 6.25e+04 | 0.0122 | 15.3 | 1.062e+05 | 2101 | 0.313 | 0.6937 | 5.72 | 0.0211 | FAIL(profile_r2) |
| I0.020 | ladder | 6.25e+04 | 0.0204 | 26.0 | 3.692e+04 | 841.8 | 0.351 | 0.6968 | 5.75 | 0.0217 | FAIL(shear_rate,profile_r2) |
| I0.040 | ladder | 6.25e+04 | 0.0389 | 56.2 | 7918 | 254.8 | 0.386 | 0.7180 | 5.92 | 0.0278 | FAIL(profile_r2) |
| I0.100 | ladder | 6.25e+04 | 0.1562 | 212.4 | 554.2 | 15.71 | 0.383 | 0.7288 | 5.93 | 0.0246 | FAIL(shear_rate) |
| I0.200 | ladder | 6.25e+04 | 0.2501 | 347.8 | 206.7 | 6.961 | 0.435 | 0.7175 | 5.67 | 0.0258 | PASS |
| I0.400 | ladder | 6.25e+04 | 0.5360 | 732.2 | 46.63 | 1.659 | 0.476 | 0.7105 | 5.41 | 0.0249 | FAIL(shear_rate) |
| I0.020_s2 | seed | 6.25e+04 | 0.0214 | 27.9 | 3.202e+04 | 801.3 | 0.366 | 0.7130 | 5.95 | 0.0228 | FAIL(profile_r2) |
| I0.200_s2 | seed | 6.25e+04 | 0.2968 | 403.0 | 153.9 | 4.886 | 0.431 | 0.7237 | 5.79 | 0.0246 | FAIL(shear_rate) |
| I0.020_dt2 | dt | 6.25e+04 | 0.0189 | 25.3 | 3.899e+04 | 1026 | 0.366 | 0.7042 | 5.84 | 0.0240 | FAIL(shear_rate,profile_r2) |
| I0.200_dt2 | dt | 6.25e+04 | 0.2946 | 403.4 | 153.7 | 5.209 | 0.453 | 0.7178 | 5.66 | 0.0250 | FAIL(shear_rate) |
| I0.100_kn2.5e5 | kn | 2.5e+05 | 0.0846 | 113.6 | 7746 | 50.72 | 0.362 | 0.5773 | 4.40 | 0.0060 | FAIL(shear_rate,profile_r2) |
| I0.100_kn1.0e6 | kn | 1e+06 | 0.0857 | 114.5 | 3.052e+04 | 38.4 | 0.279 | 0.5587 | 3.37 | 0.0015 | FAIL(shear_rate,profile_r2) |

## verdicts

| id | verdict |
|---|---|
| P1 | NOT_SUPPORTED |
| P2 | NOT_SUPPORTED |
| P3 | NOT_SUPPORTED |
| P4 | None |
| P5 | NOT_SUPPORTED |
| P6 | NOT_SUPPORTED |
