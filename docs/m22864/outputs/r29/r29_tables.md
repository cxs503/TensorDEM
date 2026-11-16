# R29 — long-window stationarity campaign (tables)

campaign: `benchmarks/plane_shear_mu_i/run_long_window_r29.py`, strain window 8 (archived rounds 24-26: 1)


## A. Per-run stationarity

| file | w | I_t | seed | γ_max | mu_wall | sem | tau | n_eff | trend z (all) | trend z (2nd half) | last-2 z | stationary | mu_inf fit | γ0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_r29_w0.5_I0.02_s0.json | 0.50 | 0.020 | 0 | 8.5 | 0.1636 | 0.0055 | 167.7 | 90 | +3.4 | +1.9 | +2.41 | no | 0.4006 | 50.00 |
| long_r29_w0.5_I0.05_s0.json | 0.50 | 0.050 | 0 | 8.5 | 0.1740 | 0.0058 | 45.0 | 135 | -0.1 | -1.0 | -0.63 | YES | 0.1717 | 0.76 |
| long_r29_w0.5_I0.05_s1.json | 0.50 | 0.050 | 1 | 8.5 | 0.1710 | 0.0061 | 49.9 | 121 | +1.3 | +0.9 | +0.95 | YES | 0.1635 | 0.60 |
| long_r29_w0.5_I0.05_s2.json | 0.50 | 0.050 | 2 | 8.5 | 0.1533 | 0.0069 | 66.5 | 91 | -4.0 | -1.2 | -0.12 | YES | -0.0895 | 50.00 |
| long_r29_w0.5_I0.2_s0.json | 0.50 | 0.200 | 0 | 8.5 | 0.2103 | 0.0103 | 16.7 | 93 | +2.2 | +0.3 | -1.21 | YES | 0.2088 | 0.58 |
| long_r29_w0.5_I0.2_s1.json | 0.50 | 0.200 | 1 | 8.5 | 0.1929 | 0.0073 | 7.9 | 197 | -0.4 | +0.5 | -0.57 | YES | 0.2031 | 0.08 |
| long_r29_w0.5_I0.2_s2.json | 0.50 | 0.200 | 2 | 8.5 | 0.2272 | 0.0080 | 11.2 | 137 | +1.5 | -0.7 | +0.88 | YES | 0.2309 | 1.65 |
| long_r29_w0_I0.05_s0.json | 0.00 | 0.050 | 0 | 8.5 | 0.0450 | 0.0036 | 32.4 | 80 | -2.2 | -0.5 | -0.41 | YES | 0.0479 | 0.23 |
| long_r29_w0_I0.2_s0.json | 0.00 | 0.200 | 0 | 8.5 | 0.1331 | 0.0085 | 7.0 | 94 | +1.7 | -0.8 | +1.44 | YES | 0.1466 | 0.44 |

## B. Seed ensemble, last half of the window

| w | I_target | seeds | I_measured | mu | seed sem | phi | phi sem | mu_ref(I) | err % | <10%? | envelope best-case err % | phi_ref | phi err % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.050 | 1 | 0.0624 | 0.0450 | nan | 0.8755 | nan | 0.4426 | -89.8 | NO | 722 | 0.6297 | +39.0 |
| 0.00 | 0.200 | 1 | 0.2528 | 0.1331 | nan | 0.8756 | nan | 0.5795 | -77.0 | NO | 218 | 0.6126 | +42.9 |
| 0.50 | 0.020 | 1 | 0.0194 | 0.1636 | nan | 0.8210 | nan | 0.3891 | -58.0 | NO | 114 | 0.6364 | +29.0 |
| 0.50 | 0.050 | 3 | 0.0490 | 0.1661 | 0.0065 | 0.8185 | 0.0015 | 0.4274 | -61.1 | NO | 119 | 0.6316 | +29.6 |
| 0.50 | 0.200 | 3 | 0.1935 | 0.2101 | 0.0099 | 0.8016 | 0.0021 | 0.5482 | -61.7 | NO | 96 | 0.6165 | +30.0 |

## B2. Curve-level falsification oracle (can *any* admissible reading of the citation fit this curve?)

| branch | points | min possible worst-point error % | can any admissible law fit? |
|---|---|---|---|
| w=0 | 2 | 722 | NO |
| w=0.5 | 3 | 119 | NO |

## C. Drift of the judged mu against the archived 1-strain window

| point | archived mu (1 strain) | R29 mu (last half of 8 strains) | change % |
|---|---|---|---|
| w0.5_I0.02 | 0.1278 | 0.1636 | +28.0 |
| w0.5_I0.05 | 0.1543 | 0.1661 | +7.6 |
| w0.5_I0.2 | 0.1616 | 0.2101 | +30.1 |
| w0.0_I0.05 | 0.0593 | 0.0450 | -24.2 |
