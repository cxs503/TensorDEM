# R29 — long-window stationarity campaign (tables)

campaign: `benchmarks/plane_shear_mu_i/run_long_window_r29.py`, strain window 8 (archived rounds 24-26: 1)


## A. Per-run stationarity

| file | w | I_t | seed | γ_max | mu_wall | sem | tau | n_eff | trend z (all) | trend z (2nd half) | last-2 z | stationary | mu_inf fit | γ0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| long_r29_w0.5_I0.05_s0.json | 0.50 | 0.050 | 0 | 0.9 | 0.1727 | 0.0106 | 4.8 | 64 | +nan | +nan | +nan | no | 0.1699 | 0.07 |

## B. Seed ensemble, last half of the window

| w | I_target | seeds | I_measured | mu | seed sem | phi | phi sem | mu_ref(I) | err % | <10%? | envelope best-case err % | phi_ref | phi err % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.50 | 0.050 | 1 | 0.0817 | 0.1727 | nan | 0.8066 | nan | 0.4627 | -62.7 | NO | 118 | 0.6272 | +28.6 |

## B2. Curve-level falsification oracle (can *any* admissible reading of the citation fit this curve?)

_fewer than two I points available - curve oracle not applicable_

## C. Drift of the judged mu against the archived 1-strain window

| point | archived mu (1 strain) | R29 mu (last half of 8 strains) | change % |
|---|---|---|---|
| w0.5_I0.02 | 0.1278 | missing | – |
| w0.5_I0.05 | 0.1543 | 0.1727 | +11.9 |
| w0.5_I0.2 | 0.1616 | missing | – |
| w0.0_I0.05 | 0.0593 | missing | – |
