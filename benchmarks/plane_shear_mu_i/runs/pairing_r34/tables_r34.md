# R34 — stress-estimator pairing audit (generated, do not hand-edit)

pre-registration: `prereg_r34.json` (6 points)

## Table A — the ledger: four normal-stress channels of the same run

| I_target | seed | p_wall/P | p_bulk/P | p_prof_contact/P | p_prof_total/P | kinetic share of p | gamma_meas/gamma_nom |
|---|---|---|---|---|---|---|---|
| 0.050 | 0 | 0.676 | 0.648 | 0.653 | 0.658 | 0.007 | 0.631 |
| 0.050 | 1 | 0.862 | 0.853 | 0.853 | 0.859 | 0.007 | 0.555 |
| 0.050 | 2 | 0.343 | 0.353 | 0.357 | 0.362 | 0.012 | 0.260 |
| 0.050 | 3 | 0.678 | 0.689 | 0.692 | 0.696 | 0.007 | 0.562 |
| 0.100 | 0 | 0.620 | 0.544 | 0.545 | 0.553 | 0.015 | 0.171 |
| 0.100 | 1 | 0.590 | 0.601 | 0.604 | 0.614 | 0.016 | 0.147 |

## Table B — the four self-consistent pairings vs the 2-D ruler

| I_target | seed | pairing | I | mu | mu_ref(I) | err % | in 10 % band? |
|---|---|---|---|---|---|---|---|
| 0.050 | 0 | (wall,wall) | 0.0424 | 0.290 | 0.262 | +10.4 | no |
| 0.050 | 0 | (bulk,bulk) | 0.0433 | 0.307 | 0.263 | +16.4 | no |
| 0.050 | 0 | (profile_contact,profile_contact) | 0.0431 | 0.307 | 0.263 | +16.5 | no |
| 0.050 | 0 | (profile_total,profile_total) | 0.0429 | 0.304 | 0.263 | +15.8 | no |
| 0.050 | 1 | (wall,wall) | 0.0331 | 0.358 | 0.253 | +41.5 | no |
| 0.050 | 1 | (bulk,bulk) | 0.0332 | 0.355 | 0.253 | +40.0 | no |
| 0.050 | 1 | (profile_contact,profile_contact) | 0.0332 | 0.355 | 0.253 | +40.0 | no |
| 0.050 | 1 | (profile_total,profile_total) | 0.0331 | 0.352 | 0.253 | +39.0 | no |
| 0.050 | 2 | (wall,wall) | 0.0248 | 0.394 | 0.245 | +60.8 | no |
| 0.050 | 2 | (bulk,bulk) | 0.0244 | 0.346 | 0.244 | +41.4 | no |
| 0.050 | 2 | (profile_contact,profile_contact) | 0.0243 | 0.346 | 0.244 | +41.5 | no |
| 0.050 | 2 | (profile_total,profile_total) | 0.0241 | 0.339 | 0.244 | +38.7 | no |
| 0.050 | 3 | (wall,wall) | 0.0377 | 0.323 | 0.258 | +25.2 | no |
| 0.050 | 3 | (bulk,bulk) | 0.0374 | 0.310 | 0.257 | +20.4 | no |
| 0.050 | 3 | (profile_contact,profile_contact) | 0.0373 | 0.310 | 0.257 | +20.5 | no |
| 0.050 | 3 | (profile_total,profile_total) | 0.0372 | 0.308 | 0.257 | +19.6 | no |
| 0.100 | 0 | (wall,wall) | 0.0236 | 0.316 | 0.244 | +29.9 | no |
| 0.100 | 0 | (bulk,bulk) | 0.0252 | 0.388 | 0.245 | +58.1 | no |
| 0.100 | 0 | (profile_contact,profile_contact) | 0.0252 | 0.388 | 0.245 | +58.1 | no |
| 0.100 | 0 | (profile_total,profile_total) | 0.0250 | 0.378 | 0.245 | +54.4 | no |
| 0.100 | 1 | (wall,wall) | 0.0208 | 0.405 | 0.241 | +68.0 | no |
| 0.100 | 1 | (bulk,bulk) | 0.0206 | 0.364 | 0.241 | +51.1 | no |
| 0.100 | 1 | (profile_contact,profile_contact) | 0.0206 | 0.364 | 0.241 | +51.1 | no |
| 0.100 | 1 | (profile_total,profile_total) | 0.0204 | 0.356 | 0.240 | +48.1 | no |

## Table C — the legacy pairing (rounds 24-33) vs its off-diagonal neighbours

| I_target | seed | pairing | I | mu | err % |
|---|---|---|---|---|---|
| 0.050 | 0 | (wall,profile_contact) | 0.0431 | 0.290 | +10.1 |
| 0.050 | 0 | (bulk,wall) | 0.0424 | 0.307 | +16.8 |
| 0.050 | 0 | (bulk,bulk) | 0.0433 | 0.307 | +16.4 |
| 0.050 | 0 | (bulk,profile_contact) **legacy** | 0.0431 | 0.307 | +16.5 |
| 0.050 | 0 | (bulk,profile_total) | 0.0429 | 0.307 | +16.6 |
| 0.050 | 0 | (profile_contact,profile_contact) | 0.0431 | 0.307 | +16.5 |
| 0.050 | 0 | (profile_total,profile_contact) | 0.0431 | 0.304 | +15.7 |
| 0.050 | 1 | (wall,profile_contact) | 0.0332 | 0.358 | +41.4 |
| 0.050 | 1 | (bulk,wall) | 0.0331 | 0.355 | +40.1 |
| 0.050 | 1 | (bulk,bulk) | 0.0332 | 0.355 | +40.0 |
| 0.050 | 1 | (bulk,profile_contact) **legacy** | 0.0332 | 0.355 | +40.0 |
| 0.050 | 1 | (bulk,profile_total) | 0.0331 | 0.355 | +40.1 |
| 0.050 | 1 | (profile_contact,profile_contact) | 0.0332 | 0.355 | +40.0 |
| 0.050 | 1 | (profile_total,profile_contact) | 0.0332 | 0.352 | +39.0 |
| 0.050 | 2 | (wall,profile_contact) | 0.0243 | 0.394 | +61.2 |
| 0.050 | 2 | (bulk,wall) | 0.0248 | 0.346 | +41.2 |
| 0.050 | 2 | (bulk,bulk) | 0.0244 | 0.346 | +41.4 |
| 0.050 | 2 | (bulk,profile_contact) **legacy** | 0.0243 | 0.346 | +41.5 |
| 0.050 | 2 | (bulk,profile_total) | 0.0241 | 0.346 | +41.6 |
| 0.050 | 2 | (profile_contact,profile_contact) | 0.0243 | 0.346 | +41.5 |
| 0.050 | 2 | (profile_total,profile_contact) | 0.0243 | 0.339 | +38.6 |
| 0.050 | 3 | (wall,profile_contact) | 0.0373 | 0.323 | +25.4 |
| 0.050 | 3 | (bulk,wall) | 0.0377 | 0.310 | +20.3 |
| 0.050 | 3 | (bulk,bulk) | 0.0374 | 0.310 | +20.4 |
| 0.050 | 3 | (bulk,profile_contact) **legacy** | 0.0373 | 0.310 | +20.5 |
| 0.050 | 3 | (bulk,profile_total) | 0.0372 | 0.310 | +20.5 |
| 0.050 | 3 | (profile_contact,profile_contact) | 0.0373 | 0.310 | +20.5 |
| 0.050 | 3 | (profile_total,profile_contact) | 0.0373 | 0.308 | +19.6 |
| 0.100 | 0 | (wall,profile_contact) | 0.0252 | 0.316 | +29.0 |
| 0.100 | 0 | (bulk,wall) | 0.0236 | 0.388 | +59.1 |
| 0.100 | 0 | (bulk,bulk) | 0.0252 | 0.388 | +58.1 |
| 0.100 | 0 | (bulk,profile_contact) **legacy** | 0.0252 | 0.388 | +58.1 |
| 0.100 | 0 | (bulk,profile_total) | 0.0250 | 0.388 | +58.2 |
| 0.100 | 0 | (profile_contact,profile_contact) | 0.0252 | 0.388 | +58.1 |
| 0.100 | 0 | (profile_total,profile_contact) | 0.0252 | 0.378 | +54.2 |
| 0.100 | 1 | (wall,profile_contact) | 0.0206 | 0.405 | +68.1 |
| 0.100 | 1 | (bulk,wall) | 0.0208 | 0.364 | +51.0 |
| 0.100 | 1 | (bulk,bulk) | 0.0206 | 0.364 | +51.1 |
| 0.100 | 1 | (bulk,profile_contact) **legacy** | 0.0206 | 0.364 | +51.1 |
| 0.100 | 1 | (bulk,profile_total) | 0.0204 | 0.364 | +51.2 |
| 0.100 | 1 | (profile_contact,profile_contact) | 0.0206 | 0.364 | +51.1 |
| 0.100 | 1 | (profile_total,profile_contact) | 0.0206 | 0.356 | +48.0 |

## Table E — is the device in steady, homogeneous state? (per run)

| I_target | seed | gamma_meas/gamma_nom | profile r^2 | p_wall/p_bulk - 1 | mu_wall/mu_bulk - 1 | mu_bulk (mean of ratios) vs ratio of means |
|---|---|---|---|---|---|---|
| 0.050 | 0 | 0.631 | 0.979 | +4.4 | -5.5 | 0.3125 vs 0.3066 |
| 0.050 | 1 | 0.555 | 0.862 | +1.0 | +1.0 | 0.3538 vs 0.3546 |
| 0.050 | 2 | 0.260 | 0.970 | -2.9 | +13.9 | 0.3224 vs 0.3456 |
| 0.050 | 3 | 0.562 | 0.993 | -1.7 | +4.1 | 0.3154 vs 0.3100 |
| 0.100 | 0 | 0.171 | 0.586 | +14.0 | -18.4 | 0.4052 vs 0.3876 |
| 0.100 | 1 | 0.147 | 0.673 | -1.9 | +11.3 | 0.3682 vs 0.3636 |

## Table F — ensemble mean +/- sd of the judged mu error per channel

| I_target | channel | n seeds | mean err % | sd err % | half-range % |
|---|---|---|---|---|---|
| 0.050 | bulk | 4 | +29.6 | 12.973 | 12.486 |
| 0.050 | profile_contact | 4 | +29.6 | 12.950 | 12.471 |
| 0.050 | profile_total | 4 | +28.3 | 12.322 | 11.627 |
| 0.050 | wall | 4 | +34.5 | 21.665 | 25.211 |
| 0.100 | bulk | 2 | +54.6 | 4.911 | 3.473 |
| 0.100 | profile_contact | 2 | +54.6 | 4.922 | 3.480 |
| 0.100 | profile_total | 2 | +51.2 | 4.392 | 3.106 |
| 0.100 | wall | 2 | +48.9 | 26.936 | 19.046 |

## Table G — can the ruler slope be re-measured from this ensemble?

weighted OLS ``mu = a0 + a1 I`` over all runs (bulk channel, achieved ``I``);
the ruler is ``mu = 0.22 + 1.00 I``.

| quantity | value | ruler |
|---|---|---|
| n points | 6 | - |
| intercept a0 | 0.435 +/- 0.033 | 0.22 |
| slope a1 | -2.939 +/- 1.049 | 1.00 |
| achieved I range | 0.0206 .. 0.0433 | 0.001 .. 0.3 |
| z(slope - 1.00) | -3.75 | 0 = consistent with the ruler |

*caveat (fixed before the runs)*: this pooled fit is **not** a measurement of the ruler slope `b`. The runs do not form a controlled `I` sweep -- the `I_target = 0.1` points never reach their nominal rate (see Table E) and land at a *lower* achieved `I` with a *higher* `mu`, so pooling them is confounded. A non-zero `z` here is evidence that the achieved `I` is not the state variable while the cell is not rate-controlled, i.e. a device-state warning, not a rheological result.


## Table D — pre-registered predictions (fixed before the runs)

| prediction | observed | held? |
|---|---|---|
| P1 kinetic share > 0.15 | max 0.016 | no |
| P2 pressure spread > 1.15 | max 1.140 | no |
| P3 error spread > 10 pp | max 28.2 pp | yes |
| P4 falsifier: spread < 2 pp | min 2.4 pp | no |

P1/P2/P3/P4 are the pre-registered statements verbatim from `prereg_r34.json`; the observed columns are computed from the archived run JSON, not typed.

