# R29 — adjudication of the archived polydispersity sweep (round 26's `cj4227`), with round-27 instruments

Data: `benchmarks/plane_shear_mu_i/runs/plane_shear_2d/poly_*.json` (9 runs, 1-strain window, round 26). No new dynamics in this table.

## A. Per-point verdict

| w | I_target | $I_{meas}$ | $\mu_{wall}$ | naive sem | corr sem | naive/corr | $\tau_{int}$ | drift z (stored) | $\mu_{ref}$ | err % | <10 %? | $\phi_{gap}$ | $\psi_6$ | $p_{bulk}/p_{wall}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.020 | 0.0259 | 0.0448 | 0.0217 | 0.0042 | 5.12 | 31.2 | -4.1 | 0.3981 | -88.7 | NO | 0.8753 | 0.937 | 0.717 |
| 0.00 | 0.050 | 0.0669 | 0.0593 | 0.0439 | 0.0104 | 4.21 | 18.5 | -7.6 | 0.4475 | -86.7 | NO | 0.8753 | 0.914 | 0.718 |
| 0.00 | 0.200 | 0.2383 | 0.1156 | 0.0634 | 0.0086 | 7.37 | 1.5 | +2.3 | 0.5725 | -79.8 | NO | 0.8733 | 0.957 | 0.783 |
| 0.25 | 0.020 | 0.0208 | 0.1379 | 0.0472 | 0.0105 | 4.51 | 58.0 | +20.2 | 0.3911 | -64.7 | NO | 0.8158 | 0.448 | 1.001 |
| 0.25 | 0.050 | 0.0502 | 0.1160 | 0.0555 | 0.0113 | 4.89 | 20.2 | +2.4 | 0.4288 | -72.9 | NO | 0.8086 | 0.436 | 0.994 |
| 0.25 | 0.200 | 0.2092 | 0.1639 | 0.0819 | 0.0139 | 5.88 | 3.6 | +2.5 | 0.5572 | -70.6 | NO | 0.8007 | 0.465 | 0.982 |
| 0.50 | 0.020 | 0.0199 | 0.1278 | 0.0434 | 0.0095 | 4.58 | 89.8 | -4.8 | 0.3899 | -67.2 | NO | 0.8148 | 0.397 | 0.987 |
| 0.50 | 0.050 | 0.0474 | 0.1543 | 0.0562 | 0.0113 | 4.97 | 31.0 | -9.6 | 0.4255 | -63.7 | NO | 0.8118 | 0.401 | 0.958 |
| 0.50 | 0.200 | 0.1730 | 0.1616 | 0.0654 | 0.0104 | 6.29 | 4.9 | +2.2 | 0.5356 | -69.8 | NO | 0.8085 | 0.376 | 0.940 |

## B. Curve-level falsification oracle

| branch | points | min possible worst-point error % | admissible law fits? | argmin $(\mu_s,\mu_2,I_0)$ |
|---|---|---|---|---|
| w = 0 | 3 | 689 | **NO** | (0.34, 0.55, 0.38) |
| w = 0.25 | 3 | 214 | **NO** | (0.34, 0.55, 0.38) |
| w = 0.5 | 3 | 174 | **NO** | (0.34, 0.55, 0.38) |

## C. What the sweep actually says about round 26's hypothesis

* at $I\approx0.020$: $\mu$ goes 0.0448 ($w=0$) → 0.1278 ($w=0.5$), +185 %; $\psi_6$ 0.94 → 0.40.
* at $I\approx0.047$: $\mu$ goes 0.0593 ($w=0$) → 0.1543 ($w=0.5$), +160 %; $\psi_6$ 0.91 → 0.40.
* at $I\approx0.173$: $\mu$ goes 0.1156 ($w=0$) → 0.1616 ($w=0.5$), +40 %; $\psi_6$ 0.96 → 0.38.

Reading: polydispersity **does** break the crystal and **raises** the effective friction, so round 25's diagnosis was directionally right; but the branch still misses the cited law by far more than the 10 % band and, per section B, **no** admissible reading of the citation can fit it. The gap is therefore neither bibliographic nor crystal-ordering alone.

