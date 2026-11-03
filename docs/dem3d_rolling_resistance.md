# Rolling resistance in TensorDEM: contact laws and analytical validation

Status: contact law implemented and verified against closed-form references; **not yet wired into the bonded-particle solver** (see *Limitations*).

## 1. Why this capability was missing

`tensordem.contact_tangential` provides normal restitution and Cundall-Strack sliding friction, but its `TangentialContactConfig.rolling_mu` field was an unused placeholder: there was no rolling-resistance torque anywhere in the package. For DEM ice work this is the dominant missing contact mechanism, because ice blocks and floes are near-spherical/lumpy bodies that, without a rolling resistance, roll far too freely. That biases two quantities the platform reports: the angle of repose of a rubble field and the resistance of a broken-ice channel.

`docs/dem3d_mechanics_benchmark_protocol.md` and `docs/dem3d_specimen_response_benchmarks.md` both list rolling resistance as not implied by the existing benchmark suite, so this document closes a recorded verification gap rather than adding a new solver feature.

## 2. Method

### 2.1 Contact laws (`src/tensordem/rolling_resistance.py`)

Let `omega_rel = omega_j - omega_i` be the relative angular velocity of a contact pair, `n` the unit normal from `i` to `j`, and `r_eff` the effective radius. Only the *rolling* part `omega_roll = omega_rel - (omega_rel.n) n` is resisted; drilling (spin) friction about `n` is out of scope. The saturated rolling limit is `M_lim = mu_r |F_n| r_eff`.

| model | torque applied to `j` | character |
|---|---|---|
| `cdt` | `-mu_r |F_n| r_eff · omega_hat` | memoryless, always saturated |
| `viscous` | `-(mu_r |F_n| r_eff / omega_ref) · omega_roll` | strictly dissipative, unbounded above `omega_ref` |
| `epsd` | `M^{n+1} = clamp(M^n - k_r omega_roll dt, |M| <= M_lim)` | elastic-plastic spring, supports static equilibrium |

Model families follow Ai et al. (2011, *Powder Technology* 206:269-282); the elastic-plastic branch follows Iwashita & Oda (1998) and Jiang et al. (2005).

### 2.2 Verification harness

A single solid sphere is integrated with a normal spring-dashpot contact, an incremental Cundall-Strack tangential spring driven by the *contact-point* slip velocity `v + R omega`, and the rolling torque above. N independent inclines run as one batched tensor loop.

| parameter | value |
|---|---|
| sphere radius | 0.01 m |
| normal stiffness `k_n` | 2e+04 N/m |
| time step `dt` | 0.0001 s |
| run duration | 0.1 s |
| acceleration fit window | `t >= 0.03` s |
| sliding friction `mu` | 0.9 |
| rolling stiffness rule | `k_r = 2.25 * mu_r * k_n * R^2` |

Two closed-form references are used, selected by the friction-cone condition `pure_rolling_feasible()`:

* pure rolling: `a = g (sin t - mu_r cos t) / (1 + I/mR^2)`, `I/mR^2 = 2/5`;
* steady sliding: `a = g (sin t - mu cos t)`;
* static threshold: `theta_c = arctan(mu_r)`.

## 3. Results

Environment: Python 3.11.6, torch 2.7.1, cpu.

### 3.1 Acceleration parity (rolling and sliding regimes)

| mu_r | theta (deg) | theta/theta_c | regime | a simulated (m/s^2) | a analytic (m/s^2) | rel. error | fit R^2 |
|---|---|---|---|---|---|---|---|
| 0.05 | 3.578 | 1.25 | rolling | 0.0876169 | 0.0876235 | 7.52e-03 % | 0.99979 |
| 0.05 | 4.2936 | 1.5 | rolling | 0.175222 | 0.175233 | 6.27e-03 % | 0.99987 |
| 0.05 | 5.7248 | 2 | rolling | 0.350341 | 0.350357 | 4.57e-03 % | 0.99993 |
| 0.05 | 8.5872 | 3 | rolling | 0.699823 | 0.69984 | 2.43e-03 % | 0.99998 |
| 0.05 | 12 | 4.19 | rolling | 1.11415 | 1.11417 | 1.72e-03 % | 0.99999 |
| 0.05 | 18 | 6.29 | rolling | 1.83209 | 1.83212 | 1.53e-03 % | 0.99999 |
| 0.1 | 7.1382 | 1.25 | rolling | 0.175431 | 0.175451 | 1.11e-02 % | 0.99953 |
| 0.1 | 8.5659 | 1.5 | rolling | 0.350776 | 0.350793 | 4.88e-03 % | 0.99989 |
| 0.1 | 11.421 | 2 | rolling | 0.700701 | 0.700714 | 1.94e-03 % | 0.99998 |
| 0.1 | 12 | 2.1 | rolling | 0.771452 | 0.771465 | 1.71e-03 % | 0.99999 |
| 0.1 | 17.132 | 3 | rolling | 1.39446 | 1.39447 | 1.18e-03 % | 1 |
| 0.1 | 18 | 3.15 | rolling | 1.49889 | 1.49891 | 1.17e-03 % | 1 |
| 0.2 | 12 | 1.06 | rolling | 0.0860558 | 0.0860629 | 8.29e-03 % | 0.99682 |
| 0.2 | 14.137 | 1.25 | rolling | 0.352499 | 0.3525 | 3.33e-04 % | 0.99988 |
| 0.2 | 16.965 | 1.5 | rolling | 0.704143 | 0.704141 | 2.66e-04 % | 0.99999 |
| 0.2 | 18 | 1.59 | rolling | 0.83249 | 0.832488 | 1.68e-04 % | 1 |
| 0.2 | 22.62 | 2 | rolling | 1.40143 | 1.40143 | 2.03e-04 % | 1 |
| 0.2 | 33.93 | 3 | rolling | 2.7484 | 2.74843 | 9.86e-04 % | 0.99999 |
| 0.3 | 18 | 1.08 | rolling | 0.166086 | 0.16607 | 1.00e-02 % | 0.99906 |
| 0.3 | 20.874 | 1.25 | rolling | 0.532597 | 0.532579 | 3.40e-03 % | 0.99995 |
| 0.3 | 25.049 | 1.5 | rolling | 1.06235 | 1.06233 | 1.37e-03 % | 0.99999 |
| 0.3 | 33.398 | 2 | rolling | 2.10214 | 2.10214 | 1.90e-04 % | 0.99999 |
| 0.3 | 50.098 | 3 | rolling | 4.02692 | 4.02698 | 1.32e-03 % | 0.99999 |
| 0.5 | 33.206 | 1.25 | rolling | 0.906088 | 0.906048 | 4.40e-03 % | 0.9999 |
| 0.5 | 39.848 | 1.5 | rolling | 1.79995 | 1.79994 | 9.70e-04 % | 0.99997 |
| 0.5 | 53.13 | 2 | rolling | 3.50359 | 3.50357 | 5.86e-04 % | 0.99998 |
| 0.5 | 79.695 | 3 | sliding | 8.07238 | 8.07238 | 9.46e-13 % | 1 |

Worst relative error over 27 supercritical rows (both regimes): **0.0111 %**; minimum fit R^2 = 0.99682. Rows whose analytical acceleration is below 0.05 m/s^2 are excluded from the relative error statistics because they are dominated by the settling transient.

### 3.2 Static threshold versus arctan(mu_r)

| mu_r | theta_c analytic (deg) | measured transition (deg) | grid cell (deg) | rel. error |
|---|---|---|---|---|
| 0.05 | 2.8624 | 2.8982 | 0.0716 | 1.25 % |
| 0.1 | 5.7106 | 5.782 | 0.143 | 1.25 % |
| 0.2 | 11.31 | 11.451 | 0.283 | 1.25 % |
| 0.3 | 16.699 | 16.908 | 0.417 | 1.25 % |
| 0.5 | 26.565 | 26.897 | 0.664 | 1.25 % |

Bisection refinement at mu_r = 0.2: transition at 11.3364 deg versus 11.3099 deg analytical, relative error **0.234 %**.

### 3.3 Model character on the same inclines

| case | model | theta/theta_c | late-window a (m/s^2) | analytic a (m/s^2) | displacement / R | end speed (m/s) |
|---|---|---|---|---|---|---|
| subcritical | cdt | 0.85 | -6.721e-05 | -- | 0.00125 | 0.000154 |
| subcritical | viscous | 0.85 | 0.01413 | -- | 0.0782 | 0.00837 |
| subcritical | epsd | 0.85 | 0.0003387 | -- | 0.000302 | 5.35e-05 |
| supercritical | cdt | 1.5 | 0.1752 | 0.1752 | 0.128 | 0.0211 |
| supercritical | viscous | 1.5 | 0.02505 | 0.1752 | 0.138 | 0.0148 |
| supercritical | epsd | 1.5 | 0.1752 | 0.1752 | 0.132 | 0.0215 |

Reading: at the subcritical angle EPSD stays at rest (no sustained acceleration, displacement at the numerical floor), whereas `cdt` and `viscous` creep downhill because they have no static branch. At the supercritical angle `cdt` and `epsd` reproduce the analytical rolling acceleration, while `viscous` brakes much more strongly and settles at a terminal speed instead of a constant acceleration -- a model-character difference, not a numerical error.

### 3.4 Torque laws (law level, |F_n| = 1.5 N, mu_r = 0.05)

| |omega| (rad/s) | limit (N m) | cdt (N m) | viscous (N m) | epsd (N m) |
|---|---|---|---|---|
| 0.05 | 0.00075 | 0.00075 | 3.75e-05 | 0.00045 |
| 0.2 | 0.00075 | 0.00075 | 0.00015 | 0.00075 |
| 0.5 | 0.00075 | 0.00075 | 0.000375 | 0.00075 |
| 1 | 0.00075 | 0.00075 | 0.00075 | 0.00075 |
| 2 | 0.00075 | 0.00075 | 0.0015 | 0.00075 |
| 5 | 0.00075 | 0.00075 | 0.00375 | 0.00075 |
| 10 | 0.00075 | 0.00075 | 0.0075 | 0.00075 |

The viscous branch exceeds the rolling limit for `|omega| > omega_ref = 1` rad/s; it is intended as a dissipation term, not as a rolling-friction coefficient.

### 3.5 Figures

| figure | content |
|---|---|
| `rolling_resistance_validation/fig_torque_laws.png` | |M| versus rolling rate for the three models with the limit line |
| `rolling_resistance_validation/fig_acceleration_parity.png` | simulated versus analytical acceleration with the +-3 % band |
| `rolling_resistance_validation/fig_threshold.png` | displacement after a fixed run versus incline angle, per mu_r |
| `rolling_resistance_validation/fig_history.png` | supercritical run: speed versus -omega R, contact slip, rolling torque |

## 4. Error judgement against the 3 % criterion

| key quantity | reference | error | below 3 %? |
|---|---|---|---|
| downhill acceleration on an incline | `g(sin t - mu_r cos t)/(1+I/mR^2)` or `g(sin t - mu cos t)` | 0.0111 % (max of 27 rows) | yes |
| static transition angle | `arctan(mu_r)` | 0.234 % (bisection), <= half a grid cell on the sweep | yes |
| rolling time-scale ratio `omega R / v` | `-1` for rolling without slip | 0.809 % (first row) | yes |

## 5. Limitations and honest caveats

* The validation harness is an independent rigid-body integration of a single sphere, not the bonded-particle solver: IceDEM3D tracks no particle rotation, so wiring rolling resistance into DEM3D requires adding angular degrees of freedom (not part of this change).
* Only the rolling component of the relative spin is resisted; drilling/spin friction about the contact normal is out of scope.
* k_r is a modelling choice (scaled with k_n), not a measured material constant; no experimental ice repose-angle calibration is included because no such dataset exists in this repository.
* The CDT and viscous branches have no static branch, so a resting particle creeps; only EPSD reproduces a true static equilibrium.
* The validation harness is not a granular-assembly benchmark: no angle-of-repose or channel-resistance calibration is claimed, because the repository holds no experimental ice data to compare against.
* `cdt`/`viscous` cannot hold a static contact; they must not be used for repose-angle work without an explicit static regularisation.

## 6. Reproduction

```bash
python scripts/validate_rolling_resistance.py   # JSON/CSV/PNG into docs/rolling_resistance_validation/
python scripts/report_rolling_resistance.py     # regenerate this document from those artifacts
python -m pytest tests/test_rolling_resistance.py -q
```

Generated by `scripts/report_rolling_resistance.py` on CPython 3.11.6 with torch 2.7.1.
