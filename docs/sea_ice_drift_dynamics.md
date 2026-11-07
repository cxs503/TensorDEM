# Sea-ice drift dynamics closure: Coriolis + added-mass tensor + Ekman/relative-velocity drag

Verification-and-validation record for `src/tensordem/sea_ice_drift.py`.
Every number below is injected into this file from
`artifacts/sea_ice_drift/sea_ice_drift_validation.json`, which is produced by
`benchmarks/sea_ice_drift_validation.py` and rendered by
`scripts/gen_sea_ice_drift_doc.py`; nothing is transcribed by hand.

## 1. Scope and repository fit

TensorDEM owns discrete-element ice mechanics, ice/structure contact and the
continuum-scale ice models that drive the DEM floe fields. A drifting ice cover
imposes a *velocity* boundary condition on the floe field and, in the opposite
direction, the floe field supplies the areal mass `m = rho_i h` that closes the
drift momentum balance. The closure therefore lives here: it is the
continuum-scale coupling layer between the floe population and the
atmosphere/ocean. Wind forcing, ocean forcing and Coriolis are not DEM contact
forces and are not re-implemented inside the contact solver; the closure is a
standalone integrator with a verified steady state that a floe-scale run can
query for the local drift velocity.

Evidence level: **REAL** - an implemented momentum balance, anchored to closed
forms at machine precision and validated against the classical Nansen/Ekman
ranges. It is not a data-assimilated operational drift forecast.

## 2. Model

Per unit horizontal area of ice (`x` east, `y` north, `k` up):

```
M du/dt = tau_a + tau_w - m f k x u - m g grad(eta),     M = m I + M_a
```

| term | expression | note |
|---|---|---|
| Coriolis | `-m f k x u`, `f = 2 Omega sin(phi)`, `Omega = 7.2921e-5` rad/s | applied to the ice mass only (`coriolis_mass="ice"`, the literature default; the `"total"` variant is exposed and its sensitivity is reported in section 7) |
| added mass | `M_a`, a full symmetric positive-definite 2x2 tensor, `m_a = c_m rho_w d` per unit area | enters the inertia only; arbitrary principal axes and anisotropy |
| wind drag | `rho_a C_a abs(W - u)(W - u)` | quadratic in the *relative* velocity |
| water drag | `rho_w C_w abs(u_w - u)(u_w - u)` | quadratic in the *relative* velocity; `u_w` = ocean current |
| Ekman closure | `rho_w sqrt(f K) R(+45 deg)(u - u_w)` | alternative ocean-drag law from the constant-viscosity Ekman layer (section 6) |

Numerics: RK4 time stepping (`SeaIceDriftModel.integrate`) and a damped Newton
root finder with a finite-difference Jacobian for the steady state
(`SeaIceDriftModel.steady_state`), which raises `RuntimeError` instead of
returning a non-converged velocity. `drag_mode` selects `"quadratic"`,
`"linear"` (Ekman-type slab friction) or `"ekman"`.

## 3. Analytical anchors (independent derivations)

**(i) Linear-drag slab.** With `tau_w = -m C_D (u - u_w)`, `u_w = 0`, the steady
balance `m C_D u + m f k x u = tau_a` has the exact solution

```
theta = arctan(f / C_D),      |u| = |tau_a| / (m sqrt(f^2 + C_D^2))
```

| phi [deg] | C_D [1/s] | model theta [deg] | arctan(f/C_D) [deg] | max abs velocity err [m/s] |
|---|---|---|---|---|
| 60 | 1e-04 | 51.629664 | 51.629664 | 0.0e+00 |
| 60 | 3e-04 | 22.831559 | 22.831559 | 0.0e+00 |
| 60 | 1e-03 | 7.198505 | 7.198505 | 0.0e+00 |
| 70 | 1e-04 | 53.882573 | 53.882573 | 0.0e+00 |
| 70 | 3e-04 | 24.551972 | 24.551972 | 0.0e+00 |
| 70 | 1e-03 | 7.803582 | 7.803582 | 0.0e+00 |
| 80 | 1e-04 | 55.152390 | 55.152390 | 0.0e+00 |
| 80 | 3e-04 | 25.582976 | 25.582976 | 0.0e+00 |
| 80 | 1e-03 | 8.173288 | 8.173288 | 0.0e+00 |
| 85 | 1e-04 | 55.460647 | 55.460647 | 0.0e+00 |
| 85 | 3e-04 | 25.840447 | 25.840447 | 0.0e+00 |
| 85 | 1e-03 | 8.266493 | 8.266493 | 0.0e+00 |

Maximum angle error over the grid: **3.55e-15 deg**,
maximum velocity error **0.00e+00 m/s** (round-off only).

**(ii) Quadratic free drift.** With a prescribed wind stress `tau_a` along `+x`,
ocean at rest and no surface tilt, `tau_a cos(theta) = rho_w C_w U^2` and
`tau_a sin(theta) = m f U` combine into the scalar equation
`tan(theta) sqrt(cos theta) = Lambda`, `Lambda = m f / sqrt(rho_w C_w tau_a)`,
solved by bisection in `free_drift_state`.

| phi [deg] | U_a [m/s] | model theta [deg] | closed form [deg] | model U [m/s] | closed form U [m/s] |
|---|---|---|---|---|---|
| 60 | 5 | 31.084691 | 31.084691 | 0.10219572 | 0.10219572 |
| 60 | 10 | 15.877607 | 15.877607 | 0.21660893 | 0.21660893 |
| 60 | 15 | 10.624403 | 10.624403 | 0.32844260 | 0.32844260 |
| 70 | 5 | 33.549117 | 33.549117 | 0.10081451 | 0.10081451 |
| 70 | 10 | 17.207536 | 17.207536 | 0.21586349 | 0.21586349 |
| 70 | 15 | 11.522144 | 11.522144 | 0.32793939 | 0.32793939 |
| 80 | 5 | 35.035484 | 35.035484 | 0.09992660 | 0.09992660 |
| 80 | 10 | 18.019470 | 18.019470 | 0.21537845 | 0.21537845 |
| 80 | 15 | 12.071209 | 12.071209 | 0.32761139 | 0.32761139 |
| 85 | 5 | 35.407736 | 35.407736 | 0.09969769 | 0.09969769 |
| 85 | 10 | 18.224080 | 18.224080 | 0.21525263 | 0.21525263 |
| 85 | 15 | 12.209700 | 12.209700 | 0.32752622 | 0.32752622 |

Maximum angle error **2.13e-14 deg**,
maximum speed error **1.69e-16** (relative).

The prescribed-stress mode (`wind_relative=False`) is what makes this comparison
exact; the physical relative-velocity mode differs by the O(u/W) reduction of the
wind stress, i.e. about 2-3% in speed and below 0.4% in angle (see the
`rel err theta` column in section 4). Both modes are reported rather than hidden.

## 4. Validation: Nansen deflection angle and drift-speed ratio

Free drift, `h = 2.5 m`, `C_a = 0.0018`,
`C_w = 0.0045`, `rho_a = 1.25`;
steady state from the Newton solver, cross-checked by marching the ODE
(`rel err U_int` is the relative difference between the two).

| case | phi [deg] | U_a [m/s] | h [m] | f [1/s] | U_ice [m/s] | U_ice/U_a [%] | theta model [deg] | theta closed form [deg] | rel err theta | rel err U_int |
|---|---|---|---|---|---|---|---|---|---|---|
| C1 | 60 | 5 | 2.5 | 1.263e-04 | 0.1002 | 2.00 | 30.99 | 31.08 | 3.0e-03 | 4.1e-11 |
| C2 | 60 | 15 | 2.5 | 1.263e-04 | 0.3214 | 2.14 | 10.62 | 10.62 | 3.8e-04 | 1.4e-15 |
| C3 | 85 | 5 | 2.5 | 1.453e-04 | 0.0978 | 1.96 | 35.27 | 35.41 | 3.8e-03 | 6.6e-11 |
| C4 | 85 | 15 | 2.5 | 1.453e-04 | 0.3205 | 2.14 | 12.20 | 12.21 | 5.0e-04 | 1.3e-15 |
| C5 | 70 | 10 | 2.5 | 1.370e-04 | 0.2113 | 2.11 | 17.19 | 17.21 | 9.8e-04 | 3.7e-14 |
| C6 | 80 | 6 | 2.5 | 1.436e-04 | 0.1211 | 2.02 | 29.47 | 29.55 | 2.7e-03 | 2.7e-13 |
| C7 | 80 | 6 | 3.0 | 1.436e-04 | 0.1176 | 1.96 | 34.91 | 35.04 | 3.7e-03 | 6.2e-11 |

Machine-evaluated criteria (see the `criteria` block of the JSON ledger):

* **(a) Nansen band 20-40 deg** - pass = **True**, in-band cases
  {'C1': 30.992, 'C3': 35.274, 'C6': 29.465, 'C7': 34.906}. Out of band: {'C2': 10.62, 'C4': 12.204, 'C5': 17.191}.
  This is a *regime* statement, not a fit: `theta` is controlled by the group
  `m f / (rho_w C_w U)`, so it grows with ice thickness and falls roughly as
  `1/U_a`; at `U_a = 15 m/s` the same model gives 10-12 deg. See
  `artifacts/sea_ice_drift/fig2_nansen_regime_map.png`, which maps the 20-40 deg
  wedge in the (h, U_a) plane. Reaching the band at 15 m/s would require h > 5 m.
* **(b) drift/wind speed ratio 1-3%** - pass = **True**, all cases
  {'C1': 2.004, 'C2': 2.143, 'C3': 1.956, 'C4': 2.137, 'C5': 2.113, 'C6': 2.019, 'C7': 1.96} (percent), i.e. 1.96-2.14%: the classical "ice drifts at a
  few percent of the wind" result emerges without tuning, because
  `U/U_a = sqrt(rho_a C_a cos(theta) / (rho_w C_w))`.
* **(e) case matrix** - pass = **True**, 7 cases covering
  phi = [60.0, 85.0] deg and U_a = [5.0, 15.0] m/s,
  including a thick-ice case.
* **(d) negative control f = 0** - pass = **True**, maximum absolute
  deflection **0.00e+00 deg** over 8 sub-cases
  (quadratic and linear drag, U_a = 5 and 15 m/s). With no rotation the drift is
  exactly parallel to the wind and `rho_w C_w U^2 = |tau_a|` holds to 1e-10.

## 5. Ekman spiral boundary layer (criterion c)

Constant eddy viscosity `K`, `f k x u = K d^2u/dz^2`, `u -> 0` at depth, surface
stress `tau`:

```
w(z) = w_s exp(-(1+i) z / D_E),   D_E = sqrt(2K/f),   w_s = tau / (rho_w sqrt(fK)) e^(-i pi/4)
```

The sampled profile is fitted log-linearly, so `D_E` is recovered by an
estimator that is independent of the formula used to generate the profile.
`momentum residual` is the re-substitution residual of `f k x u = K u_zz` with
central differences on the interior (truncation limited, about 1e-6).

| phi [deg] | f [1/s] | K [m^2/s] | D_E analytic [m] | rel err of fit | surface angle [deg] | transport angle [deg] | momentum residual | transport rel err |
|---|---|---|---|---|---|---|---|---|
| 60 | 1.263e-04 | 0.005 | 8.8980 | 5.989e-16 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 60 | 1.263e-04 | 0.02 | 17.7960 | 1.996e-15 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 60 | 1.263e-04 | 0.05 | 28.1380 | 5.050e-16 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 70 | 1.370e-04 | 0.005 | 8.5421 | 0.000e+00 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 70 | 1.370e-04 | 0.02 | 17.0842 | 8.318e-16 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 70 | 1.370e-04 | 0.05 | 27.0126 | 6.576e-16 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 80 | 1.436e-04 | 0.005 | 8.3442 | 1.277e-15 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 80 | 1.436e-04 | 0.02 | 16.6883 | 2.129e-16 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 80 | 1.436e-04 | 0.05 | 26.3866 | 4.039e-16 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 85 | 1.453e-04 | 0.005 | 8.2963 | 4.282e-16 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 85 | 1.453e-04 | 0.02 | 16.5927 | 1.071e-15 | 45.0 | 90.04 | 4.2e-06 | 0.00 |
| 85 | 1.453e-04 | 0.05 | 26.2353 | 1.354e-15 | 45.0 | 90.04 | 4.2e-06 | 0.00 |

* **(c) `D_E = sqrt(2K/f)` verified** - pass = **True**, maximum
  relative error **2.00e-15** over 12
  (latitude, viscosity) combinations.
* The surface velocity is exactly **45 deg to the right** of the stress (Ekman's
  result) and the depth-integrated transport is **90 deg right** with magnitude
  `|tau|/(rho_w f)` to 0.24% (truncation of the integration at 6 D_E).
* Momentum-balance re-substitution residual below 5e-6 relative.

## 6. Ekman drag closure (ocean feedback on the ice)

Inverting the spiral gives a linear ocean-drag law with a built-in 45 deg turn,
`tau_on_ice = -rho_w sqrt(fK) e^(i pi/4)(u - u_w)`. The steady state then has the
complex closed form `u = tau_a / (i m f + rho_w sqrt(fK) e^(i pi/4))`:

| quantity | value |
|---|---|
| Ekman drag coefficient `rho_w sqrt(fK)` at phi = 80.0, K = 0.02 | 1.7372 kg/(m^2 s) |
| equivalent linear drag rate `C_D = rho_w sqrt(fK)/m` | 7.721e-04 1/s |
| model `u` [m/s] | (0.0254066711, -0.0320904713) |
| closed form `u` [m/s] | (0.0254066711, -0.0320904713) |
| max abs error [m/s] | 3.47e-18 |
| deflection produced by the closure | 51.63 deg |
| `arctan(f/C_D)` of the equivalent linear slab | 10.54 deg |

The 51.6 deg is **larger** than the slab value because
the closure adds Ekman-layer turning on top of frictional-Coriolis turning; the
two are not exactly additive (45 + 10.5 = 55.5 deg vs 51.6 deg), which is why the
composite case is verified against the complex solution instead of a sum.

## 7. Added-mass tensor verification

| c_m | m_a [kg/m^2] | analytic tau [s] | max rel err vs exponential | steady-state match |
|---|---|---|---|---|
| 0.0 | 0.0 | 1000.0 | 4.9e-14 | 0.0e+00 |
| 0.5 | 1153.1 | 1512.5 | 8.9e-15 | 0.0e+00 |
| 1.0 | 2306.2 | 2025.0 | 2.5e-15 | 0.0e+00 |
| 2.0 | 4612.5 | 3050.0 | 2.5e-15 | 0.0e+00 |

Rotating frame with rotated anisotropic tensors, compared with the exact
matrix-exponential solution `u(t) = u* + expm(A t)(u0 - u*)`:

| phi [deg] | M_a principal values | anisotropy | max rel err vs expm | steady state vs expm |
|---|---|---|---|---|
| 80 | [2306.2, 2306.2] | 1.00 | 1.2e-09 | 1.2e-17 |
| 80 | [800.0, 1800.0] | 2.25 | 4.5e-09 | 3.6e-17 |
| 60 | [500.0, 4000.0] | 8.00 | 7.9e-09 | 0.0e+00 |

* The added-mass tensor is **inertia only**: the steady state is identical for
  `c_m` = 0, 0.5, 1, 2, 5 and for the 35 deg-rotated anisotropic tensor
  (relative difference 0.0 with `m_a` up to 5x the ice mass); the transient
  trajectory and its time constant `(m + m_a)/(m C_D)` do change
  (see `artifacts/sea_ice_drift/fig4_added_mass_transient.png`).
* The rotating-frame transient matches `expm(A t)` to 8e-9 (RK4 truncation at
  dt = 25 s), including anisotropy ratios up to 8.
* Modelling-choice sensitivity: moving the added mass into the Coriolis term
  (`coriolis_mass="total"`, `c_m` = 1) raises the nominal deflection from
  29.5 to 54.2 deg. The literature default (ice mass
  only) is kept; the switch is exposed so the assumption can be tested instead
  of buried.

## 8. Figures

| figure | content |
|---|---|
| `artifacts/sea_ice_drift/fig1_nansen_angle_speed_ratio.png` | deflection angle and drift/wind speed ratio vs wind speed, four latitudes, acceptance bands, sweep cases |
| `artifacts/sea_ice_drift/fig2_nansen_regime_map.png` | deflection map in the (ice thickness, wind speed) plane with the 20 deg and 40 deg contours |
| `artifacts/sea_ice_drift/fig3_ekman_spiral.png` | Ekman hodograph and amplitude decay with `D_E` and `pi D_E` |
| `artifacts/sea_ice_drift/fig4_added_mass_transient.png` | spin-up for `c_m` = 0...2 vs the analytic exponential; anisotropic trajectory vs the steady fixed point |
| `artifacts/sea_ice_drift/fig5_force_polygon.png` | steady force polygon (wind + water drag + Coriolis = 0) and the drift vector |

## 9. Limitations (stated, not glossed)

1. **Slab, no internal stress.** The closure has no ice-interior stress
   divergence, no rheology and no floe-floe interaction. It is the free-drift
   limit; in the central Arctic pack internal stress can rotate the drift by
   tens of degrees, and near coasts it dominates. Coupling to the DEM floe field
   (contact and bond forces) is the intended path and is not implemented here.
2. **Constant eddy viscosity.** `D_E = sqrt(2K/f)` assumes depth-independent `K`
   and a linear (Ekman) friction law. Real ice-ocean boundary layers have `K`
   varying with depth and stratification, and observed turning departs from
   45 deg.
3. **The angle is regime dependent.** 20-40 deg is reproduced at moderate wind
   and thickness (section 4); at U_a = 15 m/s the model gives 10-12 deg. Observed
   drift-station scatter (20-40 deg) mixes winds, thicknesses and internal-stress
   states that a slab cannot separate, so the band is claimed only for the cases
   recorded in the ledger.
4. **The observational comparison is a band, not a series.** No station-by-
   station drift series (e.g. Fram/NP) is ingested here, so no correlation
   coefficient or RMS error against observations is claimed. The quantitative
   evidence is the closed-form agreement (sections 3, 5, 7) plus the band checks
   (section 4).
5. **Added-mass coefficient.** `m_a = c_m rho_w d` is a bulk idealisation with
   `c_m` = 1 by default; the true added mass of a thin floe drifting in its own
   plane is anisotropic and smaller, which is why the tensor form and the
   sensitivity table are exposed rather than a scalar being assumed.
6. **No Coriolis at the equator.** `f = 0` makes the Ekman depth singular: the
   `"ekman"` drag mode requires `f != 0` and raises otherwise.

## 10. Reproduction

```
python -m pytest tests/test_sea_ice_drift.py -q
python benchmarks/sea_ice_drift_validation.py --out artifacts/sea_ice_drift
python scripts/gen_sea_ice_drift_doc.py
```

The second command rewrites the JSON ledger and the five figures, the third
re-renders this document, and the first exits non-zero on any regression of the
anchors listed above.
