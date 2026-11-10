# P272W2-N11 — Ice-structure contact with thermal coupling: from a constant
# friction coefficient to a physically self-consistent one

**Repository** TensorDEM · **module** `src/tensordem/ice_friction_thermo/` ·
**artifact** `artifacts/ice_friction_thermo/acceptance.json`
(SHA-256 `744da05a6e25fed4806fb4bb987cbe54bfcff5d86a99625dd40b6e8ea94c3da5`) ·
**driver** `benchmarks/ice_friction_thermo_evidence.py`

---

## 1. The case (physics, geometry, parameters)

An ice slab slides on a steel plane — the laboratory geometry of an ice-friction
test and the loading a DEM ice-structure contact carries.  The existing chain
(`tensordem.ice_channel`, `tensordem.ice_steel_friction`) prescribes the
coefficient of friction, either as the hard-coded `mu = 0.10` or as a fitted
function of `(v, T, p)`.  Neither knows that:

1. the contact heats itself: `Q_f = mu F_n v_slip`;
2. the heat diffuses into the ice and the steel, changing the contact
   temperature `T_contact(t)`;
3. the melting point of ice **falls with pressure**,
   `dT_m/dp = T dv/L_f = -0.07402 K/MPa`
   (Clausius-Clapeyron, derived here from `rho_ice`, `rho_water` and `L_f`, not
   fitted; 4.2 % from the tabulated
   0.0072 K/atm);
4. the water produced when it does melt removes the friction that produced it.

This node makes `mu` a *solved state* of the contact instead of an input, and
measures the consequence on friction, heat and melt production.

**Parameters.** Two friction parameters are fitted — and only two:
`mu_ref = 0.09413`, `beta = 0.0263 1/K`,
`T_ref = -10 degC` — from the Barnes, Tabor & Walker (1971)
temperature sweep alone (`Barnes, Tabor & Walker (1971) ice friction vs temperature`,
6 rows, weighted least squares, RMS residual
0.05571).  Everything else is material
data: ice `k=2.2 W/m/K`, `rho=917 kg/m^3`, `c_p=2100 J/kg/K`
(`alpha = 1.14e-06 m^2/s`,
`e = 2058 J/m^2/K/sqrt(s)`), steel
`k=45`, `rho=7850`, `c_p=470` (`e = 1.289e+04`),
`L_f = 3.34e5 J/kg`.  The Blok partition puts
`13.8 %` of the frictional heat into the ice —
86 % goes into the steel, which an ice-only thermal model would get wrong by a
factor ~7.

## 2. What was built

```
q = mu p_real v   ->  1D conduction (two columns)  ->  T_contact
                                   |
                     T_m(p_real) <= 0  (Clausius-Clapeyron)
                                   |
                    melt film h -> s = h/h_ref  ->  mu = mu_dry(T)[1 - s(1-r_wet)]
                                   |
                             back to q  (Picard loop)
```

* `clausius_clapeyron.py` — `dT_m/dp` from the Clapeyron relation, `T_m(p)`, the
  pressure-melting threshold `p = |T|/|dT_m/dp|`, the melt fraction, the Stefan
  melt rate, and the real (asperity) contact pressure `p_real = min(H, p_n/phi)`.
* `friction_law.py` — `mu_dry(T) = mu_0 exp(-beta (T - T_0))`, the melting
  softening `[1 - s(1-r_wet)]`, analytic derivatives, and the closed-form
  two-parameter fit.
* `conduction1d.py` — implicit finite-volume conductor for the ice and steel
  columns sharing a zero-capacity interface node.  Its response is **affine in
  the flux** (`T = T_a + Z q`), which is what reduces the coupled problem to a
  one-dimensional root.
* `selfconsistent.py` — the Picard loop (reports the iteration count and the
  residual), plus an independent bisection solver for the same residual.
* `sliding_case.py` — the time march: conductor response -> fixed point ->
  commit flux -> melt-film renewal `dh/dt = q_excess/(rho L) - h/t_c` -> energy
  ledger.
* `validation.py` — the holdout protocol, the sensitivity table, the acceptance
  report and its SHA-256.

**Two closures, kept separate on purpose.** The algebraic closure reads the melt
fraction off the temperature relative to `T_m(p)` (used against literature rows,
where no film history exists); the coupled solver uses the energy closure, where
`s` comes from the melt film actually produced.  Mixing them would hide which
assumption drives which number.

## 3. Results

### 3.1 Friction against independent literature (criterion C4)

Fitted on Barnes only, then **predicted** on 5 rows from
Riska (2017)/ISO 19906 and Poirier et al. (2011):

| condition | literature mu | predicted mu | error | T_contact [degC] | it | regime |
|---|---|---|---|---|---|---|
| v=0.05 m/s, T=-5.0 degC, p=0.3 MPa | 0.065 | 0.0819 | +26.0 % | -4.72 | 1 | dry |
| v=0.005 m/s, T=-20.0 degC, p=0.5 MPa | 0.130 | 0.1220 | -6.1 % | -19.87 | 1 | dry |
| v=0.02 m/s, T=-3.0 degC, p=0.1 MPa | 0.070 | 0.0780 | +11.4 % | -2.83 | 1 | dry |
| v=0.5 m/s, T=-15.0 degC, p=1 MPa | 0.055 | 0.1043 | +89.6 % | -13.89 | 2 | dry |
| v=0.2 m/s, T=-8.0 degC, p=0.5 MPa | 0.055 | 0.0879 | +59.9 % | -7.41 | 2 | dry |

`n within 30 % = 3/5`,
mean |error| 38.6 %,
max |error| 89.6 %.

The same rows scored with the constant coefficient the platform currently uses
(`mu = 0.10`):

| condition | literature mu | constant mu | error |
|---|---|---|---|
| v=0.05 m/s, T=-5.0 degC, p=0.3 MPa | 0.065 | 0.100 | +53.8 % |
| v=0.005 m/s, T=-20.0 degC, p=0.5 MPa | 0.130 | 0.100 | -23.1 % |
| v=0.02 m/s, T=-3.0 degC, p=0.1 MPa | 0.070 | 0.100 | +42.9 % |
| v=0.5 m/s, T=-15.0 degC, p=1 MPa | 0.055 | 0.100 | +81.8 % |
| v=0.2 m/s, T=-8.0 degC, p=0.5 MPa | 0.055 | 0.100 | +81.8 % |

`n within 30 % = 1/5`,
mean |error| 56.7 %.

**Honest FAIL, not hidden:** the two Poirier rows (0.5 m/s at -15 degC and
0.2 m/s at -8 degC) are over-predicted by 60-90 %.  The exponential temperature
law with a flash temperature cannot reproduce the collapse of ice friction at
0.2-0.5 m/s: that requires a *speed-dependent film*, i.e. the melt-film growth
competing with drainage, which is exactly the branch the model only reaches when
the interface is pinned at `T_m`.  Those two rows are reported as failures and
the regime column shows why (`dry`: the interface never reaches its pressure
melting point in the model).

### 3.2 The friction coefficient falls as the contact warms (C2)

| regime | v [m/s] | p_nominal [MPa] | mu (cold end) | mu (warm end) | drop | monotone |
|---|---|---|---|---|---|---|
| slow | 0.01 | 0.2 | 0.1583 | 0.0760 | 52.0 % | yes |
| fast | 1 | 1 | 0.1501 | 0.0624 | 58.4 % | yes |
| near-melting | 1 | 2 | 0.0820 | 0.0306 | 62.7 % | yes |

`3/3` strictly decreasing.  The `near-melting` row is
the one where the phase change does the work; the other two are the exponential
temperature law plus frictional heating.

### 3.3 The fixed point converges (C3)

| v [m/s] | p [MPa] | T [degC] | iterations | final \|dmu\|/mu | mu (Picard) | mu (bisection) | loop gain dT/dmu [K] | regime |
|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.2 | -10 | 1 | 3.73e-03 | 0.0938 | 0.0938 | 1.5 | dry |
| 0.1 | 0.5 | -10 | 2 | 1.39e-04 | 0.0930 | 0.0930 | 4.8 | dry |
| 1 | 1 | -5 | 2 | 1.06e-03 | 0.0800 | 0.0800 | 15.1 | dry |
| 2 | 2 | -2 | 6 | 9.55e-03 | 0.0592 | 0.0590 | 21.4 | pinned |
| 1 | 0.2 | -1 | 6 | 2.79e-03 | 0.0412 | 0.0412 | 15.1 | pressure |

Max 6 iterations (criterion < 10), max residual
9.55e-03 (criterion < 1e-2), and the two independent
solvers agree to 1.7e-11 relative.  The loop
gain printed in the last column is the physical reason the iteration is short:
`beta mu (dT/dmu)` well below 1.

### 3.4 The sliding case, with the energy ledger (C1, C5)

0.5 s of sliding at v = 1 m/s, p_nominal = 2 MPa, bulk ice at -2.5 degC
(the real pressure is the ice hardness, 20 MPa, so `T_m = -1.480 degC`):

| quantity | coupled | control (melt ignored by friction) |
|---|---|---|
| mu, start | 0.0744 | 0.07524 |
| mu, end | 0.0515 | 0.07524 |
| T_contact, end | -1.48 degC | -1.48 degC |
| melt fraction, end | 0.3322 | 0.4872 |
| film thickness, end | 3.32e-07 m | 4.87e-07 m |
| Q_f, end | 1.03e+04 W | 1.505e+04 W |
| friction work | 5.17e+05 J/m^2 | 7.52e+05 J/m^2 |
| sensible + drain | 1.039e+04 + 5.05e+05 J/m^2 | 1.039e+04 + 7.40e+05 J/m^2 |
| latent melted | 5.05e+05 J/m^2 | 7.40e+05 J/m^2 |
| energy closure error | 2.01e-12 | 1.38e-12 |
| iterations, max / mean | 4 / 3.00 | 2 / 2.00 |

The control keeps the melting *temperature cap* but removes the feedback of the
melt fraction on friction (`r_wet = 1`), so the difference is exactly what the
phase-change closure buys: **mu 0.07524 -> 0.0515**,
a 31.6 % reduction at
1 m/s, with 0.3322 of the interface melted.

`Q_f = F_friction v` is exact here (relative error 2.25e-16)
and is cross-checked through two chains (force x velocity and real-area flux x
real area) and against the DEM tangential limit
`tensordem.platen_friction.platen_friction_limit`.

### 3.5 Pressure melting: where Clausius-Clapeyron actually bites

Melting needs `p_real > |T|/|dT_m/dp|`:

| bulk temperature [degC] | pressure needed to melt [MPa] |
|---|---|
| -15 | 202.6 |
| -10 | 135.1 |
| -5 | 67.5 |
| -3 | 40.5 |
| -2 | 27.0 |
| -1 | 13.5 |
| -0.3 | 4.1 |

Sweeping the ice hardness (the only way to reach those pressures at engineering
loads of ~1 MPa) and the temperature:

* 12 of 28 grid points are in the
  pressure-melted regime, with a maximum coefficient reduction of
  95 %;
* the **Amontons check**: for a plastic contact `p_real` saturates at `H`, so
  `mu` is independent of the *nominal* pressure (spread
  0).  This is the honest negative result of
  the node: the classical pressure weakening of ice friction
  (Oksanen & Keinonen 1982, rows 0.05-2 MPa) **cannot** be explained by
  Clausius-Clapeyron melting alone — at those pressures `T_m` moves by less than
  0.15 K.  It must come from real-area growth / third-body shear, which this
  model does not contain and does not pretend to.

### 3.6 Numerical verification of the thermal solver

Against the Blok semi-infinite flash temperature
`T - T_bulk = 2 q sqrt(t/pi)/(e_ice + e_steel)`:

| dt [s] | finite volume [K] | Blok [K] | deviation | energy residual (rel.) |
|---|---|---|---|---|
| 2.00e-03 | 4.4909 | 4.7757 | -5.96 % | 4.0e-10 |
| 1.00e-03 | 5.2091 | 5.3394 | -2.44 % | 4.4e-10 |
| 2.50e-04 | 5.3074 | 5.3394 | -0.60 % | 2.1e-09 |

The deviation falls with `dt` (first-order scheme) and reaches
0.60 % at dt = 2.5e-04 s.
Steady state matches the parallel-column resistance
`q/(k_i/d_i + k_s/d_s)` to 6.36e-10 relative
(machine precision: the discretisation is exact for the linear profile).

### 3.7 What the un-fitted parameters cost

| parameter | value | holdout max error | wet-case mu | melt fraction | regime |
|---|---|---|---|---|---|
| contact_spot_radius_m | 2e-05 | 0.926 | 0.0627 | 0.175 | pressure |
| contact_spot_radius_m | 0.0001 | 0.896 | 0.0412 | 0.476 | pressure |
| contact_spot_radius_m | 0.0005 | 0.833 | 0.0126 | 0.877 | pressure |
| film_reference_m | 2e-07 | 0.896 | 0.0038 | 1.000 | pressure |
| film_reference_m | 1e-06 | 0.896 | 0.0412 | 0.476 | pressure |
| film_reference_m | 5e-06 | 0.896 | 0.0661 | 0.128 | pressure |
| hardness_mpa | 10 | 0.923 | 0.0664 | 0.104 | pinned |
| hardness_mpa | 20 | 0.896 | 0.0412 | 0.476 | pressure |
| hardness_mpa | 50 | 0.943 | 0.0040 | 1.000 | pressure |
| hardness_mpa | 100 | 0.937 | 0.0044 | 1.000 | pressure |

The dry holdout is **insensitive to the film parameter** (all three film rows
give the same error, `True`), but
the lubricated coefficient spans 0.003762 to
0.06636 across these sweeps — a factor
18.  The *wet*
regime is therefore a bounded estimate, not a prediction; only the dry branch is
validated against data.

## 4. Figures

* `fig_mu_vs_temperature.png` — the temperature law, the pressure melting point,
  calibration anchors, holdout rows and predictions.
* `fig_sliding_case.png` — `mu(t)`, `T_contact(t)` pinned at `T_m(p)`, `s(t)`,
  coupled vs control.
* `fig_pressure_melting.png` — `mu(T)` for four hardnesses and the
  Clausius-Clapeyron threshold curve.
* `fig_convergence.png` — residual of the fixed point per iteration.
* `fig_conductor_blok.png` — conductor vs Blok and the first-order convergence.
* `fig_sensitivity.png` — the wet-case coefficient vs the un-fitted parameters.

## 5. Caveats and limits

1. **Constant ice properties**: `k` and `c_p` vary ~5 % over -30..0 degC.
2. **Hardness model**: `p_real = min(H, p_n/phi)` with constant `H`; no
   strain-rate or temperature dependence of `H`.  This is why nominal-pressure
   weakening is absent (section 3.5).
3. **No drainage/refreezing physics beyond the renewal term**: drained water
   carries its latent heat out of the contact (reported separately); refreezing
   on cooling is not modelled.
4. **Interface-averaged temperature**: a real contact has a *distribution* of
   asperity flash temperatures; the onset of melting would then be smeared over
   the distribution width instead of being a sharp pin.
5. **Two un-fitted parameters dominate the wet branch**: the asperity spot
   radius (20-500 um) and the film scale `h_ref` (0.2-5 um).  Both are reported
   with their effect (3.7), neither is tuned to the validation data.
6. **Resolution requirement**: `dx <= 5e-5 m` and `dt <= a/v` to keep the
   interface impedance within ~11 % of the Blok value.  At `dx = 2.5e-4 m` the
   impedance is 46 % high and the contact temperature is over-predicted
   (measured 3.2 K instead of 1.2 K for the same flux).
7. **Not yet inside the DEM time loop**: `solve_contact_state` returns the state
   a DEM step needs (`mu`, `F_friction`, `Q_f`, `T_contact`, melt fraction); the
   tie-in is verified against the existing Cundall-Strack limit
   (`platen_friction_limit`), not by re-running `dem3d` with a thermal contact.

## 6. Error assessment (explicit)

| quantity | reference | error | < 3 % ? |
|---|---|---|---|
| Steady interface rise | `q/(k_i/d_i + k_s/d_s)` | 6.36e-10 | **yes** |
| Early-time rise, dt = 2.5e-04 s | Blok semi-infinite | 0.60 % | **yes** |
| Early-time rise, dt = 2.0e-03 s | Blok semi-infinite | 5.96 % | **no** (first-order in dt, see 3.6) |
| Energy closure of the sliding run | `int q dt` | 2.01e-12 | **yes** |
| `Q_f = F_f v` identity | algebra | 2.25e-16 | **yes** |
| Clapeyron slope | tabulated 0.0072 K/atm | 4.17 % | **no** (constant `dv`, `L`; the difference is the temperature dependence of the properties) |
| Melting point at 20 MPa | `-0.0740 K/MPa x 20 MPa` | exact (same formula) | n/a (definition) |
| Pressure-melting threshold at -5 degC | 67.5 MPa (hand arithmetic from the derived slope) | < 0.1 % | **yes** |
| `mu` vs independent literature (3 rows) | Riska / Poirier | 26.0 %, 6.1 %, 11.4 % | **no** (the campaign gate is 30 %, not 3 %; the 30 % gate is met on 3/5 rows) |
| `mu` vs independent literature (2 rows) | Poirier high-speed | 90 %, 60 % | **no — honest FAIL** (section 3.1) |

## 7. Reproduce

```bash
PYTHONPATH=src python benchmarks/ice_friction_thermo_evidence.py   # artifact + figures
PYTHONPATH=src python benchmarks/ice_friction_thermo_report.py     # this document
PYTHONPATH=src python -m pytest tests/test_ice_friction_thermo_*.py -q
```
