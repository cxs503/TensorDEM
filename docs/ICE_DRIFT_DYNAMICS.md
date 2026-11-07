# Sea-ice drift dynamics closure — Coriolis + added mass + relative-velocity drag (IN27-S6)

Module: `src/tensordem/ice_drift_dynamics.py` · tests: `tests/test_ice_drift_dynamics.py`
Artifacts: `artifacts/ice_drift_dynamics/` (acceptance JSON + 5 figures) · generator:
`scripts/ice_drift_dynamics_artifacts.py`

## 1. Case introduction

**Physical problem.** A sea-ice cover drifts freely under wind forcing in a
rotating ocean. The drift direction is not the wind direction: Nansen (1902)
observed the *Fram* ice drifting 20–40° to the right of the wind in the northern
hemisphere, and the wind factor (drift speed / wind speed) is only ~2 %. Both
facts follow from the horizontal momentum balance of the cover, closed by three
terms that are usually missing from DEM floe models: Earth rotation, the
virtual (added) mass of the water pushed aside, and quadratic drag acting on the
**relative** velocity.

Why in TensorDEM: TensorDEM owns floe-scale ice mechanics. `ice_floe_drift.py`
prescribes a constant driving force (no rotation, no relative-velocity drag) and
`floe_6dof.py` integrates attitude; neither closes the horizontal drift momentum.
This module provides that closure plus the analytic anchors (Nansen free drift,
Ekman spiral) needed to validate the terms.

**Momentum equation** (per unit ice area, northern-hemisphere tangent plane,
`x` along the wind, `y` 90° to the left):

```
(M I + M_a) dU/dt = -(M I + M_a) f (k x U)          Coriolis (+ virtual mass)
                    + rho_a C_a |U_a - U| (U_a - U)  wind stress
                    + rho_w C_w |U_w - U| (U_w - U)  ocean drag
```

with `M = rho_i h`, `M_a` the in-plane added-mass tensor, `f = 2 Omega sin(phi)`,
`k x U = (-v, u)`.

**Parameter table** (all SI; defaults of `DriftEnvironment` / `FloeInertia`):

| symbol | value | meaning |
|---|---|---|
| h | 1.5 m | ice thickness (mixed first/multi-year) |
| rho_i | 917 kg/m³ | ice density |
| rho_w | 1025 kg/m³ | sea-water density |
| rho_a | 1.30 kg/m³ | air density |
| C_a | 1.8e-3 | 10 m wind drag coefficient |
| C_w | 5.5e-3 | ice-underside water drag coefficient |
| c_a | 0.5 | horizontal virtual-mass coefficient (`M_a = c_a rho_w h`) |
| K | 0.05 m²/s | ice–ocean boundary layer eddy viscosity |
| phi | 60–85° N | latitude (f = 1.26e-4 … 1.45e-4 s⁻¹) |
| U_wind | 5–15 m/s | wind speed |

## 2. Method

1. **Coriolis.** `coriolis_parameter(phi) = 2 Omega sin(phi)`, sign carries the
   hemisphere; the term is added as `-A f (k x U)` with `A = M + M_a`
   (virtual-mass form; `A = M` is available as the classical convention).
2. **Added-mass tensor.** `added_mass_tensor(a, rho_w)` builds the diagonal
   in-plane tensor of a thin disk, `m_a = (8/3) rho_w a³` (isotropic in plane, by
   symmetry), capped by `rho_w * D_E` per unit area when a floe is large compared
   with the boundary-layer depth (a steadily drifting floe cannot drag an
   arbitrarily deep column). In the 1-D free-drift closure the bulk form
   `M_a = c_a rho_w h` is used.
3. **Drag.** Quadratic, relative-velocity form for both stresses, evaluated at
   the actual ice velocity (not at rest).
4. **Steady state — closed form.** With a prescribed wind stress and still water
   the balance is linear in `(u, v)` for fixed `R = rho_w C_w |U|`:

   ```
   u = tau R/(R² + g²),  v = -tau g/(R² + g²),  g = A f
   |U|² (A_w² |U|² + g²) = tau²        (quartic, solved in closed form)
   =>  tan(theta) = g/(A_w |U|) = f / c_d,  c_d = rho_w C_w |U| / A
   ```

   so `theta = arctan(f/c_d)` is an **identity** (`c_d` is a drag *frequency*,
   [1/s], not a dimensionless coefficient — this is the correct reading of the
   "arctan(f/C_D)" path in the campaign spec). Verified to 7e-15 deg.
5. **Steady state — full nonlinear.** Newton solve of the relative-wind +
   prescribed-current system; Jacobian `-A f J - Σ rho C (|D| I + D D^T/|D|)`,
   symmetric part negative definite ⇒ unconditionally contracting. Residual at
   the solution `< 1e-10 N/m²`.
6. **Ekman closure (vertical).** The stress the ice exerts on the ocean,
   `tau_s = -tau_w(on ice)`, drives an Ekman layer `K w'' = i f w` with
   `w(z) = w0 exp(-(1+i) z / D_E)`, `D_E = sqrt(2K/f)`,
   `|w0| = |tau|/(rho sqrt(K f))`, surface veer 45° right of the stress, column
   transport exactly `-i tau/(rho f)` (90° right). Independently verified by a
   complex tridiagonal finite-difference solve (`solve_ekman_finite_difference`).
7. **Coupled free drift.** The ocean current felt by the floe is the Ekman
   surface current it generates itself; the two problems are solved by a damped
   fixed point (relaxation 0.5, convergence 1e-12, ~32–35 iterations).
8. **Transient.** RK4 integration of the full relative-velocity system with the
   virtual-mass inertia, used to verify that added mass rescales the spin-up
   without moving the steady state.

## 3. Results

### 3.1 Acceptance criteria (all machine-checked, from `acceptance.json`)

| # | criterion | requirement | result | pass |
|---|---|---|---|---|
| a | Nansen angle (Ekman-coupled) | 20° ≤ θ ≤ 40° | 28.94 – 38.88°, mean 32.6°, reference (75°N, 10 m/s) **30.12°** | ✅ |
| b | wind factor `U_ice/U_wind` | 1 % – 3 % | 2.00 % – 2.85 %, mean 2.43 % | ✅ |
| c | Ekman decay depth | numeric `D_E` vs `sqrt(2K/f)` | rel. error ≤ **2.4e-7**; surface veer **45.0000°**; transport rel. error ≤ 9.1e-5 | ✅ |
| d | no-Coriolis negative control | `f = 0` ⇒ θ = 0 | analytic −0.0°, Newton −0.0°, RK4 −0.0° (cross-wind component 0.0) | ✅ |
| e | case matrix | φ ∈ [60°, 85°] × U_wind ∈ [5, 15] m/s, ≥ 4 cases | **12 cases** (60/70/75/85° N × 5/10/15 m/s) | ✅ |

### 3.2 Case table (programmatically injected)

| φ [°N] | U_wind [m/s] | f [1e-4 s⁻¹] | τ_a [N/m²] | θ wind-only [°] | θ Ekman-coupled [°] | wind factor | U_ice [cm/s] | \|U_s\| [cm/s] | dir U_s [°] | D_E [m] | residual [N/m²] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 5 | 1.263 | 0.0585 | 26.50 | **35.75** | 2.07 % | 10.35 | 1.78 | −73.8 | 28.14 | 0 |
| 60 | 10 | 1.263 | 0.2340 | 13.45 | **29.06** | 2.49 % | 24.88 | 7.74 | −61.4 | 28.14 | 0 |
| 60 | 15 | 1.263 | 0.5265 | 8.99 | **28.94** | 2.85 % | 42.73 | 17.64 | −57.0 | 28.14 | 1.4e-17 |
| 70 | 5 | 1.371 | 0.0585 | 28.64 | **37.53** | 2.03 % | 10.16 | 1.67 | −75.9 | 27.01 | 1.6e-17 |
| 70 | 10 | 1.371 | 0.2340 | 14.58 | **29.84** | 2.45 % | 24.49 | 7.36 | −62.6 | 27.01 | 8.4e-17 |
| 70 | 15 | 1.371 | 0.5265 | 9.75 | **29.33** | 2.80 % | 41.96 | 16.85 | −57.8 | 27.01 | 1.4e-17 |
| 75 | 5 | 1.409 | 0.0585 | 29.40 | **38.16** | 2.02 % | 10.10 | 1.63 | −76.6 | 26.64 | 2.3e-17 |
| 75 | 10 | 1.409 | 0.2340 | 14.98 | **30.12** | 2.44 % | 24.36 | 7.24 | −63.0 | 26.64 | 0 |
| 75 | 15 | 1.409 | 0.5265 | 10.02 | **29.48** | 2.78 % | 41.70 | 16.59 | −58.1 | 26.64 | 2.2e-16 |
| 85 | 5 | 1.453 | 0.0585 | 30.27 | **38.88** | 2.00 % | 10.02 | 1.59 | −77.4 | 26.24 | 6.9e-18 |
| 85 | 10 | 1.453 | 0.2340 | 15.44 | **30.45** | 2.42 % | 24.21 | 7.10 | −63.5 | 26.24 | 5.7e-17 |
| 85 | 15 | 1.453 | 0.5265 | 10.33 | **29.66** | 2.76 % | 41.42 | 16.30 | −58.5 | 26.24 | 1.1e-16 |

Two model levels are reported on purpose:

* **wind-only** (`steady_drift_analytic`, classical Nansen free drift): θ = 9–30°,
  decreasing with wind speed as `arctan(f M_eff /(rho_w C_w |U|))` since
  `|U| ∝ U_wind`. This is the textbook answer for a floe drifting in still water.
* **Ekman-coupled** (`ekman_coupled_drift`, the operational closure): the ocean
  surface current generated by the ice drag is 45–77° to the right of the wind,
  the drag turns with it, and the ice follows: θ = 28.9–38.9° — inside the
  observed Nansen band for every case.

### 3.3 Ekman layer verification

| φ [°N] | K [m²/s] | D_E analytic [m] | D_E numeric (FD) [m] | rel. err | surface veer | transport analytic [m²/s] | transport rel. err | profile max rel. err |
|---|---|---|---|---|---|---|---|---|
| 60 | 0.02 | 17.7960 | 17.7960 | 2.4e-7 | 45.00004° | −1.8075 i | 9.1e-5 | 4.5e-5 |
| 60 | 0.05 | 28.1380 | 28.1380 | 2.4e-7 | 45.00004° | −1.8075 i | 9.1e-5 | 4.5e-5 |
| 75 | 0.02 | 16.8506 | 16.8506 | 2.4e-7 | 45.00004° | −1.6206 i | 9.1e-5 | 4.5e-5 |
| 75 | 0.05 | 26.6432 | 26.6432 | 2.4e-7 | 45.00004° | −1.6206 i | 9.1e-5 | 4.5e-5 |
| 85 | 0.02 | 16.5927 | 16.5927 | 2.4e-7 | 45.00004° | −1.5713 i | 9.1e-5 | 4.5e-5 |
| 85 | 0.05 | 26.2353 | 26.2353 | 2.4e-7 | 45.00004° | −1.5713 i | 9.1e-5 | 4.5e-5 |

(τ = 0.234 N/m² along +x; the transport is purely −y, i.e. 90° to the right of
the stress, and equals `tau/(rho f)` in every case.)

### 3.4 Figures

* `nansen_angle_vs_wind.png` — turning angle vs wind speed for φ = 60/70/75/85°N:
  Ekman-coupled (solid) vs wind-only (dotted) with the observed 20–40° band.
* `ekman_spiral.png` — hodograph, speed decay with the `D_E` marker, veering with
  depth, plus the FD solve overlaid.
* `wind_factor_vs_angle.png` — the twelve cases in the (wind factor, angle) plane
  against the observed acceptance box.
* `drift_transient.png` — spin-up for three virtual-mass coefficients (same
  steady state, different time scale) and the velocity-space trajectory with the
  `f = 0` control.
* `sensitivity_angle_map.png` — θ(h, K) map; white cells inside the 20–40° band.

## 4. Notes, caveats, applicability

* **The angle is not a `f`-only number.** `tan(theta) = f/c_d` shows the angle is
  set by the ratio of the rotation rate to the drag frequency. For still-water
  free drift that gives 9–30° over this matrix; the observed 20–40° requires the
  ocean response (Ekman current) to be included. Reporting only the wind-only
  number would under-predict the angle by ~15°.
* **Relative wind matters at the 2 % level.** The closed form prescribes the wind
  stress at `U = 0` (classical convention); the physical stress acts on
  `U_a − U`, so the Newton solution drifts 1.84–1.98 % slower. Both are reported;
  the acceptance numbers use the relative-velocity solution.
* **Added mass.** With Coriolis on `M` alone the steady state is *invariant* to
  `M_a` (verified to 1e-12) and only the spin-up changes; with the virtual-mass
  form (`A = M + M_a`, default) the angle increases and the drift speed
  decreases, which is why `c_a` is an explicit, documented parameter. The
  potential-flow disk tensor is available for floe-resolved work, capped by the
  Ekman depth.
* **The `K = 0.05 m²/s` choice.** The ice–ocean boundary layer eddy viscosity in
  the literature spans 0.01–0.05 m²/s; the sensitivity map shows θ from 24° (K =
  0.08, h = 1 m) to 51° (K = 0.02, h = 3 m). h = 1.5 m with K = 0.02–0.05 m²/s is
  the region reproducing both the observed angle and the 1–3 % wind factor.
* **Not modelled:** internal ice stress and floe-floe interaction, ocean mixed
  layer inertia, geostrophic currents, waves and form-drag anisotropy, lateral
  boundaries, ice-thickness distribution, melt/growth. Free drift is a
  small-`U_ice` regime; the model is not valid for grounded ice, ridged ice under
  convergence, or near-coastal boundaries.
* The `f ≤ 0` branch of `ekman_coupled_drift` raises rather than silently
  returning a meaningless spiral (uniform `K` Ekman theory is written for `f > 0`);
  `steady_drift_analytic` and `solve_steady_drift` handle either hemisphere and
  flip the deflection sign.

## 5. Error assessment against benchmarks

| quantity | model | reference / benchmark | error | < 3 % ? |
|---|---|---|---|---|
| Nansen angle, reference case (75°N, 10 m/s) | 30.12° | observed band 20–40° (Leppäranta, *The Drift of Sea Ice*, ch. 4; Nansen 1902) | inside band; +0.4 % vs band centre (30°) | yes vs band centre; the band itself is ±10° so no sharper claim is made |
| Nansen angle, worst case (85°N, 5 m/s) | 38.88° | band centre 30° | +29.6 %, but still inside the observed band | no vs centre (band-limited) |
| wind factor, reference case | 2.44 % | canonical Nansen number 2 %, observed 1–3 % | +22 % vs 2 % centre | no vs centre; inside the observed 1–3 % range |
| wind factor, low wind (5 m/s) | 2.00–2.07 % | 2 % | +0.0 … +3.5 % | yes / borderline |
| `D_E = sqrt(2K/f)` (FD vs analytic) | 16.59–28.14 m | same, via independent tridiagonal solve | ≤ 2.4e-7 % (rel. 2.4e-9) | yes |
| Ekman surface veer | 45.00004° | 45° (Ekman theory) | 4e-5 % | yes |
| Ekman transport | −1.5713 i … −1.8075 i m²/s | `tau/(rho f)` | ≤ 9.1e-5 % | yes |
| steady-state momentum residual | ≤ 2.2e-16 N/m² | 0 | machine zero | yes |
| `arctan(f/c_d)` identity | — | closed form | ≤ 7.1e-15 deg | yes |
| RK4 transient vs Newton steady state | — | same fixed point | speed 0.0, angle 3.6e-15 deg | yes |

Honest summary: the *verification* quantities (Ekman depth/veer/transport,
momentum residual, `arctan` identity, transient convergence) close to machine
precision, i.e. the implementation does what the math says. The *validation*
quantities (Nansen angle, wind factor) sit inside the observed bands but are not
reproduced to 3 % of the band centre: the angle is 0–30 % above the 30° centre
depending on wind speed, and the wind factor is up to +42 % above the 2 % Nansen
number at 15 m/s. Those are model-physics limits (the classical 2 % number refers
to a still ocean; the coupled Ekman current adds drift), not numerical errors,
and the wind-only column of the case table shows the classical limit.
## 6. Reproduction

```bash
python -m pytest tests/test_ice_drift_dynamics.py -q        # 42 passed
python scripts/ice_drift_dynamics_artifacts.py              # acceptance.json + 5 figures
```
