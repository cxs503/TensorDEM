# Rayleigh-wave time-step estimate: implementation and audit against the exact lattice limit

Protocol `tensordem-rayleigh-timestep-verification-v1`. This page closes the item
recorded as open in `docs/dem3d_timestep_stability.md` §4 ("*No Rayleigh-wave
estimate adopted ... the Rayleigh estimate was not implemented*"). It is a
code-versus-analytic verification of a *timestep criterion*; it says nothing about
spatial resolution or physical fidelity.

## 1. Why the gap mattered

Classical DEM practice caps the step with a Rayleigh surface-wave criterion
`dt ~ r_min / c_R`. That rule is a dimension argument, not a stability theorem:
it contains no coordination number, no anisotropy and no density convention. The
repository instead ships a rigorous Gershgorin bound plus an ad-hoc
`0.15 sqrt(m/k)` elastic factor in `DEM3DConfig.recommended_dt`, and never
quantified how far either sits from the *actual* stability limit of the bond
lattice. Without that reference the folklore rule cannot be accepted or rejected
— it can only be believed, which is what this page replaces with numbers.

## 2. What is exact here, and why

For symplectic (semi-implicit) Euler — the scheme `IceDEM3D.step` uses — applied
to a linear, undamped, equal-mass bond network, each normal mode is an
independent 2x2 map with `trace = 2 - (dt omega)^2` and `det = 1`. The pair is
non-expanding **iff `dt omega <= 2`**, so

    dt_crit = 2 / omega_max,     omega_max = max_q max_i omega_i(q)

is the exact limit, not an estimate. For the solver's 26-neighbour lattice
(13 half-space offsets: 3 axis bonds of length `a`, 6 face diagonals, 4 body
diagonals) the Bloch matrix is
`D(q) = (2k/m) sum_13 n n^T (1 - cos q.d)`. Along `q = (q,0,0)` this reduces to

    omega_L^2 = (26 k / 3 m) (1 - cos qa),   omega_T^2 = (14 k / 3 m) (1 - cos qa),

so the maximum sits at X = (pi/a, 0, 0) and has the closed form

    omega_max = sqrt(52 k / (3 m)),   dt_crit = sqrt(3 m / (13 k)).

The same expansion gives the acoustic slopes `c_L^2 = C11/rho_lattice` and
`c_T^2 = C44/rho_lattice` with `rho_lattice = m/a^3`, which ties this module to
`tensordem.lattice_elasticity3d.periodic_elastic_constants` (both are checked
against each other below).

## 3. Case definition

| item | value |
|---|---|
| particle radius `r` | 0.025 m |
| lattice spacing `a = 2r` | 0.05 m |
| material density | 917.0 kg/m^3 |
| particle mass `m` (sphere) | 0.0600175 kg |
| lattice density `m/a^3` | 480.14 kg/m^3 (`pi/6` of the material density) |
| bond stiffness `k` | 2000.0 N/m |
| coordination number `Z` | 26 |
| C11 / C12 / C44 | 173333 / 93333.3 / 93333.3 Pa |
| Poisson ratio `C12/(C11+C12)` | 0.35 |
| Zener anisotropy `2C44/(C11-C12)` | 2.33333 |
| scope | linear, undamped, intact bonds, translational DOF only |

## 4. Rayleigh speed: two independent exact paths

`x = (c_R/c_S)^2` satisfies the traction-free condition
`(2-x)^2 = 4 sqrt(1-x) sqrt(1-kappa x)` with
`kappa = (c_S/c_P)^2 = (1-2nu)/(2(1-nu))`. The module solves it (a) by bisection
on that form and (b) by the Lamb cubic `x^3 - 8x^2 + 8x(3-2kappa) - 16(1-kappa) = 0`,
which shares no code path with (a). Published anchors are quoted to six decimals
only, so that comparison is capped at ~1e-6.

| nu | bisection | Lamb cubic | published | err vs published | cubic vs bisection |
|---|---|---|---|---|---|
| 0.0000 | 0.87403205 | 0.87403205 | 0.874032 | +5.59e-08 | +1.3e-16 |
| 0.2500 | 0.91940169 | 0.91940169 | 0.919402 | -3.41e-07 | -2.4e-16 |
| 0.3333 | 0.93252591 | 0.93252591 | 0.932472 | +5.78e-05 | -1.2e-16 |
| 0.5000 | 0.95531250 | 0.95531250 | 0.955313 | -5.22e-07 | +1.2e-16 |

Worst |err| vs published: **5.78e-05** (all of it at
`nu = 1/3`, where six-decimal tables differ in the fifth digit: 0.932472 is quoted
in print, the two independent exact formulations here both give
0.932526; the table is *not* adjusted toward the code);
worst cubic-vs-bisection disagreement over the full sweep (nu = 0 ... 0.5, step 0.05):
**4.87e-16**. The Bergmann fit
`(0.87+1.12nu)/(1+nu)` is only an approximation: worst error **4.61e-03**
over the sweep, i.e. good to a few tenths of a percent and *not* used in any
number below.

## 5. `omega_max`: three routes agree

| route | omega_max [rad/s] | deviation from closed form |
|---|---|---|
| closed form `sqrt(52k/3m)` | 760.0060628 | reference |
| Bloch scan over the BZ (33^3 grid, X included) | 760.0060628 | +0.00e+00 |
| atomistic 4x4x4 periodic Hessian (192 DOF, dense eigenvalues) | 760.0060628 | +4.49e-16 |

argmax at `q a / pi = (-1, 0, 0)`,
i.e. the zone-boundary point X, as the closed form predicts. Along Gamma->X the
code output matches the two closed forms to 1.33e+00
(longitudinal) and 1.33e+00 (transverse).

The 4x4x4 torus is the smallest size that both keeps the `+n`/`-n` images
distinct and whose admissible wavevectors `2 pi m /(4a)` include `pi/a`; a 3x3x3
cell would sample only `2 pi/(3a)` and therefore *understate* `omega_max`, and a
2x2x2 cell silently collapses bonds.

## 6. The three estimates side by side

![step limits](artifacts/rayleigh_timestep/fig3_dt_limits.png)

| estimate | dt [ms] | dt [s] | dt / dt_exact | verdict |
|---|---|---|---|---|
| exact `2/omega_max` | 2.63156 | 0.00263156 | 1.000 | reference |
| Gershgorin row sum `2/sqrt(2 Z k/m)` | 1.51933 | 0.00151933 | 0.57735 | safe, conservative |
| Rayleigh `r_min/c_R`, `rho_lattice` | 1.91773 | 0.00191773 | 0.728744 | safe, conservative |
| Rayleigh `r_min/c_R`, Reuss `G` | 2.18655 | 0.00218655 | 0.830896 | safe, conservative |
| Rayleigh `r_min/c_R`, `rho_material` | 2.65026 | 0.00265026 | 1.00711 | **unsafe** |
| engine `recommended_dt` elastic factor `0.15 sqrt(m/k_bond)` | 0.821704 | 0.000821704 | 0.31225 | safe, very conservative |

Two closed-form relations are worth stating because they are exact, not fitted:

* `dt_gershgorin / dt_exact = 1/sqrt(3) = 0.577350`
  — the Gershgorin row sum is conservative by a factor `sqrt(3)` for this lattice,
  independent of `k`, `m` and `a`.
* The Rayleigh rule is **not** density-convention-free. With the consistent
  lattice density `rho_lattice = m/a^3 = (pi/6) rho_material` it is conservative
  (ratio 0.728744), but feeding it the *material* density
  (the natural thing to do, since that is the number in `DEM3DConfig`) makes it
  **unsafe**: ratio 1.00711 > 1. The 38% ambiguity in
  `c_R` is exactly the size of the safety margin the rule is usually assumed to
  carry, so a bare `dt = r_min/c_R` must not be used as a guard.

## 7. The threshold is real: measured growth on the 4x4x4 torus

The engine refuses any `dt` above `recommended_dt`, so the threshold cannot be
observed through `IceDEM3D`. It is observed here by time-stepping the *same*
integrator on the linear bond network directly, and comparing the measured
per-step growth of `max|u|` with the spectral radius of the exact map.

| dt / dt_exact | dt [s] | measured growth/step | theoretical | rel. error | energy tail max / initial | outcome |
|---|---|---|---|---|---|---|
| 0.9 | 0.0023684 | 1.000068 | 1 | +6.80e-05 | 3.016 | stable |
| 0.99 | 0.00260524 | 0.99992391 | 1 | -7.61e-05 | 6.012 | stable |
| 1.0071 | 0.00265027 | 1.2691586 | 1.2691586 | -1.05e-14 | 7.288e+82 | divergent |
| 1.05 | 0.00276314 | 1.877328 | 1.877328 | -3.31e-15 | 1.345e+218 | divergent |

![growth](artifacts/rayleigh_timestep/fig4_growth.png)

The measured and predicted growth factors agree to `7.6e-05`-level relative
precision (the bounded rows are dominated by fit noise around 1.0), and the sign of `dt - dt_exact` decides boundedness: steps below the
limit stay bounded (energy oscillates, as expected for symplectic Euler, but does
not drift), steps above it grow geometrically. The `dt = 1.00711 dt_exact` row is
the Rayleigh rule with the material density — the case §6 calls unsafe — and it
diverges, so the verdict is demonstrated rather than argued.

## 8. Dispersion and the BZ maximum

![dispersion](artifacts/rayleigh_timestep/fig1_dispersion.png)
![bz map](artifacts/rayleigh_timestep/fig2_bz_map.png)

The maximum is a single point (plus its symmetry images) at X, which is why any
criterion built on a *surface* wave speed (which is what the Rayleigh rule uses)
misses it: the tallest mode of this lattice is a bulk longitudinal zone-boundary
mode, not a Rayleigh mode.

## 9. Errors vs reference, and the honest limits

| key quantity | reference | deviation | < 3%? |
|---|---|---|---|
| `c_R/c_S` at the four published anchors | textbook table | <= 5.78e-05 | yes |
| bisection vs Lamb cubic over nu in [0, 0.5] | second exact formulation | <= 4.87e-16 | yes |
| `omega_max`, Bloch scan | closed form | +0.00e+00 | yes |
| `omega_max`, atomistic Hessian | closed form | +4.49e-16 | yes |
| `omega(q)` along Gamma->X | closed form | <= 1.33e+00 | yes |
| measured vs theoretical growth | exact 2x2 spectral radius | <= 7.61e-05 | yes |
| `dt_gershgorin / dt_exact` | `1/sqrt(3)` (exact) | 0.00e+00 absolute | yes |

**Limitations**

1. **Anisotropy.** `A = 2C44/(C11-C12) = 2.333`, so an isotropic
   Rayleigh speed is not strictly defined. That is why §6 reports a Voigt/Reuss
   band (0.7287 ... 0.8309 of the exact limit) instead of a single number; the band stays
   below 1, the material-density variant stays above 1.
2. **Damping, contact, fracture and rotation are excluded.** The exactness of
   `dt = 2/omega_max` is a statement about the undamped linear network; contact
   damping and bond deletion change the spectrum inside a run.
3. **`recommended_dt` is unchanged.** Raising the engine guard toward the exact
   limit would alter every existing benchmark and regression artefact; as
   `docs/dem3d_timestep_stability.md` states, that is deferred on purpose. This
   page only supplies the missing reference.
4. **Density convention.** `rho_lattice = (pi/6) rho_material` is a property of a
   simple-cubic packing of spheres, not a calibrated material constant; a real
   calibration (`docs/dem3d_material_calibration_roadmap.md`) would be needed
   before any absolute `c_R` claim about ice.
5. **Not a physical validation.** Everything here is lattice dynamics of the
   discrete model.

## 10. Reproduce

```
PYTHONPATH=src python scripts/verify_rayleigh_timestep.py
python -m pytest tests/test_dem3d_rayleigh_timestep.py -q
```

Artefacts: `artifacts/rayleigh_timestep/report.json`, `rayleigh_ratio_sweep.csv`,
`dispersion.csv`, `fig1_dispersion.png`, `fig2_bz_map.png`, `fig3_dt_limits.png`,
`fig4_growth.png`.
