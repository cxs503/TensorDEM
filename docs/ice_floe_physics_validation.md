# Ice-floe physics verification: buoyancy, heave period, friction, restitution

Verification record for `tensordem.buoyancy_wet` and `tensordem.contact_tangential`.
The harness is `benchmarks/ice_floe_physics.py` (schema `tensordem.ice-floe-physics/2`),
the figure/table producer is `benchmarks/ice_floe_physics_figures.py`, the
regression guard is `tests/test_ice_floe_physics.py`, and every number quoted
below is machine-generated into
`artifacts/ice_floe_physics/validation_report.json` (nothing is transcribed by
hand — re-run the two scripts to reproduce).

Scope statement: this verifies *the implemented force laws and their coupling to
an explicit integrator* against closed forms. It is **not** a validation against
ice-tank or full-scale data, and it says nothing about whether a 2-D extruded
disk with centre-force bonds is an adequate model of a real ice floe.

## 1. Why this work exists (audit of the previous revision)

The file `benchmarks/ice_floe_physics.py` was merged as "three-anchor validation
for ice-floe physics" but nothing in the tree imported it: no test executed it,
no document referenced it, and it wrote its report to the *same*
`benchmarks/validation_report.json` path that `benchmarks/fracture_gen_ledger.py`
owns, so the two produced reports that clobbered each other (the committed file
still contains anchor keys from one run and a `"task": "m154 ..."` block from the
other). Reading the anchors against their claims:

| anchor | previous claim | what it actually did | grade before |
|---|---|---|---|
| 1 buoyancy equilibrium | ice disk settles at `f_sub = rho_i/rho_w` | real: integrated `compute_buoyancy_force` + drag, 200 000 steps | REAL, but the reported equilibrium draught was `-(1-rho_i/rho_w)*R = -0.0500 m` while the correct value is `-0.3378 m` (factor 6.8 wrong); the anchor was unreachable by any test (≈250 s runtime) |
| 2 restitution | rebound velocity vs `e` | round-trip `e -> gamma_n -> e` through two inverse formulas (exact identity) plus a hand-written ODE integration | partially REAL (the ODE part is a genuine independent check of the formula) |
| 3 friction angle | "slip onset angle vs arctan(mu)" | computed `m*g*sin(theta) > mu*m*g*cos(theta)` in floating point and bisected it — **no repository function was called at all**; the result equals `arctan(mu)` to 1e-14 by construction | MOCK |

The lesson generalises: an anchor whose "measured" value is a symbolic
rearrangement of its own reference is not evidence, and a benchmark that no test
runs decays without anyone noticing.

## 2. Method

Common setup: SI units, `float64`, symplectic (semi-implicit) Euler
`v += a*dt; x += v*dt` — the same update ordering as the repository's 2-D DEM
integrator. Sea water `rho_w = 1025 kg/m^3`, ice `rho_i = 917 kg/m^3`, disk
radius `R = 0.5 m`, extruded thickness `t = 1.0 m`.

**Independent geometry reference.** The repository computes the submerged
fraction from the circular-segment formula `f = (theta - sin theta)/(2*pi)`,
`theta = 2*arccos(u/R)`. The reference used here is derived on a different path,
by substituting `y = R sin(phi)` in `A_sub = int_{-R}^{y_max} 2*sqrt(R^2-y^2) dy`:

```
A_sub(u) = R^2 * (pi/2 + phi + sin(phi)cos(phi)),   phi = arcsin(-u/R)
```

Anchor A compares the code force with `rho_w g A_sub t`; anchor H solves
`A_sub(h_eq) = (rho_i/rho_w) pi R^2` by bisection on that closed form.

## 3. Results

### Anchor H — hydrostatic equilibrium (6 release heights, one run)

| quantity | reference | measured | rel. error |
|---|---|---|---|
| submerged fraction `f_sub` | 0.8946341 | 0.8946341 (worst of 6 floats) | 3.45e-05 |
| centre height `h_eq` | -0.3377671 m | per-float | 6.57e-05 of `R` |
| residual velocity at end of run | 0 | 2.45e-05 m/s | — |

Release heights `+0.6, +0.3, 0, -0.3, -0.6, -1.0 m` all converge to the same
waterline (`dt = 2e-3 s`, 3000 steps, `C_wet = 4000 N s/m`). The equilibrium
value is reached from both dry and fully submerged starts, which is the part
that a single-start run cannot show.

### Anchor A — Archimedes force vs independently derived displaced volume

23 centre heights spanning fully dry, partially submerged and fully submerged:
max relative error `7.7e-16`, i.e. the two geometry paths agree to machine
precision. This is an *algebraic* equivalence check, not a discretisation check —
report it as such.

### Anchor T — small-amplitude heave period

Linearised restoring stiffness at the waterline
`k = rho_w g t 2*sqrt(R^2 - h_eq^2) = 7414.02 N/m`, mass `m = rho_i pi R^2 t =
720.210 kg`, so `T = 2*pi*sqrt(m/k) = 1.9583147 s`.

| `dt` [s] | measured `T` [s] | rel. error |
|---|---|---|
| 4e-3 | 1.9583042 | 5.37e-06 |
| 2e-3 | 1.9583143 | 2.26e-07 |
| 1e-3 | 1.9583168 | 1.06e-06 |

Amplitude `2e-3 R`, no damping, period from linear interpolation of zero
crossings. The residual is *not* monotone in `dt`: it bottoms out at ~2e-7 and
then rises to ~1e-6, which is the resolution of the zero-crossing estimator
rather than an integration error. No convergence order is claimed beyond
"< 1e-5 for `dt <= 4e-3`".

### Anchor F — Cundall-Strack friction onset and sliding acceleration

`integrate_tangential_displacement` + `compute_tangential_force`, 1-DOF disk on a
fixed inclined plane, quasi-static normal load `N = m g cos(theta)`,
`k_t = 1e7 N/m`, `mu` in {0.2, 0.5}.

| `mu` | `arctan(mu)` [deg] | onset bracket [deg] | sliding acceleration (meas / ref) [m/s^2] | rel. error |
|---|---|---|---|---|
| 0.2 | 11.309932 | (11.209932, 11.309932] | 1.73722435 / 1.73722435 | 1.05e-14 |
| 0.5 | 26.565051 | (26.465051, 26.565051] | 1.90455818 / 1.90455818 | 2.87e-14 |

Grid resolution 0.1 deg, so the honest statement is "onset lies inside the
0.1 deg bracket that ends at `arctan(mu)`". The acceleration agreement to 1e-14
is expected — once the tangential spring is saturated by the Coulomb cone the
equation of motion has constant acceleration, which symplectic Euler reproduces
exactly — so this row verifies the *magnitude and direction of the clip*, not the
integrator. The stick control (`theta = arctan(mu) - 5 deg`) keeps
`travel = 2.17e-7 m` against a stick scale `mu*N/k_t = 1.93e-7 m`, i.e. bounded.

### Anchor R — restitution mapping against the collision ODE

| target `e` | `gamma_n` [N s/m] | `e` from ODE integration | rel. error |
|---|---|---|---|
| 0.5 | 43.0908 | 0.49999961 | 7.83e-07 |
| 0.8 | 14.1701 | 0.79999994 | 7.94e-08 |

The round-trip `e -> gamma_n -> e` is still reported in the JSON but is labelled
as formula consistency only. Engine-level contact restitution for the 3-D solver
is a different question and is covered elsewhere.

## 4. Error judgement (criterion: < 3 %)

| key quantity | reference | relative error | < 3 %? |
|---|---|---|---|
| equilibrium submerged fraction (H) | `rho_i/rho_w` | 3.45e-05 | yes (3 orders below) |
| equilibrium centre height (H) | closed-form `h_eq` | 6.57e-05 | yes |
| buoyancy force over 23 submergences (A) | `rho_w g A_sub t` | 7.70e-16 | yes |
| heave period, `dt <= 4e-3` (T) | `2 pi sqrt(m/k)` | 5.37e-06 | yes |
| friction onset angle (F) | `arctan(mu)` | 0.00 deg (bracket 0.1 deg = 0.3 % of 11.3 deg) | yes |
| sliding acceleration (F) | `g(sin - mu cos)` | 1.05e-14 | yes |
| restitution (R) | damped-oscillator rebound | 7.83e-07 | yes |

Every anchor is far inside 3 %. The remaining error is dominated by estimator
resolution (zero-crossing interpolation, 0.1 deg angle grid), not by the force
laws.

## 5. Negative controls (anti-tautology)

Each physics anchor is also run with the code under test replaced by a stub, and
the anchor must fail:

| control | stub | result |
|---|---|---|
| H without buoyancy | `compute_buoyancy_force -> 0` | fails (disk sinks; note the fraction metric saturates at `1 - rho_i/rho_w = 11.8 %`, so the control is judged on "clearly outside criterion") |
| A zero / half scale | force `-> 0`, `-> 0.5x` | fails (max rel. error 1.0, 0.5) |
| T free fall | force `-> 0` | raises: no zero crossing exists — no oscillation without a restoring force |
| F frictionless | `mu = 0` | every lane at or above 0 deg slides, stick control fails |
| R zero damping | `gamma_n` frozen at the `e = 0.5` value | target `e = 0.9` is missed |

These controls are the reason the anchors can no longer silently degrade into
identities: a stub provably breaks them.

## 6. Limitations and open items

- **Model scope.** Buoyancy here is per-particle Archimedes on a 2-D extruded
  disk. No added mass, no free-surface deformation, no wave damping, no surface
  tension. Sub-floe hydrodynamics and floe-floe collision in water are outside
  this harness.
- **Small-amplitude only.** The period anchor uses `2e-3 R`; the nonlinear
  (finite-amplitude) heave period is not verified and is expected to differ.
- **Prescribed normal load.** Anchor F imposes `N = m g cos(theta)` rather than
  resolving the normal contact spring, so it verifies the tangential law in
  isolation. Coupling with a resolved normal force (normal/tangential
  interaction, rolling) is a separate increment (see the rolling-resistance
  work on `agent/m2157`).
- **Constant-acceleration exactness.** The friction acceleration row cannot
  detect integrator error by construction; use it as a clip test.
- **Report channel.** `benchmarks/ice_floe_physics.py` now writes
  `artifacts/ice_floe_physics/validation_report.json`. The older shared path
  `benchmarks/validation_report.json` belongs to
  `benchmarks/fracture_gen_ledger.py` and is still polluted by the historical
  collision; it is left untouched rather than rewritten from here.
- **Runtime.** Full configuration ≈ 28 s on one CPU thread; `--fast` ≈ 17 s;
  the unit test uses reduced step counts (≈ 17 s) and looser onset tolerance
  (documented in the test).