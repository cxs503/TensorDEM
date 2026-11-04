# Hertzian restitution target in `tensordem.dem3d`: reference solution, measured bias, and a verified mapping

Mission: m3608 (TensorDEM self-driven work, strategy seat). Branch `agent/m3608`.

## 1. Case and motivation

`DEM3DConfig(contact_model="hertz")` replaces the linear normal spring by the
Hertz law `F = A δ^(3/2)` with `A = (4/3) E* sqrt(R*)`, and divides the force by
the *secant* stiffness when a dissipation model is built:

    k_s(δ) = A sqrt(δ),        γ(δ) = 2 ζ sqrt(k_s(δ) m_eff),
    F_n    = max(0, A δ^(3/2) - γ(δ) v_n)          (v_n = closing speed)

The damping ratio `ζ` is obtained from the requested coefficient of restitution
(COR) with the **linear free-oscillator identity**

    ζ = |ln e| / sqrt(π² + ln²e)      (inverse of e = exp(-πζ/√(1-ζ²))),

which `docs/dem3d_contact_impact_analytic.md` §5.3 declares out of scope for Hertz
("this validates the linear law only"). A previous campaign (m3578) measured that
the Hertz branch misses a requested `e = 0.5` by +16.4 % in the shipped solver,
but the cause was not isolated and no correction existed.

This document closes that debt. It provides (i) a reference integration of the
exact law the solver implements, (ii) the measured bias across the COR range,
(iii) the proof that the bias is **impact-speed independent**, and (iv) a
verified re-mapping plus its solver validation.

### Parameter set (SI)

| quantity | symbol | value | source |
|---|---|---|---|
| Young's modulus | `E` | 1.0e9 Pa | `DEM3DConfig.young_modulus` |
| Poisson ratio | `ν` | 0.30 | `DEM3DConfig.poisson_ratio` |
| particle radius | `r` | 0.025 m | `DEM3DConfig.radius` |
| density | `ρ` | 917 kg/m³ | `DEM3DConfig.density` |
| particle mass | `m` | 0.06001751 kg | `ρ·(4/3)πr³` |
| effective modulus | `E*` | 5.494505e8 Pa | `E/(2(1-ν²))` (sphere–sphere) |
| effective radius | `R*` | 0.0125 m | `r/2` |
| reduced mass | `m_eff` | 0.03000876 kg | `m/2` |
| Hertz prefactor | `A` | 8.1907252e7 N·m^-3/2 | `(4/3)E*√R*` |
| closing speed (base case) | `v_in` | 1.0 m/s | `v0 = 0.5 m/s` per sphere |

## 2. Method

**Reference model** (`src/tensordem/hertz_restitution.py`). The relative motion of
the pair is integrated with classical RK4 at fixed step count over ≈5 elastic
contact durations:

    m_eff d²δ/dt² = -(A δ^(3/2) + γ(δ) dδ/dt),   γ(δ) = 2ζ sqrt(A sqrt(δ) m_eff),

with the force clipped at zero exactly as `dem3d` does. The COR is `|dδ/dt|/v_in`
at the instant the overlap returns to zero. `HertzImpactSpec` fixes the geometry
and `hertz_impact_restitution(ζ, spec)` returns `e` together with `steps`,
`dt_s`, `contact_time_s` and the elastic peak overlap
`δ_max = (15 m_eff v²/(16 A))^(2/5)`.

**Inverse mapping.** `e(ζ)` is monotone decreasing, so `hertz_zeta_for_target(e)`
bisects it to `|e - e_target| < 1e-5`; `classic_zeta_from_restitution` is kept as
the shipped mapping for comparison.

**Solver validation.** `measure_solver_impact()` (in the same module, deferred
import of `dem3d`) builds two free-flying `IceDEM3D` spheres on the Hertz branch,
breaks every lattice bond, and impacts them at `±v0` along z. The COR is the
rebound-speed ratio; the normal force history is recovered from the momentum
change per step.

**Reproduction.** `python scripts/verify_hertz_restitution.py --output
docs/m3608/outputs/hertz_restitution --figure` regenerates every number and the
figure below (JSON + 4 CSV files). `tests/test_dem3d_hertz_restitution.py`
(14 tests) asserts the claims.

## 3. Results

### 3.1 Reference model: the shipped mapping overshoots, and the bias is set by `e_target` alone

Base case `v_in = 1.0 m/s`.

| `e_target` | `ζ_classic` (shipped) | `e` produced (reference) | bias | `ζ_eff` (verified) | `e` produced (verified) | bias |
|---|---|---|---|---|---|---|
| 0.95 | 0.01634 | 0.95561 | +0.59 % | 0.01840 | 0.95000 | < 0.001 % |
| 0.90 | 0.03352 | 0.91172 | +1.30 % | 0.03831 | 0.90000 | -0.0002 % |
| 0.80 | 0.07085 | 0.82579 | +3.22 % | 0.08315 | 0.80000 | -0.0009 % |
| 0.70 | 0.11281 | 0.74225 | +6.04 % | 0.13666 | 0.70000 | -0.0001 % |
| 0.50 | 0.21545 | 0.58232 | **+16.46 %** | 0.28545 | 0.50000 | +0.0004 % |
| 0.30 | 0.35786 | 0.43122 | **+43.74 %** | 0.55603 | 0.30000 | +0.00001 % |

The 3 % accuracy band is exceeded from `e_target ≤ 0.82` downwards. The
`e_target = 0.5` row reproduces the +16.4 % reported by m3578 to two digits,
which is the first independent check that this reference model is the model the
solver implements.

**Why one scalar fixes it.** Substituting `δ = s²δ̃, t = s⁻¹t̃` leaves
`m_eff δ̈ = -(Aδ^(3/2) + γ(δ)δ̇)` form-invariant: both the elastic and the
`δ^(1/4)`-scaled damping term pick up the same factor `s³`. The clipped law is
scale-free as well, so the dimensionless COR cannot depend on `v_in`, and it does
not (reference model: identical to 1e-9 over `v_in = 0.1 … 8 m/s`; solver: see
§3.3). A single `ζ → ζ_eff(e_target)` re-mapping is therefore complete, with no
speed- or stiffness-dependent correction table.

### 3.2 Shipped solver (`IceDEM3D`, `hertz_restitution_verified=False`)

Free-flying pair impact, `dt = 0.35·recommended_dt = 1.64e-6 s`,
333–408 steps inside contact.

| `e_target` | `e` measured | bias | contact steps | peak overlap |
|---|---|---|---|---|
| 0.90 | 0.911729 | +1.30 % | 333 | 0.177 mm |
| 0.80 | 0.825789 | +3.22 % | 336 | 0.170 mm |
| 0.70 | 0.742209 | +6.03 % | 341 | 0.163 mm |
| 0.50 | 0.582167 | **+16.43 %** | 354 | 0.148 mm |
| 0.30 | 0.430918 | **+43.64 %** | 375 | 0.131 mm |

Solver vs reference at `e_target = 0.5`: 0.582167 vs 0.582318 → **0.026 %**
disagreement. The reference is not validated against itself.

### 3.3 Verified mapping in the same solver (`hertz_restitution_verified=True`)

Only the damping-ratio scalar changes; geometry, `dt` and the contact law are
untouched.

| `e_target` | `e` measured | bias |
|---|---|---|
| 0.90 | 0.900004 | +0.0004 % |
| 0.80 | 0.799979 | -0.003 % |
| 0.70 | 0.699939 | -0.009 % |
| 0.50 | 0.499770 | -0.046 % |
| 0.30 | 0.299543 | -0.152 % |

Worst case over the grid: **0.152 %** (was 43.6 %).

**Discretisation evidence** (`e_target = 0.5`):

| mapping | `dt_scale` | `dt` [s] | contact steps | `e` | bias |
|---|---|---|---|---|---|
| classic | 0.50 | 2.34e-6 | 248 | 0.581755 | +16.351 % |
| classic | 0.35 | 1.64e-6 | 354 | 0.582167 | +16.433 % |
| classic | 0.20 | 9.38e-7 | 620 | 0.582233 | +16.447 % |
| verified | 0.50 | 2.34e-6 | 255 | 0.499281 | -0.144 % |
| verified | 0.35 | 1.64e-6 | 364 | 0.499770 | -0.046 % |
| verified | 0.20 | 9.38e-7 | 637 | 0.499872 | -0.026 % |

Both columns converge monotonically to the reference values (16.46 % / 0.00 %),
so the residual is O(dt) semi-implicit-Euler truncation, not a modelling artefact.

**Impact-speed independence** (`e_target = 0.5`, `v0` = speed of each sphere):

| mapping | `v0` = 0.25 m/s (`v_in` 0.5) | `v0` = 0.5 (1.0) | `v0` = 1.0 (2.0) |
|---|---|---|---|
| classic | +16.416 % | +16.433 % | +16.385 % |
| verified | -0.064 % | -0.046 % | -0.103 % |

All three classic runs land within 0.05 % of each other, confirming that the
error is a mapping error and not a velocity-dependent one.

### 3.4 Figure

`docs/m3608/outputs/hertz_restitution/hertz_restitution_bias.png` — three panels:
(1) |relative COR error| vs requested COR, shipped vs verified, log scale, with
the 3 % band; (2) measured vs requested COR in `IceDEM3D` for both mappings with
the 1:1 line; (3) contact trace (overlap and normal force vs time) for
`e_target = 0.5`, solver symbols over the RK4 reference line — the two traces
overlap to line width, which is the direct visual form of the 0.026 %
solver/reference agreement.

## 4. Error verdict

| key quantity | baseline | error | < 3 %? |
|---|---|---|---|
| shipped-mapping COR, `e_target ∈ {0.9, 0.8, 0.7, 0.5, 0.3}` | requested COR | +1.30 … +43.6 % | **No** (fails from 0.8 downwards) |
| verified-mapping COR, same grid | requested COR | -0.003 … -0.152 % | **Yes** |
| solver vs reference model, `e_target = 0.5` | RK4 reference (0.582318) | 0.026 % | **Yes** |
| COR speed independence (classic, `v_in` 0.5→2 m/s) | 1.0 | ≤ 0.05 % | **Yes** |
| dt convergence, verified, `dt_scale` 0.5→0.2 | 0.5 | -0.144 → -0.026 % | **Yes** |

## 5. Notes, limits and traps

1. **This is numerical consistency, not physics.** The reference integrates the
   *model the solver implements* (rigid Hertz spring + linear viscous damper,
   compressive-only). Reproducing a requested COR of that model says nothing
   about whether a linear damper is the right dissipation law for ice: a
   Hunt–Crossley (`∝ δ^(3/2) δ̇`) damper or a velocity-dependent COR would be a
   different model and needs its own validation. Adhesive (JKR/DMT) and
   elasto-plastic contacts are out of scope.
2. **The mapping is speed independent only for this law.** For a *constant*
   dashpot (`γ = const`, e.g. `contact_damping_model="absolute"`) the COR is
   velocity dependent; no scalar re-mapping can fix that, and the reference
   module will report the actual dependence (integrate at the case's `v_in`).
3. **`e(n)` is still an inverse problem.** `hertz_zeta_for_target` costs ~13 RK4
   impacts and is `lru_cache`d; with `steps=2000` that is ~0.1 s once per
   distinct `e_target`, i.e. negligible next to a DEM run. It is *not* a
   closed-form formula, and the module says so rather than tabulating a fit.
4. **The legacy default is unchanged.** `hertz_restitution_verified=False` keeps
   every existing Hertz case bit-for-bit identical; the flag is rejected for
   `contact_model="linear"` and for `contact_damping_model="absolute"` so that a
   silent no-op cannot occur. The linear law keeps its own (different) clamped
   mapping from `tensordem.contact_restitution`.
5. **Platen/tool Hertz branches are not covered.** Only the particle–particle
   branch builds `γ` from the restitution target; the rigid-platen branches take
   an absolute damping coefficient. Extending the correction there needs a
   separate reference (rigid-counterpart `E*`, `R*`), which is not done here.
6. **`peak_overlap` bug caught by the test suite.** The first version of the
   reference used `δ_max = (5 m_eff v²/(4A))^(2/5)`; the energy identity
   `(8/15)A δ^(5/2) = ½ m_eff v²` gives `(15 m_eff v²/(16A))^(2/5)` (13.8 %
   smaller). It only affected the integration window and the reported diagnostic,
   but the test `test_peak_overlap_matches_energy_identity` now pins it.

## 6. Files

| path | role |
|---|---|
| `src/tensordem/hertz_restitution.py` | reference integration, inverse mapping, solver-level measurement |
| `src/tensordem/dem3d.py` | `hertz_restitution_verified` flag + corrected `ζ` on the Hertz branch |
| `scripts/verify_hertz_restitution.py` | regenerates all tables + figure (JSON/CSV/PNG) |
| `tests/test_dem3d_hertz_restitution.py` | 14 tests (reference model, solver, config gates) |
| `docs/m3608/outputs/hertz_restitution/` | `*.json`, 4 `*.csv`, `hertz_restitution_bias.png` |