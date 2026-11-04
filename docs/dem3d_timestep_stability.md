# Explicit time-step stability criteria (CFL-class) for the 3-D bonded-sphere DEM

Status: implemented in `src/tensordem/stability.py`, verified by
`tests/test_dem3d_stability.py`, reproducible audit in
`benchmarks/timestep_stability_audit.py`, generated numbers and figures in
`docs/dem3d-timestep-stability/`.

This page closes a gap that previously had **no coverage at all**: before this
change the repository contained no stability criterion, no growth rate, and no
time-step limit other than the ad-hoc `DEM3DConfig.recommended_dt` heuristic
(0.15 safety factors on `sqrt(m/k)`, `m/c` and an indenter-travel bound). The
absence was confirmed by `grep -ri "cfl|critical time|rayleigh|dt_crit"` over
`src/ tests/ docs/ scripts/` returning no matches.

## 1. Physical and numerical background

`IceDEM3D.step` advances the particle system with **semi-implicit (symplectic)
Euler**:

```text
v_{n+1} = v_n + dt * f(x_n, v_n) / m
x_{n+1} = x_n + dt * v_{n+1}
```

The force model is a linear bond spring (`bond_stiffness`, applied to the bond
extension), a linear unilateral contact spring (`contact_stiffness`, applied to
the overlap, active only in compression), velocity-proportional contact and tool
damping (`contact_damping`) and linear drag (`drag`). Bond failure at
`breaking_strain` / `shear_breaking_strain` removes bonds irreversibly, so the
stiffness distribution changes with the damage state and the stable step is not
a property of the initial configuration alone.

Semi-implicit Euler is a first-order, conditionally stable scheme: the step must
stay below a critical value set by the stiffest mode of the discrete system. In
bonded-particle DEM that limit scales like `sqrt(m/k)`, which is why ice
simulations at engineering scale (small particle mass, stiff bonds) are step
limited. An explicit criterion is therefore a first-class verification object,
not a tuning knob.

## 2. Canonical single-degree-of-freedom problem (exact reference)

For `x'' = -omega^2 x - gamma_rate x'` (`omega^2 = k/m`, `gamma_rate = c/m`) the
scheme is a linear map with `d = dt*omega`, `g = dt*gamma_rate`:

```text
[x_{n+1}]   [ 1 - d^2       dt (1 - g) ] [x_n]
[v_{n+1}] = [ -dt omega^2     1 - g    ] [v_n]

det A = 1 - g,        tr A = 2 - d^2 - g,
rho   = max(|lambda_+|, |lambda_-|),   lambda_\pm = (tr A \pm sqrt(tr^2 - 4 det A)) / 2.
```

Two consequences follow analytically and are the backbone of the module:

1. **Stability region.** `rho <= 1` exactly when `0 <= g <= 2` **and** `d <= 2`.
   The two limits are independent, hence

   ```text
   dt_crit = min( 2/omega , 2*m/c ).
   ```

2. **Marginal case.** At `d = 2` the matrix has a repeated eigenvalue `-1` and
   is defective (a Jordan block). `rho = 1` yet the amplitude grows linearly in
   `n` and the energy quadratically — the discrete step is "stable" in the
   spectral sense but useless in practice. Any engineering margin must therefore
   be *strictly* below `dt_crit`; see §5.

For the damped case with `d < 2` and `g < 2`, `rho = sqrt(1-g)`: the step adds
artificial damping of rate `g/2` per unit time. At the audited engine settings
`g ~ 1e-3`, so the numerical damping contribution is negligible compared with
the physical drag.

## 3. Many-body bound (Gershgorin on the assembled stiffness)

For the translation-only bonded system the assembled stiffness matrix has
diagonal `S_i` (sum of the stiffness magnitudes of every alive bond, active
contact, platen and tool coupling attached to particle `i`) and off-diagonals
`-k_ij`. Its row sum of absolute values is `2 S_i`, so

```text
omega_max <= sqrt( 2 * max_i S_i / m ),     dt_crit <= min( 2/omega_max , 2 m / c_max ).
```

Notes on rigour:

* The bound is **exact for an infinite simple-cubic lattice with only nearest
  neighbours** (`2*6k = 12k = lambda_max`), and on a finite `N^3` periodic-free
  lattice it is conservative by `1/cos`-type factors that shrink with `N`
  (7.2% at `N = 6`, see the generated tables).
* Fixed (Dirichlet) particles are excluded from the governing maximum (they have
  no free degree of freedom) but their couplings still enter `S_i`, which keeps
  the bound valid.
* Broken bonds contribute nothing; active contacts are linearised as springs,
  which is the conservative choice for stability. Fracture can therefore only
  *raise* `S_i` through fragment contact, and the reported bound covers it.
* For damping proportional to identity (pure `drag`) the bound is exact
  mode-by-mode because `C` commutes with `K`. For contact damping acting along
  contact normals `C` does not commute with `K`, so `min(2/omega_max, 2m/c_max)`
  is a **sufficient** condition rather than a mode-exact threshold
  【推测】, supported by the random-lattice checks (bound never underestimates
  the dense eigenvalue) but not proved here.
* Rotational degrees of freedom are not modelled by `IceDEM3D` (spherical
  particles with central forces only), so no torsional/frictional contact branch
  enters the limit. If a rotational contact model is added, its stiffness must
  be included in `S_i` with the corresponding moment of inertia.

## 4. API

```python
from tensordem import stability
from tensordem.dem3d import DEM3DConfig, IceDEM3D

model = IceDEM3D(DEM3DConfig(nx=9, ny=5, nz=3))
report = stability.stability_report(model)   # read-only, JSON serialisable
print(report["critical_dt_s"], report["margin_factor"], report["binding_limit"])

# closed-form building blocks
stability.symplectic_euler_growth_factor(dt=1e-4, omega=1.3e3, damping_rate=33.0)
stability.critical_timestep(omega=1.3e3, damping_rate=33.0)
stability.gershgorin_omega_bound(stiffness_sums, mass, free_mask=~model.fixed)
```

`stability_report` returns the per-state maximum stiffness/damping sums, the
elastic and damping limits, the governing (`binding_limit`) branch, the margin
`dt_crit/dt`, the growth factor of the step in use, the engine's own
`recommended_dt` guard and a kinematic (travel) heuristic. It never advances the
solver.

Command line (read-only, writes `stability_report.json`, no solver history):

```bash
tensordem-3d --nx 9 --ny 5 --nz 3 --stability-report --output ./audit
```

## 5. Verification

`benchmarks/timestep_stability_audit.py` regenerates
`docs/dem3d-timestep-stability/report.json` (machine-readable),
`summary.md` (tables) and two figures. All reference values are independent of
the module under test: the step matrix is *reconstructed by probing a
self-contained integrator*, the lattice top frequency comes from the closed-form
dispersion relation, dense eigenvalues are cross-checked with
`torch.linalg.eigvalsh`, per-particle stiffness sums are cross-checked against
brute-force `O(N^2)` pair enumeration, and the engine audit is asserted against
the solver's own state.

Findings, with evidence state:

| Claim | Status | Evidence |
| --- | --- | --- |
| Closed-form growth factor equals the reconstructed step-matrix spectral radius | 【已证实】 | max relative error `4.1e-16` over `d in [0.2, 2.9]`, `g in [0, 2.1]` (generated table 1); `test_growth_factor_matches_independent_eigenvalues`, `test_step_matrix_matches_hand_derivation` |
| Stability region is exactly `d <= 2 and g <= 2` | 【已证实】 | boundary predicates and `test_stability_region_is_exactly_the_closed_form_boundary`; damping decay matches `(1-g)^n` to `1e-12` |
| Marginal step `dt = dt_crit` grows algebraically (`E \propto n^2`) | 【已证实】 | `E(1000)/E(500) = 3.998`, `E(2000)/E(1000) = 3.999` (generated table 2); `test_marginal_step_grows_only_polynomially` |
| Row-sum bound is exact for the infinite simple-cubic lattice | 【已证实】 | dense `lambda_max = 12k/m` matches `sqrt(2*6k/m)^2` to `1e-12`; converged family for finite lattices |
| Row-sum bound is conservative for finite/random lattices | 【已证实】 | 20 random lattices, bound never below the dense eigenvalue; 1-D chain ratio bound/exact = 1.0012 at N = 32 |
| Chain threshold follows the closed-form dispersion limit | 【已证实】 | top-mode excitation at `0.5 dt_crit` keeps peak energy below `2.5 E0`; at `1.05 dt_crit` the run overflows (i.e. diverges) |
| Engine step is always inside the audited limit | 【已证实】 | 13 configuration families (platens, cylinder geometry, stiff bond/contact, heavy damping, 5 mm particles): `margin_factor >= 13.5`, `engine recommended_dt <= dt_crit` |
| Engine closed system conserves bounded energy at the audited step | 【已证实】 | 2000 steps, peak/initial mechanical energy `1.000000`, no bond failure, remaining energy removed by drag |
| Non-proportional contact damping keeps the row-sum condition sufficient | 【推测】 | random-lattice eigenvalue checks + coupled-chain runs; no proof, and no test at the threshold (see below) |
| Engine-level threshold crossing verified dynamically | 【已验证】 | `dt > dt_crit` 只能通过验证专用逃生舱 `DEM3DConfig.allow_dt_above_recommended` 取得；引擎自带力律的每步放大因子与辛 Euler 闭式在 10 个 `dt/dt_crit` 上一致到 ≤7.3e-9，引擎二分把阈值定到 `2/omega_max` 的 4.66e-10，1.05 `dt_crit` 下每步增长 = `rho`（6.6e-10），0.95 `dt_crit` 下 2000 步二次不变量漂移 5.1e-7。见 `docs/dem3d_engine_stability_threshold.md` |
| Criterion validated against ice experiments | 【未验证】 | no experimental data are involved; this is a numerical-stability statement, not a material-validation statement |

### Engineering guidance (measured, not assumed)

Peak-energy amplification of a closed elastic system grows sharply as the step
approaches the limit (generated table 3 and figure `stability_domain.png`):

| `dt/dt_crit` | 0.25 | 0.5 | 0.8 | 0.9 | 1.0 |
| --- | ---: | ---: | ---: | ---: | --- |
| peak energy / E0 (32-mass chain, top mode) | 1.33 | 2.00 | 5.00 | 10.0 | diverges |

So `dt <= 0.5 dt_crit` keeps the artificial energy oscillation within a factor
~2 of the physical result, while being "spectrally stable" is not enough near
the boundary. For the default engine configuration the step actually used is
~2.2% of `dt_crit`, i.e. the existing guard is conservative by roughly a factor
45 in step size (about 2000 in stiffness) — safe, but a statement about
robustness, not about accuracy.

## 6. Limitations and honest gaps

1. **Linear, fixed-topology bound.** The criterion is derived from a linearised
   state. Bond rupture and fragment contact change `S_i` nonlinearly within a
   step; the audit recomputes the bound from the *current* state
   (`stability_report(model)` can be called at any time), but a validated
   recommendation is to re-audit after fracture events instead of trusting the
   initial configuration.
2. **Threshold not exercised through the engine API.** As stated above, the
   engine refuses any step above its own conservative heuristic, so an
   engine-level demonstration of divergence above `dt_crit` is not possible
   without changing the public guard. Relaxing `recommended_dt` toward the
   audited limit would change the dynamics of every existing benchmark and
   regression artifact in `docs/` and `benchmarks/`, so it is deliberately
   deferred; if it is ever done, the accuracy (not stability) study in
   `docs/DEM3D_FIXED_DURATION_TIMESTEP_STUDIES.md` must be repeated.
3. **Rotational DOF and friction absent.** The solver is translational with
   central forces, so torsional/frictional contact stiffness does not enter the
   bound. Adding a tangential contact model (`contact_tangential.py` exists for
   the 2-D path) will require extending `particle_stiffness_sums` with the
   rotational diagonal.
4. **No Rayleigh-wave estimate adopted.** Classical DEM practice uses a Rayleigh
   surface-wave limit `dt ~ r_min/c_R`. It needs a calibrated shear modulus of
   the lattice, which this repository does not have (§ `dem3d_material_calibration_roadmap.md`);
   the Gershgorin bound is rigorous and needs no such calibration, so it was
   preferred and the Rayleigh estimate was not implemented.
5. **Accuracy vs stability.** Passing the stability audit says nothing about
   spatial resolution, calibration, or physical fidelity; see
   `docs/dem3d_mechanics_benchmark_protocol.md` and the energy-ledger caveats in
   `docs/dem3d_energy_audit.md`.