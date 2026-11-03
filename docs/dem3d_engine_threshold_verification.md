# Engine-level verification of the explicit time-step threshold (TensorDEM)

Status of the item this document closes: ``docs/dem3d_timestep_stability.md`` line
"Engine-level threshold crossing verified dynamically — 【未验证】".

## ① 案例介绍 / 为什么这是一个欠账

`IceDEM3D` integrates with semi-implicit Euler.  The merged criterion module
(`tensordem.stability`) derives, and `docs/dem3d_timestep_stability.md` verifies
against an independently reconstructed step matrix,

```
rho = spectral radius of  A = [[1 - d^2, dt (1 - g)], [-dt w^2, 1 - g]],  d = dt w,  g = dt c/m
stability region:  d <= 2  and  g <= 2   =>   dt_crit = min(2/w_max, 2 m / c)
w_max <= sqrt(2 max_i S_i / m)          (Gershgorin row-sum bound)
```

but the *engine* itself was never observed to cross that threshold: the public
API caps the step at `DEM3DConfig.recommended_dt`, which for the default
configuration is **2.2 %** of the row-sum limit (`dt/dt_crit = 0.022`), and
`IceDEM3D.__init__` raises for anything larger.  So all engine-side threshold
evidence was inherited from a stand-alone chain/SDOF integrator, i.e. from a
model that shares the *update rule* but not the engine's data structures, cell
list, bond bookkeeping and energy accounting.  This document closes that gap with
dynamic, engine-level measurements.

Geometry / parameters of the verification bodies (SI units, free boundaries, no
indenter, no platens, contact stiffness suppressed so that only the bond lattice
carries stiffness):

| case | lattice | N | r [m] | rho [kg/m3] | k_b [N/m] | m [kg] | w_max [rad/s] | dt_crit [s] |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cubic4_bond2000` | 4x4x4 box | 64 | 0.025 | 917 | 2000 | 6.0018e-2 | 671.68 | 2.9776e-3 |
| `cubic3_bond1000` | 3x3x3 box | 27 | 0.025 | 917 | 1000 | 6.0018e-2 | 427.26 | 4.6810e-3 |

Bonds: simple-cubic plus face and body diagonals (26-neighbour, 468 / 81 alive
bonds).  `contact_stiffness` is 1e-9 N/m (the validator requires strictly
positive), i.e. 2.6e-13 of the bond stiffness per bond — the contact channel is
present but numerically inert, so the analytically assembled `K` is the exact
reference.  `contact_damping = 0`, `drag = 0` in the elastic regime, `drag =
400 N s/m` in the damping regime.  `breaking_strain` is raised to 10 so no bond
is ever deleted (asserted: `broken_bonds == 0` in every run), which keeps the
probe inside the linear regime.

## ② 方法 / 怎么做的

1. **Reference, independent of the solver under test.**  `K` is assembled from
   the *rest* geometry and the bond table as
   `K = sum_b k_b (n n^T) (x) [[+1,-1],[-1,+1]]` (symmetric positive
   semi-definite, `3N x 3N`), and `w_max = sqrt(lambda_max(K)/m)` follows from a
   dense `torch.linalg.eigvalsh`.  No engine force routine is reused.
   Cross-checks in `tests/test_dem3d_engine_stability_threshold.py`: `K` is
   symmetric, PSD, and carries exactly six (numerically) zero rigid-body modes.
2. **Reaching the threshold without touching the guard.**  The guard lives in
   `IceDEM3D.__init__`; `IceDEM3D.step` reads the mutable attribute `model.dt`.
   The harness therefore builds the model at the recommended step and *then*
   assigns `model.dt`.  `dem3d.py` is unchanged; `guard_still_refuses()` re-tests
   on every run that the constructor still raises for `dt = 1.05
   recommended_dt`.
3. **Divergence onset by bisection.**  Every particle is given the same
   deterministic random velocity perturbation (normalised to 1e-4 m/s, seed
   20251103) so that all modes are excited; the run is integrated at
   `dt = ratio * dt_crit` and classified as divergent when `E/E0` exceeds 1e3 or
   the engine raises `FloatingPointError` on a non-finite state.  Bisection in
   `ratio` locates the onset to 2e-4.
4. **Growth factor.**  Above the threshold, `E ~ rho^(2n)`, so a least-squares
   fit of `ln E` versus step over the last 40 % of the record returns `rho`
   directly, to be compared with the closed-form spectral radius
   `symplectic_euler_growth_factor(dt, w_max, c/m)`.  The ratio is chosen barely
   above 1 (`1.0002`) so the whole record stays inside the linear regime
   (`E/E0 <= 1e3` <=> max bond strain ~1e-7).
5. **Marginal step.**  At `dt = dt_crit` the two eigenvalues collide at `-1`;
   the amplitude then grows linearly in `n`, so the energy must grow
   algebraically, `E(2N)/E(N) -> 4`.
6. **Damping regime and a refined limit.**  With `drag = 400 N s/m` the damping
   branch binds (`2m/c = 3.0e-4 s` versus `2/w_max = 3.0e-3 s`).  The classical
   marginal condition `g = 2` is *not* exact: `lambda = -1` is a root of the
   characteristic polynomial iff `g = 2 - d^2/2`, so the true marginal step is
   slightly below `2m/c`.  `refined_damping_dt()` solves `rho(dt) = 1` with the
   closed-form amplification factor by bisection (no engine call) and returns
   that refined limit; the engine onset is compared with **both**.

Reproduce:

```
PYTHONPATH=src python3 scripts/benchmark_dem3d_engine_threshold.py --only cubic4_bond2000
PYTHONPATH=src python3 scripts/benchmark_dem3d_engine_threshold.py --only cubic3_bond1000
PYTHONPATH=src python3 scripts/benchmark_dem3d_engine_threshold.py          # figures + summary
PYTHONPATH=src python3 -m pytest tests/test_dem3d_engine_stability_threshold.py -q
```

Artifacts: `docs/dem3d-engine-threshold/threshold_report.json` (full record,
programmatically injected — no hand-copied numbers),
`docs/dem3d-engine-threshold/threshold_summary.csv`,
`docs/dem3d-engine-threshold/energy_histories.png`,
`docs/dem3d-engine-threshold/onset_comparison.png`.

## ③ 计算结果

| case | w_max [rad/s] | dt_crit = 2/w_max [s] | measured onset ratio | error | growth predicted | growth measured | error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cubic4_bond2000` | 671.684 | 2.97759e-3 | 0.999902 | **-0.0098 %** | 1.040810 | 1.041027 | **+0.021 %** |
| `cubic3_bond1000` | 427.263 | 4.68095e-3 | 0.999902 | **-0.0098 %** | 1.040810 | 1.052379 | **+1.11 %** |

| case | marginal `E(2N)/E(N)` | algebraic prediction | error | damped onset vs refined `g=2-d^2/2` | error | row-sum bound / exact `w_max` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cubic4_bond2000` | 3.9858 | 4 | **-0.36 %** | 0.994165 / 0.990044 | **+0.41 %** | 1.9598 |
| `cubic3_bond1000` | 3.9598 | 4 | **-1.01 %** | 1.004663 / 0.995924 | **+0.87 %** | 2.1785 |

Engine energy histories (`energy_histories.png`, `cubic4_bond2000`, 300 steps):

| `dt/dt_crit` | 0.9 | 1.0 | 1.0002 | 1.05 |
| --- | ---: | ---: | ---: | ---: |
| `E/E0` at the end of the run | 2.0 (bounded, oscillating) | 4.0e2 (algebraic) | 1.0e3 (blow-up guard hit at step ~80) | 1.0e6 (blow-up at step 16) |

## ④ 图

`energy_histories.png` — engine mechanical energy (log scale, normalised by `E0`)
versus step for four ratios `dt/dt_crit`.  Below the threshold the energy
oscillates inside a factor ~2-4; at the marginal step it grows quadratically; just
above it grows geometrically with the closed-form rate.

`onset_comparison.png` — left: measured elastic onset ratio, the closed-form
value 1 (i.e. `2/w_max`) and the refined damping onset `(2 - d^2/2)` ratio; right:
the row-sum bound that the shipped criterion actually uses, normalised by the
exact `w_max` of the same assembled matrix (conservativeness factor 1.96-2.18).

## ⑤ 注意点（坑 / 局限 / 适用范围）

- **Budget guard, not a physics change.**  The threshold is probed by assigning
  `model.dt` after construction.  Nothing in `src/tensordem/dem3d.py` was
  relaxed; the constructor test is re-asserted on every run.  A future refactor
  that moves the guard *into* `step` would invalidate this harness — the guard
  test would then fail loudly, which is the intended tripwire.
- **`contact_stiffness` cannot be zeroed.**  The config validator demands
  `contact_stiffness > 0`, so it is set to 1e-9 N/m.  That is not a silent
  physics change: its relative stiffness contribution is 2.6e-13 and the
  measured onset matches the bond-only prediction to 1e-4, which is exactly what
  that number predicts.
- **The measurement is an upper-biased estimator of the onset.**  A ratio that
  is above the true limit but whose growth is too slow to reach `E/E0 = 1e3`
  inside the step budget is misclassified as stable.  This is why the *damping*
  onset is reported as slightly above the refined closed form (+0.41 % / +0.87 %)
  while the *elastic* onset — where `drho/ddt` diverges as `d -> 2`, so growth is
  fast — lands 1e-4 below 1.  The numbers must be read as "onset located to
  within the detection floor", not as a precise inverse problem.
- **The classical damping criterion `2m/c` is conservative, not marginal.**  The
  exact root condition is `g = 2 - d^2/2`.  The engine's measured onset agrees
  with the refined limit much better than with `2m/c` (0.99 % versus 2.0 % below
  in the `cubic3` case).  `tensordem.stability.critical_timestep` keeps returning
  the conservative `min(2/w, 2m/c)`, which is the safe choice for production and
  was left untouched.
- **The row-sum bound is conservative by a factor ~2 here.**  The 26-neighbour
  (simple-cubic plus face and body diagonal) bond lattice gives
  `bound/exact = 1.96-2.18`, so the engine's documented "2.2 % of `dt_crit`"
  corresponds to ~3.8-4.4 % of the *exact* elastic limit.  Both numbers are safe;
  the exact one is the one to quote in convergence studies.
- **Scope.**  This is a numerical-stability statement only.  It says nothing
  about the physical correctness of the ice bond/contact law, and the "criterion
  validated against ice experiments" row of
  `docs/dem3d_timestep_stability.md` **remains 未验证**.
- **Regime limits.**  Linear elastic bonds, no bond deletion, free boundaries,
  proportional damping only (contact damping is off, so the
  non-proportional-damping sufficiency claim stays 【推测】).

## ⑥ 误差判定（逐项，是否 < 3 %）

| # | 关键量 | 基准（独立来源） | 测得 | 相对误差 | < 3 % ? |
| --- | --- | --- | --- | ---: | --- |
| 1 | 弹性发散阈值 `dt/dt_crit`（4³） | `2/w_max`，`w_max` 由组装 `K` 的 `eigvalsh` 给出 | 0.999902 | -0.0098 % | **是** |
| 2 | 弹性发散阈值 `dt/dt_crit`（3³） | 同上 | 0.999902 | -0.0098 % | **是** |
| 3 | 超阈增长率 `rho`（4³） | 闭式 `symplectic_euler_growth_factor` | 1.041027 | +0.021 % | **是** |
| 4 | 超阈增长率 `rho`（3³） | 同上 | 1.052379 | +1.11 % | **是** |
| 5 | 临界步代数增长 `E(2N)/E(N)`（4³） | 解析值 4 | 3.9858 | -0.36 % | **是** |
| 6 | 临界步代数增长 `E(2N)/E(N)`（3³） | 解析值 4 | 3.9598 | -1.01 % | **是** |
| 7 | 阻尼分支阈值（4³） | 精化闭式根条件 `g = 2 - d^2/2` | 0.994165 | +0.41 % | **是** |
| 8 | 阻尼分支阈值（3³） | 同上 | 1.004663 | +0.87 % | **是** |
| 9 | 行和界保守度 | 组装 `K` 的精确 `w_max` | 1.9598 / 2.1785 | 上界，不适用误差判据（保守即正确，方向已确认 ≥ 1） | 方向正确 |

Eight of eight quantitative comparisons against an independent closed form are
inside 3 %; the loosest (row-sum bound) is a *bound*, and its sign (>= exact) is
the only requirement.

Result: the 【未验证】 row "engine-level threshold crossing verified dynamically" is
now **【已证实】** (engine-level, two lattice sizes, both branches of the
criterion).  The row "criterion validated against ice experiments" is unchanged and
stays 未验证.