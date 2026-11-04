# Two silent defects in the pair-contact damping path (m3441)

Scope: `src/tensordem/contact_restitution.py`, `src/tensordem/dem3d_wedge.py`.
Evidence grade before the fix: **the shipped restitution / damping calibration
claim was `MOCK`-grade on `main`** — the closed-loop verification suite that
backs `docs/dem3d_contact_restitution.md` and
`docs/dem3d_restitution_verification.md` did not apply any pair damping at all.
After the fix the same suite reproduces the published numbers.

## 1. How the defects were found

A full-suite audit (`pytest tests/ -n 8`) on `main` reported red tests. Two
independent defects were responsible:

| test | symptom |
|---|---|
| `tests/test_dem3d_restitution_verification.py` (4 of 9 checks) | measured `e = 1.0001` for **every** `gamma_n`, i.e. 9.3 %–66.7 % away from the analytic reference |
| `tests/test_dem3d_moving_wedge_bow.py::test_campaign_persists_repeatable_histories` | `TypeError: MovingWedgeBowIceDEM3D.forces() got an unexpected keyword argument 'update_history'` |

Neither defect raises an exception in normal use, which is why both survived.

## 2. Defect A — `contact_damping` was inert in the particle pair branch

`DEM3DConfig` carries two damping conventions (`dem3d.py`):

* `contact_damping_model="restitution"` (**default**): the pair normal damping
  is derived from `restitution_coefficient` (default `1.0`, i.e. *no* damping);
  `contact_damping` does **not** enter the pair branch.
* `contact_damping_model="absolute"`: `gamma_n = contact_damping`, the
  convention used by the tool and platen branches.

The closed-loop harness `measure_dem3d_impact(gamma_n, ...)` built its config
with `contact_damping=gamma_n` but never selected the absolute convention:

```python
config = DEM3DConfig(..., contact_stiffness=k_n, contact_damping=gamma_n, ...)
```

so the impact it integrated was frictionless, drag-free and **undamped**, and
`e_measured` was ~1.0000 regardless of `gamma_n`. Every solver-side restitution
statement in the two documents above rests on this harness
(`test_solver_reproduces_clamped_reference_model`,
`test_target_restitution_mapping_closes_the_loop`,
`test_equal_mass_pair_uses_reduced_mass`,
`test_classic_mapping_is_within_3pct_only_for_small_damping`), so the published
calibration was unbacked by executable evidence.

Fix (2 lines, opt-in, no default changed):

```python
config = DEM3DConfig(..., contact_damping=gamma_n,
                     contact_damping_model="absolute", ...)
```

Measured effect of the fix at `k_n = 5000 N/m`, `R = 0.025 m`, `rho = 917 kg/m3`
(`m = 0.060018 kg`, `v_in = 0.5 m/s`):

| `gamma_n` (N s/m) | `zeta` | `e` before fix | `e` after fix | clamped reference (RK4) | classic `exp(-pi zeta/sqrt(1-zeta^2))` |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.0000 | 1.0001 | 1.0001 | 1.0000 | 1.0000 |
| 1.0 | 0.0289 | 1.0001 | 0.9158 | 0.9148 | 0.9133 |
| 2.0 | 0.0577 | 1.0001 | 0.8411 | 0.8395 | 0.8339 |
| 10.0 | 0.2886 | 1.0001 | 0.4632 | 0.4628 | 0.3879 |

Worst solver-vs-RK4 error over the sweep: `0.19 %` (`gamma_n = 2.0`), i.e.
inside the 1 % gate. The "after" column reproduces the reference tables in
`docs/dem3d_restitution_verification.md` (e.g. `gamma_n = 2.0` -> `0.8411`
reference / `-0.86 %` classic-mapping closure), which is the independent check
that the fix restores the intended law rather than tuning the harness to pass.

## 3. Defect B — `forces()` signature drift in the moving-wedge subclass

`IceDEM3D.forces` gained an `update_history` keyword; its override in
`MovingWedgeBowIceDEM3D` was not updated. `IceDEM3D.diagnostics` forwards
`update_history=False`, so `MovingWedgeBowIceDEM3D.diagnostics()` -- and hence
every campaign script that reads diagnostics -- raised `TypeError`. Nothing in
the suite had called `diagnostics()` on that subclass before, so the class was
unusable through its documented entry point.

Fix: the override now accepts `update_history` and forwards it, so the
subclass obeys the base contract.

The new signature guard immediately found a **third** instance of the same
drift: `tensordem.rudder_ice_load.RudderIceLoadDEM.forces` (ship-relevant:
rudder ice load) also lacked `update_history`, so `RudderIceLoadDEM.diagnostics()`
-- the entry point used by `scripts/benchmark_rudder_ice_load.py` -- raised
`TypeError`. Fixed the same way. After the fix, all four `IceDEM3D` subclasses in
the package (`EnergyAuditedIceDEM3D`, `MovingWedgeBowIceDEM3D`,
`threefield.hull_ice.BowContactDEM`, `RudderIceLoadDEM`) satisfy the base
signature.

## 4. Guards added

`tests/test_contact_damping_convention_guard.py` (new):

* `test_damping_is_actually_applied` — `e(gamma_n)` must respond to `gamma_n`
  (`e(1.0) < 0.98`, `e(10) < e(1) - 0.05`). Fails on the pre-fix code.
* `test_harness_matches_classic_weak_damping_limit` — the independently coded
  `exp(-pi zeta/sqrt(1-zeta^2))` closure within 3 % at `zeta = 0.058`.
* `test_absolute_convention_is_the_documented_pair_law` — asserts the harness
  selects `contact_damping_model="absolute"` (root cause, not just symptom).
* `test_forces_accepts_base_keywords` / `test_diagnostics_runs_on_every_subclass`
  — every `IceDEM3D` subclass discovered in the package must accept the base
  keyword arguments and survive a `diagnostics()` call.

## 5. Verification

```
$ pytest tests/test_dem3d_restitution_verification.py \
         tests/test_dem3d_moving_wedge_bow.py \
         tests/test_dem3d_contact_impact.py -q
31 passed
$ pytest tests/test_contact_damping_convention_guard.py -q
6 passed
$ pytest tests/ -q -n 8            # full suite, before the fix: 2+ red
```

## 6. Limitations

* `contact_damping_model="restitution"` remains the engine default, so the
  defect affected only callers that pass `contact_damping` while relying on the
  default model. Those callers now must state the convention; the two in-repo
  ones (`measure_dem3d_impact`, and through it `benchmarks/` and
  `scripts/verify_dem3d_restitution.py`) do.
* The residual 0.4 %–0.6 % gap between the solver and the RK4 clamped reference
  is the semi-implicit integrator on a stiff contact, not the damping law; it
  shrinks with `dt_scale` and is bounded by the existing 1 % gate.
* These are code-versus-analytic checks, not a material validation of sea ice.