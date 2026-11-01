# TensorDEM main CI red (2026-10-31 → ): external nodal loads dropped in `IceDEM3D.forces`

**Status:** root-caused, fixed, full suite green. **Classification: physics regression (not a
stale test assumption). No assertion, tolerance or acceptance threshold was modified.**

## 1. Symptom

`main` was red for three consecutive runs (`9c9ff9a`, `bfdb5e3`, `99c49c0`, all ≥ 2026-10-31).
Exactly 4 tests failed with one signature — every failure came from the same acceptance check
inside the DEM3D bending fixtures:

| test | failing signature |
| --- | --- |
| `tests/test_dem3d_three_point_bending.py::test_load_transfer_is_finite_and_repeatable` | `RuntimeError: three-point-bending benchmark failed: ['support_load_transfer', 'nonzero_midspan_deflection']` |
| `tests/test_dem3d_three_point_bending.py::test_fixed_duration_time_step_refinement_preserves_duration` | same (fixture raises before the duration assertion) |
| `tests/test_dem3d_bending_resolution.py::test_campaign_is_finite_repeatable_and_exports_artifacts` | same, via `verdict != "PASS"` |
| `tests/test_dem3d_bending_load_convergence.py::test_refinement_campaign_emits_ordered_levels_and_comparisons` | same, via `verdict != "PASS"` |

Reproduced locally at `99c49c0` with `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/ -q`:
`4 failed, 656 passed, 1 skipped` (the 4 failures above).

Note: the dispatch hint pointed at PR #69/#70 (`d134312`/`e2baeab`). That is **not** the culprit —
`a68fa8a` (PR #70 merge) is green. The regression entered three merges later.

## 2. Bisect evidence chain

```
git bisect start origin/main a68fa8a          # 24 commits in range
bad  : 098c655  Merge pull request #76        (2 failed, 2 passed)
bad  : a91da9d  Merge pull request #73        (2 failed, 2 passed)
good : 94e3b06  Merge pull request #71        (all passed)
good : b7c5544  Merge pull request #72        (all passed)
# first bad commit: a40e29a  "m670: matured mission deliverables"   (PR #73)
```
Criterion: `python3 -m pytest tests/test_dem3d_three_point_bending.py -q`.

`a40e29a` is a direct child of `a68fa8a` and is the only commit in `a68fa8a..origin/main`
that touches `src/tensordem/dem3d.py` (`git log -S "force += loads" -- src/tensordem/dem3d.py`
returns it as the removal commit).

## 3. Root cause

`a40e29a` refactored `IceDEM3D.forces()`/`step()` (docstring/`last_reaction` cleanups, explicit
Euler with acceleration masking) and, in the same edit, deleted:

```diff
-        if loads is not None:
-            force += loads
```
and in `step()`
```diff
-        self.velocities[self.fixed] = 0
-        self.positions[self.fixed] = self.initial_positions[self.fixed]
-        if not ...: raise FloatingPointError("non-finite DEM3D state; ...")
```
and flipped one sign in the bottom-platen damper
```diff
-                          - cfg.platen_damping * bottom_relative_v).clamp_min(0.0)
+                          + cfg.platen_damping * bottom_relative_v).clamp_min(0.0)
```

`forces()` still *validated* the incoming `external_forces` tensor through `self._loads()` and
then discarded it. Consequence: **every externally loaded DEM3D case became a silent no-op** —
no exception, no NaN, just zero motion. That is why the failure signature is "zero support
reaction / zero midspan deflection" rather than a numerical blow-up, and why the three bending
campaigns (the only tests that drive the lattice with nodal loads) are the only casualties:
the tool/indenter and platen code paths never go through `loads`.

Derivation for the platen sign: with inward normal `n` the contact force is
`F = (k·ov − c·(v_particle − v_platen)·n) n`. For the top platen `n = −ẑ`, so
`F_z = −(k·ov + c·(v_z − v_platen))` (unchanged, correct). For the bottom platen `n = +ẑ`, hence
`F_z = k·ov − c·(v_z − v_platen)`: the new `+c` term is a **negative damper** (it feeds energy in
on approach). The existing platen test uses `platen_damping = 0.0`, so it could not catch this.

## 4. Fix (minimal, 15 lines in one file)

`src/tensordem/dem3d.py`
1. `forces()`: re-add the external nodal loads to the internal force vector
   (`force = force + loads`), with a comment marking it as a regression-critical term.
2. `step()`: re-pin fixed supports (`velocities[fixed] = 0`, `positions[fixed] =
   initial_positions[fixed]`) and restore the non-finite state guard — the released contract of
   `a68fa8a`, and what the `EnergyAuditedIceDEM3D` subclass already does. (Not required for the
   4 red tests; restored for contract consistency.)
3. bottom platen: damping back to `− c·(v_z − v_platen)`.

`tests/test_dem3d_external_load_balance.py` (new, 6 tests) pins the invariants with exact
identities, so the same silent loss cannot recur:
`forces(loads) − forces() == loads` (bitwise), `Σ F_z == Σ loads_z` for `drag = 0`,
peak reaction inside the applied-load envelope, fixed supports byte-identical to their initial
positions after 20 loaded steps, and the bottom/top platen damper deltas
`ΔF = ± c·Δv·N_bottom` (exact to 1e-12 N).

**Guard check:** on the pre-fix tree 4 of the 6 new tests fail
(`external_loads_are_added…`, `vertical_force_resultant…`, `loaded_benchmark_transfers…`,
`bottom_platen_damper_opposes_approach`); after the fix all 6 pass. The two that pass either way
are the pinned-support test (velocity never becomes non-zero spontaneously) and the top-platen
test (that sign was never broken) — reported as such rather than removed.

## 5. Numerical evidence (`docs/dem3d_ci_regression_metrics.json`, `docs/ci_regression_evidence.py`)

Identical three-point-bending fixture (`benchmarks/dem3d/three_point_bending_cases.json`,
81 particles / 572 bonds / 6 support particles / 3 loaded particles, `dt = 1.805915e-05 s`,
24 steps, `peak_load = 0.02 N`):

| solver state | peak \|R_z\| (N) | peak \|w_mid\| (m) | acceptance |
| --- | --- | --- | --- |
| PRE_REGRESSION `a68fa8a` | 6.582665e-06 | 5.986521e-08 | PASS |
| BROKEN `a40e29a` | 0.0 | 0.0 | FAIL (`support_load_transfer`, `nonzero_midspan_deflection`) |
| FIXED (this branch) | 6.582665e-06 | 5.986521e-08 | PASS |

`FIXED` vs `PRE_REGRESSION`: peak-reaction relative error **0.0**, peak-deflection relative error
**0.0**, per-step max abs difference in `R_z` **0.0 N** — the solver reproduces the last green
release bit-for-bit, i.e. the repair restores the original physics instead of relaxing the test.

Post-fix load-convergence campaign (`base_steps = 8`, `levels = 2`, `peak_load = 0.01 N`):
verdict `PASS`, levels `[8, 16]`, support-reaction relative change
8→16 steps = **91.95 %**, midspan-deflection relative change = **70.35 %**, no bond breakage.
These are the expected non-converged values of an 8-step ramp (effective loading rate doubles with
step count) — the campaign measures the correct dynamic response again, and the ordering
(`runs[1].ramp_duration_s > runs[0].ramp_duration_s`) holds.

Bottom-platen damper probe (10 bottom particles, `k_p = 4000 N/m`, `c_p = 50 N·s/m`,
`Δv = 0.03 m/s`, expected damper delta `c·Δv·N = 15 N`):

| state | ΔF on approach (N) | ΔF on separation (N) |
| --- | --- | --- |
| BROKEN | **−15** (anti-damping) | +15 |
| FIXED | **+15** | −15 |

## 6. Acceptance

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/ -q
666 passed, 1 skipped, 64 subtests passed        # 0 failed
```
Baseline before the fix (same checkout, `99c49c0`): `4 failed, 656 passed, 1 skipped`.
666 = 656 recovered + 4 repaired + 6 new regression tests.

## 7. Caveats / scope

* The 4 red tests only needed fix #1. Fix #3 (platen sign) is a latent negative-damping defect in
  the same commit found during the audit; it is covered by new tests but no CI-green test depends
  on it. Fix #2 (pinning + guard) is a contract restoration, likewise not required for the reds.
* Timing-dependent flakiness was not observed: the suite is deterministic here (single-threaded
  torch, hashed signatures are reproducible), so the red CI was a genuine logic regression.
* Scope: `src/tensordem/dem3d.py` + one new test module + this note. `benchmarks/`, `scripts/` and
  the four failing test files are untouched (no threshold change anywhere).
