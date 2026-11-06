# Restart boundary matrix for `IceDEM` (interruption ±1 step)

Scope: `tests/test_coupling_restart.py` — the existing restart contract
(`test_restart_bitwise_and_fail_closed`) is kept unchanged; the new tests add
**boundary interruption points** and compare the restart record
**leaf by leaf, with exact equality (no tolerance)** before and after the
interruption step.

## Why leaf-wise, exact comparison

`snapshot()` is the complete JSON restart record of the bonded-disk lattice
(`positions`, `velocities`, `fixed`, `alive`, failure ledger, tool/hull motion,
accounting ledger, `last_step` diagnostics, clock). A restart that only
"resumes approximately" is indistinguishable from a solver regression once the
simulation is restarted as part of a long coupled run, so the contract here is
bitwise: after `from_snapshot()` the record must be identical, and one step
later the two branches must still produce identical records.

`_state_diffs()` walks dicts/lists recursively and reports the failing path,
e.g.

```
interruption state at step 1: 1 field(s) disagree;
state.last_step.external_work_J: 2.1487603305820196e-06 != 2.14876033058202e-06
```

so a mismatch never degrades into an opaque `dict == dict` failure.

## Scenario matrix

| Test | Interruption point | What is compared | Fail-closed checks |
|---|---|---|---|
| `test_restart_boundary_fieldwise_before_and_after_step[fix_edges, k]` | `k ∈ {0, 1, 2, 5, 11, 13}`, `fix_edges ∈ {False, True}` | JSON round-trip; state at the interruption point; state after 1 step; states after 3 further steps | `load_time = t ± dt` rejected on both sides of the interruption; clock/step_count preserved |
| `test_restart_at_first_failure_boundary_preserves_ledger` | the very step that writes the first fractured bond (mixed failed/alive ledger) | full record plus `failure_mode`, `failure_time_s`, `failure_extension_m`, `failure_energy_J`, `alive`; `fracture_release_J` after one further step (no loss, no double count) | checkpoint shifted by one step (`step_count + 1`) rejected |
| `test_restart_boundary_with_hull_contact_active` | after 6 steps with a prescribed hull polyline in contact (reaction > 0 asserted first) | full record, `hull_profile_m`, hull clock; state after 1 step | — |
| `test_state_comparator_flags_single_leaf_drift` | comparator self-check (metamorphic) | six single-leaf mutations: 1-ulp position, 1-ulp velocity, 1-ulp impulse component, 1-ulp `last_step` work, flipped `failure_mode`, drifted `step_count` | each mutation must be reported with its own path |

`k = 0` covers a checkpoint taken *before the first step* (empty `last_step`,
`time_s = 0`), `k = 1` the smallest non-trivial history, `k = 5/11/13` cover
steps straddling the 12-step marker used by the pre-existing test, and
`fix_edges=True` covers the constrained support path (support impulse and the
`positions[fixed] == initial_positions[fixed]` invariant).

## Negative controls (proof the new tests can go red)

`scripts/verify_restart_boundary_negative_controls.py` injects three
single-point defects into `IceDEM.from_snapshot` (restored coordinate off by one
ulp; restored `last_step.external_work_J` off by one ulp; fracture ledger
zeroed on restore), reruns the target file, and records the outcome in
`artifacts/coupling_restart_negative_controls.json`. The source file is
restored in a `finally` block and re-hashed.

Reproduce:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/verify_restart_boundary_negative_controls.py
```

Recorded outcome (see the artifact for the full list of failed test ids):

| injection | result |
|---|---|
| clean tree | `19 passed` |
| `ulp_position` | `15 failed, 4 passed` |
| `last_step_work_ulp` | `12 failed, 7 passed` |
| `drop_failure_ledger` | `2 failed, 17 passed` (the fracture-boundary and hull-contact cases) |

## Notes / limits

* Comparison is exact on the JSON record, not on in-memory tensors: the
  tolerances of the physical checks (`< 15 %` vs. reference, momentum/energy
  invariants) are untouched — no existing assertion was relaxed or removed.
* Only `IceDEM` (2-D bonded disks) is covered here. The calibrated per-bond and
  wet-coupling records (`CalibratedIceDEM`, `CalibratedWetCoupling`) have their
  own restart tests; extending this matrix to them is the natural next step.
* Repo-wide `ruff check .` is red on the parent commit as well (561 findings,
  mostly unused imports in tests); this change introduces none and keeps the
  touched files clean (`ruff check tests/test_coupling_restart.py scripts/...` →
  `All checks passed!`).