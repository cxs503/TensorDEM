# Moving-wedge bow: the inherited `forces()` probe contract was narrowed (m3296)

Protocol: `tensordem-probe-contract-repin-v1`
Scope: `src/tensordem/dem3d_wedge.py` (one signature), `tests/test_dem3d_wedge_bow_forces_contract.py` (new).
No numerical path, no default parameter and no published artifact was changed.

## 1. The defect: a subclass that silently narrowed its parent's API

`tensordem.dem3d.IceDEM3D.forces()` owns two keyword-only ownership flags:

| flag | meaning |
|---|---|
| `update_fracture=True` | commit the fracture/bond-topology history of this evaluation |
| `update_history=True` | advance the Cundall-Strack tangential slip and the diagnostic rolling-torque accumulator |

The base-class docstring states the *contract*: **probing routines (diagnostics, energy audits, invariance checks) must pass
`update_history=False`**, because with the default every call adds one `v_tangent * dt` slip increment, i.e. merely
monitoring a simulation changes its trajectory.

`MovingWedgeBowIceDEM3D` (`src/tensordem/dem3d_wedge.py`) overrides `forces()` to append the rigid bow contact, but it kept
the older two-argument signature (`external_forces`, `update_fracture` only). A subclass override that is *narrower* than
the parent is an API-contract break, and its failure mode is not a wrong number — it is a `TypeError` on every probe:

```
TypeError: MovingWedgeBowIceDEM3D.forces() got an unexpected keyword argument 'update_history'
```

A pristine `main` checkout of this worktree therefore carried **45 red tests**. Grouped by root cause they are:

| # failing | test module(s) | root cause | class |
|---|---|---|---|
| 6 | `test_dem3d_moving_wedge_bow.py` (3), `test_dem3d_wedge_resistance_sensitivity.py` (1), `test_threefield.py` (2) | **this document** — narrowed `forces()` signature | real `src` bug |
| 14 | `test_fracture_gen_audit.py` | `FractureGenConfig.__init__() got an unexpected keyword argument 'face_extent'`; `mass_mode` alias set rejects `cell_volume` / `equal_volume_spheres` | test+doc merged, `src` hunks absent |
| 16 | `test_ice_floe_physics.py` | `benchmarks/ice_floe_physics.py` lacks `analytic_submerged_fraction`, `anchor_archimedes_static`, `anchor_heave_period`, `anchor_friction_slip`, `anchor_restitution_mapping`, `run_all` | test merged, script hunks absent |
| 4 | `test_dem3d_restitution_verification.py` | accuracy assertions fail: `0.1669 > 0.03` and `0.2766 > 0.015` on the classic-vs-clamped restitution mapping | open numerical question |
| 4 | `test_doc_evidence_pointers.py` | unresolved evidence pointers (e.g. `docs/dem3d_contact_impact_dashpot_convention.md` → `scripts/…` token) | doc evidence chain |
| 1 | `test_wake_lbm_analytic_verification.py` | independent Smagorinsky evaluation differs by `1.83e-3` against a `1e-12` tolerance | open numerical question |

Only the first row is repaired here; the other four groups are recorded above as the next tickets' anchors (see §5).

## 2. Fix (minimal)

```python
def forces(self, external_forces=None, *, update_fracture: bool = True,
           update_history: bool = True):
    force, _ = super().forces(external_forces,
                              update_fracture=update_fracture,
                              update_history=update_history)
    bow_force, bow_reaction, _ = self._wedge_contact()
    ...
```

The bow contribution itself is unchanged: it is evaluated from the *current* positions/velocities and carries no history
state, so `update_history` is a pure pass-through. Nothing else in the module refers to the flag.

## 3. Verification

`tests/test_dem3d_wedge_bow_forces_contract.py` (new, 6 tests) pins the contract rather than a number:

| test | assertion |
|---|---|
| `test_signature_exposes_base_kwargs` | both flags exist and are `KEYWORD_ONLY`, mirroring the parent |
| `test_probe_call_does_not_raise` | `forces(update_history=False)` returns a finite `(N, 3)` force and a finite bow reaction |
| `test_stepping_call_still_works` | the default path is unaffected; `diagnostics()` bow reaction is not NaN |
| `test_repeated_probe_is_idempotent` | two probe evaluations are bit-identical (with `update_fracture=False`, which isolates the friction channel from the one-shot topology commit) |
| `test_probe_does_not_advance_slip_history` | the stored `_tangential_state` displacement is unchanged after three probes |
| `test_stepping_call_advances_slip_history` | a stepping call *does* move the stored displacement (skipped when the test lattice has no load-bearing contact pair) |

Results (this worktree, `PYTHONPATH=src`, CPU):

| selection | before | after |
|---|---|---|
| `test_dem3d_wedge_bow_forces_contract.py` | module absent | **5 passed, 1 skipped** |
| `test_dem3d_moving_wedge_bow.py` + `test_dem3d_wedge_resistance_sensitivity.py` + `test_threefield.py` | 6 failed, 27 passed | **33 passed** |
| full suite `pytest -q` (baseline, before the fix) | 45 failed, 1187 passed, 2 skipped (704 s, CPU) | not re-run in this ticket (11 min single-process run; the five affected modules above were re-run instead) |

## 4. Why this matters for the ship-ice work

`MovingWedgeBowIceDEM3D` is the engine path behind the moving-wedge bow-resistance campaign
(`docs/dem3d_moving_wedge_bow.md`, `docs/dem3d_wedge_resistance_sensitivity.md`). With the narrowed signature, **no
side-effect-free probe of that path existed**: any attempt to measure a force/reaction snapshot, or to audit energy at a
given state, either raised `TypeError` or silently took the stepping branch and perturbed the friction history it was
supposed to observe. The `test_threefield.py` coupling ledgers (`HullIceTests`, `CouplingTests`) were failing for the same
reason, i.e. the coupled mass ledger could not even be evaluated.

## 5. Residual (honest, not closed here)

* `test_fracture_gen_audit.py` (14) and `test_ice_floe_physics.py` (16) are the familiar "tests+docs merged, `src`/`scripts`
  hunks dropped" debt. They need feature restoration (`face_extent` plane-extent area/mass mapping; the ice-floe anchor
  functions), not a signature fix, and are left as the next ticket's anchor. Their evidence grade on `main` today is
  `WITHHELD`, not REAL, no matter what the documents claim.
* `test_dem3d_restitution_verification.py` (4): the classic Sudduth/`ζ`-mapping disagreement (16.7 % / 27.7 % vs the 3 % /
  1.5 % assertions) is *not* the C4 closure reported earlier (that closed the engine-loop round trip, ≤0.27 %); the failing
  assertions compare two *analytic mappings* and are a genuinely open numerical question.
* `test_wake_lbm_analytic_verification.py` (1): a `1.8e-3` closure mismatch against a `1e-12` tolerance is either a real
  Smagorinsky formula defect or an over-tight assertion; both readings need their own ticket.
* `test_doc_evidence_pointers.py` (4): doc→script pointer chain, unrelated to the engine.