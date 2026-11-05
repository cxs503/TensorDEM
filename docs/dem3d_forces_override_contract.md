# The `forces()` override contract: a silent capability outage in the 3-D DEM solver

Scope: `src/tensordem/dem3d_wedge.py`, `src/tensordem/rudder_ice_load.py`
Evidence: `artifacts/forces_override_contract/` (`audit.json`, `probe_invariance.csv`, `forces_override_contract.png`)
Guards: `tests/test_dem3d_forces_override_contract.py` (7 tests, 5 subtests)
Reproduce: `PYTHONPATH=src python3 scripts/verify_forces_override_contract.py`

## 1. The defect: the solver's monitoring path was dead on `main`

`IceDEM3D.forces` takes two keyword-only switches:

| switch | meaning |
|---|---|
| `update_fracture=False` | evaluate the stored bond topology, commit nothing |
| `update_history=False` | evaluate constitutive forces **without** advancing the Cundall–Strack tangential displacement or the diagnostic rolling-torque accumulator |

`update_history` exists because of a previously fixed reproducibility defect: with
the default, merely *monitoring* a friction simulation adds one `v_tangent · dt`
slip increment per call and therefore changes the trajectory it is measuring
(see `src/tensordem/contact_invariants.py`). `IceDEM3D.diagnostics` therefore probes with

```python
self.forces(external_forces, update_fracture=False, update_history=False)
```

Two subclasses override `forces` with the **pre-`update_history` signature** and so
reject that call. The consequence on a pristine checkout was not a subtle drift but
a hard `TypeError` on *every* diagnostic snapshot:

```
TypeError: MovingWedgeBowIceDEM3D.forces() got an unexpected keyword argument 'update_history'
TypeError: RudderIceLoadDEM.forces() got an unexpected keyword argument 'update_history'
```

Six tests were red on `main`, in three modules, all from this one cause:

| test module | red tests |
|---|---|
| `tests/test_threefield.py` | 2 |
| `tests/test_dem3d_moving_wedge_bow.py` | 3 |
| `tests/test_dem3d_wedge_resistance_sensitivity.py` | 1 |

That is the whole moving-wedge bow-resistance capability plus the three-field
hull–ice pipeline: `run_hull_ice_case` could not complete a single pass, because it
calls `sim.diagnostics()` in its step loop.

Impact by evidence grade, before and after:

| capability | before | after |
|---|---|---|
| moving-wedge bow resistance (冰区航行) | **STUB** — importable, untraversable | REAL (runs, anchored) |
| three-field hull–ice coupling | **STUB** — `TypeError` per pass | REAL (runs, anchored) |
| rudder DEM–FSM load report (`diagnostics`) | **WITHHELD** — unreachable | REAL (reachable, anchored) |

## 2. Why the rudder case was *invisible*

`RudderIceLoadDEM` never failed a test, because its tests drive `step()`, and
`step()` calls `forces` with the defaults. Only the *probe* path passed
`update_history`. Reproduced against the pristine `HEAD` sources:

```
pre-fix step():        ok
pre-fix diagnostics(): TypeError -> RudderIceLoadDEM.forces() got an unexpected keyword argument 'update_history'
```

So the class was fully usable for integrations but could not report a reaction,
pressure or FSM state — its entire public monitoring surface, and the only route to
`blade_load_N` / `blade_effective_pressure_MPa`, was unreachable. No test asserted
that `diagnostics()` is *callable* on a subclass, which is exactly the gap the new
guard closes.

## 3. The fix (minimal, 25 lines)

1. `MovingWedgeBowIceDEM3D.forces` accepts and forwards `update_history`.
2. `RudderIceLoadDEM.forces` accepts and forwards `update_history`.
3. The wedge override also stopped rebinding `self.last_reaction = bow_reaction` and
   now copies into the preallocated tensor, gated on the owning call:

```python
if update_fracture:
    self.last_reaction.copy_(bow_reaction)
```

Reason (3) is a real second defect, not cosmetics. The base class documents that a
probe leaves "the stored last reaction" untouched, and checkpoint restore relies on
`last_reaction.zero_()` — an **in-place** write that only works if the tensor keeps
its identity. Rebinding broke that in three ways: the probe committed a reaction it
was only asked to measure; a caller's returned reaction aliased live solver state, so
scribbling on the result moved the solver; and any cached reference desynchronised.
The bow contact is a memoryless function of (positions, velocities, time), so
`update_history` has nothing extra to guard in the wedge term itself — it is
forwarded because the inherited friction/rolling accumulators do accumulate.

## 4. Verification

```
$ python -m pytest tests/test_dem3d_moving_wedge_bow.py \
      tests/test_dem3d_wedge_resistance_sensitivity.py \
      tests/test_threefield.py tests/test_rudder_ice_load.py -q
57 passed                                  # was 6 failed, 51 passed

$ python -m pytest tests/test_dem3d_forces_override_contract.py -q
7 passed, 5 subtests passed

$ PYTHONPATH=src python3 scripts/verify_forces_override_contract.py
```

**Negative control** (the new tests must fail on the old code, or they prove nothing):
restoring the two pristine `HEAD` sources gives *8 failed, 3 passed*; with the fix,
all pass. The static signature test alone catches both offenders.

Analytic anchors — exact identities, no fitted constants:

| anchor | worst relative error | < 3 % ? |
|---|---|---|
| bow reaction = −Σ nodal bow force | 0.0 (bitwise) | yes |
| `bow_contact_energy_J` = ½ k Σ overlap² | 0.0 (bitwise) | yes |
| repeated probe is bitwise identical | 0.0 (bitwise) | yes |
| stored `last_reaction` drift across probes | **0.0 N, exactly** | yes |

The probe-invariance figure shows the discriminating construction: the bow is
advanced between probes so the *measured* reaction rises, while the stored trace
stays flat — with the old rebinding code the two coincide.

Capability check after the repair (unit-width, `n_length=5`, `n_width=2`, 40 steps):

| ice thickness | layers | peak resistance | max contact particles | broken bonds |
|---|---|---|---|---|
| 0.15 m | 3 | 80.04 N | 2 | 40 |
| 0.25 m | 5 | 80.04 N | 2 | 21 |

## 5. Limitations (do not over-read these numbers)

- **No absolute resistance benchmark.** These are short (40-step) engagement
  windows showing the pipeline traverses end to end; they are *not* a validation of
  ice resistance against mT–Uikku or model tests. The two cases peak at the same
  80.04 N, so **no thickness sensitivity is claimed** — the contact set saturates at
  2 particles before the crushing branch develops. A resistance trend needs the full
  400-step campaign, which is a separate task.
- **Lattice-parity artefact.** `layers = round(thickness / 2r)`; an even layer count
  with `tip_height_fraction = 0.5` puts the wedge tip exactly *between* two layers, so
  a 0.10 m case (2 layers) records zero contact in the same window. That reads as
  "zero resistance" but means "no engagement". Only same-parity thicknesses are
  comparable; the audit asserts `max_contact_particles > 0` so a null result cannot
  be plotted as physics.
- `resolution` of the guard is limited to `IceDEM3D` subclasses whose modules are
  imported; the test imports `ridge_dem` and `threefield.hull_ice` for registration.
- Not touched: the 2-D hierarchy (`dem.py`) and `calibrated_dem`/`continuum_boundary_dem`
  wrappers, which have independent `forces` signatures and do not share this probe
  protocol.