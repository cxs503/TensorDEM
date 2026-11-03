# Coulomb friction-cone verification for the DEM tangential contact model

## Purpose

`tensordem.contact_tangential` ships the Cundall-Strack tangential spring with a
Coulomb cap. The stateless helpers (`compute_tangential_force`,
`integrate_tangential_displacement`) clamp only the *force*; the stored
tangential displacement keeps growing while a contact slides. This round
verifies the corrected state-projected stepper
(`advance_tangential_contact`) against closed-form Coulomb anchors, and
quantifies the error of the previous stateless usage so the fix is evidenced, not
asserted.

No DEM3D production case is re-run here: the verified object is the contact
constitutive state machine, which is unit-scale and analytic.

## Model and the corrected increment

Per contact, with tangential displacement `delta_t`, unit normal `n`, tangential
relative velocity `v_t`, normal force magnitude `F_n`, tangential stiffness `k_t`
and friction coefficient `mu`:

```
delta_t <- delta_t + v_t * dt                      (integrate_tangential_displacement)
delta_t <- delta_t * min(1, delta_s / |delta_t|)   (project_tangential_displacement, NEW)
F_t      = k_t * delta_t
delta_s  = mu * F_n / k_t                          (saturation_displacement, NEW)
```

Contacts with `F_n = 0` are inactive: their stored displacement is reset
(`reset_tangential_displacement`), which removes the spurious tangential force a
new or detached contact would otherwise inherit.

## Analytic anchors

With the reference parameter set below, the closed-form predictions are

| quantity | analytic value |
|---|---|
| cone radius `delta_s = mu*N/k_t` | 0.003 m |
| saturation force `mu*N` | 30 N |
| sliding onset travel | `delta_s` = 0.003 m |
| force zero crossing after reversal | `delta_s` = 0.003 m |
| force reaches `-mu*N` after reversal | `2*delta_s` = 0.006 m |
| friction work to travel `s` | `k_t*delta_s^2/2 + mu*N*(s-delta_s)` |

Reference parameter set (constant normal load, block on plane, no rotation):

| parameter | value |
|---|---|
| normal force `N` | 100 N |
| tangential stiffness `k_t` | 10000 N/m |
| friction coefficient `mu` | 0.3 |
| tangential speed `v` | 0.02 m/s |
| time step `dt` | 0.0001 s |
| phase duration | 0.6 s |

## Results

Projected stepper, measured vs analytic:

| check | analytic | measured | relative error |
|---|---|---|---|
| stick force (delta_t = 1e-3 m) | 10 N | 10 N | 0 |
| sliding onset travel | 0.003 m | 0.003002 m | 0.000666667 |
| saturated force at end of phase | 30 N | 30 N | 0 |
| max abs(F_t) / (mu*N) over the phase | 1.0 | 1 | 0 |
| friction work to s = 0.012 m | 0.315 J | 0.31503 J | 9.52381e-05 |
| reversal: force zero crossing travel | 0.003 m | 0.003 m | 0 |
| reversal: travel to `-mu*N` | 0.006 m | 0.005997 m | 0.0005 |

Same protocol with the legacy stateless usage (force clamped, displacement left
unprojected):

| check | analytic | legacy measured | relative error |
|---|---|---|---|
| reversal: force zero crossing travel | 0.003 m | 0.012 m | 3 |
| reversal: travel to `-mu*N` | 0.006 m | 0.014997 m | 1.4995 |

Aggregate: max projected-stepper relative error is
0.000666667 (tolerance 0.03), legacy max
relative error 3. Verdict **PASS**.

The legacy error is a pure state-history artefact: the unprojected displacement at
the end of the first phase is `v*T = 0.012` m,
i.e. 4 times the cone
radius, so the contact only re-sticks after that much backward travel. Monotonic
sliding alone hides the defect (the force is capped either way); any reversal,
demoulding, or unloading path exposes it.

## Artifacts and reproduction

```bash
python scripts/verify_coulomb_friction_cone.py --output outputs/friction-cone --plot
python -m pytest tests/test_friction_cone_verification.py tests/test_contact_tangential.py -q
```

Outputs (all values above are read from the JSON, none transcribed by hand):

- `outputs/friction-cone/friction_cone_verification.json` — parameters, per-case
  metrics, error budget, verdict.
- `outputs/friction-cone/friction_cone_summary.csv` — one row per case.
- per-step histories `monotonic_sliding_history.csv`, `reversal_history.csv`,
  `legacy_reversal_history.csv` (F_t, abs(delta_t), travel, sliding flag) are
  written by the script but not committed (26k rows each); regenerate with the
  command above.
- `outputs/friction-cone/friction_cone_verification.png` — left: force vs backward
  travel for projected vs legacy; right: sliding onset at `delta_s`.

Tests: 32 pass (`tests/test_friction_cone_verification.py` plus the pre-existing
`tests/test_contact_tangential.py`). `pytest` runs the analytic screens directly
from the same script modules, so the doc and the tests cannot drift apart.

## Error judgement

Every projected-stepper anchor is within 0.03 relative
error; the largest is 0.000666667. **Is every key quantity < 3%? Yes.**
The residual is one time step of tangential travel
(`v*dt = 2e-06` m vs `delta_s = 0.003` m,
ratio 0.000667), i.e. discretisation of
the onset step, not a model error.

## Limits and remaining debt

- **Not yet wired into DEM3D.** `tensordem.dem3d` still has no tangential contact
  in its force loop (`hull_contact` / `dem3d_wedge` are frictionless). This round
  verifies the constitutive state machine only; coupling it to the solver pair
  list and re-running wedge resistance is separate work, and rolling resistance
  (`rolling_mu`) remains an unused placeholder.
- **No rotation term.** The Cundall-Strack increment here ignores particle spin
  (`omega x r`), so the model is exact for a block sliding on a plane and
  approximate for rolling/spinning grains. Torsional friction is not modelled.
- **No ice calibration.** `mu`, `k_t` are taken as given. Nothing here says which
  coefficients represent ice-steel or ice-ice contact; that requires an
  independent friction experiment, which this repo does not contain.
- **Not a solver benchmark.** These are closed-form single-contact checks. They
  do not certify DEM3D wedge/platen results, mesh/size sensitivity, or energy
  conservation in a full assembly.
