# Engine-closure restoration: the pair dashpot was never in the pair law (m2779)

**Repository** TensorDEM · **unit** selfstart m2779 · **protocol** `tensordem-engine-closure-restoration-v1`
**Artifacts** `artifacts/engine_closure_restoration/` · **Reproduce**

```
PYTHONPATH=src python scripts/verify_engine_closure_restoration.py
PYTHONPATH=src python -m pytest tests/test_dem3d_contact_impact.py \
    tests/test_dem3d_restitution_verification.py tests/test_restitution.py \
    tests/test_dem3d_pair_contact_damping.py -q     # 56 passed
```

## 1. Case description

`IceDEM3D` (the bonded-sphere 3-D engine behind every UCS, Brazilian, platen and
moving-wedge bow case in this repository) offers two normal-dashpot
conventions, documented in `DEM3DConfig`:

| `contact_damping_model` | pair damper `gamma_n` | who uses it |
|---|---|---|
| `"restitution"` (default) | derived from `restitution_coefficient`; `contact_damping` does **not** enter the pair law | legacy, bit-for-bit default |
| `"absolute"` | `gamma_n = contact_damping` [N s/m] | tool / platen branches, the clipped law `F = max(k*delta - c*v_n, 0)` |

The clipped law is exactly the object the two impact harnesses are meant to
measure, and `scripts/verify_dem3d_restitution.py` selects `"absolute"`
explicitly. The **library** harnesses did not:

* `tensordem.contact_restitution.measure_dem3d_impact` — the public measurement
  function used by `docs/dem3d_restitution_verification.md` and by the
  restitution-calibration loop,
* `tensordem.contact_impact._build_pair` — the two-sphere rig behind
  `docs/dem3d_contact_impact_analytic.md`.

Both built `DEM3DConfig(contact_damping=gamma_n, ...)` and left the convention
at its default. With `restitution_coefficient = 1.0` the pair law then computes
`gamma_n = 0`, so:

> every "measured restitution" produced by these two harnesses was a perfectly
> elastic impact (`e ≈ 1`) regardless of the damping installed.

Harness configuration: `K_n = 5000 N/m`, `rho = 917 kg/m^3`, `R = 0.025 m`
(`m = 6.003e-2 kg`), impact speed `0.5 m/s`, `dt = dt_crit`, drag `0`.

## 2. What was done

Two one-hunk source changes, no numerical default touched:

1. `src/tensordem/contact_restitution.py` — `measure_dem3d_impact` now passes
   `contact_damping_model="absolute"` (with a comment stating why: the
   `"restitution"` convention silently ignores `contact_damping`).
2. `src/tensordem/contact_impact.py` — `_build_pair` does the same.

The engine kernel is unchanged, so the shipped default trajectories are
bit-for-bit identical; what changed is that the verification apparatus now
measures the law it claims to measure. Nine previously red trunk tests turn
green (`tests/test_dem3d_contact_impact.py` 5, `tests/test_dem3d_restitution_verification.py` 4).

## 3. Results

Measured with the restored harness against the independent RK4 integration of
the clipped law (`clamped_restitution_ratio`), table produced by
`scripts/verify_engine_closure_restoration.py` (zero hand-copied numbers):

| `gamma_n` [N s/m] | `zeta` | `e` solver | `e` clipped reference | rel. error | `e` classic unclamped | rel. error of classic |
|---|---|---|---|---|---|---|
| 1 | 0.0289 | 0.91581 | 0.91480 | **0.11 %** | 0.91328 | 0.17 % |
| 3 | 0.0866 | 0.77470 | 0.77261 | **0.27 %** | 0.76105 | 1.50 % |
| 5 | 0.1443 | 0.65833 | 0.65972 | **0.21 %** | 0.63243 | 4.14 % |
| 10 | 0.2886 | 0.46323 | 0.46276 | **0.10 %** | 0.38787 | 16.2 % |
| 20 | 0.5773 | 0.25842 | 0.25901 | **0.23 %** | 0.10851 | 58.1 % |

Calibration loop check (target `e = 0.6`): `damping_for_target_restitution`
returns `gamma_n = 6.7349 N s/m` and the solver measures `e = 0.59749`
(**0.42 %** error); the classic unclamped mapping returns
`gamma_n = 5.5605 N s/m` and the solver measures `e = 0.63295` (**5.5 %** error,
13x worse). Elastic negative control (`gamma_n = 0`): `e = 1.00007`
(6.6e-5 from unity), i.e. the restored convention adds no spurious dissipation.

![restitution closure](artifacts/engine_closure_restoration/engine_closure.png)

Left: solver vs the clipped-law reference vs the classic unclamped mapping.
Right: closure error per `gamma_n` against the 3 % acceptance band.

## 4. Error verdict

| quantity | benchmark | worst relative error | < 3 % ? |
|---|---|---|---|
| `e` for `gamma_n` in [1, 20] N s/m | RK4 clipped-law reference | 0.27 % | **yes** |
| `e` at target 0.6 via calibrated `gamma_n` | requested target | 0.42 % | **yes** |
| `e` via classic unclamped mapping at `zeta = 0.577` | clipped-law reference | 58.1 % | **no** (documented limit of the classic mapping, not of the solver) |

The classic-mapping column is deliberately kept: it reproduces the published
finding that `2 zeta sqrt(k m)` is only valid in the weak-damping limit and that
the discrepancy is model (not time-step) error.

## 5. Open debt found and NOT landed: the platen contact mask

While auditing the red-test surface (42 failures on trunk `8cc88f7`) I found the
platen contact-mask fix documented as landed in
`docs/dem3d_material_calibration_roadmap.md` is **not on trunk**: it exists only
on the unmerged salvage commits `8ff1e7b / 0543b4a / 26e3447`, while its
regression test `tests/test_dem3d_platen_contact_mask.py` (4 contracts) *is* on
trunk and fails (3 red). Measured today on `8cc88f7` with
`artifacts/engine_closure_restoration/evidence.json`: a 192-particle block
between platens that touch nothing (`platen_damping = 15 N s/m`, rigid-body
`v_z = 0.05 m/s`) reports a top-platen reaction of **+144 N**, matching the
fictitious-load closed form `n * gamma_p * |v_z| = 192 * 15 * 0.05 = 144 N`
exactly. So every platen-driven benchmark (UCS, Brazilian, 3-point bending) is
currently loaded by a global drag-like term.

Applying the mask (2 lines, verified to turn the 3 platen tests green) also
breaks **2 published tests** in `tests/test_dem3d_restitution.py`
(`test_measured_restitution_tracks_closed_forms`, tolerance 3.5–5 %; measured
deviation 5.2 / 12.4 / 22.8 % after masking, plus
`test_unclamped_formula_fails_at_moderate_damping`). Those tolerances were fitted
against the leaky kernel, so the mask cannot be landed without re-deriving a
platen-specific restitution reference. That is a physics-contract decision, not
a line edit, so this unit deliberately does **not** change them: the mask is
left unlanded and the conflict is recorded here rather than papered over by
loosening the acceptance numbers.

Recommended next unit: derive the platen-impact reference for a *rigid
half-space* counterpart (the platen branch's effective mass is `m`, not `m/2`,
so the pair closed form does not apply as written), re-baseline the two platen
restitution tests against it, then land the mask together.

## 6. Remaining red surface (honest ledger, trunk `8cc88f7` + this unit)

| test module | failing | root cause | in scope? |
|---|---|---|---|
| `tests/test_dem3d_contact_impact.py`, `tests/test_dem3d_restitution_verification.py` | 9 → **0** | dashpot convention (this unit) | yes, closed |
| `tests/test_dem3d_platen_contact_mask.py` | 3 | mask only on unmerged salvage branch (§5) | no, needs reference re-derivation |
| `tests/test_ice_floe_physics.py` | 15 | `ice_floe_physics` is imported as a top-level module but the salvaged implementation (`2b31c69`) was never merged; `analytic_submerged_fraction` etc. are absent | no, separate salvage |
| `tests/test_fracture_gen_audit.py` | 17 | mass-mapping/footprint API collision (`721041a`); merging it silently deletes the audited `pi/6 - 1` honest-FAIL | no, owner decision |
| `tests/test_fracture_gen_mass_footprint.py` | collection error | same family | no, owner decision |

Full-suite state: `1070 collected, 1 collection error; 42 failed -> 33 failed` on
the run used for this audit (xdist `-n 8`, one 90 s pass), the reduction being
exactly the 9 tests above.