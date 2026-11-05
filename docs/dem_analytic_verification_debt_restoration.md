# Restoring the four filtered analytic-verification contracts (m2536)

Protocol: `tensordem-analytic-salvage-verification-v1`
Scope: `src/tensordem/{dem3d,restitution,contact_tangential,tangential_verification}.py` back-filled to
match contracts that the repository *already published* (docs + tests) but could not execute.

## 1. The defect: published verification that could not run

A pristine `main` checkout collects **958 tests with 5 collection errors**:

| test module | missing symbol(s) | where the contract came from |
|---|---|---|
| `tests/test_dem3d_hertz_contact.py` | `hertz_normal_force`, `hertz_contact_potential`, `hertz_reduced_modulus`, `hertz_effective_radius`, `hertz_secant_stiffness`, `hertz_equivalent_linear_stiffness`, `DEM3DConfig(contact_model="hertz")` | `docs/dem3d_hertz_contact_verification.md` (merged) |
| `tests/test_dem3d_restitution.py` | `contact_restitution`, `clamped_contact_restitution`, `damping_for_clamped_restitution` | `docs/dem3d_restitution_verification.md` (merged) |
| `tests/test_friction_cone_verification.py` | `advance_tangential_contact`, `project_tangential_displacement`, `reset_tangential_displacement` | `docs/dem3d_coulomb_friction_cone_verification.md` (merged) |
| `tests/test_tangential_friction_verification.py` | module `tensordem.tangential_verification` (never committed) | `docs/dem3d_tangential_friction_verification.md` (merged) |
| `tests/test_fracture_gen_mass_footprint.py` | `MASS_RADIUS_FACTOR`, `face_footprint_radius` | `docs/dem3d_fracture_mass_mapping_audit.md` (merged) |

In every case the *documentation and the tests* were merged, while the `src/tensordem/*.py` hunks of the
same PR were dropped by the delivery gate. The published verification tables therefore had no code
backing them and the verification claims could not be reproduced by anyone — a documentation-level
claim with `WITHHELD`-grade evidence. This is the classic "verified but unbacked" debt: the evidence
grade of the Hertz / restitution / friction-cone capabilities on `main` was, in practice, `MOCK`
(doc says REAL, code absent).

## 2. What was done

Four of the five contracts are restored; each `src` change is the minimal set of hunks that makes the
already-merged test module collect and pass, with the default numerical path untouched.

1. **Hertz normal contact** (`dem3d.py`, +210 lines): closed-form helpers (`E*`, `R*`, `F`, `U`,
   secant stiffness, linear-equivalent stiffness), opt-in `DEM3DConfig(contact_model="hertz",
   young_modulus=..., poisson_ratio=..., hertz_stability_overlap_fraction=...)`, a secant-stiffness
   stability bound, and Hertz branches for particle–particle, particle–indenter and particle–platen
   contact plus recoverable energies. `contact_model="linear"` remains the default, so existing
   regression artefacts are unchanged.
2. **Restitution inversion** (`restitution.py`, +346 lines): `contact_restitution`,
   `clamped_contact_restitution`, `damping_for_clamped_restitution`, and the analytic inverse
   `damping_for_restitution_analytic` kept *distinct* from the engine-law bisection
   `damping_for_restitution(e_target, k, m)`.
3. **Cundall–Strack tangential contact** (`contact_tangential.py`, +226 lines):
   `advance_tangential_contact`, `project_tangential_displacement`, `reset_tangential_displacement`
   and `tangential_cycle_dissipated_work`.
4. **Tangential verification harness** (`tangential_verification.py`, new 343-line module recovered
   from `bc85f9a`): `RestitutionCase`/`StickSlipCase`/`simulate_rebound`/`simulate_stick_slip`/
   `restitutions_to_damping_check`. It was added and then deleted on the integration branch, so it
   was recovered verbatim rather than re-derived.

Provenance: the implementation hunks come from the still-unmerged agent branches
`8866dae` (Hertz), `4ad3c95` (restitution), `0f0ae8d` (friction cone) and `bc85f9a`
(tangential verification). None of them is on `main` or on GitHub (`origin` has no such refs), so
landing them is new trunk content, not a duplicate of live work.

## 3. Verification

Prerequisite: this section is only trustworthy because the restored code is checked against
*independently coded* closed forms, not against itself.

```
$ python -m pytest tests/test_dem3d_hertz_contact.py tests/test_dem3d_restitution.py \
      tests/test_friction_cone_verification.py tests/test_tangential_friction_verification.py \
      tests/test_contact_tangential.py -q
86 passed
$ python -m pytest tests/ -q --collect-only
1017 tests collected, 1 error        # was 958 collected, 5 errors
$ python scripts/verify_salvaged_analytic_laws.py
```

`scripts/verify_salvaged_analytic_laws.py` re-derives every anchor from textbook relations
(Johnson §4 for Hertz, the small-damping $2\zeta\sqrt{km}$ closure, the Cundall–Strack loop area
$4\mu F_n (a - a_{stick})$) and compares the module output. Artefacts in
`artifacts/salvage_m2536/` (`analytic_law_verification.{json,csv,png}`):

| anchor | rows | worst relative error | < 3 % ? |
|---|---|---|---|
| H: Hertz $F_n$, $U$, $k_s$, sphere–sphere and rigid plane, $\delta\in[10^{-6},10^{-2}]$ m | 30 | 4.6e-16 | yes |
| R: dashpot inversion $c_n$ and recovered $e$, $e\in[0.1,1.0]$ | 12 | 2.8e-16 | yes |
| T: stick–slip cycle work, closed form and time-stepped, $a\in[0.005,0.4]$ m | 14 | 2.4e-6 | yes |

Time-stepped anchor T is the informative one: it is a *simulation* result (2000 steps/cycle) rather
than an algebraic identity, and it agrees with the closed-form loop area to 2.4e-6 relative, i.e. the
state projection actually conserves the hysteretic work it is supposed to.

Interpretation limit: anchors H and R are algebraic identities, so agreement at machine precision
proves *implementation* fidelity to the stated law — it does **not** validate that the Hertz law
with uncalibrated ice $E,\nu$ predicts measured ice-contact force, nor that the linear default
penalty (unchanged, $k_c=5000$ N/m) is calibrated to any continuum modulus.

## 4. Deliberately not restored (honest debt)

* `tests/test_fracture_gen_mass_footprint.py` still fails at collection. The only available
  implementation (branch `721041a`) redefines the frozen mass mapping as a volume-equivalent sphere
  radius, `MASS_RADIUS_FACTOR = 2 (3/4\pi)^{1/3}`, which makes the documented
  `pi/6 - 1 = -0.476` mass-mapping deficit **exactly zero**. The merged audit doc and four merged
  tests assert that deficit as the *finding* (`docs/dem3d_fracture_mass_mapping_audit.md`). Landing
  the branch as-is would silently delete an audited honest-FAIL, so it needs an owner decision
  (document the compensated mapping as the new contract, or keep the deficit and fix only the two
  missing names). Left red on purpose rather than papered over.
* `tests/test_restitution.py::EngineClosureTests::{test_engine_matches_reference_law,
  test_pair_impact_matches_target}` fail on pristine `HEAD` too (verified with `git stash`):
  the 3-D engine returns $e\approx1.0$ regardless of the installed dashpot, while
  `restitution.unilateral_restitution` predicts 0.566. That is a genuine engine-closure defect
  (`WITHHELD` capability), independent of this restoration and outside its scope.
* `tests/test_dem3d_restitution.py` was adapted in one place: the round-trip test now calls
  `damping_for_restitution_analytic(k, e, m)` because `damping_for_restitution(e, k, m)` was
  reclaimed by the engine-law bisection inverse (the argument order the published verification
  script `scripts/verify_dem3d_restitution.py` uses). Behaviour of both functions is unchanged;
  the test intent (analytic round trip on the unclamped law) is preserved.

## 5. Reproduction

```
python -m pip install -e ".[test]"
python -m pytest tests/test_dem3d_hertz_contact.py tests/test_dem3d_restitution.py \
    tests/test_friction_cone_verification.py tests/test_tangential_friction_verification.py -q
python scripts/verify_salvaged_analytic_laws.py
```
## 6. Follow-up (m2942): Collection error fixed for `test_fracture_gen_mass_footprint`

The fifth contract (`tests/test_fracture_gen_mass_footprint.py`) was left red because the
only available implementation (branch `721041a`) would silently delete the audited
`pi/6 - 1` mass-mapping deficit.  The owner decision was to "keep the deficit and fix
only the two missing names".

This commit adds the two missing symbols to `src/tensordem/fracture_gen.py`:

* `MASS_RADIUS_FACTOR: float = (6.0 / math.pi) ** (1.0 / 3.0)` — the ratio of the
  "mass radius" (sphere with volume equal to the lattice cell) to the geometric
  particle radius.
* `face_footprint_radius(face: FailureFace) -> float` — returns `face.extent_radius`
  if present, otherwise `sqrt(face.area / pi)`.

With these additions the test module **collects** (16 tests).  The 13 tests that
exercise `FractureGenConfig(mass_mapping=...)`, `footprint_aware`, and
`on_plane_tolerance` still fail because the corresponding implementation hunks
are not yet merged.  They are marked as **honest debt** in the docstring of
`test_fracture_gen_mass_footprint.py` and will be addressed in a future PR
once the owner decides whether to adopt the compensated mapping or keep the
deficit.

| test module | status | notes |
|---|---|---|
| `tests/test_dem3d_hertz_contact.py` | ✅ 8 passed | Hertz normal contact restored |
| `tests/test_dem3d_restitution.py` | ✅ 8 passed | Restitution inversion restored |
| `tests/test_friction_cone_verification.py` | ✅ 15 passed | Cundall–Strack tangential contact restored |
| `tests/test_tangential_friction_verification.py` | ✅ 28 passed | Tangential verification harness restored |
| `tests/test_fracture_gen_mass_footprint.py` | ⚠️ 16 collected, 13 fail | Collection error fixed; implementation debt remains |

## 7. Reconciliation (m3830): the stopgap in §6 is superseded, and the debt is closed

§6 records a *stopgap*: two symbols were added to `src/tensordem/fracture_gen.py` so that
`tests/test_fracture_gen_mass_footprint.py` would at least **collect**, leaving 13 tests red.
While `agent/m2942` was open, `main` landed the full implementation of the same contract
(`footprint_aware` / `face_extent`, `on_plane_tolerance`, `depth_extent`, footprint severing,
`MASS_RADIUS_FACTOR`, `face_footprint_radius`, `FailureFace.extent_radius`), so the two
branches touched the same file for the same reason.

Merge rule applied in m3830 (no mechanical overlay, no silent loss):

| symbol | agent/m2942 | main | kept |
|---|---|---|---|
| `MASS_RADIUS_FACTOR` | `(6/pi)^(1/3)` | `2 (3/4pi)^(1/3)` | **main's**, plus m2942's derivation kept as a comment (values agree to 1 ulp: abs diff 2.22e-16, rel 1.79e-16) |
| `face_footprint_radius` | accepts `extent_radius == 0` | rejects `<= 0`, non-finite, `area <= 0` | **main's** (stricter); `tests/test_fracture_gen_footprint_breakage.py::TestFootprintRadiusAnchor::test_invalid_footprints_rejected` requires the rejection |
| `FailureFace.extent_radius` | added (no docstring) | added (with field docstring) | **main's** (same field, documented) |

Consequence: the 13 "honest debt" tests listed in §6 now **pass** — the debt is closed by the
merged implementation rather than deferred. Reproduce with

```
python -m pytest tests/test_fracture_gen_mass_footprint.py \
                 tests/test_fracture_gen_footprint_breakage.py \
                 tests/test_fracture_gen_audit.py -q
python scripts/m3830_merge_cure_evidence.py --out-dir docs/m3830-merge-cure
```
