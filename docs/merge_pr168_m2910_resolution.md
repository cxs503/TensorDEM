# PR #168 (`agent/m2910`) → `main`: three-way merge record

Mission m3802. The PR branch `agent/m2910` (fracture-generation contract
restoration) went unmergeable against `main` because both sides extended
`src/tensordem/fracture_gen.py` and its verification surface independently after
the common ancestor `8cc88f7`. This note records the merge, the per-file
resolution and the test evidence, so a reviewer can check that no addition from
either side was dropped.

## 1. Topology

| item | value |
|---|---|
| repository | `TensorDEM` |
| `main` at merge time | `e12a2ab` (Merge #203 / m3317 via merge-governor) |
| `agent/m2910` (PR #168) before | `535d81e` |
| merge base | `8cc88f7` |
| merge result | `e90d6e5` (`Merge origin/main into agent/m2910: unify fracture_gen contract`, parents `e12a2ab` + `535d81e`) |
| `agent/m2910` after | `e90d6e5` |
| `agent/m3802` (this worktree) | `e90d6e5` |

Scope of each side since the base: `agent/m2910` = 15 files / +1122 / −32
(fracture_gen contract restoration: mass mapping reporting, finite face
footprint, layer-coincident plane, contract report script, artifacts, docs);
`main` = 92 files / +21912 / −430 (contact invariants, shear convergence,
three-field LBM transport, ridge geotech, fracture-gen footprint/breakage audits
and *its own* rewrite of `fracture_gen.py`).

## 2. Conflicting files and how each was resolved

Four files conflicted; the merge had 17 conflict hunks, 14 of them inside
`src/tensordem/fracture_gen.py`. Resolution rule: keep the union of behaviour —
where the two sides had overlapping implementations of the *same* function, the
stricter/superset implementation was kept and the other side's API surface
(names, keys, aliases) was restored on top of it.

### 2.1 `src/tensordem/fracture_gen.py`

| # | region | `main` side | `agent/m2910` side | resolution |
|---|---|---|---|---|
| 1 | module constants | `MASS_RADIUS_FACTOR`, `MASS_MAPPING_ALIASES` (9 spellings), `_FOOTPRINT_TOLERANCE`, `FACE_EXTENTS`, `normalise_mass_mapping`, `normalise_face_extent` | `MASS_RADIUS_FACTOR`, 4-spelling alias table, `MASS_MODE_TO_MAPPING`, `VALID_FACE_EXTENTS` | main's normalisers (one validator) **+** branch's `MASS_MODE_TO_MAPPING` and `VALID_FACE_EXTENTS` re-added |
| 2 | `FailureFace.extent_radius` doc | in-plane extent semantics | optional producer-supplied extent | main's docstring (same semantics, keeps `extent_radius` field) |
| 3 | `FractureGenConfig` fields | `mass_mapping`/`footprint_aware: bool`/`face_extent: str\|None`/`on_plane_tolerance` | `mass_mapping`/`face_extent: str`/`footprint_aware: bool\|None`/`on_plane_tolerance` | main's block (accepts both spellings, `normalise_face_extent` resolves them) |
| 4 | `FractureGenConfig.__post_init__` | `normalise_*` + finite/non-negative tolerance check | hand-rolled alias lookup, `on_plane_tolerance < 0` only | main's (strict superset: same validation, plus finiteness) |
| 5 | `FractureField.__init__` | `mass_mapping` echoed as the caller's spelling, `footprint_aware` normalised | `mass_mapping` = canonical descriptive name, `face_extent` validated | main's validation **with** branch's canonical reporting: `mass_mapping = MASS_MODE_TO_MAPPING[mass_mode]` |
| 6,7 | `mass_radius`, `volume_fraction` docstrings | equivalent | equivalent | main's (identical formula `2 (3/4π)^{1/3} r ≡ (6/π)^{1/3} r`; main adds the zero-cell guard) |
| 8 | `build_bond_topology` pair sweep | blocked `torch.cdist` | blocked broadcast + `norm` | main's; both give the identical pair set (verified: 300/882/1944 bonds identical at 5³/7³/9³ lattices), timings are within run-to-run noise |
| 9 | `face_footprint_radius` | finite/positive validation of area and extent | positive-only validation | main's (strict superset) |
| 10 | `identify_bonds_crossing_face` signature | `on_plane_tolerance, footprint_aware, face_extent=None` | `face_extent="plane", on_plane_tolerance, footprint_aware=None` | main's order; `face_extent`/`footprint_aware` still accepted as keywords (every caller in the tree uses keywords) |
| 11 | footprint / tolerance body | `_FOOTPRINT_TOLERANCE` + **bond-midpoint** in-plane projection | exact **plane-crossing point** projection | main's (trunk tests assert the midpoint drift `≤ R`; branch adds no analytic count that the crossing-point form satisfies and the midpoint form does not) **and** the resolved spelling is propagated (`face_extent = "disc" if footprint_aware else "plane"`) so the branch's `VALID_FACE_EXTENTS` check accepts the default `face_extent=None` path — this was the one real defect the first merge attempt produced (`ValueError: face_extent must be one of ('plane','disc'), got None` on 24 tests) |
| 12,13 | `generate_fracture_field` → `identify_bonds_crossing_face` / `FractureField(...)` kwargs | passes `footprint_aware` + `on_plane_tolerance` + `mass_mapping` | passes `face_extent` + `on_plane_tolerance` | main's block (covers both spellings, keeps `mass_mapping` propagation) |
| 14,15 | `energy_accounting` body and return dict | full analytic bond count, unit-bond-energy error, input-consistency error | same quantities via local names | main's (superset: adds `bond_count_relative_error` beside `unit_bond_energy_relative_error`) |
| 16,17 | `check_bonds_at_faces` re-identification | `footprint_aware=field.footprint_aware` | `face_extent=field.face_extent` | main's (`field.face_extent` is a property derived from `footprint_aware`, so both agree) |

Non-conflicting additions of both sides survive automatically: the branch's
`extent_radius` field, `face_extent` property, analytic energy keys, contract
report script, artifacts and docs; `main`'s `normalise_*` helpers, `_FOOTPRINT_TOLERANCE`,
`num_broken_bonds`, `relative_error_magnitude`, `on_plane_tolerance` criterion and
the finite-input validation of the face geometry.

One compatibility key was added so the branch's published tables stay
reproducible: `FractureLedger.mass_accounting()` now reports the signed
`relative_error` **and both** magnitude spellings, `relative_error_magnitude`
(trunk) and `absolute_relative_error` (contract-restoration tables).

### 2.2 `tests/test_fracture_gen_audit.py`

Conflict in `test_lattice_quantisation_is_reported`: `main` asserted the signed
`relative_error` plus `relative_error_magnitude`; the branch asserted the signed
`relative_error` plus `absolute_relative_error`. Resolution keeps **all** of them
and adds a cross-check that the two magnitude spellings are the same number.

### 2.3 `tests/test_fracture_gen_mass_footprint.py`

Two conflicts, both about the spelling of the frozen mapping (`"sphere"` on
`main`, `"equal_volume_spheres"` on the branch). `MASS_MAPPING_ALIASES` accepts
both, so the resolution builds the field with both spellings, asserts equality of
`total_mass`/`mass_mode`, and asserts the reported canonical name is
`equal_volume_spheres`; the second hunk uses the trunk spelling only.

### 2.4 `benchmarks/fracture_gen_mass_footprint_audit.py`

Conflict over the alias used for the frozen mapping. Same mapping either way;
the file now requests `equal_volume_spheres` with a comment that
`equal_volume_sphere` is an alias, so the audit is explicit about the spelling
that the branch published.

### 2.5 Auto-merged files (no manual edit)

`benchmarks/fracture_gen_audit_report.json`, `artifacts/fracture_gen_audit/*`,
`artifacts/fracture_gen_contract_restoration/*`, `docs/fracture_gen_contract_restoration.md`,
`scripts/fracture_gen_contract_report.py` — branch-only additions, taken as is.
The contract-restoration artifacts were regenerated from the merged module
(`PYTHONPATH=src python scripts/fracture_gen_contract_report.py`) so the
committed numbers belong to the merged code; the only values that moved are the
wall-clock topology timings (machine/load dependent) and a float repr
(`1.2407009817988` → `1.2407009817988002` for `mass_radius / r`).

## 3. Verification

- `python -m pytest tests -q` on the merged tree vs the same suite on `main`
  (`e12a2ab`) — see `runs/m3802_merged_suite.log` and the numbers in §4.
- Targeted: `pytest tests/test_fracture_gen*.py` → 82 passed (both sides'
  fracture-generation contracts green at once).
- Static: file parses (`ast.parse`), no duplicate top-level definitions or
  methods, no conflict markers left anywhere in `src/`, `tests/`, `benchmarks/`,
  `scripts/`.
- Reproducibility: `scripts/fracture_gen_contract_report.py` runs on the merged
  module and regenerates the committed artifact tables.

## 4. Result

Full-suite evidence (`python -m pytest tests -q`, 13.4 min per run, the two runs
executed back to back on the same host):

| tree | passed | failed | skipped | subtests | source |
|---|---|---|---|---|---|
| `main` `e12a2ab` (baseline) | 1224 | 1 | 3 | 152 | timing test `test_hull_mesh_contact.py::TestComplexity::test_linear_scaling` (wall-clock scaling, host load) |
| merge `e90d6e5` | 1230 | 3 | 2 | 152 | the three `test_doc_evidence_pointers` gates, all caused by the regenerated artifacts being uncommitted at run time; they pass once the artifacts are committed |

So the merged tree passes **1230** tests against the baseline's **1224** (the
branch's six extra tests are the restored contract tests), and no test that
passes on `main` fails on the merge: the only merged-run failures were
bookkeeping gates about uncommitted artifacts, fixed by committing them, and the
single baseline failure is the host-load-dependent scaling test.

Contract-level checks that pin the resolution:

- `pytest tests/test_fracture_gen*.py -q` → **82 passed** (trunk's footprint,
  tolerance, alias and mass-radius contracts *and* the branch's contract
  restoration tests, in one run).
- `mass_accounting()` reports `relative_error`, `relative_error_magnitude` and
  `absolute_relative_error` with the two magnitude spellings equal to 15 digits;
  `FractureField.mass_mapping` is the canonical descriptive name
  (`cell_volume` / `equal_volume_spheres`) whichever spelling was supplied.
- `face_extent=None` (the trunk default) no longer trips the restored
  `VALID_FACE_EXTENTS` check; `footprint_aware` / `face_extent` are still
  accepted interchangeably.
- `scripts/fracture_gen_contract_report.py` regenerates
  `artifacts/fracture_gen_contract_restoration/{summary.json,tables.md,*.png}`
  from the merged module (committed).