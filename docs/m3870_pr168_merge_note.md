# PR #168 cure: three-way merge of `main` into `agent/m2910`

Mission m3870. Repository `cxs503/TensorDEM`, branch `agent/m2910`, pull
request #168 was reported `mergeable=false` by `merge_governor`. This note
records the merge topology, the conflict set, the per-file resolution, and the
evidence that nothing from either side was silently dropped.

Everything numeric below is produced by
`scripts/merge_m3870_evidence.py` into `runs/m3870_merge/evidence.json` and the
figure `runs/m3870_merge/merge_evidence.png`; nothing is transcribed by hand.

## 1. Case: what had to be merged

| item | value |
|---|---|
| merge base | `8cc88f7f1bc681cb028269014e61c1fc265d9d2e` (merge #137) |
| old tip of `agent/m2910` | `420355924a5c4eeae6df7f36fdf05ead2ad82488` |
| trunk tip merged in (`origin/main`) | `e12a2abd636ab5a2998054dbe0fcf3c18b6a8564` (merge #203) |
| cure merge commit | `c047f83b4964bb9297db98f14e4e3b5aa8695051` |
| conflicts | 4 files (listed below) |
| branch-side additions kept | `MASS_MODE_TO_MAPPING` canonical reporting, `absolute_relative_error`, `domain_volume` report key, `FractureLedger.lattice_volume()`, `equal_volume_spheres` spelling in the audit, `docs/fracture_gen_contract_restoration.md`, `scripts/fracture_gen_contract_report.py` |
| trunk-side additions kept | superset alias table (`space`-free 11 spellings), `FACE_EXTENTS` / `normalise_face_extent`, `footprint_aware` + `on_plane_tolerance` crossing criteria, `relative_error_magnitude`, `_lattice_volume()`, `tests/test_fracture_gen_footprint_breakage.py`, whole non-fracture trunk work (92 files) |

Physics/model content: the module builds the DEM ice field from FEM cohesive
failure faces (SI-2 campaign). Both sides had evolved the *same* two physics
decisions independently:

* mass per particle: inscribed sphere `rho * 4/3 pi r^3` (frozen prereg
  mapping, −47.64 % of the represented ice volume) vs the volume-consistent
  cell mapping `rho * h^3`, `h = 2r` (trunk default since the m2536
  restoration);
* bond severing: the frozen *infinite plane* sign test vs the footprint *disc*
  criterion restricted to `sqrt(A/pi)` around the face centroid.

Both sides agreed on the physics and disagreed only on the *names* of the
mappings (`cell_volume` vs `lattice_cell`, `equal_volume_spheres` vs `sphere`)
and on the *report keys* for the mass residual
(`relative_error_magnitude` vs `absolute_relative_error`). That is what the
conflicts were about, and it is what the union resolution fixes: one mapping,
one canonical reported name, and every published spelling still accepted.

## 2. Method

1. Worktree-based three-way merge: `git merge --no-ff origin/main` on a branch
   created from `origin/agent/m2910`; no rebase, no force, no overwrite.
2. Conflicts were resolved **per hunk, not by overlay**: the file that carried
   17 hunks (`src/tensordem/fracture_gen.py`) was resolved on trunk's superset
   and then patched with each branch-side addition (patched by exact-string
   replacement, so an unexpected context aborts instead of guessing), followed
   by an AST parse plus a line-by-line audit of the diff against the *branch*
   parent to prove that every dropped branch-side hunk is either a comment, an
   equivalent expression, or a superseded validation path.
3. `python3 -m pytest tests` on the merged tree and on an untouched `main`
   checkout (`e12a2ab`), run in parallel, full 1235-test suite.
4. Evidence regenerated from the merged module and from the merge commit's own
   parents (`runs/m3870_merge/evidence.json`).

### 2.1 Conflict files and per-file resolution

| file | conflict | resolution (semantic union) |
|---|---|---|
| `src/tensordem/fracture_gen.py` | 17 hunks: alias table, canonical-name reporting, `mass_accounting` keys, `lattice_volume`, energy-ledger docstrings, `face_extent` plumbing | kept trunk's alias table (11 spellings, superset of the branch's 4), trunk's `FACE_EXTENTS`/`normalise_face_extent` crossing criteria and trunk's `energy_accounting`; **added** the branch's `MASS_MODE_TO_MAPPING` + `VALID_FACE_EXTENTS` constants, `absolute_relative_error` and `domain_volume` keys, and a public `FractureLedger.lattice_volume()` returning `N h^3` (branch semantics: material-mapping reference volume) while trunk's `_lattice_volume()` (bounding box grown by `r`) stays the default of `mass_accounting`; `FractureField.mass_mapping` now reports the canonical name and keeps the requested spelling in `mass_mapping_alias` |
| `tests/test_fracture_gen_audit.py` | mass-residual keys in the quantisation test | both assertions kept: signed `relative_error`, trunk's `relative_error_magnitude`, the branch's `absolute_relative_error`, plus a new equality check between the two magnitude spellings and the new `domain_volume` key |
| `tests/test_fracture_gen_mass_footprint.py` | spelling of the frozen mapping (`sphere` vs `equal_volume_spheres`), docstring paragraph | frozen mapping requested under its canonical name `equal_volume_spheres` (accepted alias of both `sphere` and `equal_volume_sphere`); trunk's mapping-selection paragraph kept and extended with the branch's contract pointer; the branch-only test `test_default_mapping_is_volume_consistent` kept |
| `benchmarks/fracture_gen_mass_footprint_audit.py` | which spelling names the frozen mapping | trunk's 79-line audit kept verbatim; the frozen mapping is requested as `equal_volume_spheres` so the audit keeps measuring the frozen `pi/6` deficit (requesting the default would measure 0 %) |

Branch-only files that merged without conflict and are preserved:
`docs/fracture_gen_contract_restoration.md`,
`scripts/fracture_gen_contract_report.py` (re-run after the merge, it still
produces `artifacts/fracture_gen_contract_restoration/tables.md` and the three
figures).

### 2.2 Non-functional differences that the audit had to justify

| branch-side construct | status after merge |
|---|---|
| `MASS_RADIUS_FACTOR = (6/pi)^(1/3)` | algebraically identical to trunk's `2 (3/4pi)^(1/3)` = 1.240701; trunk's expression kept |
| `if mass_mapping not in MASS_MAPPING_ALIASES: raise` | trunk validates through `normalise_mass_mapping(...)`, which raises `ValueError` for unknown input and *additionally* accepts the extra trunk spellings; test `test_mass_mapping_rejects_unknown_value` passes on both |
| `face_extent=` keyword plumbed into `identify_bonds_crossing_face` | trunk's `footprint_aware=` does the same job and `normalise_face_extent` maps `face_extent="disc"` onto it; both spellings are accepted by the merged signature |
| disc criterion measured at the **exact plane crossing** (branch) vs the **bond midpoint** (trunk) | trunk's midpoint form kept; the two differ only in sub-cell measure and both parents' disc-count tests (to `pi R^2/h^2`) pass on the merged tree. Recorded here as the one remaining modelling divergence |
| `mass_accounting` default reference volume | branch used `N h^3`; merged default is trunk's bounding-box volume (`_lattice_volume()`). For a full rectangular lattice they are identical (`N h^3`), so every published number is unchanged; the branch's reference volume remains available through `FractureLedger.lattice_volume()` and by passing `continuum_volume`/`domain_volume` explicitly |

## 3. Results

Lattice used by the evidence script: `rho = 917 kg/m^3`, `r = 0.05 m`,
`h = 2r = 0.1 m`, `10^3 = 1000` particles, domain `1 m^3`.

### 3.1 Mass mapping (both parents' semantics, one reported name)

| requested spelling | reported `mass_mapping` | `mass_mode` | `M_DEM` [kg] | `rho V` [kg] | signed residual [%] | < 3 % | 15 % bar |
|---|---|---|---|---|---|---|---|
| `cell_volume` | `cell_volume` | `cell` | 917.000000 | 917.000000 | +0.0000 | yes | PASS |
| `lattice_cell` | `cell_volume` | `cell` | 917.000000 | 917.000000 | +0.0000 | yes | PASS |
| `equal_volume_spheres` | `equal_volume_spheres` | `sphere` | 480.140077 | 917.000000 | -47.6401 | no | FAIL |
| `sphere` | `equal_volume_spheres` | `sphere` | 480.140077 | 917.000000 | -47.6401 | no | FAIL |

The alias pairs collapse onto one behaviour and one reported name, which is the
property PR #168 was trying to establish: an audit can no longer quote two
numbers for "the same" mapping. The `-47.6401 %` row is the *frozen prereg*
mapping and is expected to fail the 15 % bar — that deficit is the finding the
branch documented (contract restoration note §4), not a regression.

### 3.2 Bond severing (`A = 0.16 m^2` face, `G = 500 J/m^2`)

| `face_extent` | severed bonds | analytic `A/h^2` | bond-count error [%] | `U_bond` [J] | analytic `G h^2` [J] | < 3 % |
|---|---|---|---|---|---|---|
| `plane` | 100 | 16.00 | +525.0000 | 0.800000 | 5.000000 | no |
| `disc` | 16 | 16.00 | +0.0000 | 5.000000 | 5.000000 | yes |

The `plane` row is the frozen SI-2 criterion (infinite plane: over-breaks by
the plane/face area ratio `(1.0/0.4)^2 = 6.25`), the `disc` row is the
footprint criterion that the trunk can now select. Both criteria survive the
merge, which is the point: the merged module can reproduce the frozen numbers
*and* the corrected ones.

![merge evidence](runs/m3870_merge/merge_evidence.png)

Figure (generated by `scripts/merge_m3870_evidence.py`):
`runs/m3870_merge/merge_evidence.png`. Panel (a) — mass residual per mapping
spelling with the 3 % research target and the 15 % platform bar; panel (b) —
severed-bond count for the two surviving crossing criteria against `A/h^2`.

## 4. Test evidence

| run | command | result |
|---|---|---|
| merged tree, full suite | `python3 -m pytest tests -q` | **1 failed, 1232 passed, 2 skipped, 152 subtests passed** in 547 s |
| the one failure | `tests/test_doc_evidence_pointers.py::...::test_committed_ledger_is_current` | `docs/doc_evidence_ledger.json` was stale **because of the branch's own new document**: the ledger lacked the pointers of `docs/fracture_gen_contract_restoration.md`. Fixed by re-running `scripts/audit_doc_evidence.py`; the test then passes (14 passed) |
| main baseline (untouched `e12a2ab` checkout) | `python3 -m pytest tests -q` | runs in parallel, partial log kept: `runs/m3870_merge/test_baseline_main_partial.log` (§4.1) |
| fracture-gen focused | `pytest tests/test_fracture_gen*.py -q` | 82 passed |
| branch contract script | `python3 scripts/fracture_gen_contract_report.py` | runs, writes `artifacts/fracture_gen_contract_restoration/` (tables + 3 figures); bond topology vectorised-vs-brute-force identical (4.4x/15.5x/30.0x at n = 4/6/8) |

The merged tree therefore does not fall below the `main` baseline: the only
suite failure was a branch artefact (a stale documentation ledger, introduced
by the branch's own new document) and it is fixed inside this cure.

### 4.1 Baseline comparison (honest scope)

`main` at `e12a2ab` was launched on a pristine detached checkout
(`/tmp/base_m3870`, `git worktree add --detach e12a2ab`) with the same
interpreter and the same command, in parallel with the merged-tree run. The
shared machine is heavily loaded (other agents' suites running), so the
baseline suite had reached **40 % with 0 failures** when this mission's window
closed; the partial log is kept as
`runs/m3870_merge/test_baseline_main_partial.log` and the comparison is *not*
claimed as a finished full-suite diff.

What is claimed instead:

* merged tree full suite: **1232 passed, 1 failed, 2 skipped** (547 s), the one
  failure being the stale documentation ledger named above — a defect that
  `main` cannot have, because the document whose pointers are missing lives only
  on the branch;
* after regenerating `docs/doc_evidence_ledger.json` the documentation gate is
  green: `tests/test_doc_evidence_pointers.py` 14 passed;
* the fracture-gen core set (both parents' suites together, 82 tests) is green
  on the merged tree, and every baseline-only test file that touches this module
  (`tests/test_fracture_gen.py`, `..._footprint_breakage.py`,
  `..._mass_mapping.py`) passes unchanged.

So no test that passes on `main` is known to fail on the merged tree; the
residual risk is the tail of the baseline suite that did not finish in this
window (no failure was observed in the 40 % that did).

## 5. Notes, limits, pitfalls

* **Overlay is lethal here.** A mechanical "take theirs" on
  `src/tensordem/fracture_gen.py` silently deletes the branch's canonical
  reporting name and `absolute_relative_error` key — the branch's own two tests
  and its contract report would then fail without any merge conflict being
  reported. That is why the resolution is hunk-by-hunk with an AST check plus a
  diff audit against both parents.
* **A green merge can still be red in CI for a non-code reason.** The stale
  doc-evidence ledger is an *evidence-chain* gate: adding a document that cites
  code without regenerating `docs/doc_evidence_ledger.json` fails
  `test_doc_evidence_pointers.py`. Any future doc-adding branch must re-run
  `scripts/audit_doc_evidence.py`.
* **Two volumes named "lattice volume".** `lattice_volume()` is `N h^3`
  (material-mapping reference); `_lattice_volume()` is the bounding box grown by
  a half cell (default reference of `mass_accounting`). They agree for a full
  rectangular lattice and diverge only for a ragged one; the two names are kept
  rather than silently unified, and the difference is documented on both
  methods.
* **Alias tables must stay in sync.** The accepted spellings now live in one
  table (`MASS_MAPPING_ALIASES`) with one canonical reporting map
  (`MASS_MODE_TO_MAPPING`); adding a spelling to the audit layer without adding
  it here is what produced this conflict in the first place.
* Scope limit: this cure merges and unifies; it does not re-derive the frozen
  `pi/6` deficit or change any default physics. The disc-criterion divergence
  (exact crossing point vs bond midpoint) is recorded, not changed.

## 6. Error verdict (key quantities vs reference)

| quantity | reference | merged value | error | < 3 %? |
|---|---|---|---|---|
| `M_DEM` with `cell_volume` mapping | `rho V` = 917 kg (analytic, exact) | 917.000000 kg | 0.00 % | **yes** |
| `M_DEM` with frozen `equal_volume_spheres` mapping | `rho V` = 917 kg | 480.140077 kg | −47.64 % | no (expected: frozen prereg mapping, `1 - pi/6`) |
| severed bonds, `disc` criterion | `A/h^2` = 16 (analytic) | 16 | 0.00 % | **yes** |
| `U_bond`, `disc` criterion | `G h^2` = 5.0 J (analytic) | 5.000000 J | 0.00 % | **yes** |
| severed bonds, frozen `plane` criterion | `A/h^2` = 16 | 100 | +525 % | no (frozen infinite-plane over-break by `(1.0/0.4)^2`) |
| alias identity `sphere` ≡ `equal_volume_spheres` | same mapping | identical masses, one reported name | 0.00 % | **yes** |

The three quantities that the merge actually governs — mass of the default
mapping, disc-criterion bond count, and per-bond energy — reproduce their
analytic references to machine precision (0.00 %, well under 3 %). The two
"FAIL" rows are the *frozen* contract's documented defects (the branch's own
finding), reproduced on purpose so that the merged tree can still regenerate the
pre-audit numbers; they are not errors of the merge.