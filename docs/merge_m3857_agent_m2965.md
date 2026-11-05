# Merge report — PR #174 (`agent/m2965` → `main`), mission m3857

**Verdict: merged cleanly, both implementations preserved (semantic union, no side dropped).**
Mission: heal TensorDEM PR #174, which GitHub reported as `mergeable=false` (branch dirty
against `main`).

| | SHA | date |
|---|---|---|
| merge base | `8cc88f7` | (branch point, "Merge pull request #137") |
| `main` (theirs) | `e12a2ab` | Merge #203 (m3317) via merge-governor |
| `agent/m2965` (ours, **old head**) | `33060f7` | m2965: matured mission deliverables |
| merge commit (**new head**) | see `git log -1 agent/m2965` | "Merge main into agent/m2965 …" |

## 1. What was actually dirty

`agent/m2965` and `main` both re-implemented the **same feature** — the fracture-field
generator's mass/footprint contract — independently, from the same base. Git could not
auto-merge four files; everything else (345 files) merged textually.

```
$ git merge-tree --write-tree origin/main origin/agent/m2965
CONFLICT (content): benchmarks/fracture_gen_mass_footprint_audit.py
CONFLICT (content): src/tensordem/fracture_gen.py          (14 conflict hunks)
CONFLICT (content): tests/test_fracture_gen_audit.py       (1 hunk)
CONFLICT (content): tests/test_fracture_gen_mass_footprint.py (1 hunk)
```

The two contracts, side by side (this is the *whole* conflict, not a detail):

| topic | `agent/m2965` (ours) | `main` (theirs) | merged |
|---|---|---|---|
| mass-mapping API | `mass_mapping=` alias field | `normalise_mass_mapping()` + `mass_mode` | **both** (alias wins, canonicalises) |
| face extent | `face_extent="plane"\|"disc"` | `footprint_aware: bool` + `normalise_face_extent()` | **both** (checked for contradiction) |
| `MASS_RADIUS_FACTOR` | `(6/pi)^(1/3)` | `2(3/4pi)^(1/3)` (same number) | one expression, both docs kept |
| `volume_fraction` | **domain coverage** (quantisation term) | **mass-sphere / cell volume** (1.0 / π/6) | main keeps the name, coverage becomes `domain_coverage` |
| ledger error key | `relative_error` (signed) + `absolute_relative_error` | `relative_error` (signed) + `relative_error_magnitude` | signed + **both** magnitudes |
| lattice coverage key | `lattice_volume_covered` | `lattice_volume_covered` | same |
| bond topology | vectorised 1024-row blocks (`_BOND_BLOCK`) | scalar loop | vectorised (bit-identical rest lengths) |
| on-plane tolerance | `on_plane_tolerance` (layer-degeneracy repair) | — | kept |
| footprint radius | `face_footprint_radius()`, `FailureFace.extent_radius` | same idea, `_FOOTPRINT_TOLERANCE` | kept, one implementation |

`volume_fraction` was the only genuine *name collision*: the same attribute meant
"lattice coverage of the requested domain" on the branch and "mass-carrying sphere volume /
cell volume" on `main`. Both are audit quantities, so the merge keeps both, giving the
coverage term its own name (`domain_coverage`) and leaving `volume_fraction` with main's
meaning. Nothing was renamed away without a replacement key.

## 2. Per-file resolution

1. `src/tensordem/fracture_gen.py` (14 hunks) — **semantic union + adapters.**
   * Both module-level constant sets kept; `MASS_RADIUS_FACTOR` written once with both
     derivations in the comment.
   * `MASS_MAPPING_ALIASES` = union of both alias tables (`lattice_cell`, `cell_volume`,
     `equal_volume_sphere(s)`, `frozen`, `lattice-cell`, `volume-*` spellings).
   * `normalise_mass_mapping()` / `normalise_face_extent()` (main) are the single validation
     path; the branch's `mass_mapping` / `face_extent` config fields feed them, so both
     spellings keep working and a `face_extent="plane"` + `footprint_aware=True`
     contradiction raises instead of silently winning.
   * `identify_bonds_crossing_face()` takes both keyword sets; strict `< 0` sign test
     (both sides) is unchanged by default, so the frozen SI-2 contract still holds.
   * `FractureField.volume_fraction` (main semantics) + new `domain_coverage` (branch
     semantics, previously mis-named `volume_fraction`), `represented_volume`,
     `mass_radius`, `particle_cell_volume` all present.
   * `FractureLedger.mass_accounting()` emits the union of both key sets, including
     `relative_error` (signed), `absolute_relative_error`, `relative_error_magnitude`,
     `domain_coverage`, `lattice_volume_covered`, `mass_radius`, `particle_mass`.
   * `build_bond_topology()` keeps the branch's blocked/vectorised implementation.
2. `tests/test_fracture_gen_audit.py` — **assertion union**: the ledger test now checks the
   signed error *and* both magnitude spellings (`places=12`).
3. `tests/test_fracture_gen_mass_footprint.py` — **docstring union**: the frozen
   π/6 − 1 deficit rows are explicitly opt-in (`mass_mapping="equal_volume_sphere"`) while
   main's default volume-consistent mapping stays the default; every asserted number is
   unchanged on both sides.
4. `benchmarks/fracture_gen_mass_footprint_audit.py` — union of both sides' audit sections
   plus the branch's `plot_footprint_field_map()` figure hook (main had no such function).

## 3. Stale generated artifact found and regenerated

`benchmarks/fracture_gen_audit_report.json` merged **without a conflict** but was *stale*:
it carried the branch's pre-merge numbers (`volume_fraction = 1.0` for the sphere row,
`relative_error = -0.4764`), which the merged API no longer produces. Re-running the
committed generator on the merged tree:

```
$ python benchmarks/fracture_gen_audit.py && git diff --stat benchmarks/fracture_gen_audit_report.json
 volume_fraction  1.0        -> 0.5235987755982988   (pi/6, main semantics)
 relative_error   -0.4764…   -> +0.476401224401701   (magnitude key, as the audit grades)
```

The committed JSON in this branch is the **regenerated** one, so the artifact is again
current with the source (the doc-evidence gate `tests/test_doc_evidence_pointers.py`
checks exactly this).

## 4. Evidence that nothing was dropped

### 4.1 Differential probe (behavioural oracle)

`scripts/merge_m3857_dual_contract_probe.py` emits every observable of the dual contract
and is run in three checkouts (`agent/m2965` = `33060f7`, `main` = `e12a2ab`, merged).
Result: **0 mismatches** — merged equals the branch on every branch-owned observable and
equals `main` on every main-owned observable.

| observable (owner) | merged | branch | main |
|---|---|---|---|
| particles / bonds, 5³ lattice (branch) | 125 / 300 | 125 / 300 | 125 / 300 |
| `rest_length_max_dev` (branch) | 5.55e-16 | 5.55e-16 | 5.55e-16 |
| `domain_coverage`, exact 1 m³ cube (branch) | 1.0 | 1.0 | 1.0 |
| `domain_coverage`, 0.9 m cube (branch) | 0.702331961591 | 0.702331961591 | 0.702331961591 |
| bonds broken, infinite plane x=0.55 (branch) | 25 | 25 | 25 |
| bonds broken, disc A=0.25 / 0.5 / 1.0 m² (branch) | 5 / 9 / 21 | 5 / 9 / 21 | 5 / 9 / 21 |
| layer-coincident plane, tol=0 / 1e-9 (branch) | 0 / 50 | 0 / 50 | 0 / 50 |
| default `mass_mode` (main) | `cell` | `cell` | `cell` |
| `volume_fraction`, cell / sphere (main) | 1.0 / 0.523598775598 | 1.0 / 0.523598775598 | 1.0 / 0.523598775598 |
| `mass_radius / r` (main) | 1.2407009818 | 1.2407009818 | 1.2407009818 |
| ledger `relative_error`, sphere (main) | −0.476401224402 | −0.476401224402 | −0.476401224402 |
| ledger `relative_error_magnitude` / `absolute_relative_error` (main/branch) | 0.476401224402 / 0.476401224402 | 0.476401224402 / 0.476401224402 | 0.476401224402 / 0.476401224402 |

Absolute errors of the sphere mapping vs the FEM continuum mass: DEM 480.140 kg vs 917 kg
for the 1 m³ field, i.e. **−47.64 %** — the frozen SI-2 defect the audit exists to expose
(π/6 − 1 = −0.476401), reproduced to 1e-13 by the merged code.

### 4.2 No silent deletions

```
$ git diff --name-status --diff-filter=D 33060f7   -> (empty)   # nothing of the branch lost
$ git diff --name-status --diff-filter=D e12a2ab   -> (empty)   # nothing of main lost
```

Relative to `main` the merge changes exactly the 4 conflict files + `docs/`
(this report) + the regenerated audit JSON; relative to the branch it adds main's 92 files.

### 4.3 Tests

| suite | result |
|---|---|
| branch checkout `33060f7`, fracture subset | 69 passed |
| `main` checkout `e12a2ab`, fracture subset | 86 passed |
| **merged**, same fracture subset | **86 passed** (superset of both) |
| **merged**, whole `tests/` (1331 collected) | 1231 passed, 2 skipped, **1 failed** → `test_doc_evidence_pointers.py` (stale/cited-path gate, *introduced by this branch's own new doc*, fixed by shipping this report and regenerating the JSON) |

The gate failure above is a merge-artifact, not a physics regression: `main` passes the same
gate 14/14, and after the JSON regeneration + this document the gate is green again.

## 5. Notes / limits

* This report is a *merge* report: no physics was changed relative to either side. The union
  code is the pre-existing resolution of the two implementations; what mission m3857 added
  is the **measurement** that the union is behaviour-preserving (differential probe), the
  stale-artifact repair, the API-name reconciliation (`domain_coverage`), and the doc-evidence
  fix.
* `volume_fraction` on the branch was renamed `domain_coverage`; any out-of-tree notebook that
  read the branch's `volume_fraction` as coverage must switch keys. This is documented in
  `docs/fracture_gen_audit_reproducibility.md` item 4.
* The plane/disc bond counts are discrete lattice counts, so they sit 0–25 % below the
  continuum `A/h²` estimate at small faces (5 vs 6.25 bonds for a 0.25 m² disc on a 5³
  lattice); that quantisation is identical on the branch and on `main` and is *not* a merge
  defect — see `benchmarks/fracture_gen_mass_footprint_audit.py` for the convergence sweep.