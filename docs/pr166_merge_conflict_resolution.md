# PR #166 (`agent/m2942`) × `main` — conflict-resolution record

Mission: heal TensorDEM PR #166, which had gone `mergeable=false` against `main`.
Target branch: `agent/m2942` (base `8cc88f7`, PR head `4b54f9b`, `main = e12a2ab`).

## 1. Why it went dirty

`agent/m2942` was opened to close the fifth validation-debt contract
(`tests/test_fracture_gen_mass_footprint.py`) by adding two *missing names* to
`src/tensordem/fracture_gen.py`: `MASS_RADIUS_FACTOR` and
`face_footprint_radius()` (+ the `FailureFace.extent_radius` field). It was parked at
base `8cc88f7`. In the meantime `main` advanced 29 commits and landed the whole
mass-mapping / face-footprint implementation (audit `docs/dem3d_fracture_mass_mapping_audit.md`,
breakage `docs/dem3d_fracture_footprint_breakage.md`, `tests/test_fracture_gen_footprint_breakage.py`),
so `main` already defined the very symbols the PR was adding — on 372 inserted lines
inside the same file. Hence the content conflict on `src/tensordem/fracture_gen.py`.

## 2. Conflict inventory

| file | merge state | sides |
|---|---|---|
| `src/tensordem/fracture_gen.py` | **CONFLICT (content)**, 1 hunk + 1 shadowing duplicate block | `main` rewrote the mass-mapping / footprint layer; PR re-declared 2 symbols |
| `docs/dem_analytic_verification_debt_restoration.md` | auto-merged (PR appended §6 at EOF, `main` touched other sections) but the appended text became factually wrong | PR-only edit |
| `report.json` | auto-merged, PR-only addition (new file) | PR-only edit |

Only one textual conflict marker appeared; the two "clean" files still needed
*semantic* repair because the PR text described a world (`13 tests fail, debt open`)
that no longer exists on `main`.

## 3. Per-file resolution (semantic union — nothing dropped silently)

### `src/tensordem/fracture_gen.py`

The PR's block (an "Analytic constants & helpers" section placed *before* `FailureFace`)
re-defined both symbols that `main` already defines — module constant at line 46 and
`face_footprint_radius()` at line 411 — and would have **shadowed** `main`'s stricter
implementations (dead duplicate + weaker validation). Resolution:

1. Removed the PR's provisional duplicate block (the single deletion).
2. Folded the PR's *derivation* into `main`'s canonical constant as a comment:
   the PR wrote `r_m/r = (6/π)^{1/3}` (from `4/3 π r_m³ = h³ = (2r)³`), `main` writes
   `2 (3/4π)^{1/3}`; the two are algebraically identical — measured difference
   `2.22e-16` (relative `1.79e-16`, machine precision), pinned by a new test
   (`MassRadiusFactorMergeTest`).
3. Kept `main`'s `face_footprint_radius()`. It is a **strict superset** of the PR
   contract: explicit `extent_radius` wins, otherwise `sqrt(area/π)` (identical
   behaviour, `sqrt(0.04/π)` reproduced with 0 relative error), *plus* rejection of
   non-finite / non-positive inputs. The PR's `extent_radius >= 0` bound is satisfied
   by it; `extent_radius = 0` is rejected on both sides' contracts
   (`tests/test_fracture_gen_footprint_breakage.py::...zero_extent_rejected`).
4. Kept `main`'s `FailureFace.extent_radius` field comment (the PR side of that hunk
   was empty, so the `=======` side carried no content).

Net delta of the merged file against `main`: 6 comment lines. No behaviour change, no
API loss, and the two symbols the PR promised are exported with the PR's documented
contract (verified by the PR's own test module, which passes 16/16 on the merge, and
18/18 with the two merge-guard tests added here).

### `docs/dem_analytic_verification_debt_restoration.md`

Kept §6 (the PR's narrative: why the contract was red, what the two names mean) and
corrected it to the post-merge truth: the debt table is no longer "16 collected, 13
fail" but green, and the resolution of the duplicated symbols is documented row by row.
The PR's original claim is preserved as history rather than deleted.

### `report.json`

Kept verbatim from the PR (repo `main` had no root `report.json`); it is the mission's
own submission payload, not code.

## 4. Evidence

Reproduce everything below with
`python3 artifacts/merge166/merge166_verification.py` (writes
`artifacts/merge166/merge166_metrics.json` and
`artifacts/merge166/merge166_verification.png`).

| check | main baseline | merged tree | verdict |
|---|---|---|---|
| `tests/test_fracture_gen_mass_footprint.py` (PR's contract module) | 16 passed (PR file on `main` src) | 18 passed (16 + 2 merge guards) | ≥ baseline |
| `tests/test_fracture_gen_footprint_breakage.py` | 17 passed | 17 passed | = baseline |
| `tests/test_fracture_gen_mass_mapping.py` | 16 passed | 16 passed | = baseline |
| `tests/test_fracture_gen_audit.py` | 16 passed | 16 passed | = baseline |
| `tests/test_fracture_gen.py` | 16 passed | 16 passed | = baseline |
| full `pytest tests -q` | see §4 note | 100 % green after the doc-evidence ledger refresh | ≥ baseline |

Audited numbers regenerated on the merged tree (programmatic, no hand-copying):

| quantity | value | reference | error |
|---|---|---|---|
| `MASS_RADIUS_FACTOR` trunk vs PR #166 closed form | 1.2407009817988002 vs 1.2407009817988 | same algebra | 2.2e-16 abs (1.8e-16 rel) |
| `face_footprint_radius(area=0.04)` | 0.11283791670955126 m | `sqrt(A/π)` = 0.11283791670955126 m | 0.0 |
| DEM mass, `mass_mode='sphere'` (frozen SI-2) | 3841.12 kg | continuum 7336.0 kg | **−47.64 %** (>15 %, fail) |
| DEM mass, `mass_mode='cell'` (default) | 7336.0 kg | continuum 7336.0 kg | 0.00 % (<3 %) |
| severed bonds, full plane `A/s²` | 64 | 64 analytic | 0.00 % |
| severed bonds, footprint-aware | see panel (a) | exact lattice enumeration | tracks enumeration |
| layer-coincident plane, strict / tolerant | 0 / 128 | 2n² = 128 | 0 % with tolerance |

## 5. Caveats

* `main`'s stricter `face_footprint_radius()` rejects `extent_radius = 0`, which the
  PR's provisional version accepted. This is a deliberate union choice (strict wins),
  backed by `tests/test_fracture_gen_footprint_breakage.py`; a caller relying on a
  zero-radius face would have been severing bonds inside a degenerate footprint.
* The `main` full-suite baseline was not re-run from a pristine worktree inside this
  mission; the merged tree's `src/` differs from `main` only by comments (§3), and
  every added test is additive, so the merged counts are lower-bounded by the baseline
  counts listed above.
## 6. Branch state

| ref | SHA | note |
|---|---|---|
| `origin/main` (merge target) | `e12a2ab` | first parent of the healed merge |
| `origin/agent/m2942` (PR #166 head, before) | `4b54f9b` | second parent of the healed merge |
| healed merge commit | tip of `agent/m3801` (subject `Merge main (e12a2ab) into agent/m2942: ...`) | old→new for `agent/m2942`; commit is identified by its parents, since this record's own amend advances the tip |
| merge base of the two sides | `8cc88f7` | PR was parked on this base |

The merge commit has parents `e12a2ab` (main) and `4b54f9b` (PR head), so
re-pointing `agent/m2942` at it is a fast-forward for the governor's next cycle: no
rebase, no history rewrite, and `git merge-base --is-ancestor e12a2ab <tip>` holds.
Publishing the ref is left to the platform auto-push / merge-governor
(the mission work tree is not permitted to push to `origin`); the healed tree is
carried on branch `agent/m3801` at the same commit.
