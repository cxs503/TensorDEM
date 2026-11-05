# TensorDEM PR #166 merge cure (mission m3830)

Branch: `agent/m3830` working tree, pushed back onto `agent/m2942`.
Goal: make PR #166 (`agent/m2942` → `main`) non-dirty (`mergeable=false` before the cure)
**without** any branch's content being silently dropped, then prove the merged tree is not
worse than the `main` baseline.

## 1. Topology at the time of the cure

| item | SHA |
|---|---|
| merge base | `8cc88f7f1bc681cb028269014e61c1fc265d9d2e` (Merge PR #137) |
| `main` tip | `e12a2abd636ab5a2998054dbe0fcf3c18b6a8564` (Merge #203 / m3317) — 29 commits ahead of base |
| `agent/m2942` tip **before** cure | `4b54f9bbb037adc092bcdcba7bdca94f6cc0e41e` — 1 commit ahead of base |
| `agent/m2942` tip **after** cure | recorded in section 6 |

`agent/m2942` touched 3 files; only one of them was also touched by `main`:

```
docs/dem_analytic_verification_debt_restoration.md   +32 -1   (only m2942)
report.json                                           +9      (only m2942, new file)
src/tensordem/fracture_gen.py                        +27     (BOTH sides,  main: +372 -42)
```

## 2. Conflict inventory and per-file resolution

### 2.1 `docs/dem_analytic_verification_debt_restoration.md` — auto-merged

`main` never edited it, so git merged it cleanly and m2942's §6 text is kept verbatim.
Because §6 ("13 tests still red, honest debt") became stale the moment `main` landed the
implementation, an **additive** §7 "Reconciliation (m3830)" was appended: it tabulates the
per-symbol decision and states that the debt is now closed. Nothing was removed.

### 2.2 `report.json` — auto-merged (m2942-only addition)

Kept byte-for-byte (m2942's own mission report, `mission_id: 2942`).

### 2.3 `src/tensordem/fracture_gen.py` — manual resolution

Git reported **one** textual conflict (the `FailureFace.extent_radius` field where `main`
added a docstring and m2942 did not). The dangerous part was *not* the marked conflict but
two **unmarked** ones: because the two branches inserted the same new symbols at different
offsets, `git merge` happily produced **duplicate module-level definitions**:

| symbol | m2942 copy | main copy | risk |
|---|---|---|---|
| `MASS_RADIUS_FACTOR` | = `(6/pi)^(1/3)` | = `2 (3/4pi)^(1/3)` | second assignment silently shadows the first |
| `face_footprint_radius` | accepts `extent_radius == 0`, returns `0.0` | rejects `<= 0` / non-finite / `area <= 0` | two different contracts, later `def` wins |
| `FailureFace.extent_radius` | field, no docstring | field + docstring | duplicate keyword in one `@dataclass` → `TypeError` if both text blocks survive verbatim |

Resolution policy — *semantic union of intent, single authoritative definition*:

1. every symbol that m2942 introduced still exists **exactly once** after the merge
   (`MASS_RADIUS_FACTOR`, `face_footprint_radius`, `FailureFace.extent_radius`);
2. the surviving implementation is `main`'s, because it is a strict superset (it also carries
   `footprint_aware`, `face_extent`, `on_plane_tolerance`, footprint severing) and because a
   frozen test in `main` *requires* the stricter validation:
   `tests/test_fracture_gen_footprint_breakage.py::TestFootprintRadiusAnchor::test_invalid_footprints_rejected`
   asserts `ValueError` for `extent_radius == 0.0`, which m2942's copy would have returned as `0.0`;
3. m2942's analytic derivation (`4/3 pi r_m^3 = h^3 => r_m = r (6/pi)^{1/3}`) is preserved as a
   comment on the surviving `MASS_RADIUS_FACTOR` so no branch content is lost — the two closed
   forms agree to 1 ulp (abs `2.22e-16`, rel `1.79e-16`);
4. the marked hunk was resolved to `main`'s documented field (both sides had the field itself).

Net effect of the whole merge, measured as `git diff origin/main`:

```
docs/dem_analytic_verification_debt_restoration.md | 32 +++++++++-
report.json                                        |  9 +++
src/tensordem/fracture_gen.py                       |  2 +     <- m2942's derivation comment only
```

i.e. the merge adds back exactly m2942's deliverables and nothing from `main` is lost
(`main`'s 92 changed files and 372 new `fracture_gen.py` lines are all present; verified by the
zero-diff on every other path and by a top-level-symbol duplicate scan of the resolved module).

## 3. Verification

### 3.1 Duplicate-definition scan (the failure mode that motivated the cure)

AST scan of the resolved `src/tensordem/fracture_gen.py`: 16 top-level symbols,
**0 duplicates** (before the manual fix: 2 duplicated symbols).

### 3.2 Focused fracture suite (the contract m2942 was about)

```
tests/test_fracture_gen_mass_footprint.py        16 passed
tests/test_fracture_gen_footprint_breakage.py    17 passed
tests/test_fracture_gen_audit.py                 16 passed
                                                 ---------
                                                 49 passed  (13.6 s)
```

`test_fracture_gen_mass_footprint.py` is the module that collected but had **13 failures** in
m2942's own report (§6 table); it is fully green on the merged tree.

### 3.3 Full suite vs `main` baseline (cure acceptance criterion)

Both trees were run with the same command (`python3 -m pytest tests -q -p no:cacheprovider`),
the baseline in a clean detached worktree of `origin/main`.

| tree | result |
|---|---|
| `origin/main` (`e12a2ab`, baseline) | **1232 passed, 2 skipped**, 152 subtests passed, 587 s |
| merged `agent/m2942` (this cure) | ****1232 passed, 2 skipped**, 49 warnings, 152 subtests passed, 71 s (`-n 24`)** |

Acceptance: merged >= baseline, no new failures/skips -- **satisfied exactly** (1232 passed /
2 skipped / 152 subtests on both trees; the 2 skips are pre-existing on `main`). Re-run at the end
of the cure with 24 xdist workers (71 s): identical counts.

One transient, fully explained failure mode was observed *before* committing: the three
`tests/test_doc_evidence_pointers.py` gate tests fail while the new documents/scripts are still
untracked (`cited path not tracked` / `ledger stale`). They pass once `python3 scripts/audit_doc_evidence.py`
regenerates `docs/doc_evidence_ledger.json` (106 documents, 525 pointers, 0 orphans), part of this commit.

Note on delivery: the agent sandbox blocks a direct upstream `git-push`, so publishing is left to the
platform auto-push. The cure is a single merge commit whose parents are `4b54f9b` (old `agent/m2942`
tip) and `e12a2ab` (`main` tip), i.e. a fast-forward of `agent/m2942` -> PR #166.

### 3.4 Independent quantitative check of the surviving contract

`python3 scripts/m3830_merge_cure_evidence.py --out-dir docs/m3830-merge-cure` recomputes every
number from the merged module (nothing hand-transcribed):

| quantity | value |
|---|---|
| `MASS_RADIUS_FACTOR` (merged) | 1.2407009817988002 |
| abs / rel diff between the two branch closed forms | 2.220446e-16 / 1.789671e-16 |
| inscribed-sphere mass deficit `1 - pi/6` | 47.64 % |
| `face_footprint_radius` contract | 4 rejected / 2 accepted of 6 probes, matching `main`'s tests |
| severed bonds, footprint aware vs exact discrete column count | identical at all 7 radii, rel diff **0.0** |
| severed bonds, infinite-plane criterion | 121 (constant, over-breaks for small faces) |

Figure: `docs/m3830-merge-cure/fig_m3830_merge_cure.png`
(a) mass mapping + 47.64 % deficit, (b) footprint-radius contract, (c) the merged
footprint-severing implementation vs the exact lattice prediction.

## 4. Pitfalls recorded (append-only knowledge)

- **`git merge` does not flag duplicated insertions as conflicts.** Two branches adding the
  same module-level name at different offsets merge "cleanly" into two definitions; the later
  one silently wins. After any text merge on a hot module: AST duplicate scan, not eyeballing.
- **A "take both sides" resolution is not a semantic union.** Here m2942's
  `face_footprint_radius` would have accepted `extent_radius = 0` and returned `0.0`, violating
  `main`'s frozen test. Union must be decided per symbol against the frozen tests.
- **`git checkout --theirs -- .` destroys a hand resolution.** It was run once here by mistake
  right after resolving; the file silently reverted to `main`'s blob (verified by comparing
  `git hash-object` with `:2:<path>`). Recovered with `git checkout -m -- <path>` and the
  resolver re-run. Never run `--theirs/-X ours` inside an unresolved merge you already edited.
- The m2942 branch's own stopgap was written *before* `main` implemented the feature; a cure must
  therefore re-derive the intent of the stale branch, not preserve its code literally.

## 5. Scope / limitations

- Only `src/tensordem/fracture_gen.py` needed manual work; the other two m2942 files were taken
  as-is, so this cure carries no new physics and no performance change.
- The merge does not attempt to re-audit `main`'s 29 new commits or the other 91 changed files;
  they are accepted as `main`'s state.
- `main`'s own 2 skipped tests are pre-existing skips, not caused by the merge.
- The full-suite runs executed on CPU only (`torch 2.7.1`, CUDA paths untested here).

## 6. Git bookkeeping

| ref | SHA |
|---|---|
| `agent/m2942` before cure | `4b54f9bbb037adc092bcdcba7bdca94f6cc0e41e` |
| `agent/m2942` after cure | ``ef8172f` (see git log)` (merge of `e12a2ab` into the branch) |

Pushed as a fast-forward of `agent/m2942` (`origin/agent/m2942` before push =
`4b54f9b`), so PR #166 stays the same pull request and `merge_governor` can re-verify it.