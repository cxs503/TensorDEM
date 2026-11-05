# Merge record: PR #165 (`agent/m2937`) into `main`

Status: **resolved locally, awaiting publication** (the merge commit lives on
`agent/m3856`; see "Handover" below).

| item | value |
|---|---|
| repository | TensorDEM |
| merge base | `8cc88f7f1bc681cb028269014e61c1fc265d9d2e` |
| main (ours) | `e12a2abd636ab5a2998054dbe0fcf3c18b6a8564` (29 commits ahead of base) |
| PR head (theirs) | `7aa61a28c1bcda4f7e0d0554639682de242ea74d` — `agent/m2937`, 1 commit ahead of base |
| conflict class | 2 × add/add (same path invented independently), 3 × content, 2 files auto-merged |
| strategy | semantic union — both sides' content, APIs and numbers preserved; mechanical overlay explicitly rejected |
| renames | none of the branches' files removed; two functions and one module-level constant per side renamed so both APIs coexist in one module |

## 1. What both branches did

Both sides independently fixed the *same* physical defect in the ridge-grounding
subgrade model: the undrained branch of `MohrCoulombSoil.vertical_pressure_capacity`
applied the Skempton (1951) embedment factor `5.14 (1 + 0.2 z/B)` **and** a
Meyerhof depth factor `(1 + 0.2 k)`, counting the `0.2 z/B` term twice and using
the square-footing cap `9.0` for a strip footing. The over-prediction is 20 % at
`z/B = 1` and 44 % at the cap.

* `main` (via the audit campaign) introduced
  `skempton_undrained_nc(depth, width, *, length_ratio)` with the strip cap `7.5`,
  the shape-factor route to `9.0`, and the audit document
  `docs/ridge_capacity_skempton_audit.md`.
* `agent/m2937` introduced `skempton_nc(z_over_b, *, footing="strip")` with the
  published table values (strip `5.14 → cap 7.5`, square/circular `6.2 → cap 9.0`),
  the verification module `src/tensordem/ridge_geotech_verification.py`
  (43 reference checks rolled up by evidence grade) and the document
  `docs/ridge_geotech_capacity_verification.md`.

The two APIs are the same closed form on the strip default, so the union keeps
both public entry points rather than dropping either branch's interface.

## 2. Conflict files and the resolution applied to each

| file | sides | resolution |
|---|---|---|
| `src/tensordem/ridge_geotech.py` | content (3 hunks) | Union. `main`'s `skempton_undrained_nc` (+ `SKEMPTON_NC_STRIP_SURFACE/MAX`, `SKEMPTON_SQUARE_TO_STRIP`) is the single reference implementation; `agent/m2937`'s `skempton_nc(z_over_b, footing=...)` is retained as a thin wrapper — strip branch delegates to `skempton_undrained_nc(z, 1.0)` (bit-identical), square/circular branch keeps the *tabulated* `N_c0 = 6.2` via the new constants `SKEMPTON_NC_SQUARE_SURFACE = 6.2`, `SKEMPTON_NC_SQUARE_MAX = 9.0` (the `5.14 × 1.2 = 6.168` shape route is 0.5 % away, i.e. below the 3-significant-figure resolution of the table; the difference is documented in the docstring instead of being silently averaged). Both symbols are exported in `__all__`. The `vertical_pressure_capacity` docstring is the union of both wordings and the code keeps `main`'s `length_ratio`-aware call, with the branch's regression comment (20 % at `z/B = 1`, cap lifted to 10.8) preserved. |
| `scripts/verify_ridge_geotech_capacity.py` | add/add | Union in one runner with two stages: `run_reference_audit()` (`main`: Vesic table, Skempton strip table, Prandtl anchor, pre-fix/post-fix curves → `artifacts/ridge_capacity_verification/`) and `run_reference_ledger()` (`agent/m2937`: the 43-check ledger, impact table, spring reaction figure → `artifacts/ridge_geotech_capacity/`). `main()` runs both by default; `--audit` / `--ledger` select one. Renames required to coexist: `OUT → OUT_AUDIT/OUT_LEDGER`, `FIG → FIG_LEDGER`, the imported ledger table → `SKEMPTON_STRIP_TABLE_REF`, and one `main()` per stage. |
| `tests/test_ridge_geotech_capacity_verification.py` | add/add | Union of both suites in one module: 9 tests from `main` (table-driven factors, cap/shape scaling, embedment-enters-once, gradient, drained branch, WITHHELD lateral anchor) + 12 tests from `agent/m2937` (published-table checks through `skempton_nc`, cap/rejection, cap bound, spring fractions, legacy-impact numbers, whole-ledger roll-up, defect quantification). Shared `_undrained()` helper widened to the superset signature; the duplicated `test_undrained_capacity_is_linear_in_shear_strength` name resolved by renaming the per-length variant to `..._per_length_...`; ledger table imported as `SKEMPTON_STRIP_TABLE_LEDGER`. |
| `tests/test_ridge_grounding.py` | content (1 hunk) | Union: both assertions kept — `30.0e3 * 5.14 * 1.2` (`main`, rel 1e-12) and `30.0e3 * 6.168` (`agent/m2937`, rel 1e-9) are the same value from two independent hand computations, plus `main`'s anti-double-factor bound `p < 30.0e3 * 5.14 * 1.2 * 1.2`. |
| `docs/ice_ridge_pipeline_grounding.md` | content (1 hunk) | Union: `main`'s paragraph (once-only embedment, strip cap 7.5 / square 9.0 via the shape factor, 20–44 % over-statement, pointer to the audit) kept verbatim, followed by the branch's table-form statement (`min(5.14 (1 + 0.2 z/B), 7.5)`, square `min(6.2 (1 + 0.2 z/B), 9.0)`) and its verification evidence (43 checks, worst error 0.071 %). |
| `docs/doc_evidence_ledger.json` | generated | Regenerated with `scripts/audit_doc_evidence.py` **after** the doc union, with `artifacts/` absent (the state of a fresh clone), so the evidence gate stays green in CI rather than baking local `UNTRACKED` statuses into the ratchet. 102 unresolved pointers, 0 orphans. |
| `src/tensordem/ridge_geotech_verification.py` | theirs only | Auto-merged, content unchanged. |
| `docs/ridge_geotech_capacity_verification.md` | theirs only | Auto-merged, content unchanged. |

No file was dropped and no side was resolved by "take ours"/"take theirs"
wholesale.

## 3. No-silent-loss evidence

`scripts/audit_merge_content_survival.py` takes every line each parent *added*
relative to the merge base and tests whether the merged tree still carries it,
allowing only the declared renames and the declared rewrites (each with a
reason). Machine output: `runs/m3856/content_survival.json`.

* `main` side: 100 % of added lines survive (every file, no exceptions).
* PR side: survives verbatim except the declared rename/regeneration set
  (`OUT`/`FIG` renames, the two `main()` entry points, the ledger table alias,
  the duplicated test name, the `skempton_nc` body rewritten onto the new
  constants, and the rewritten module docstrings).

## 4. Test evidence

| command | result |
|---|---|
| `PYTHONPATH=src pytest tests/test_ridge_geotech_capacity_verification.py tests/test_ridge_grounding.py -q` | 56 passed |
| `PYTHONPATH=src pytest tests/test_doc_evidence_pointers.py -q` | 14 passed (after regenerating the ledger; the pre-regeneration failure is recorded in `runs/m3856/`) |
| `PYTHONPATH=src python scripts/verify_ridge_geotech_capacity.py` | exit 0; both stages written; 43 reference checks all `ok`, worst relative error `1.2e-4` (Vesic factors reproduced to the two-decimal table) |
| `PYTHONPATH=src pytest <14 test modules, chunk 0> -q` | 187 passed, 6 subtests passed |
| `PYTHONPATH=src pytest <7 test modules, chunk 0 of remainder> -q` | 83 passed, 1 skipped |

The full 1246-test suite could not be completed inside the agent's 120 s shell
window (the run reaches ~85 % before the harness reaps the process group); the
chunked runs above are the evidence obtained, and they are all green. A complete
suite run is the remaining open item — it is required before the branch is
declared fully verified.

## 5. Notes and limitations

* The undrained factor is the *only* quantity whose number changed in either
  branch; the merge does not change any value relative to either parent. The two
  square-footing surface constants (`5.14 × 1.2 = 6.168` vs the tabulated `6.2`)
  are both kept and cross-referenced, because silently picking one would change a
  published-table comparison by 0.5 %.
* `skempton_nc` is not a second implementation: it is a documented wrapper, with
  one deviation (the tabulated `6.2`) that is stated in its docstring.
* Handover: `git push` is blocked for the agent by the platform security policy,
  so the merge commit was not pushed to `agent/m2937`. It is on `agent/m3856`
  (parents `e12a2ab` + `7aa61a2`); publishing it fast-forwards
  `agent/m2937` from `7aa61a2` to the merge commit, after which merge_governor's
  re-check sees a clean, non-conflicting PR.