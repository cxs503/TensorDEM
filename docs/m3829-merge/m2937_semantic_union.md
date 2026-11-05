# PR #165 cure: semantic union of `agent/m2937` into `main`

Mission m3829 (cure branch `agent/m3829`).  PR #165 was reported dirty
(`mergeable=false`) because both `main` and `agent/m2937` independently closed the
**same** piece of verification debt — the undrained embedment factor of the
ridge/pipeline subgrade being applied twice — and therefore wrote to the same
file names.  This page records the three-way merge, the per-file resolution and
the evidence that nothing from either side was dropped.

## 1. Provenance

| item | value |
|---|---|
| merged side (ours) | `agent/m2937` @ `7aa61a28c1bcda4f7e0d0554639682de242ea74d` (old SHA) |
| merged-into side | `main` @ `e12a2abd636ab5a2998054dbe0fcf3c18b6a8564` |
| merge base | `8cc88f7f1bc681cb028269014e61c1fc265d9d2e` |
| result | merge commit on `agent/m2937` (new SHA in the commit subject/message) |
| conflicts | 5 files (3 content, 2 add/add) |

Both sides found the identical defect (Skempton 1951 `N_c` already carries the
`0.2 z/B` embedment term, so an additional Meyerhof/Hansen depth factor counts
the same physics twice and overstates undrained capacity by 20 % at `z/B = 1`,
up to 44 % at the cap).  They fixed it with two different but strip-equivalent
APIs, and they named their runners and test modules identically.  The merge is
therefore resolved by **union**, never by overlay.

## 2. Conflict list and per-file resolution

| file | conflict | resolution |
|---|---|---|
| `src/tensordem/ridge_geotech.py` | content (3 hunks) | **union of both APIs.** `skempton_nc(z_over_b, footing=...)` (branch) *and* `skempton_undrained_nc(depth, width, length_ratio=...)` + `SKEMPTON_*` constants (main) are both kept and both exported; the undrained branch of `MohrCoulombSoil.vertical_pressure_capacity` uses the `length_ratio`-aware helper (a strict superset: `length_ratio=0`, the pipeline case, is numerically identical to the branch form, max difference 8.9e-16 over `z/B` in [0, 6]). Both docstrings keep both derivations; the 0.5 % square-footing difference (6.168 vs the tabulated 6.2) is documented rather than silently picked. |
| `docs/ice_ridge_pipeline_grounding.md` | content | **union paragraph**: both Skempton forms described, and both verification pages cited (`docs/ridge_geotech_capacity_verification.md` 43 checks / 0.071 % worst error, *and* `docs/ridge_capacity_skempton_audit.md` 20–44 % defect magnitude). |
| `tests/test_ridge_grounding.py` | content | **union assertions**: the branch's `pytest.approx(30.0e3 * 6.168, rel=1e-9)` and main's `pytest.approx(30.0e3 * 5.14 * 1.2, rel=1e-12)` plus the "not double counted" upper bound all kept; they are mutually consistent (5.14 x 1.2 = 6.168). |
| `scripts/verify_ridge_geotech_capacity.py` | add/add (two independent runners) | main's runner keeps the canonical path (it is the producer cited by `scripts/build_ridge_capacity_audit.py` and by `docs/ridge_capacity_skempton_audit.md`). The branch's runner is preserved **verbatim** as `scripts/verify_ridge_geotech_capacity_embedment.py` — different artifact directory (`artifacts/ridge_geotech_capacity/` vs `artifacts/ridge_capacity_verification/`), so the two artifacts cannot overwrite each other. The branch page's three citations were repointed. |
| `tests/test_ridge_geotech_capacity_verification.py` | add/add (two independent suites, disjoint test names) | main's suite keeps the canonical path; the branch's 12-test suite is preserved verbatim as `tests/test_ridge_geotech_capacity_embedment_verification.py`. The branch page's two citations were repointed. |
| `docs/doc_evidence_ledger.json` | (auto) | regenerated with `scripts/audit_doc_evidence.py` after the union: the new/unrepointed pointers of the branch page enter the ratchet instead of failing the gate. |

Non-conflicting branch additions (`src/tensordem/ridge_geotech_verification.py`,
`docs/ridge_geotech_capacity_verification.md`) merged cleanly and are untouched.

Rationale for keeping two runners/suites instead of fusing them into one file:
the branch's and main's suites import *different* helpers
(`skempton_nc` vs `skempton_undrained_nc`, `ridge_geotech_verification` vs
inline tables).  Fusing them would have meant rewriting verified test bodies,
which is exactly the "mechanical overlay" hazard the mission forbids; keeping
both under explicit names preserves every assertion and every artifact.

## 3. What each side contributes (nothing dropped)

* branch: `ridge_geotech_verification.py` (reference-check harness, 43 checks),
  `skempton_nc` discrete-footing form, embedment runner + 12-test suite,
  `docs/ridge_geotech_capacity_verification.md`.
* main: `skempton_undrained_nc` continuous shape-factor form,
  `docs/ridge_capacity_skempton_audit.md` + its builder, 138-line capacity test
  suite, doc-evidence ledger/gate.

## 4. Verification evidence

Run in this worktree (merged tree) and in a `main` worktree at `e12a2ab`
(baseline), same interpreter and flags:

| suite | main baseline `e12a2ab` | merged `agent/m2937` |
|---|---|---|
| `tests/test_ridge_grounding.py` + `tests/test_ridge_geotech_capacity_verification.py` + `tests/test_doc_evidence_pointers.py` | 58 passed | 58 passed |
| `tests/test_ridge_geotech_capacity_embedment_verification.py` | n/a | 12 passed |
| total | 58 | **70 passed, 0 failed** |
| `scripts/audit_doc_evidence.py --check` | clean | clean |

Both runners were executed in the merged tree; they regenerate the JSON ledgers
and figures below (artifacts stay out of the clone by repo convention, copies of
the numeric output are kept under `docs/m3829-merge/evidence/` and reproduced by
the merge commit message):

* branch runner (now `scripts/verify_ridge_geotech_capacity_embedment.py`) ->
  `artifacts/ridge_geotech_capacity/verification.json`: 43 reference checks,
  worst relative error **0.071 %** (acceptance 15 %), roll-up `WITHHELD: 1`
  (the lateral limit pressure `N_p = 2 + 1.5 z/D` has no independent anchor).
* merged runner (`scripts/verify_ridge_geotech_capacity.py`) ->
  `artifacts/ridge_capacity_verification/capacity_verification.json`: Vesic
  factors max abs error 5.1e-3, Skempton strip sequence max relative error
  **0.071 %**, pre-fix defect magnitude **44 %** at the cap; reference case
  `D = 0.914 m`, cover `0.5 m`, `s_u = 30 kPa`, `z/B = 1.547`, `N_c = 6.730`,
  `q_ult = 201.9 kPa`, capacity 184.5 kN/m against 221.5 kN/m pre-fix (+20.0 %).

Error verdict: every reference-checked quantity is far below the 3 % line
(worst 0.071 %, i.e. 42x margin); the large errors quoted above are the *pre-fix*
defect that both branches removed, not residual error.

## 5. Caveats / honest limits

* The **full** `pytest tests` suite could not be run in this session: the host
  was at load average ~1900 (other tenants), and two full-suite attempts were
  reaped before completion.  The comparison above is therefore scoped to the
  files this merge touches (ridge grounding, both capacity suites, doc-evidence
  gate) plus the branch's own suite; no test *file* was removed by the merge, so
  the baseline surface is a subset of the merged surface.
* `tests/test_doc_evidence_pointers.py` is only meaningful inside a git
  worktree; the archive export used for the first baseline attempt reported 6
  errors for that reason and was replaced by a real `main` worktree.
* `skempton_nc(footing="square")` (6.2, tabulated) and
  `skempton_undrained_nc(..., length_ratio=1)` (6.168 = 1.2 x 5.14) differ by
  0.5 % by construction.  Both are kept, documented, and identical on the strip
  branch that the pipeline model actually uses.
* The two verification runners write to different `artifacts/` directories; those
  are not committed (repo convention: regenerable artifacts stay out of the
  clone), which is why copies of the numeric evidence and the two figures sit
  under `docs/m3829-merge/evidence/`.