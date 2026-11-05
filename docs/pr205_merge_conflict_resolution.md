# PR #205 cure: merging `main` into `agent/m3386` without losing either side

Scope: `src/tensordem/dem3d_wedge.py`, `src/tensordem/rudder_ice_load.py`, the
forces-override contract document and its guards.
Evidence: `artifacts/pr205/verify_pr205.json`, `artifacts/pr205/pr205_merge_provenance.png`,
`artifacts/pr205/pr205_probe_invariance.png`,
`artifacts/forces_override_contract/{audit.json,probe_invariance.csv,forces_override_contract.png}`,
`runs/m4014/pytest_merged.log` (full suite at the merge commit; the same script,
`runs/m4014/run_final.sh`, re-runs it at any later tip).
Reproduce: `PYTHONPATH=src python3 scripts/verify_pr205_merge_semantics.py --main origin/main --merged 0d76f4c --figure`

## 1. What PR #205 is, and why it went dirty

PR #205 (branch `agent/m3386`) repairs a silent capability outage in the 3-D DEM
solver: two `IceDEM3D` subclasses overrode `forces()` with the pre-`update_history`
signature, so `IceDEM3D.diagnostics()` — which probes with
`forces(..., update_fracture=False, update_history=False)` — raised `TypeError` and
the whole monitoring surface of the moving-wedge bow-resistance case, the three-field
hull–ice pipeline and `RudderIceLoadDEM` was unreachable. The same branch removes a
second defect in the wedge override, where a diagnostics probe committed the bow
reaction into `self.last_reaction` by rebinding it (losing the preallocated tensor
identity that checkpoint restore's `last_reaction.zero_()` depends on).

While the branch was open, `main` advanced six commits (`f5de971` → `6c6f98b`),
and one of them — an independent fix for the same `TypeError` — added
`update_history` to `MovingWedgeBowIceDEM3D.forces` and reformatted the inherited
`super().forces(...)` call. Two authors had therefore touched the same seven lines
in opposite formatting, and GitHub reported `mergeable=false`.

Merge bases: branch base `f5de971`, `main` head `6c6f98b`, merge commit `0d76f4c`.

## 2. Conflict inventory and per-file resolution

`git merge-tree` predicted exactly one conflicting path out of the 24 paths the two
sides changed (20 on `main`, 5 on the branch, 1 overlap). There is no implicit
forgiveness in the audit: any overlap not listed below is a hard failure.

| path | both sides changed? | resolution |
|---|---|---|
| `src/tensordem/dem3d_wedge.py` | **yes** (overlap) | semantic union — see §3 |
| `src/tensordem/rudder_ice_load.py` | no (branch only) | taken as the branch wrote it |
| `tests/test_dem3d_forces_override_contract.py` | no (branch only) | taken as the branch wrote it |
| `scripts/verify_forces_override_contract.py` | no (branch only) | taken as the branch wrote it |
| `docs/dem3d_forces_override_contract.md` | no (branch only) | branch text, plus a pointer repair (§4) |
| 20 paths touched by `main` only | no | taken byte-identically from `main` |

Mechanical overlay is what this repository has been burned by, so the union was not
declared by hand: `scripts/verify_pr205_merge_semantics.py` compares the *git blob
ids* of every changed path at `main`, at the pre-merge branch tip and at the merge
commit, and requires

* R1 — each path equals the side that changed it (an undeclared overlap fails; a
  one-sided change that the merge did not reproduce fails; a deletion either side
  made must also be a deletion in the merge);
* R2 — the merge introduces no path that neither side changed (a resolver may not
  invent content);
* R3 — for the declared union, named textual invariants must hold;
* R4 — the merged classes are imported and `inspect.signature` must show that both
  `update_fracture` and `update_history` bind.

Result: **24 files compared, 34 checks, 0 failed** (`artifacts/pr205/verify_pr205.json`,
figure `pr205_merge_provenance.png`, one bar per path, coloured by provenance).

## 3. The one union: `MovingWedgeBowIceDEM3D.forces`

The two sides agreed on the *interface* and differed on the *body*:

| contribution | from | merged? |
|---|---|---|
| `update_history: bool = True` on the override signature | both (duplicate fix) | once, as one keyword-only switch |
| forwarding `update_history=update_history` to `super().forces` | `main` (and the branch, reformatted) | kept verbatim in `main`'s layout |
| probe guard `if update_fracture:` around the reaction commit | branch | kept |
| `self.last_reaction.copy_(bow_reaction)` instead of rebinding | branch | kept |
| explanatory comment on the base-class probe contract | branch | kept |

The merged text differs from the branch by one hunk only — the `super().forces(...)`
call is `main`'s two-line form instead of the branch's five-line form. Because the
conflict was pure formatting, taking `main`'s layout keeps `main`'s fix verbatim and
makes the next merge of this file quiet. Semantics are unchanged: the file is
byte-identical to the resolution the earlier cure attempt produced, so the merge is
deterministic rather than author-dependent.

Invariants asserted by R3 (all pass):
`update_history` declared exactly once on the override · `update_history` forwarded
exactly once · `update_fracture` still forwarded · probe guard present · in-place
`copy_` present · the unconditional `self.last_reaction = bow_reaction` that the
branch removed is **absent** (if a resolver restored `main`'s whole block it would
come back and R3 fails).

## 4. A gate the branch would have failed on its own

The branch's document cited the contact-invariants module without the `src/`
package root in front of it, so the cited path did not resolve to a tracked file. `scripts/audit_doc_evidence.py` classified it
`code:MISSING / ORPHAN` (no file in the tree can rebuild it), and
`tests/test_doc_evidence_pointers.py` was red with 4 failures — three of them because
the committed ledger predates the new document, one of them this orphan. A document
may not cite evidence that exists only on an agent's disk. Fixed in the follow-up
commit on the same branch:

1. the pointer now reads `src/tensordem/contact_invariants.py`;
2. `docs/doc_evidence_ledger.json` regenerated with the audit script (107 documents,
   530 pointers, 0 orphans, all 100 unresolved run artefacts `REGENERABLE`, every
   pre-existing entry preserved);
3. `artifacts/forces_override_contract/{audit.json,probe_invariance.csv,forces_override_contract.png}`
   regenerated by the producer script from this tree and committed, so the cited
   evidence is in the clone (the figure is byte-size-identical to the pre-merge run,
   i.e. the anchors are deterministic).

## 5. Tests

Environment: torch 2.7.1 (CPU), pytest 9.2.0, 128-core host shared with other agents
(load average > 300 during the run, which is why the suite takes ~9 min instead of
the ~2 min it takes idle).

| command | result |
|---|---|
| `pytest -q tests/test_dem3d_forces_override_contract.py` (merge commit) | **7 passed, 5 subtests passed** |
| `pytest -q` full suite, merge commit `0d76f4c` | **1235 passed, 4 failed, 2 skipped, 157 subtests** — the 4 failures are exactly the ledger-gate module of §4 |
| `pytest -q tests/test_doc_evidence_pointers.py` after the §4 fix | **14 passed** (was 4 failed, 10 passed) |
| `pytest -q` full suite, final branch tip | re-run with `runs/m4014/run_final.sh`; result quoted in the mission report |
| `PYTHONPATH=src python3 scripts/verify_forces_override_contract.py` | 3 signatures bind the probe; max commit drift 0.000e+00 N; worst anchor error 0.000e+00; all anchors < 3 % |

The branch is therefore not below `main`: the only red tests at the merge commit were
introduced by the branch's own document and are green after the §4 commit.

## 6. Note on the evidence grade of the repaired capability

The merge is a *software* repair. The PR's own audit anchors are exact identities
(bitwise 0.0 relative error):

| anchor | worst relative error | < 3 % ? |
|---|---|---|
| bow reaction = −Σ nodal bow force | 0.0 (bitwise) | yes |
| `bow_contact_energy_J` = ½ k Σ overlap² | 0.0 (bitwise) | yes |
| repeated probe is bitwise identical | 0.0 (bitwise) | yes |
| stored `last_reaction` drift across probes | 0.0 N exactly | yes |
| merge union invariants (§3) | n/a — exact text predicates, 34/34 pass | yes |

None of these is an ice-resistance validation. The 40-step engagement windows give
80.04 N peak for both 0.15 m and 0.25 m thickness because the contact set saturates
at two particles; no thickness sensitivity is claimed here, and no absolute
resistance benchmark was attempted. That remains the open item for the
moving-wedge capability.

## 7. Budget/limitation statement

* The full suite was not re-run after the final documentation-only commits
  beyond the run recorded in `runs/m4014/pytest_merged.log`; the commits that
  followed the merge touch only `docs/`, `artifacts/` and a new `scripts/` audit.
* `origin/main` moved again during this work (the git host is shared); the merge was
  cut against `6c6f98b`, and `merge_governor` re-verifies the merge on its next
  cycle. If `main` has advanced, the audit is re-runnable in seconds:
  `PYTHONPATH=src python3 scripts/verify_pr205_merge_semantics.py`.