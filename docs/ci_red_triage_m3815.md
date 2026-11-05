# CI-red triage for TensorDEM PRs #195 / #196 / #199 / #201 (mission m3815)

**Scope**: four open pull requests, all cut at merge base `f5de971`, all red on the
`unit-tests` job. This page records (i) that they are red for **one shared reason**,
(ii) the second, independent numerical defect found while reproducing, and
(iii) exactly what each branch now changes. Nothing here relaxes a physical
criterion: the only test assertion touched is shown, with numbers, to have been
below the resolution of the quantity it measured.

## 1. Four red PRs, one root cause

| PR | branch | branch subject | shared files touched by the branch |
|---|---|---|---|
| #195 | `agent/m3242` | closed-form dynamic moving-plane `tool_impulse` | none (additive) |
| #196 | `agent/m3254` | engine-level time-step stability boundary | `src/tensordem/stability.py`, `tests/test_dem3d_stability.py` |
| #199 | `agent/m3292` | Winkler subgrade modulus vs Vesić (1961) | `src/tensordem/ridge_geotech.py` |
| #201 | `agent/m3319` | bounded-domain boundaries of the D3Q19 propeller-race solver | none (additive) |

The red tests were identical on all four branches, and they are **not** the tests
the branches add:

```
tests/test_doc_evidence_pointers.py::DocumentedEvidenceGateTests::
  test_cited_source_docs_and_tests_are_tracked          FAILED
  test_committed_ledger_is_current                      FAILED
  test_no_orphan_pointers                               FAILED
  test_uncommitted_artifacts_have_a_producer            FAILED
```

Diagnosis: the four branches forked at `f5de971`; `main` has since repaired three
citations in documents the branches never touched, and regenerated the ratchet
ledger `docs/doc_evidence_ledger.json`:

| pointer reported by the gate | state on the stale base | state on `main` |
|---|---|---|
| `docs/dem3d_contact_impact_dashpot_convention.md`, line 18 — a stray `_probe_duration` script token (no such file was ever committed) | MISSING | pointer rewritten by the main-side repair |
| `docs/wake_lbm_analytic_verification.md`, line 14 — `coupling.py` cited under a bare `threefield/` prefix | ORPHAN (no producer) | resolves to `src/tensordem/threefield/coupling.py` |
| `docs/dem3d_fracture_footprint_breakage.md`, line 96 — `…/footprint_field_map.png` | ORPHAN | produced by the merged footprint audit |

So the gate was reporting **main-side drift, not branch-side damage** — the same
class of failure on four branches, fixed the same way on four branches:

```bash
git merge origin/main --no-edit          # inherit the repaired citations + ledger
python scripts/audit_doc_evidence.py     # re-ratchet the ledger over the new docs
```

After the merge each branch's own new documentation adds pointers, so the ledger
must be regenerated **on the branch** (the audit is a ratchet: it freezes the exact
pointer set). Unresolved pointers after regeneration: 102 / 100 / 100 / 103 for
#195 / #196 / #199 / #201, **zero orphans on all four** (the same `REGENERABLE`
run-artifact set `main` already carries — uncommitted figures with a producer).

### #196 needed two extra, branch-local repairs

1. The reproduction block of `dem3d_engine_threshold_crossing.md` (line 130, #196 only) contained
   ```bash
   ... 重写 metrics.json (原文档此处无分隔符) summary.md 两图
   ```
   The Chinese prose ran the two file names together without a separator, so the
   pointer scanner read one token joining `metrics.json` + `/summary.md`, which
   resolves to nothing. Rewritten as `重写 metrics.json、summary.md 与两张图` — same meaning, one
   unambiguous pointer.
2. `#196` had committed an agent scratch helper `.tmp_api.py` (platform login /
   report upload, 35 lines) at the repository root. It is referenced by nothing in
   the tree (`grep -rn tmp_api` → empty) and was removed: a solver library should
   not ship an agent's API client.

## 2. The second defect: an absolute mass-conservation bound below one ulp

Reproducing the suite locally exposed one further failure, present on `main` as
well, that no branch introduced:

```
tests/test_threefield_lbm_transport.py::test_streaming_is_mass_conserving
assert 3.637978807091713e-12 < 1e-12          # 1 failed
```

`periodic_stream` is an exact permutation of the 19 discrete velocities plus a
periodic shift, so every population is preserved bit for bit; the only error is
the rounding of the two `Tensor.sum()` reductions. For the test grid (`n = 12`,
`12³ × 19 = 32 832` populations, mean ½) the total mass is `M = 16 490.1761`, and

$$\mathrm{ulp}(M) = 2^{-52}\cdot 2^{14} = 3.638\times10^{-12}$$

— the measured defect **is exactly one ulp**. A bound of `1e-12` is therefore
`1e-12 / 3.638e-12 = 0.27 ulp` at that grid: *smaller than the smallest error a
float64 reduction can achieve*, and it only "passed" while the thread-dependent
reduction order happened to round exactly. `scripts/audit_ci_red_mass_tolerance_m3815.py`
measures this on `n = 12 … 32` (`artifacts/ci_red_mass_tolerance_m3815/mass_tolerance_audit.json`,
figure `mass_tolerance_vs_grid.png`):

| n | populations | M [–] | ulp(M) | \|ΔM\|, 1 thread | \|ΔM\|, 128 threads | old `<1e-12` | new `≤1e-13·M` |
|---|---|---|---|---|---|---|---|
| 12 | 32 832 | 16 490.18 | 3.638e-12 | 0 (0 ulp) | 3.638e-12 (1 ulp) | fails at 128 thr | pass |
| 16 | 77 824 | 38 981.67 | 7.276e-12 | 7.276e-12 (1 ulp) | 0 | fails at 1 thr | pass |
| 20 | 152 000 | 76 046.73 | 1.455e-11 | 1.455e-11 (1 ulp) | 0 | fails at 1 thr | pass |
| 24 | 262 656 | 131 323.32 | 2.910e-11 | 2.910e-11 (1 ulp) | 0 | fails at 1 thr | pass |
| 32 | 622 592 | 311 474.77 | 5.821e-11 | 4.075e-10 (7 ulp) | 0 | fails at 1 thr | pass |

Summary over the 10 measured cases: the old absolute bound is violated in **5/10**
cases (4/5 grids at one thread, 1/5 at 128 threads); the new relative bound in
**0/10**. At the largest grid the old bound is only `0.017 ulp` of the total, i.e.
unsatisfiable by construction.

The assertion is therefore rewritten to conserve mass *relative to the mass it
conserves*:

```python
total = f.sum()
streamed = periodic_stream(inst, f).sum()
# one ulp of the ~1.6e4 total is 3.6e-12: an absolute 1e-12 bound sits below the
# resolution of the quantity it measures.
assert float((streamed - total).abs()) <= 1e-13 * float(total.abs())
```

`1e-13` relative is still ≈ 450 ulp — far tighter than any physical leak, and the
negative control is unambiguous: a 1e-9 relative loss (5 ppm of the mass, still
physically tiny) now fails, where the old bound would also fail, as it should.
**Nothing in the solver changed**; only the yardstick was put above the resolution
of the quantity.

## 3. Verification actually run

| branch | doc-evidence gate | branch-specific suites | mass-conservation test |
|---|---|---|---|
| #195 `m3242` | 14/14 pass | `test_moving_plane*, test_moving_plane_ledger_verification` 36 passed | 1 passed |
| #196 `m3254` | 14/14 pass | stability / hertz / buoyancy-coupling / energy-timestep / integration-order 202 passed (+6 subtests) | 1 passed |
| #199 `m3292` | 14/14 pass | `test_ridge_grounding, test_ridge_geotech_capacity_verification, test_ridge_winkler_subgrade_verification` 58 passed | 1 passed |
| #201 `m3319` | 14/14 pass | `test_threefield*, test_threefield_boundary_verification` | 1 passed |

## 4. Notes and limits

* The four `unit-tests` failures are one defect, so they were fixed with one
  recipe (merge `main`, regenerate the ledger) applied branch-by-branch — the
  branches stay textually disjoint, as the repair touches only the ratchet file
  plus (on #196) two branch-local lines.
* The mass-conservation bound is a *repository-wide* defect, not a property of any
  of the four PRs: it reproduces on `main`. It is repaired identically on all
  branches so that the four merges are order-independent (an identical hunk merges
  cleanly whichever lands first).
* `main` and the four branches were not re-validated by a full serial `pytest`
  run on the CI runner: this host carried a load average of ~6000 on 128 cores
  while the work was done, so the suites above were run per branch on the files
  whose behaviour the branch can change. The gate tests, which were the reported
  failure, are deterministic and were run in full.