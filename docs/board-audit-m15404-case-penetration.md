# Board audit m15486 — m15404 CASE-PENETRATION (TensorDEM)

**Audited** mission 15404 · anchor CASE-PENETRATION · repo TensorDEM · branch `agent/m15404`
(head `41c42bc2` "Merge #1299 (m15404) via merge-master") · **Verdict: L2 — reopened**
**Auditor** mission 15486 (board-audit) · evidence `artifacts/audit_m15486/audit_m15404.json`

This is a **cross-check, not a re-run**: the mission's own numbers are recomputed from the
raw `F(z)`/`KE(z)` profile it published, the acceptance flags are re-derived, and the
evidence chain (git, gate test, repro entry, platform judge row) is inspected. The task's
stated judge — `tensor_benchmark_cases` where `anchor_slug='CASE-PENETRATION'` — is queried
live.

## 1. What was audited (raw inputs, sha256-bound)

| input | path | sha256 (first 16) |
|---|---|---|
| production run artifact | `artifacts/case_penetration/base_rd2_v060/result.json` (worktree m15404) | `7a41bd45f98ea610` |
| rate-doubling control | `artifacts/case_penetration/rate2x_rd2_v120/result.json` | `feb62720dd260649` |
| teleport negative control | `artifacts/case_penetration/teleport_control/result.json` | `591b97c63eb5a460` |

Copies (byte-identical) live in `artifacts/audit_m15486/source/` so this audit is
self-contained — the mission's own artifacts are **not** in git (finding F4).

## 2. Independent recomputation of the benchmark

Reference law: Brzinski III, Mayor & Durian, PRL **111**, 168002 (2013), Eq. (2)/(3c)/(3d),
`alpha = 35 ± 5`. Recomputed from the published profile with `artifacts/audit_m15486/audit_m15404.py`
(no mission code on the metric path):

| quantity | mission | audit (independent) | delta | gate | verdict |
|---|---|---|---|---|---|
| `alpha_fit` (piecewise-shape fit, headline) | 349.34 | **341.11** | −2.4 % | | |
| `alpha_fit_deep` (deep-branch constrained) | 348.36 | **340.09** | −2.4 % | ≤ 15 % err | **FAIL (871–898 %)** |
| `alpha_fit_free` (free slope+intercept) | 90.91 | 69.24 | −24 % | | FAIL |
| mean force ÷ Brzinski(alpha=35) in window | 11.21 | **11.13** | −1.2 % | | |
| KE max, headline (all steps) | 0.365 J | n/a (unsampled) | — | ≤ 0.1 J | **FAIL** |
| KE max, sampled (profile) | 0.201 J | **0.201 J** | 0 | ≤ 0.1 J | **FAIL** |
| force variation coefficient | 0.476 | **0.476** (resid. of shape fit) | 0 | ≤ 0.10 | **FAIL** |
| max contact overlap ratio | 0.01396 | 0.01396 | 0 | ≤ 0.05 | PASS |

Recomputed gate set = `{KE: false, CV: false, force_error: false, overlap: true}` —
**identical to the mission's own flags** (`disagreements = {}`). The alternative
estimators (deep-branch, free, independent smoothing with a different edge convention)
all land between alpha ≈ 69 and 349, i.e. **outside the [30, 40] band by 2–10×**; the
FAIL is structural, not a fitting-convention artefact.

**Artifact ↔ code self-consistency:** re-running the mission's own `evaluate_profile` on the
published profile reproduces `alpha_fit`, `alpha_fit_deep`, `alpha_fit_free`, CV and
`penetration_force_error` to **abs_delta = 0.0**. The artifact was produced by the shipped
code, honestly — the numbers are not the problem, the physics is.

**Teleport negative control:** reproduced exactly (KE 5.6976 J vs mission 5.70 J), 15.6×
the velocity-BC headline KE — the m15367 energy-injection phenomenon is real and the
loading-path attribution is correct.

## 3. Findings (severity-ranked)

**F1 · BLOCKING for L4 — benchmark not met.** 3 of 4 predeclared gates fail; the anchor
quantity `alpha` is off by 871–898 % against a tolerance of 15 %. The mission states this
honestly (docs `## ⑥`, report 14510); the audit confirms it arithmetically.

**F2 · HIGH — the task's judge row does not exist.** `SELECT ... FROM tensor_benchmark_cases
WHERE anchor_slug='CASE-PENETRATION'` returns **0 rows** (16 anchors registered:
GAP-*/H1..H6/REPRO-FVM-SOD1978/CASE-EULER-VORTEX, none for penetration). The "唯一裁判" is
absent, so the tolerable criterion had to come from the mission's own `ACCEPTANCE` block
(`src/tensordem/penetration_quasistatic.py:118-123`) plus the literature anchor. That makes
the acceptance line self-declared, which is exactly what a benchmark row exists to prevent.
*Remediation:* register the row (metric = alpha vs Brzinski 35±5 with ≤15 % tolerance + KE
and CV gates), then re-adjudicate.

**F3 · HIGH — the quasi-staticity claim is falsified by the mission's own control.**
`fig2` is captioned "倍速对照 … 准静态性证据：贯入力曲线与 alpha 不随速率变化". The two
artifacts say otherwise: doubling the drive speed 0.6 → 1.2 m/s gives

| | base (0.6 m/s) | rate2x (1.2 m/s) | change |
|---|---|---|---|
| `alpha_fit` | 349.34 | 559.64 | **+60 %** |
| KE max (sampled) | 0.201 J | 0.961 J | **×4.8** |

So the run is rate-dependent in precisely the metric that was supposed to prove
quasi-staticity. The honest statement is the opposite of the caption; the correct speed
scale is `v/sqrt(gR)` = 2.14 (base) — an order of magnitude above the quasi-static regime.
This is the mechanism behind F1 and should be the head of any rework (reduce speed and/or
raise damping until `alpha` stops moving with rate).

**F4 · HIGH — reproducibility chain broken on the branch.** From a fresh checkout of
`41c42bc2`: `git ls-files artifacts/case_penetration` → **0 files**; `artifacts/case_penetration/`
is not even present (untracked, worktree-local). The only production runner
(`artifacts/case_penetration/tools/run_case.py`, `bed_recipes.py`, `post_report.py`) is
untracked, so **no `repro_cmd` exists in the repo** and the artifact cannot be regenerated
from the branch. Additionally the gate test is wired to a path no run produces:
`tests/test_penetration_quasistatic.py:37` → `artifacts/case_penetration/base/result.json`,
while the labels actually used are `base_rd2_v060` / `rate2x_rd2_v120` / `teleport_control`
→ `test_cached_case_meets_acceptance` is **permanently skipped** (it can never see an
artifact, and would fail if it did — which is why the skip is currently silent-green).

**F5 · MEDIUM — the live A/B evidence test is red in `main`, and the accepted check
deselects exactly it.** Full file, this worktree:
`python -m pytest tests/test_penetration_quasistatic.py -q` →
**1 failed, 11 passed, 1 skipped in 196.17 s**;
`test_quasistatic_loading_stays_in_equilibrium_while_teleport_does_not` fails at
`assert np.isfinite(metrics["alpha_fit"])` because `alpha_fit = nan` (degenerate profile,
137.9 s). The mission's accepted check string is
`-k "not quasistatic_loading and not velocity_bc"` → 10 passed, 1 skipped (verified, 5.05 s).
Green is therefore obtained by exclusion of the very test the module docstring calls "the
A/B contrast that makes the fix evidence rather than an assertion". Either fix the reduced
case (smaller/faster config that keeps the fit identifiable) or move the contrast into a
cheap analytic assertion.

**F6 · MEDIUM — the headline KE is not reproducible from the published profile.** The
profile is decimated (`sample_every_steps=5`) while the headline `kinetic_energy_max_J =
0.365 J` is the all-step max; the maximum visible in the published series is 0.201 J
(×1.8 short). Verdict unchanged (both > 0.1 J), but the artifact is not self-contained:
post the unsampled trace or define the metric on the sampled series.

**F7 · LOW — literature provenance not in repo.** `benchmarks/case_sphere_penetration/run_case.py`
cites `docs/refs/brzinski2013_1307.4638v2.pdf`; `docs/refs/` does not exist on the branch.
The reference constants themselves are correct (checked against the paper's Eq. (2)/(3c)/(3d)
and alpha = 35 ± 5), but the "zero hand-transcription" claim is not checkable offline.

**F8 · LOW — engine integration is real and verified (positive finding).**
`tests/test_sphere_pack_dem.py` → **7 passed in 9.13 s** on the merged branch, covering the
two engine fixes claimed in `docs ## ⑤` (neighbour-list truncation for a large fixed body;
per-wall array history for Mindlin tangential memory). This part of the delivery stands.

## 4. Level = L2

| level | requirement | this mission |
|---|---|---|
| L0 | docs only | no — 819-line module + engine diff |
| L1 | math/unit validation | yes (Brzinski branches, C1 continuity, fit inversion: 5 tests) |
| **L2** | **independent functional implementation: input → compute → output** | **yes** — 810-grain bed, 18 810 steps, real `F(z)` artifact, second/minute-scale tests green |
| L3 | integration into main solver / platform | **no** — `benchmarks/case_sphere_penetration/run_case.py` (the repo's case harness) does **not** import `penetration_quasistatic`; the only importers are this module and its tests; the production runner is untracked (F4) |
| L4 | benchmark validation + error analysis + reproducible | **no** — F1 (871–898 % vs 15 %), F3, F4 |
| L5 | research/engineering result | no |

Supporting integration evidence that *does* exist (engine defect fixes reused by the rest of
the DEM test suite) is recorded under `integration_callers` but does not upgrade the case
capability itself: the audited case is not callable from the repo's case harness and its
validation gate fails.

## 5. Disposition

**Request changes / reopen (L2).** Keep the honest-FAIL record (it is good work and the
negative control is real), but the claim set must shrink:

1. **Close F3 first** — the reported "rate independence" is false; re-run at
   `v/sqrt(gR) ≪ 1` (or state plainly that the case is rate-dependent and alpha is an
   effective, protocol-defined quantity).
2. **Commit the artifacts + runner** (F4) so `repro_cmd` exists in-repo, and repoint the
   gate test at a real label (or delete the dead skip).
3. **Make the A/B test green for a reason** (F5) rather than deselecting it, or replace it
   with an analytic-equilibrium assertion that costs seconds.
4. **Register the benchmark row** (F2) with the anchor quantified (alpha vs 35 ± 5 ≤ 15 %,
   KE ≤ 0.1 J, CV ≤ 0.10) so the next round is judged by the platform, not by the author.
5. Then and only then re-open L3/L4: wire the module into
   `benchmarks/case_sphere_penetration/` and drive R/d = 3/4 (the mission's own §⑦).

Not blocking, but worth doing: state the KE metric on the sampled series (F6) and land the
reference PDF or a DOI-anchored quotation block (F7).

## 6. Checks not run / limitations

- The production run (18 810 steps, 191 s wall) was **not** re-executed; the audit reads the
  published profile and cross-checks it with the shipped estimator (exact match). Regenerating
  it is impossible from the branch anyway (F4).
- Only the reduced-config live tests were executed (196 s full file). Cluster-scale
  re-runs (R/d = 3/4) were out of scope for an audit.
- Independence limits: same host, same Python env as m15404; no second reviewer. The
  recomputation uses an independently written smoothing (`edge`-padded convolution, 2.4 %
  from the module's convention) — the spread of the three estimators is reported above
  rather than hidden.