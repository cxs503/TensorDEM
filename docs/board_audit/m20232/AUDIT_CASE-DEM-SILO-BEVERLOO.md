# Board audit — m20055 / CASE-DEM-SILO-BEVERLOO (TensorDEM)

**Mission audited**: m20055 `case-dem-silo-beverloo`, status `succeeded` (finished 2026-11-14 10:26Z,
`fail_stage = wall_exhausted+forced_finish`), worktree `/root/tp-agent/work/m20055`, branch
`agent/m20055`, base `5101c393` (main-line merge of #1516).
**Audit mission**: m20232 (this worktree, `/root/tp-agent/work/m20232`, branch `agent/m20232`).
**Type**: cross-check / 交差核验 (not a redo). **Verdict**: **L2**, benchmark anchor **FAIL**.

---

## 0. What was reviewed (identities)

All mission deliverables are **uncommitted** in the mission worktree
(`git status`: `M src/tensordem/silo_discharge.py` + 7 untracked paths); `tensor_dev_missions.head_commit`
for m20055 is NULL, so no commit id binds anything. Everything was therefore reviewed as byte copies with
sha256 recorded in `docs/board_audit/m20232/refs/`:

| reviewed file (mission) | byte copy | sha256 |
|---|---|---|
| `runs/m20055_beverloo/D4.json` (the only run artefact) | `refs/m20055_D4.json` | `ad594807536c5e28…` |
| `runs/m20055_beverloo/summary.json` | `refs/m20055_summary.json` | `bb0e944ce22b27c3…` |
| `benchmarks/m20055_beverloo/ref_beverloo_1961.json` | `refs/m20055_ref_beverloo_1961.json` | `5c16cf5630c00742…` |
| `benchmarks/m20055_beverloo/run_case.py` | `refs/m20055_run_case.py` | `a883cb4041081cd5…` |
| `docs/CASE-DEM-SILO-BEVERLOO_m20055.md` | `refs/m20055_report.md` | `cf1b9b350cf199ea…` |
| `prereg/m20055_beverloo.md` | `refs/m20055_prereg.md` | `9778d68498d2bc1d…` |
| `tests/test_silo_discharge_steady.py` | `refs/m20055_test_silo_discharge_steady.py` | `f7ecf83f702ab831…` |

Independent re-derivation: `scripts/audit/audit_m20055_beverloo.py` →
`docs/board_audit/m20232/audit_m20055_beverloo.json` (figures:
`scripts/audit/fig_m20055_beverloo.py` → `fig_m20055_rate_vs_time.png`, `fig_m20055_estimators.png`).

## 1. Scope and checks performed

| # | check | result |
|---|---|---|
| C1 | Does the filed `W` really come from the recorded discharge-event stream? | **yes** — re-implemented the quantile estimator from the raw `(t, N)` record: 1.8175329 kg/s vs filed 1.8175329 kg/s, abs diff **2.2e-16** |
| C2 | Is `W_ref` arithmetic correct on the *measured* density basis? | **yes** — 0.58·1298.485·√9.81·0.052^2.5 = **1.454480 kg/s**, reproduces filed `W_ref` exactly; no density inflation (see §5) |
| C3 | Is the verdict robust to the rate estimator? | **yes for FAIL** — 8 estimators, 7 independent of the filed one: +19.6 … +28.8 % (§3, M3); only the legacy fixed-window mean including the plugged-phase zero bin gives −5.5 % |
| C4 | Are the predeclared thresholds met? | **no** — `pass_10pct = False`, `pass_15pct = False` in the mission's own artefact |
| C5 | Do all 6 predeclared cases have artefacts? | **no** — 1 of 6 (§2, B2) |
| C6 | Is there a registered referee row for the anchor? | **no** — 0 rows (§4, M1) |
| C7 | Are the tests real invariants or artifact sniffing? | **real** — 22 passed in 13.3 s; assertions recompute physics, none reads a stored JSON (§5) |
| C8 | Does the delivered report have all six sections? | **present but hollow** — ①–⑥ headings exist, ③/④ carry one row, ⑤(a)/⑤(b) are header-only empty tables, `power_law = {}` |
| C9 | Is the delivered `repro_cmd` executable? | `repro.sh` **is** valid bash; the report's own quoting of it (`PYTHONPATH=src python …/repro.sh`) is **not**. Underlying path re-verified end-to-end by re-running D4 on the compute gate (cj2578, §6) |
| C10 | Rounding/units/consistency of the report table vs the JSON | consistent (W, W_ref, error, C_rec, CV, φ all match the artefact) |

## 2. Findings — blocking

### B1 (blocking, correctness of the claim) — the anchor case FAILS, by every honest estimator
The one filed case **D4 (D/d = 4, μ_p = 0.30)** gives

| quantity | value |
|---|---|
| `W_ref` (Beverloo 1961, Eq. 4 p.265, C=0.58, k=1.4, ρ_b measured = 1298.5 kg/m³) | 1.4545 kg/s |
| `W_dem` (mass quantile 0.2–0.8 of the discharge stream) | 1.8175 kg/s |
| relative error | **+25.0 %** |
| recovered C | **0.7248** (+25 % over 0.58; outside the spherical band 0.55–0.60) |
| steadiness CV (5 sub-windows) | 0.060 |

+25.0 % is outside the case's own predeclared 10 % gate **and** outside the platform's 15 %
acceptability bound. It is not an estimator artefact: see §3. **L4 is refused on this alone.**

### B2 (blocking, completeness) — 5 of the 6 predeclared cases were never produced
`runs/m20055_beverloo/` holds `D4.json`, `summary.json`, `report_tables.md` — nothing else. The artefact's
own `command` field records `… --shard 1 --nshards 6`, and the mission result states that cluster job
cj2460 landed **only shard 1** (killed) and that cj2555 (the remaining 5 cases) was **never fetched**
("在 n315 运行中未取回"). Consequences, all visible in the delivered report:

* `docs/CASE-DEM-SILO-BEVERLOO_m20055.md:43-51` — §⑤(a) time-step convergence and §⑤(b) friction
  sensitivity are **header-only tables with no rows** (only §⑤(c), which needs no new run, has rows);
* `power_law = {}` in `summary.json` — the 5/2 exponent test needs ≥ 3 diameters
  (`analyse.py:power_law_slope`), with one diameter it silently returns empty;
* the prereg's stated case matrix (D4/D5/D6/D5_mu15/D5_mu45/D5_dthalf) is therefore ⅙ delivered.

### B3 (blocking, unsupported claim) — the headline verdict names D5, which has no artefact
`docs/CASE-DEM-SILO-BEVERLOO_m20055.md:3` states "**判定: 主算例 D/d = 5 FAIL**" and `:86` "总判定：…
主算例 D5 未通过". No D5 artefact exists on disk or in the DB. Worse, the artefact that *does* exist is
D/d = 4, which sits exactly at the lower edge of the reference's own declared validity domain
(`ref_beverloo_1961.json: validation_domain.D_over_d_min = 4.0`, applicability "D/d ≳ 4"). The audited
evidence therefore supports only: **"the D/d = 4 point, at the edge of the correlation's validity, is
+25 % high"**; a D5 verdict is *unestablished*, not merely unfavourable.

## 3. Findings — the estimator question (what the +25 % is, and what it is not)

Same raw record, eight ways of turning it into a rate
(`audit_m20055_beverloo.json["estimators"]`, figure `fig_m20055_estimators.png`):

| estimator | W [kg/s] | rel % | C_rec | <10 % | <15 % |
|---|---|---|---|---|---|
| quantile 0.20–0.80 (predeclared) | 1.8175 | +25.0 | 0.725 | ✗ | ✗ |
| quantile 0.10–0.60 | 1.7402 | +19.6 | 0.694 | ✗ | ✗ |
| quantile 0.35–0.65 | 1.8162 | +24.9 | 0.724 | ✗ | ✗ |
| quantile 0.05–0.95 | 1.8026 | +23.9 | 0.719 | ✗ | ✗ |
| quantile 0.30–0.90 | 1.8445 | +26.8 | 0.736 | ✗ | ✗ |
| **filed fixed-window mean incl. plug-phase zero bin** | **1.3739** | **−5.5** | 0.548 | ✓ | ✓ |
| filed fixed-window mean, zero bins dropped | 1.7174 | +18.1 | 0.685 | ✗ | ✗ |
| whole-stream mean over discharge | 1.8116 | +24.6 | 0.722 | ✗ | ✗ |
| longest ±5 % plateau (0.75 s at t ≈ 1.79–2.29 s) | 1.8728 | +28.8 | 0.747 | ✗ | ✗ |

Two things follow.

1. **The FAIL is robust**: every estimator that respects "steady discharge only" lands at +19.6 … +28.8 %.
2. **The single passing number is an artefact of the defect m6296 already diagnosed**: the legacy
   fixed-window mean averages across the period in which the orifice was still plugged. Evidence:
   `flow_rate_windows_kg_s[0] = 0.0` while the first discharge event is at
   `t = 0.665 s ≥ settle_time = 0.60 s`. Any future reading of this case must not quote −5.5 %.

Corollary for the record: the mission's protocol change (measured bed ρ_b at the instant of opening,
plug retained during settling, start-up/tail rejection) is a **real improvement** over the filed
`beverloo_discharge.py` lineage — see §5 — even though the resulting verdict is FAIL.

## 4. Findings — medium

### M1 — no machine referee exists for this anchor
```sql
SELECT case_name, metric, tolerance FROM tensor_benchmark_cases
WHERE anchor_slug = 'CASE-DEM-SILO-BEVERLOO';   -- 0 rows
```
`tensor_benchmark_cases` holds 127 rows; **none** matches any `%SILO%`/`%BEVERLOO%` anchor, so this
anchor could not have been judged by a tolerance row, and the mission record's own
`server_quality_audit = {gate_valid: false, key_present: false, gate_missing_warning: true}` agrees. The
"benchmark" evidence in the maturity ledger must therefore be case-side only
(`ref_beverloo_1961.json` vs `D4.json`, tolerance 10 % as predeclared).

### M2 — reference coefficients are cited, not re-digitised
`ref_beverloo_1961.json: provenance.verification_status = "cited, NOT re-digitized"`. Honestly declared;
still, the pass band rests on C = 0.58 (band 0.55–0.60 is ±4 % on W) and k = 1.4 (1.2–1.6 moves D_eff).

### M3 — the predeclared steadiness gate (CV < 0.15) is too weak at this system size; "steady" is not established
`steady_rate_cv = 0.060` looks healthy because the five quantile sub-windows are each ≈ 0.53 s and
contiguous, i.e. they integrate the same slow drift. The 0.25 s binned rate
(`fig_m20055_rate_vs_time.png`, 18 bins) swings **1.20 → 2.12 kg/s**, CV = **0.126**, and the longest
interval over which the binned rate stays inside ±5 % of its mean is only **0.75 s**. The mission's own
band-sensitivity row shows the same fragility: the tight 0.35–0.65 band reports CV = 0.164, so the
predeclared CV < 0.15 gate is itself band-dependent (passes on 0.20–0.80, fails on 0.35–0.65). So:
* the +25 % verdict is **not** noise: the 0.363 kg/s gap is **7.5 σ** of the sub-window standard error
  (0.049 kg/s) and is stable across all five quantile bands;
* but a claim of *agreement* below ~10 % would be **below the resolution** of this case
  (n_discharged = 766 over 4.43 s). Any remediation must add particles/hour of statistics, not only
  more diameters.

### M4 — reproducibility identity is incomplete
Artefact `git_sha = "unknown"` (dirty worktree), all 8 paths uncommitted, `head_commit` NULL,
`run_case.py` does not record `nshards` in the artefact, and `host = node315` is recorded but the queued
job id (cj2460) is not. A replay can check the physics (C9/§6) but cannot bind the result to a commit.

### M5 — the report's own repro command is not executable as written
`docs/CASE-DEM-SILO-BEVERLOO_m20055.md:77` gives
`PYTHONPATH=src python benchmarks/m20055_beverloo/repro.sh` — that invokes a bash script through the
Python interpreter (fails). The correct form is `bash benchmarks/m20055_beverloo/repro.sh`, and the file
itself is sound (6 cases + analyse + render + pytest).

## 5. Findings — what is genuinely good (credit, and evidence it is real)

* **REAL DEM, no formula shortcut**: 1525 particles, 70 945 steps, dt = 7.19e-5 s
  (dt/T_contact = 0.0354), 763 discharge events, 766 particles discharged.
* **The change is small and backward compatible**: `git diff --stat src/tensordem/silo_discharge.py` =
  +156/−26 (steady_rate + the settled/plug options + a plate-contact rewrite with rim circles and a
  closed-plate gate); the three pre-existing `tests/test_silo_discharge_beverloo.py` tests still pass
  against the new plate geometry.
* **Plug integrity is measured, not asserted**: first discharge at t = 0.665 s > settle 0.60 s; the
  test `test_no_discharge_before_the_orifice_opens` pins it.
* **The density-basis defect of the previous audit is fixed here**: `W_ref` uses the **measured bed**
  ρ_b = 1298.485 kg/m³ (φ = 0.519) taken at the instant of opening, not the hard-coded φ = 0.64 that
  inflated the reference in `src/tensordem/beverloo_discharge.py:354` (finding of the m15418 board audit).
  My recompute reproduces W_ref = 1.454480 kg/s from the ref JSON + that measured ρ_b: no inflation factor.
* **Tests are invariants, not snapshot readers** (22 passed in 13.33 s, `tests/test_silo_discharge_steady.py`
  19 + pre-existing `tests/test_silo_discharge_beverloo.py` 3): exact recovery of a constant-rate stream
  (rel 1e-12) and of a plateau between a slow start and a slow tail (2 % of the plateau rate, correct
  interval); degenerate safety (0.0 / NaN); the plate gate over the aperture centre (0.999 r, closed vs
  open); settled packing born without overlaps (min pair distance > 2r) while legacy HCP packing is unchanged
  (min pair distance 2r·1.02 to rel. 2e-3); invalid packing mode rejected; no discharge before opening; legacy open-from-step-0
  protocol unchanged. **No test asserts a stored JSON value**, so the suite is green *because the physics
  holds*, not because a number was copied.

## 6. Reproduction performed by this audit

* `PYTHONPATH=src python -m pytest tests/test_silo_discharge_steady.py tests/test_silo_discharge_beverloo.py -q`
  → **22 passed, 13.33 s** (mission worktree, 1 core).
* `scripts/audit/audit_m20055_beverloo.py` → independent recompute of the filed W to 2.2e-16 and the
  estimator table/figures above.
* **cj2578** (compute gate, 1 core, node n330, `stage_dir=/root/tp-agent/work/m20055`,
  `run_case.py --case D4 --out-dir runs/m20232_repro`) — **succeeded** in 734.6 s wall. The fresh artefact
  (`refs/cj2578_D4.json`, sha256 `81c14f3baa0a54c4…`) is **byte-equal in every number** to the filed one:
  `W_got = 1.8175328703858848` kg/s, `W_ref = 1.454479691076926` kg/s, `rel_err = +0.2496103`, ρ_b =
  1298.4846 kg/m³, `n_discharged = 766`, CV = 0.0598263 — differences exactly 0.0. So the DEM result is
  **deterministic and reproducible** (same seed 20055, same case block); the only difference is the wall
  clock (749.0 s vs 734.6 s). **This reproduces the FAIL, it does not remove it.**

## 7. Checks NOT run (unknown, not "passed")

* D5 / D6 / D5_mu15 / D5_mu45 / D5_dthalf: no artefacts exist and re-running the matrix is a remediation
  order, not an audit step — the time-step-convergence and friction-sensitivity claims are **not run**,
  i.e. **unknown**.
* Beverloo (1961) original table: no paper copy in the worktree and no outbound network → the C/k pair is
  **not re-digitised** (declared by the mission too).
* Platform/API exposure of `tensordem.silo_discharge`: not part of this case; no API/CLI/export caller was
  found (`grep` over the repo shows only the mission's own scripts), so L3 integration is absent by
  inspection rather than by absence of a test.
* The mission result's local pilot ("553 particles, D/d = 5, μ = 0.30 → +3.0 %, CV 0.166") could not be
  reproduced: no artefact, no recorded parameters beyond that sentence. The prereg explicitly excludes
  pilots from the verdict table, which is correct — but note that its CV 0.166 would have failed the
  mission's own CV < 0.15 stability gate had it been a matrix case, so it cannot be used as a
  "the law is actually fine" counter-claim.

## 8. Overstated claims (to be corrected, in the report and the ledger)

1. "判定: 主算例 D/d = 5 FAIL" — no D5 run exists; the audited evidence is the D4 point only (B3).
2. "报告六段齐全" is formally true but §⑤(a)/(b) are empty and `power_law = {}`: the delivered
   sensitivity/convergence study is **absent**, not "thin" (B2).
3. `summary.json: reference.provenance` is honest, but `silo_discharge.py`'s module docstring still
   advertises "C ≈ 0.58, k ≈ 1.4 for glass beads" as if verified; this mission's own result (+25 %,
   C_rec = 0.725) refutes that for the shipped protocol — the docstring needs the honest caveat.
4. The `flow_law` figure draws a "Beverloo exponent 5/2 (anchored)" curve through a **single** point;
   it is decorative and must not be read as exponent evidence.

## 9. Level determination

| level | test | verdict |
|---|---|---|
| L0 documents | — | passed (but see §8) |
| L1 math/unit verification | real invariant tests on the estimator, packing, plug | **passed** (22 tests, §5) |
| L2 independent functional implementation | real code: geometry → DEM contacts (granular3d_core) → discharge event stream → W → error; entry `run_case.py:165-210`; replayed to 1e-16 | **PASSED → level = 2** |
| L3 system integration | no API/CLI/export/solver caller; additions are called only by the mission's own benchmark (and the pre-existing `silo_discharge` kernel is what is reused) | refused |
| L4 benchmark validation | anchor FAIL at +25 % (> 15 % unacceptable, > 3 % target), 1/6 cases, no registered referee row, no power-law test | refused |
| L5 research/engineering result | — | not reachable from L4 |

**Verdict = L2.** The level is a statement of function, not of effort: the estimator + packing + plug
protocol are real, tested, and reproduce a genuine DEM number to machine precision. What is missing for
L4 is not polish but *the measurement itself*: a 25 % disagreement, measured on 1 of 6 predeclared cases,
at the edge of the correlation's validity, against a referee row that does not exist in the DB.

Gate expectation: JUDGE_SCHEME confirms an L2 row on real evidence pointers **while the anchor stays
FAIL** (as it did for m15418). The FAIL, the missing 5 cases and the empty §⑤ tables need a separate
remediation order, not a level change.

## 10. Disposition

**Approve with notes at L2** — the delivered code is real and the honest-FAIL is well founded; the
mission's own report already says FAIL, so nothing is being hidden. Required remediation for any L4
attempt (for the ledger):

1. run the 5 missing predeclared cases (D5 primary, D6, μ-sweep, dt/2) **and** retrieve them from the
   cluster; without D5/D6 the 5/2 exponent and the primary verdict are unestablished;
2. interpret the +25 % on the correct basis: is it a contact-law/rolling-friction artefact
   (μ_r = 0.02 fixed, unswept) or a finite-size/orifice-plate artefact (plate thickness 0.5 d, R_silo/D =
   3.25)? The μ and dt arms were predeclared exactly for this and are the missing measurements;
3. register the referee row (`tensor_benchmark_cases`, anchor `CASE-DEM-SILO-BEVERLOO`, metric
   `rel_err(W)`, tolerance `0.10`) so the anchor can be machine-judged at all;
4. strengthen the steadiness gate (e.g. binned-rate CV, or require a ≥ 2 s plateau within ±10 %) and
   report the resolution limit (n_discharged) alongside W, because CV < 0.15 passed while the binned rate
   swings ±12 %;
5. fix `repro.sh` quoting in the report, record `git_sha`, `nshards` and the job id in every artefact, and
   add the honest caveat to `silo_discharge.py`'s docstring.