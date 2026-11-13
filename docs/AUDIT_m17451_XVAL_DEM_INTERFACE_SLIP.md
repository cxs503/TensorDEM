# Board audit — m17451 `XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE` (TensorDEM)

**Audit mission**: m17493 (board-audit, 交差核验单 — verification only, not a redo)
**Subject mission**: m17451 · repo TensorDEM · status `succeeded` · branch `agent/m17451`
**Auditor worktree**: `/root/tp-agent/work/m17493` (TensorDEM @ `f2e227f6`, merge #1445)
**Subject worktree**: `/root/tp-agent/work/m17451` (same HEAD `f2e227f6`)
**Verdict table row**: `tensor_maturity_verdicts` **id=574** (repo=TensorDEM, scope=`m17451 XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE`, level=**3**, verdict_by=`board-audit`, gate_state=pending)
**Report lake**: `tensor_reports` **id=16460** (4 embedded figures)
**Anchor judicial basis**: `SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE'` → **0 rows** (see F1)

---

## 0. Audit contract and independence limitations

* I audited the *delivered artefacts*, not the mission transcript. Evidence read: `tensor_dev_missions` row 17451 (`result`), subject worktree files, delivered `results.json`, delivered report.
* I did **not** import the delivered harness to produce any reference value. `runs/m17451_audit/independent_check.py` re-derives E\*, G\*(rigid), a, k_t, δ_s and W = 0.8 μF_n δ_s from the textbook formulae (Mindlin & Deresiewicz 1953; Johnson 1985 §4.4) with plain Python arithmetic, and applies physics necessary-condition invariants to the subject code.
* **Independence limit 1**: the reference is the *closed form of the same theory the subject implements*. There is no experimental data and no executed third-party code (LIGGGHTS / YADE / MercuryDPM are absent from this host) — the delivered "OSS comparison" is a literature/behaviour table, not a run. Verified by `which liggghts yade mercury` → none; no OSS checkout under `/root`.
* **Independence limit 2**: the 1953 original was not consulted by the auditor (no network, not on disk); the formulae used are the standard secondary-source transcription (Johnson 1985 §4.3–4.4). The delivery itself discloses the same limitation (§⑥ item 3).
* **Independence limit 3**: the auditor runs on the same host, same Python 3.11.9 / torch (float64) — bit-identical reproduction is expected and does not prove portability across architectures (loong64 cluster not used for this audit).

---

## 1. Three-piece check (取证的核验，不是复述)

| 件 | 声明位置 | Auditor finding |
|---|---|---|
| run_artifacts | `artifacts/xval_interface_slip_pseudo_steady_state/results.json` + `figures/fig1..5.png` | **exists in subject worktree, but untracked** (`git status` shows `??`; `git log --all` touching these paths = 0 commits) → F2. Rescued into this worktree; sha256 in `runs/m17451_audit/artifact_sha256.txt` |
| benchmark ref vs got | `results.json` keys `closed_form_vs_analytical`, `engine_vs_closed_form`, `parameter_sensitivity` | numbers present and internally consistent; **no platform-side case row exists** (`tensor_benchmark_cases`: 0 rows for the anchor; also 0 rows in `tensor_xval_cases`) → F1 |
| repro_cmd | doc §⑥: `PYTHONPATH=src python benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py` | **executable as documented** — auditor re-ran it: numbers bit-identical to the delivered `results.json` (all 5 numeric blocks equal, only the figure `path` strings differ). Bare `python` (no `PYTHONPATH`) died `ModuleNotFoundError` → F3 (fixed, R1) |

Reproduction transcript: `runs/m17451_audit/repro_stdout.log`; numeric-identity proof:
`runs/m17451_audit/independent_check.json` → `delivery_vs_auditor_rerun_numeric_identity` = all `true`.

---

## 2. What the delivery actually claims, and what survives audit

Case: ice-like elastic sphere on a rigid plane, E = 9 GPa, ν = 0.33, R = 0.05 m, μ = 0.3, F_n = 10 N
→ E\* = 1.0100e10 Pa, G\*(rigid) = 2.0260e9 Pa, a = 3.3361e-4 m, k_t = 5.4072e6 N/m, δ_s = 8.3222e-7 m.

Auditor re-derivation (independent of the harness), from `runs/m17451_audit/independent_check.json`:

| quantity | auditor value | delivered value | rel. error | <3 %? | <15 %? |
|---|---|---|---|---|---|
| E\*, G\*, a, k_t, δ_s (all five constants) | see `S1_constants` | identical | **0.0** | ✅ | ✅ |
| module quadrature W | 1.9973352474987245e-06 J | 1.997335091959933e-06 J | 7.79e-08 | ✅ | ✅ |
| engine (**mindlin** unloading) W | 1.9973352474987245e-06 J | 1.997319802386954e-06 J | **7.73e-06** | ✅ | ✅ |
| engine (**linear**, legacy arm) W | same | 1.5365292076558017e-06 J | **2.31e-01** | ❌ | ❌ (by design, disclosed) |
| loading law δ(F_t) vs Cattaneo–Mindlin | — | — | 4.71e-15 (max over 2001 pts) | ✅ | ✅ |
| reversal end point F(2δ_s) = F\*−2μF_n | −3.0 N | −3.0 N | 0.0 | ✅ | ✅ |
| dF/dδ at δ=0 ÷ k_t | 1 | 0.99999975 (secant, finite h) | 2.5e-07 | ✅ | ✅ |
| dF/dΔ at Δ=0 ÷ (−k_t) | 1 | 0.99999987 (secant) | 1.3e-07 | ✅ | ✅ |
| stick ratio c/a = (1−t)^{1/3} at t=0.5 | 0.793700526 | 0.793700526 | 0.0 | ✅ | ✅ |
| scaling δ_s ∝ μ¹ (auditor refit) | 1 | 1.000 | 3.3e-16 | ✅ | ✅ |
| scaling δ_s ∝ F_n^{2/3} (refit) | 0.66667 | 0.66667 | 1.1e-16 | ✅ | ✅ |
| scaling δ_s ∝ E^{−2/3} (refit) | −0.66667 | −0.66667 | 1.1e-16 | ✅ | ✅ |

Unit/integration test re-runs by the auditor:
`pytest tests/test_mindlin_partial_slip.py -q` → **11 passed in 11.47 s** (`runs/m17451_audit/test_log_mindlin.txt`);
`pytest tests/test_contact_tangential.py -q` → **17 passed in 4.08 s** (`runs/m17451_audit/test_log_tangential.txt`).

Integration callers verified by grep in the worktree (main 3D solver, not a bespoke script):
* `src/tensordem/dem3d.py:215,337,340,1193-1210,1436-1445,1472` — `DEM3DConfig.tangential_model == "mindlin_partial_slip"` branch inside the production 3D DEM solver (default remains `cundall_strack`, so the validated law is **opt-in**).
* `src/tensordem/protocols/sandpile.py:833,1109` — `mindlin_tangential_step` in the sandpile protocol.
* `tests/test_mindlin_partial_slip.py:182` exercises the dem3d path with `tangential_model="mindlin_partial_slip"`.

---

## 3. Findings (severity-ranked)

### F1 · medium · benchmark registry gap (blocks L4, not the delivery's numbers)
The audit's own self-check of the platform judge (`maturity_gate.judge(3, evidence, 'TensorDEM', 17493)`) returns `gate_level=3, reasons=[]`, i.e. every L3 key resolves against this worktree, and no key was fabricated to reach a higher level.
The anchor has **no row** in `tensor_benchmark_cases` (query in the header) and none in `tensor_xval_cases`. The board's "唯一裁判" therefore cannot rule on the anchor: there is no `metric`/`tolerance` to check `got` against. Reported here as WITHHELD rather than pass/fail. *Reproduction*: `psql -d tensorplatform -c "SELECT ... WHERE anchor_slug='XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE'"` → empty. Remediation owner: the anchor/mission side must register `metric='steady_state_cycle_area_J'`, `tolerance='15%'` (and optionally the exponent criteria) so future audits are machine-checkable.

### F2 · high · delivered evidence was never committed — the delivery does not exist in the repository
`git -C /root/tp-agent/work/m17451 status --short` lists all four deliverables as untracked (`artifacts/…`, `benchmarks/…`, `benchmarks/artifacts/…`, `docs/XVAL_DEM_INTERFACE_SLIP_PSEUDO_STEADY_STATE.md`); `git log --all -- <paths>` returns **0 commits**; branch `agent/m17451` exists but carries none of them. The gate's own `server_quality_audit` for the mission recorded `gate_valid=false, key_present=false, gate_missing_warning=true`, i.e. the acceptance was auto-approved while the evidence sat outside version control. After worktree recycling the whole delivery would have been lost, and the ECO clone (`/root/tensor-ecosystem/TensorDEM`, merge #1147) does not even contain `src/tensordem/mindlin_partial_slip.py`.
**Remediation performed by this audit**: verbatim rescue into this worktree (doc md5 `9351d0ceb3b6bdac8cdd14fdd05a0381` identical to the subject copy), harness + reproduced artefacts committed with this audit branch.

### F3 · low · `run_xval.py` sys.path off-by-one (fixed, R1)
Delivered line 33: `sys.path.insert(0, ...parents[1]/"src")` → resolves to `benchmarks/src`, so a bare `python benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py` raises `ModuleNotFoundError: No module named 'tensordem'`. The doc's `PYTHONPATH=src` workaround hides it. Auditor fix (R1) to `parents[2]`; re-run after the fix is numerically identical to the delivered `results.json` (see §1), so the fix is behaviour-preserving.

### F4 · low · summary banner contradicts the report (fixed, R2)
Delivered `main()` OR-ed the engine's **linear** arm (documented 23 % legacy deviation) into `all_pass`, so the harness always prints `Overall: SOME FAILURES` while the report's 误差判定总结 claims everything passes. Any machine reading the stdout would flag the anchor as failing. Auditor fix (R2) separates validity checks (closed form + sensitivity) from documented design deviations; stdout now `ALL PASS (closed-form + sensitivity checks)` + an explicit note for the linear arm.

### F5 · medium · report §4.1 time-step convergence table has no artefact behind it
`docs/XVAL_DEM_INTERFACE_SLIP_PSEUDO_STEADY_STATE.md` §4.1 lists Δt factors 0.5 / 1.0 / 2.0 with 7.40e-16 / 3.70e-16 / 3.70e-16. `run_xval.py` never performs a Δt sweep (no `dt_factor`/`convergence` code path; `run_engine_cycle` takes a fixed `n_steps=1200`), and `results.json` has no convergence block — the three numbers are hand-transcribed (零手抄 violation). The claim "algebraic law, Δt only changes sampling density" is defensible in itself, but the three digits are unsupported. Repair: compute the sweep in the harness or delete the table.

### F6 · medium · no paper-level report was published for m17451
`tensor_reports` has **no row** with `mission_id=17451`; `tensor_dev_missions.report_id` is NULL and `report_owed=false`, so the owed-report watchdog will not chase it either. Neighbouring missions 17440–17470 each have one report (only 17445 and 17451 are missing), so this is a mission-specific miss, not a systemic outage. The six-element report is therefore absent from the report lake even though the markdown doc exists in the worktree (and only there — F2).

### F7 · info · the validated branch is opt-in and the legacy arm exceeds the 15 % standard
`DEM3DConfig().tangential_model` defaults to `cundall_strack` (`tests/test_mindlin_partial_slip.py:236` asserts this). The "cross-validated" claim therefore applies to the opt-in `mindlin_partial_slip` branch; the default linear-unloading behaviour deviates **23.1 %** on loop area (>15 %). Disclosed by the delivery, but a downstream user reading "XVAL … PASS" must not assume the default configuration is the validated one.

### F8 · info · exact zeros in `closed_form_vs_analytical` are tautological
`gross_slip_displacement.rel_err = 0.0`, `stick_radius_ratio.max_abs_err = 0.0`: the harness compares the module against the same formula transcribed a second time in the same file, so 0.0 carries (almost) no information. The informative evidence in the delivery is `engine_vs_closed_form` (7.7e-6, a genuine 1200-step incremental integration vs the closed form) and the sensitivity exponents. This audit adds a third, independent leg (`independent_check.py`) rather than relying on the 0.0s.

---

## 4. Claimed vs supported

| delivery claim | audit disposition |
|---|---|
| "steady-state cycle area matches the M&D closed form" | **supported** for the `mindlin` arm: 7.7e-6 (engine) / 7.8e-8 (module) vs the auditor's independent value |
| "loading / reversal branches match the analytic law" | **supported**: necessary conditions (endpoints, both k_t slopes, monotonicity) and the Cattaneo–Mindlin law agree to ≤4.7e-15 |
| "scaling δ_s ∝ μ¹, F_n^{2/3}, E^{−2/3}" | **supported**: auditor refit on the delivered tables to ≤3.3e-16 |
| "cross-validated against LIGGGHTS / YADE / MercuryDPM" | **not supported as an execution claim** — it is a literature/behaviour comparison table; no third-party code was run (independence limit 1). Overstated wording; keep, but label as "documented model comparison", not cross-validation |
| "all checks pass" (§误差判定总结) | **conditionally supported**: pass for the mindlin arm; the legacy linear arm fails 15 % and the harness's own banner used to say `SOME FAILURES` (F4) |
| §4.1 convergence numbers | **unsupported / not reproducible** (F5) |

---

## 5. Level decision (L0–L5) — level = functional substance, not report length

| level | criterion (gate scheme) | verdict |
|---|---|---|
| L0 docs only | — | no: real 355-line module + 738-line engine module |
| L1 math/unit verification | `tests` | **met** — `tests/test_mindlin_partial_slip.py` (11) and `tests/test_contact_tangential.py` (17) pass on auditor re-run |
| L2 independent implementation | `impl_files` + `entry` + `run_artifacts` | **met** — `src/tensordem/mindlin_partial_slip.py` (closed form) and `src/tensordem/contact_tangential.py::mindlin_tangential_step` (stateful engine); entry `benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py` runs and writes artefacts |
| L3 system integration | + `integration_callers` | **met** — `dem3d.py` production 3D solver branch (`tangential_model="mindlin_partial_slip"`, lines 1193-1210 / 1436-1445), `protocols/sandpile.py:833,1109` |
| L4 benchmark validation | + `benchmark` + `repro_cmd` | **NOT claimed** — error analysis and reproducibility are L4-grade (verified bit-identical re-run, three independent error legs), but (a) the anchor has no registered benchmark case, so the platform judge cannot rule (F1); (b) the reference is the closed form of the same theory, with no independent external data and no executed cross-code run; (c) the delivery reached the repo only through this audit (F2). Promote to L4 after registering the case + adding either experimental/digitised-paper data or an executed OSS comparison |
| L5 research/engineering result | + `external_anchor` | **not met** — no full multi-contact simulation result (no ice-hull DEM case), no experimental anchor |

**Level = 3.** The scientific quality of the error analysis is above the level awarded; the award is held at 3 by the two machine-checkable gaps above, not by any doubt about the numbers (all of which this audit reproduced and re-derived independently).

---

## 6. Checks NOT run (explicitly unknown, not "passed")

* No LIGGGHTS / YADE / MercuryDPM execution (binaries absent) — the OSS arm of the anchor remains unverified.
* 1953 original paper not consulted (no network); reference is secondary-source closed form.
* No Δt/h convergence recomputation (the harness implements no sweep; delivering one would be new work, not verification).
* No multi-reversal loading path (the delivery documents this as a module limitation; auditor did not construct such a case).
* No full `dem3d` multi-contact simulation with `tangential_model="mindlin_partial_slip"` (only the single-contact unit test path), no cluster/loong64 run, no cross-architecture reproducibility.

---

## 7. Disposition

**Approve with notes — L3 confirmed for scope `m17451 XVAL-DEM-INTERFACE-SLIP-PSEUDO-STEADY-STATE`; L4 request changes.**
Blocking for L4 (ordered): F1 (register benchmark case with metric + tolerance) → F2 (commit the delivery; done by this audit for the subject files) → F5 (compute or delete the convergence table) → F6 (publish the six-element report). F3/F4 are fixed here (behaviour-preserving) so the harness no longer misleads a machine reader.

## 8. Artefacts produced by this audit (all inside `work/m17493`)

| path | content |
|---|---|
| `docs/AUDIT_m17451_XVAL_DEM_INTERFACE_SLIP.md` | this report |
| `docs/XVAL_DEM_INTERFACE_SLIP_PSEUDO_STEADY_STATE.md` | subject report, verbatim rescue (md5 9351d0ce…, matches subject copy) |
| `benchmarks/xval_interface_slip_pseudo_steady_state/run_xval.py` | subject harness + R1 (sys.path) + R2 (banner), numerically identical output |
| `artifacts/xval_interface_slip_pseudo_steady_state/results.json`, `figures/fig1..5.png` | auditor reproduction (sha256 in `runs/m17451_audit/artifact_sha256.txt`) |
| `runs/m17451_audit/independent_check.py`, `independent_check.json` | independent re-derivation + physics invariants |
| `runs/m17451_audit/repro_stdout.log`, `test_log_mindlin.txt`, `test_log_tangential.txt` | reproduction and test transcripts |
| `runs/m17451_audit/fig_audit_m17451_errors.png`, `make_fig.py` | error budget vs 3 %/15 % gates + cycle schematic |