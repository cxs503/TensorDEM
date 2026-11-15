# CASE-DEM-PLANE-SHEAR-MUIDI — Round 19: three latent defects in the round-18 higher-N sweep

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池·平面剪切 mu(I) 流变律标定), status `open`
**Repo:** TensorDEM · mission m21228 · branch `agent/m21228`
**Ref:** da Cruz et al. 2005 PRE 72:021309; GDR MiDi 2004 Eur. Phys. J. E 14:341
**Acceptance (ledger):** mu(I) and phi(I) errors < 10 % over I = 1e-3 ... 0.3

## ① Case & context

Dense plane shear of frictional spheres held at constant normal stress P by a
servo platen — the standard device for the local rheology
`mu = tau/P`, `I = gamma_dot d / sqrt(P/rho)`, with
`mu(I) = mu_s + (mu_2 - mu_s)/(I_0/I + 1)` and a *decreasing* `phi(I)`
(dilatancy).

Round 18 (m21111) opened the higher-N verification debt (round-14
recommendation R4, "N > 1000") and shipped
`benchmarks/plane_shear_mu_i/high_n_sweep.py` plus a report whose results
table was still ``(pending)``.  This round audits that script **before** its
sweep is assembled, because the assembly is the only place where the ledger
verdict is computed.

## ② What was found (independent re-read of the shipped code)

| id | defect | effect on the verdict |
|---|---|---|
| D-A | the script re-declared its own `(mu_s, mu_2, I_0) = (0.36, 0.84, 0.30)` and its own `ref_mu`/`ref_phi`, although `reference_law.py` is documented as the single source of truth for this benchmark ("do not copy this triple into another module") | silent drift: a future correction of the disputed triple in `reference_law`/`reference_provenance` would not reach the sweep |
| D-B | that local `ref_phi` was `0.58 + 0.06/(I_0/I + 1)` — **increasing** with I, i.e. compaction instead of dilatancy; this is the round-2 defect D2 reintroduced | every phi residual in the assembly would carry the wrong sign and be ~9 % low at the low-I end (see ③) |
| D-C | `TOL = 0.10` was compared against residuals already expressed in **percent** (`err_mu_pct`) | the ledger band "error < 10 %" was enforced as "error < 0.1 %" → any assembly would have printed a **spurious FAIL**, at every point |

A fourth, methodological issue (D-D): the abscissa was
`res["inertial_number"]`, i.e. the **platen-rate** I.  Rounds 9/10 established
that the physical abscissa is I built from the *measured* velocity-profile
shear rate (`inertial_number_measured_rate_final`): wall slip makes the bed
shear slower than the platen drives it, so the platen-rate abscissa labels a
material state at a higher I than it experienced and biases mu low at high I —
which is exactly the shape of the 13-33 % high-I deficit reported since round
12.  The sweep now reports both abscissae and takes the verdict on the
measured one.

## ③ Numbers (injected from `runs/high_n_reference_audit/result.json`)

`phi(I)` under the two definitions, same (I, triple) inputs:

| I | round-18 local ref_phi | reference_law ref_phi | diff | rel. err of the old one |
|---|---|---|---|---|
| 0.001 | 0.5802 | 0.6398 | -0.0596 | -9.3 |
| 0.01 | 0.5819 | 0.6381 | -0.0561 | -8.8 |
| 0.03 | 0.5855 | 0.6345 | -0.0491 | -7.7 |
| 0.1 | 0.5950 | 0.6250 | -0.0300 | -4.8 |
| 0.3 | 0.6100 | 0.6100 | +0.0000 | +0.0 |

Band check, old code: `TOL = 0.1` against a percent residual ⇒
band = 0.1 %; correct band = 10 %.

![defects](runs/high_n_reference_audit/fig_reference_defects_r19.png)

## ④ Method of the fix

* `high_n_sweep.py` now **imports** `ref_mu`, `ref_phi`, `MU_S`, `MU_2`,
  `I_0`, `PHI_MAX`, `PHI_MIN`, `MU_REF_STATUS`, `PHI_REF_STATUS` from
  `reference_law`; the local definitions are deleted (D-A, D-B).
* `TOL_PCT = 10.0` replaces `TOL = 0.10`, and every comparison names the unit
  (D-C); the verdict records `tolerance_pct`.
* the steady-state summariser is factored out as `summarise(res, I_target)` —
  pure, no DEM run — reporting `I_imposed_rate`, `I_measured_rate`,
  `I_abscissa`, both residuals, and the three phi estimators
  (`phi_bed_steady`, `phi_extent_final`, `phi_container_final`) instead of one
  number, so the estimator spread of rounds 13-15 stays visible (D-D).
* `assemble(runs_dir=None)` takes a directory, so the verdict logic is
  testable without touching the archived runs.

## ⑤ Verification

`benchmarks/plane_shear_mu_i/test_high_n_reference.py` — **11 tests, all
green** (`PYTHONPATH=src pytest … test_high_n_reference.py`), each defect
pinned by a *counter-example* as well as by the fix:

* D-A: `ref_mu is reference_law.ref_mu`; an **AST** guard rejects any
  re-introduced module-level copy of the triple (prose may quote it, code may
  not);
* D-B: `ref_phi` strictly decreasing, limits `phi_max`/`phi_min`, plus
  `test_regression_round18_phi_formula_was_inverted` which asserts the *old*
  formula increases and is 5.6 pp low at I = 0.01;
* D-C: `test_tolerance_band_is_in_percent_not_fraction` — a 9 % residual now
  passes, an 11 % residual still fails (with `TOL = 0.10` both would have
  failed);
* D-D: the measured-rate abscissa is preferred, the imposed-rate fallback is
  tested, and the two abscissae must differ materially in the residual;
* `assemble` PASS/FAIL paths on synthetic rows, and `P3_honest_verdict`.

Full benchmark test file as well: `test_plane_shear_mu_i.py` +
`test_high_n_reference.py` = **16 passed, 2 skipped**.

## ⑥ Error judgement (honest)

The key quantity this round can judge is the **reference-side error of the
shipped script**, not the physics:

| quantity | vs baseline | error | < 3 %? |
|---|---|---|---|
| `ref_phi` at I = 0.01 (old code) | `reference_law.ref_phi` = 0.6381 | **-8.8 %** | **NO** |
| `ref_phi` at I = 0.001 (old code) | 0.6398 | **-9.3 %** | **NO** |
| `ref_phi` at I = 0.3 (old code) | 0.6100 | 0.0 % | yes (both branches coincide at the `phi_min` end) |
| tolerance band (old code) | ledger 10 % | 0.1 % vs 10 %, **100× too tight** | **NO** |

So the round-18 script would have produced a wrong-signed, ~9 %-low phi column
and a guaranteed spurious mu FAIL: **the ledger verdict of an assembly from it
would have been an artefact of the harness, not a statement about the
material.**  No archived number is affected — round 18 never assembled a row
(the results table in `R18.md` is still `(pending)`), which is also why the
defects were latent rather than published.

**The mu(I)/phi(I) curve verdict itself is NOT delivered this round.**  The
corrected sweep is submitted as cluster job `cj3277` (4 shards, stage_dir =
this worktree, `high_n_sweep.py --shard i/4`) and was still `queued` at
hand-off; `runs/high_n/` is empty, so no simulation number is claimed here.
`analyse_r19.py` (tables + two figures from `result.json`, zero hand-copied
numbers) is committed and dry-run-tested on synthetic rows, so the assembly
is one command once the shards land.

## ⑦ Caveats / next steps

1. Run `high_n_sweep.py --assemble` then `analyse_r19.py` when `cj3277`
   returns, and write `R20` with the P1/P2 verdict on the **measured-rate**
   abscissa.
2. If the deficit survives at N ≈ 1300 on the measured-rate abscissa, D-D is
   refuted as the explanation and the remaining candidates are the wall-slip /
   servo-compaction pair (round-14 root causes 1-2); if it collapses, then a
   large part of the 13-33 % "high-I mismatch" of rounds 12-18 was an abscissa
   artefact.
3. The reference triple is still QUOTED, NOT ARCHIVED (`MU_REF_STATUS`), and
   the phi magnitudes are still PROVENANCE-UNVERIFIED: the phi acceptance
   remains WITHHELD on provenance grounds independent of this round.
4. This round changes no simulator code and no archived data — only the
   harness that would have judged them.
