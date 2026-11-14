# AUDIT · XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT — independent verification

**Anchor**: `XVAL-DEM-ROUND2-DEPOSITION-RIGID-LIMIT` · repo **TensorDEM** · mission 19458
**Reviewed**: `docs/XVAL_DEM_ROUND2_DEPOSITION_RIGID_LIMIT.md` (mission 18189),
`benchmarks/xval_dem_round2_deposition_rigid_limit.py`,
`src/tensordem/deposition_packing.py::stiffness_limit` /
`::extrapolation_selfconsistency`, `benchmarks/case_rcp_deposition/runs/*.json`
at base commit `b99c206f` (branch `agent/m19458`).
**Verifier artefact**: `benchmarks/xval_dem_round2_deposition_rigid_limit_audit.py`
→ `artifacts/xval_dem_round2_deposition_rigid_limit_audit/` (`results.json`,
`table_main.md`, `figures/`), regenerable with
`PYTHONPATH=src python benchmarks/xval_dem_round2_deposition_rigid_limit_audit.py`.
**Role**: verifier seat, zero new physics assumed; every number quoted below is
printed by that script from the committed run JSONs.

---

## ① What was reviewed, and why it needed verifying

The round-2 report claims that Richardson extrapolation of the soft-sphere
volume fraction over a stiffness ladder gives a hard-sphere limit
`phi_inf(mu=0) = 0.63535`, agreeing with the literature RCP `0.6366` to
**-0.20%**, and that the observed convergence order `p = 1.007` matches the
theoretical `p = 1`. Verdicts M1, M3, M4, M5 were marked ✅ and M2 ❌.

The anchor's acceptance is only `< 15%` (owner precision standard), so the
regression itself is comfortably inside tolerance. What was *not* verified is
the **evidence chain**: whether the quoted numbers are reproducible from
committed artefacts, whether the quoted order belongs to the fit that produced
the quoted limit, and whether a 0.2% agreement is even a resolvable statement
for this protocol. Those are the questions below.

## ② Independence and its limits

* The extrapolation used here is **re-implemented from the paper formula**
  (`richardson()` in the verifier script), not imported; it reproduces
  `deposition_packing.stiffness_limit` to `|Δ| = 0.0` on identical inputs
  (`library_crosscheck.agree = true`). Agreement therefore says the audit did
  not inherit a library bug, and that the library does what its docstring says —
  it does **not** validate the method against an external code.
* **Zero new simulations** in the audit tables. All φ values come from the
  committed runs; the only numbers that are not measured are the "OSS reference"
  values, which are literature numbers — no LIGGGHTS/YADE binary was executed,
  in the round-2 report or here.
* The mu = 0.1 tier is **not** cross-validated against a frictionless reference
  in this audit; see finding F8.
* Two gap-closing runs (extra seed and dt/2 at `kn = 256000`) were submitted to
  the cluster; see §⑤ for what landed and what did not.

## ③ Findings, ranked

### F1 · *Major · evidence integrity* — the declared artefact directory does not exist

`docs/XVAL_DEM_ROUND2_DEPOSITION_RIGID_LIMIT.md` declares
`artifacts/xval_dem_round2_deposition_rigid_limit/` (`results.json`,
`table_main.md`, `figures/`) under **Artifacts**. In the base commit that path
is absent (`ls artifacts/` → only `pr166/`). The producer script *does* write
those files, so the directory is simply uncommitted.

Reproduction: `ls artifacts/xval_dem_round2_deposition_rigid_limit` → *No such
file or directory*; then run the producer and the three files appear as
untracked.

Consequence: none of the report's numbers could be re-derived from a committed
artefact, and — as F2 shows — the prose does not match what the script now
emits. **Fixed in this mission**: the audit's own artefacts are committed, and
the producer's output is captured in `artifacts/…_audit/`.

### F2 · *Major · claim–evidence mismatch* — the quoted order belongs to a different fit than the quoted limit

The report's E1 table row for mu = 0 reads `phi_inf = 0.63535`,
`p = 1.007 (回退)`. Both halves of that cell cannot come from the same fit:

| quantity | value from the pipeline | source |
|---|---|---|
| φ_inf quoted (4 levels, seed-mean) | 0.63536 | `results.json → B_error_budget.fit4_limit` |
| observed order of that 4-level fit | **None** (out of bounds `0.2 < p < 4`) | `B_error_budget.fit4_order` |
| order actually used by that fit | **1.0** (assumed, fallback) | `B_error_budget.fit4_p_used` |
| observed order of the 3-level sub-ladder | 0.9852 (seed-mean) / 1.0439 (seed 1) | `A_ladder_sensitivity.variants` |
| what the report prints | **1.007 "回退"** | `REPORT_CLAIMS.p_mu0` |

The 4-level ladders have deltas `-0.004463, -0.001143, -0.001057`; the last two
give `p = log(1.081)/log(4) = 0.057` (0.013 for seed 1) — a factor 20–75 below
the assumed `p = 1`, i.e. the ladder has already flattened and the fit silently
substitutes the *assumption*. `p = 1.007` comes from `debt_audit.json`
(`stiffness["0"].order = 1.0072992`), a 3-level fit whose aggregation is itself
mixed (3-seed mean at `kn = 4000` but seed 1 only at `kn = 64000`). And the
label "(回退)" is self-contradictory: a fallback is exactly `1.0`, never
`1.007`.

Consequence: the sentence "观测阶数 p=1.007 与理论 p=1 一致 ⇒ 外推方法有效"
mutually supports two different fits. At the level where the limit is actually
taken, the order is **assumed, not measured**, so H2 as written is not
supported. (The limit itself barely depends on `p` here — see F4 — which is why
the conclusion survives but the stated evidence does not.)

### F3 · *Major · precision not resolvable* — "-0.20%" sits ~3× below the protocol noise floor

Error budget, all in φ units (`results.json → B_error_budget`):

| source | magnitude | kind |
|---|---|---|
| density definition D1..D7 within one run | **4.06e-3** | systematic |
| N sweep, `kn=4000`, N=1000…10000 (std) | **2.18e-3** | systematic |
| ladder bracket, 3-level vs 4-level limit | 1.45e-3 | extrapolation |
| limit shift when the stiffest level is admitted | 1.01e-3 | extrapolation |
| out-of-sample error (fit 3 levels, predict stiffest) | 7.63e-4 | extrapolation |
| Richardson correction applied | 3.51e-4 | correction |
| seed scatter, `kn=64000` (n=3) | 2.56e-4 | random |
| **claimed deviation φ_inf vs RCP** | **1.25e-3** | claim |

The advertised agreement is 3.2× smaller than the largest systematic and 1.7×
smaller than the finite-size scatter. Verdict `V2 = FAIL`. The defensible
statement is "consistent with RCP within ≈0.7% protocol systematic", not
"-0.20%". The search space is symmetric too: the alternative defensible
aggregations give `-0.022% … -0.250%` (fig. 1), so the quoted 0.63535 is one cell
of a family, not a converged value.

### F4 · *Major · the correction is smaller than its own error bar*

`|Richardson correction| = 3.51e-4` but the model's own held-out error is
`7.63e-4` (`V4 = FAIL`). The extrapolated limit is therefore **statistically
indistinguishable from the uncorrected stiffest measurement** (0.63571, which is
already -0.14% from RCP). Two consequences:

1. E2's "外推值与直接高刚度值差异 <0.4% ⇒ 外推方法自洽" is true but vacuous —
   a method that moves the number by 0.06% agrees with doing nothing.
2. The correct headline for the method is "the correction is real in sign, and
   resolvable only if the protocol systematic is brought below ~3e-4", i.e. a
   finite-size/definition work item, not a code-accuracy claim.

### F5 · *Major · the repo's own coverage gate passes on the tiers that don't matter*

`E_coverage` shows the controlling tier (stiffest, `kn = 256000`, mu = 0) has
**1 seed**, and `kn = 16000` has 1 seed. The report's §② further claims
"mutually `64000 (1)`", but `kn = 64000` has **3 seeds (1, 2, 3)** — 3/1/3/1, not
3/1/1/1.

Worse, `benchmarks/case_rcp_deposition/debt_audit.py` (lines 98–107) computes
`G16_seed_coverage` as `min(t["n_seeds"] for t in s["tiers"].values())`, and
`s["tiers"]` is built per-μ from the **base** stiffness tier only. So
`debt_audit.json` reports `G16 … pass: true, got: 3` while the tier that sets
the extrapolated limit is single-seed. The gate cannot see the debt it exists to
measure. (Severity major because gates are the mechanism that is supposed to
catch exactly this class of drift.)

### F6 · *Moderate · the temporal-convergence check has no power, and skips the level that matters*

E4 varies `dt/t_c` only at `kn = 4000` (the softest level). The stiffest level —
whose measurement the limit is anchored on — has **no** dt test
(`D2_temporal.untested_levels = [64000, 256000]`, `V7 = FAIL`). At `kn = 4000`
the measured difference is `2.6e-4` against a seed scatter of `1.8e-4`
(`diff/scatter = 1.4`), so the check cannot resolve any dt dependence at the
precision being discussed. "时间步减半引起的变化 <0.0003 ⇒ 生产时间步可接受" is
an over-read of a non-discriminating test.

### F7 · *Moderate · two extrapolations from one ladder land on inconsistent states (open question)*

Applying the *identical* extrapolation to the bulk coordination number
(`C_coordination_regime`):

| kn | Z (bulk) | std | seeds |
|---|---|---|---|
| 4000 | 6.3576 | 0.0125 | 3 |
| 16000 | 5.9538 | 0 | 1 |
| 64000 | 5.7669 | 0.0370 | 3 |
| 256000 | 5.4094 | 0 | 1 |
| → ∞ | **5.2903** (4-level, p falls back to 1) | | |

`Z_inf = 5.29 < Z_iso = 6` for frictionless spheres. The φ ladder extrapolates
onto the literature RCP while the Z ladder extrapolates to a **hypostatic**
(unjammed) state; at least one of them is outside its asymptotic regime. I do
**not** claim this proves the φ limit wrong — two competing explanations must
first be separated, and both are cheap for the owner of the case:

* the bed is wall-bounded (`n_wall_contacts` is logged), and isostaticity `Z ≥ 6`
  is a statement about *periodic bulk* packing — walls can carry load and hold a
  hypostatic interior;
* `coordination_bulk` may be `kn`-dependent by construction (contact = overlap >
  0, while mean overlap `δ ∝ 1/kn`), in which case the metric, not the physics,
  drifts down the ladder.

Either way, this is a second independent observable of the same state that the
round-2 report never looked at before quoting a 0.2% limit. Adjudication needs a
periodic-BC (bulk) deposition at two stiffness levels. Recorded as an open
question, `V6 = FAIL` meaning "unresolved", not "wrong".

### F8 · *Moderate · M2 compares a frictional packing to a frictionless reference*

M2 judges `phi_inf(mu=0.1) = 0.61705` against the band `[0.62, 0.66]` — a
*frictionless* RCP band — and reports ❌, then attributes the failure to
"Cundall-Strack 一阶积分误差" with no diagnostic. The repo's own friction scan at
fixed N and kn (`F_friction_reference`, N=2000, `kn=4000`) is monotone:

| μ | 0.0 | 0.1 | 0.3 | 0.5 |
|---|---|---|---|---|
| φ | 0.6375 | 0.6246 | 0.6045 | 0.5931 |

Friction *lowers* the packing fraction by construction, and the mu = 0.1 shift
at `kn = 4000` is `-2.5%` — so a frictional limit below the frictionless RCP is
the expected behaviour, and the ❌ is a mis-specified band, not a physics defect.
The stated causal attribution (first-order integration error) is unsupported:
with 1 seed at two of the three levels, `p = 0.512` is not a measured order
either (the same ±sensitivity that produced F2). Since no frictional reference
value is citable from this repo, the mu = 0.1 tier should be judged **by
ordering** (φ(mu=0.1) < φ(mu=0), same extrapolation sign) and its precision
marked NOT_SUPPORTED rather than pass/fail against `[0.62, 0.66]`.

### F9 · *Minor* — mu = 0.1 has no `kn = 256000` level

So the 4-level self-consistency test is impossible for that tier, and its limit
is structurally weaker than the mu = 0 one. The report says "3 级" correctly, but
presents both limits in one table without that asymmetry in the verdict column.

## ④ Checks not run (marked unknown / not_run, not passed)

| check | status | why |
|---|---|---|
| dt/2 at `kn = 256000` | **not_run** | submitted (`cj2148`, then `cj2150` staged) — see §⑤ |
| second seed at `kn = 256000` | **not_run** | submitted (`cj2149`, retried as `cj2150`) — see §⑤ |
| periodic-BC (bulk) deposition to settle F7 | **not_run** | outside a single-seat budget; needs a new case variant |
| executed OSS binary (LIGGGHTS/YADE) comparison | **not_run** | literature values only, in the original report and here |
| frictional RCP reference value | **NOT_SUPPORTED** | no citable value in-repo; mu = 0.1 precision cannot be judged |
| pressure/stress-based hard-sphere limit (independent protocol) | **not_run** | would adjudicate F4 independently of φ |

## ⑤ Gap-closing runs attempted in this mission (honest status)

Two cluster jobs were submitted to close F5/F6 (extra seed, dt/2 at the
controlling level). Their execution exposed two infrastructure defects worth
recording, and **neither produced usable data**:

* `cj2148` (dt/2, `ref=main`): exit `rc=2` —
  `can't open file '…/repo/benchmarks/case_rcp_deposition/run_case.py'`. The node
  checkout of `main` (`03e857d`) does not contain the case script (it is
  gate-filtered out of the pushed branch), so `ref=main` cannot run case files.
  The fetched tarball contains the repo but **no** `case_rcp_deposition/runs/`.
* `cj2149` (seed 2, `ref=main`): stage failed —
  `cannot lock ref 'refs/remotes/origin/main'`: two jobs fetching into the same
  node checkout raced on the git ref. The gate nevertheless reported
  `state = succeeded` with an empty tarball (0 KB), i.e. the scheduling layer's
  success status did not reflect the stage failure. **Do not treat
  `state=succeeded` as evidence that a command ran.**
* `cj2150` (seed 2, staged working tree, no git fetch) — final state recorded in
  the mission report; if it did not return in time the F5/F6 gaps remain open and
  are listed as such.

## ⑥ Verdicts

| id | check | result |
|---|---|---|
| V1 | φ_inf(mu=0) vs RCP within 15% | **PASS** (-0.195%; every ladder variant passes, worst -0.250%) |
| V2 | quoted precision (-0.20%) is resolvable | **FAIL** — 3.2× below dominant systematic |
| V3 | observed order equals theoretical p = 1 | **FAIL** — holds only on the 3-level sub-ladder; 4-level p is assumed |
| V4 | Richardson correction ≥ its own held-out error | **FAIL** — 3.5e-4 < 7.6e-4 |
| V5 | stiffest tier has ≥ 3 seeds | **FAIL** — 1 seed |
| V6 | coordination limit above isostatic | **FAIL** — Z_inf = 5.29 < 6 (caveats in F7) |
| V7 | temporal convergence tested at the controlling level | **FAIL** — tested only at `kn = 4000` |
| V8 | density plateaued when `settled` | **PASS** — max drift 4.4e-4 < settle_tol 5e-4 |
| V9 | mu = 0.1 judged against a valid reference | **PASS** (by ordering; precision NOT_SUPPORTED) |

**Anchor acceptance (< 15%)**: **PASS**, with ≈77× margin, on every defensible
aggregation of the data. This audit does not overturn the round-2 conclusion;
it downgrades the *precision claim* that accompanies it.

## ⑦ Error judgement against the < 3% bar (explicit)

| quantity | got | reference | error | < 3%? |
|---|---|---|---|---|
| φ_inf(mu=0), canonical (4-level, seed-mean) | 0.63536 | RCP 0.6366 | **-0.195%** | **yes** |
| φ_inf(mu=0), family range | 0.63501…0.63646 | RCP 0.6366 | -0.250%…-0.022% | yes |
| φ_inf(mu=0.1) | 0.61705 | frictionless RCP 0.6366 | -3.07% | **no** — and the reference is invalid for a frictional system (F8) |
| observed order p (family) | 0.510…1.044, 2/6 fits fall back | 1.0 | up to -49% | **no** as a single "p" claim |
| Z_inf | 5.29 | Z_iso 6.0 | -11.8% | **no** (applicability caveated, F7) |

## ⑧ Overstated claims in the round-2 report

1. "-0.20%，远小于 15%" — true vs the 15% gate, but used rhetorically as evidence
   of method accuracy; it is below the protocol noise floor (F3).
2. "观测阶数 p=1.007 与理论 p=1 一致 ✅" — different fit from the quoted limit
   (F2).
3. "E2 说明外推方法自洽" — self-consistent with doing nothing (F4).
4. "E4 无收敛漂移，生产时间步可接受" — non-discriminating test, wrong level (F6).
5. "G16_seed_coverage … PASS" in `debt_audit.json` — blind to the controlling
   tier (F5).
6. "mu=0.1 的 p=0.512 偏离归因于 Cundall-Strack 一阶积分误差" — unsupported
   causal claim; the sign of the effect is expected from friction alone (F8).

## ⑨ Disposition and remediation

**Disposition: approve with notes.** The anchor's quantitative acceptance
(< 15%) is met with large margin and is robust to every aggregation choice, so
no regression or wrong physics is being asserted. **Request changes** on the
report's evidence presentation (F1–F4, F6, F8) and **escalate** F5 as a
small, contained fix in this repo.

Remediation, in value order:

1. **Commit the artefacts** and regenerate the doc's tables from
   `artifacts/…_audit/table_main.md` (done for the audit; the round-2 doc should
   point at it).
2. **Reword the precision claim** to "consistent with RCP within ≈0.7% protocol
   systematic" and drop "-0.20%" as an accuracy statement; replace the E1 `p`
   column with the order *actually used by that fit* plus a `fallback` flag
   (the audit's table A already has this shape).
3. **Fix `G16_seed_coverage`** in `debt_audit.py` to take the minimum over all
   stiffness tiers, not just the base tier, and let it go red — a red gate here
   is the correct signal (F5).
4. **Add the dt/2 probe at the stiffest level** and quote the check's power
   (`diff / seed_scatter`), not just the raw difference (F6).
5. **Adjudicate F7** with a periodic-BC deposition at two stiffness levels; until
   then, quote Z alongside φ as the second observable of the same state.
6. **Re-specify M2**: judge mu = 0.1 by ordering against the mu = 0 ladder and
   mark its precision NOT_SUPPORTED until a frictional reference is citable (F8).