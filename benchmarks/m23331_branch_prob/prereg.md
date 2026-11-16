# m23331 — branch-probability ensemble for the frozen Beverloo case (pre-registration)

Anchor: `CASE-DEM-SILO-BEVERLOO` (ledger status checked before any work: `open`).
Kernel: `tensordem.silo_flow` (kernel B). Reference: Beverloo, Leniger & van de
Velde (1961), *Chem. Eng. Sci.* **15**, 260–269, Eq. (4),
`W = C·ρ_b·√g·(D − k·d)^(5/2)`, `C = 0.58`, `k = 1.4` (literature, never refitted).

This file is written **before any discharge record of this mission exists**.
Nothing below is changed after data are looked at. The honest-FAIL clause of the
campaign applies: a record that contradicts the pre-registered verdict rule is
reported as such, never replaced by a re-selected seed or a re-tuned protocol.

## 1. The open question this mission closes

`docs/CASE-DEM-SILO-BEVERLOO_m23261_ulp_dt_divergence.md` established:

* the two archived records of the *same* frozen case (`W` = 0.0307498 and
  0.0033577 kg/s, factor **9.2**) differ in one last-bit value of `dt` —
  `0.05·T_c` (`Material.dt`) vs `T_c/20` (`run_level.py`), 1 ULP apart;
* a 1-ULP `dt` shift decorrelates a paired copy of the same bed exponentially
  (λ ≈ 1.3e2 s⁻¹), so the two branches cannot be the same trajectory.

What that mission explicitly **could not** decide (its §5) is *whether the
branch flip is a property of the `dt` value or a property of the bed
realization*: it had exactly one record per `dt` and one bed. Without that
attribution, no `W` reported for this anchor (any round, any seed) is a
measurement — it is one sample from an unknown mixture.

**Question Q**: at fixed `dt`, does the frozen case flow for every packing
realization, or does the FLOWING/ARRESTED split appear across seeds too?

## 2. Design — a 2 × N factorial, one factor changed at a time

| factor | levels | how it is set |
|---|---|---|
| `dt` spelling | **F** = `0.05·T_c` (legacy: what the archived m22885 driver used), **Q** = `T_c/20` (`Material.dt` after m23261, i.e. `dt_for_steps`) | bit-exact, `struct`-checked per record |
| packing seed | `1 … N`, `N = 6` | `build_bed(..., seed=s)` — the only other degree of freedom |

**Correction made before any record existed** (this mission has no DEM data yet
at the time of this edit): `kernel.Material.dt` was changed by m23261 to delegate
to `dt_for_steps`, so `Material.dt` now evaluates the **`T_c/20`** spelling, not
`0.05·T_c`. The arms are therefore named after the *bit-patterns*, not after a
driver: `F` = `0.05·T_c` (the legacy spelling of the archived m22885 record),
`Q` = `T_c/20` (the canonical spelling today). Nothing else changes.

Consequence for the reproduction check (P0 below): arm `Q` at seed 1 is the
spelling of the **arrested** archive (`W = 0.0033577 kg/s`) and arm `F` at seed 1
of the **flowing** archive (`W = 0.0307498 kg/s`).

Everything else is the campaign's frozen primary case, untouched:
`d = 3 mm`, `D = 15 mm` (`D/d = 5`), `H = 36 mm`, `μ = 0.5`, `μ_rolling = 0.05`,
smooth wall (`μ_w = 0`), `k_n = 5e3 N/m`, settle 0.06 s, discard 0.12 s,
8 × 0.05 s windows, `W` = mean of the last 4 windows.

`2 × 6 = 12` records, one shard each.

Why seeds and not a broader parameter sweep: m22628 owns the orifice scan, m22885
owns the powered/seed ensemble at one spelling, m23114 owns the `dt`/`k_n`
ladder. This mission owns exactly the **cross of the two factors that m23261
showed to interact** and nothing else.

Domain proof (asserted by a test, not by inspection): every record satisfies
`D/d ≥ 4`, `silo_D/D ≥ 2.5`, `H/D ≥ 1.5`, and the two spellings differ by
exactly 1 ULP.

## 3. Classification and verdict rule (fixed now)

Classification per record: the campaign's own `beverloo.discharge_status`
(thresholds declared in `src/tensordem/silo_flow/beverloo.py`, never tuned here):
`ARRESTED` if any of {no particle exited, ≥ 50 % empty windows, rate < 20 % of
the Beverloo reference, last-half/first-half rate < 0.20}, else `FLOWING`.

`p_F`, `p_Q` = fraction `FLOWING` in each arm (n = 6 each), reported with a
95 % Wilson score interval. Two-sided Fisher exact test on the 2 × 2 table
(implemented exactly in `analyse.py`, hypergeometric, no scipy dependency).

```text
B1 SPELLING-DOMINATED : one arm is pure (p = 1 or p = 0) and the other is pure
                        at the opposite value        -> the branch is a property
                        of the dt VALUE, not of the bed
B2 SEED-DOMINATED     : both arms have 0 < p < 1, OR both arms are pure and
                        equal, with Fisher p >= 0.05 -> the branch is a property
                        of the packing realization; dt is irrelevant here
B3 INCONCLUSIVE       : anything else (e.g. 5/6 vs 1/6, Fisher p < 0.05 but
                        neither arm pure)            -> needs more seeds
```

Conditional accuracy is quoted on the **canonical** arm `Q` (the spelling a
driver gets today from `Material.dt`) when it has `n >= 2` FLOWING records, and
on arm `F` otherwise; both arms always get their own row. This is the ledger's
< 10 % tolerance, applied for the first time to a **mean with a CI** instead of
a single record:

```text
W_F      = mean of the FLOWING records of arm F
CI95     = mean ± t(0.975, n-1) · s/√n   over those records
verdict  = PASS if |W_F/W_1961 − 1| < 10 % and the CI half-width < 10 %
           FAIL otherwise
```

If an arm has `n < 2` FLOWING records no CI is reported (undefined) and the
accuracy verdict for that arm is `NOT_DECIDABLE` — the collapse of the mixture
to a single branch is itself the finding, and a `W` computed from one record of
a mixture is not quoted as agreement with the literature.

## 4. Predictions, stated as falsifiable claims

* **P0** (reproduction, `< 1e-12` relative): arm `F` seed 1 reproduces
  `W = 0.030749793997350797 kg/s` (m22885 `fill36_seed1.json`, git `db2bdaf`) and
  arm `Q` seed 1 reproduces `W = 0.0033577361261474805 kg/s` (m23114
  `dt_nom20_s1.json`, git `1fb2a46`). If P0 fails the kernel changed and every
  other statement of this mission is void.

* **P1** (spelling-dominated): `p_F = 1`, `p_Q = 0`. Arch formed in every Q
  record, in none of the F records → the 9.2 × is a numerics-on-a-knife-edge
  artifact and all past single-record verdicts are void, but `W` (F) is
  measurable.
* **P2** (seed-dominated): `0 < p_F, p_Q < 1` and Fisher `p ≥ 0.05` → `D/d = 5`
  is a marginal (arching) configuration in this kernel; the anchor cannot be
  graded at this orifice at all and must move to `D/d ≥ 8`.
* **P3** (both pure, equal, FLOWING): `p_F = p_Q = 1` → m23261's flip was a
  consequence of its bed being at a *transient* arch threshold; `W` is
  seed-insensitive within the CI → anchor gradeable at `D/d = 5`.
* Any outcome not listed is reported as found (B3 or honest-FAIL).

## 5. Scope and what is deliberately not done

* One configuration, monodisperse spheres, flat bottom, smooth wall. Nothing
  here transfers to conical hoppers, wall friction, or polydisperse beds.
* `C`, `k` are never refitted (m22628 owns that inversion); no case is dropped;
  seed list is `1..6`, fixed here.
* No overlap/stiffness argument: m23114 §3 bounded the soft-contact bias at
  ≤ 3.24 % arithmetically (0 % of the 9.2 × spread) and the `k_n` ladder is its
  mission, not this one.
* Arithmetic of `dt`: BOTH spellings are integrated; the mission does not
  declare one of them "physically right".
## 6. Addenda (each one dated by the order of events, not back-dated)

**A0 — archival ensemble (before the harvest code was written).** The campaign's
earlier rounds left frozen-case records in other worktrees. Reading them is
cheaper than re-running them, so the ensemble is also assembled **from archived
records** by `harvest_archives.py`, under rules fixed before it ran:
the arm of a record is decided by the **IEEE-754 bits of its own `dt`** (never by
a file name); a record is accepted only if geometry, friction, `k_n`, window
layout and reference density match the frozen case, else it is *reported as
rejected with its reason*; the FLOWING/ARRESTED label is re-derived from the
stored window series with `beverloo.discharge_status`; two records of the same
`(arm, seed)` from different commits are kept as a reproducibility check and must
agree to `1e-12` relative.  The archival part is **observational**: its records
come from two commits (`db2bdaf`, `1fb2a46`) and were produced on a different
machine, so it is reported next to the prospective ensemble, never pooled into
it anonymously.

**A1 — B1 must pass its own significance test (made after the archival harvest
existed, before any prospective record did).** As first written, §3 calls the
branch `B1_SPELLING_DOMINATED` whenever the two arms are pure and opposite.  The
archival subset has `n = 1` in arm `Q`, and a 4/4-vs-0/1 table has Fisher
`p = 0.20`: the rule as written would have emitted a factor effect that its own
test contradicts.  Therefore `B1` now **additionally requires Fisher
`p < 0.05`**; a pure-and-opposite split that fails it is `B3_INCONCLUSIVE`
(underpowered).  The archival result is reported both ways in the status
document, and the pre-registered 6-vs-6 design reaches `p = 0.0022`, so the
amendment does not weaken the prospective test — it only stops the archival
subset from over-claiming.

**A2 — accuracy and precision are reported separately.** "PASS" requires both
`|W̄/W_1961 − 1| < 10 %` (accuracy) and `CI95 half-width / W̄ < 10 %` (precision);
an unbiased mean with a wider interval is `INCONCLUSIVE_CI_TOO_WIDE`, not `FAIL`,
because the estimator has not shown a bias — it has shown a sample that is too
small. `required_n = min{ n : t(n−1)·CV/√n < 10 % }` is reported for each arm as
the budget such a verdict needs.
