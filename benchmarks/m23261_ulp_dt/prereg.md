# Pre-registration — 1-ULP integration step vs. discharge verdict (m23261)

Written **before** any DEM run of this mission was read. Analyses live in
`analyse.py` and are pinned on synthetic data by `tests/test_m23261_ulp_dt.py`.
Anchor: `CASE-DEM-SILO-BEVERLOO` (`|W/W_ref − 1| < 10 %`, Beverloo et al. 1961,
`C = 0.58`, `k = 1.4`, never refitted). Ledger status at checkout: `open`.

## 0. The open question this mission closes

m23184 found two archived records of the *same* frozen case and the *same* seed
with a 9.2 × disagreement in `W`:

| record | `W` [kg/s] | status |
|---|---|---|
| `m22885_beverloo_powered/fill36_seed1` (git `db2bdaf`) | 0.0307498 | FLOWING (+5.5 %) |
| `m23114_siloflow_ladder/dt_nom20_s1` (git `1fb2a46`) | 0.0033577 | ARRESTED (−88.5 %) |

m23184 left the cause open, between (a) a different code state and (b) genuine
bistability of the bed under an initial-state difference far below the stored
precision, and noted the record carries no bed checksum to separate them.

## 1. Hypothesis (stated before the runs)

**H1 (arithmetic, no DEM).** The two drivers spell "20 steps per contact
period" differently — `Material.dt = 0.05 * T_c` (used by the `study beverloo`
CLI, hence m22885) versus `dt = T_c / 20` (used by `m23114/run_level.py`) —
and the two results differ by exactly **1 ULP**
(`1.86767398828964e-05` vs `1.8676739882896397e-05`, rel. 1.8e-16). The two
archived `dt` fields match the two spellings **bit for bit**, and the
`src/tensordem/silo_flow/` diff between the two git shas is purely additive
diagnostics (no physics line touched). Therefore cause (a) is excluded and the
9.2 × spread must be an *amplification* of the 1-ULP `dt` difference.

**H2 (DEM).** Starting from a bit-identical bed, a `dt` difference of 1 ULP is
enough to flip the discharge between the jet branch (`W ≈ 0.0307`) and the
delayed-onset branch (`W ≈ 0.0034`). The frozen configuration is therefore
*marginal*: its discharge onset is not resolved by the campaign's 0.12 s discard
window, and a single record cannot decide the anchor even at fixed seed.

**P1 (prediction, falsifiable).** Re-running the archive with a single shared
bed reproduces, bit for bit, each archived `W` when the archived `dt` spelling
is used (0.030749793997350797 / 0.0033577361261474805). If either reproduction
misses by more than 1e-12 relative, H1 is falsified and the difference is a code
state after all.

**P2.** ≥ 1 of the 1-ULP-perturbed branches flips the FLOWING/ARRESTED label, and
the two labels are separated by ≥ 5 × the counting-noise floor (7–8 %) of either
record.

**P3 (control).** A *finite* 0.1 % change of `dt` at the same bed also changes
the label, i.e. the sensitivity is not an artefact of the last bit but a
property of a marginal configuration near its onset threshold. (If P3 fails and
P2 holds, the finding is "chaotic amplification of the last bit" only, and the
protocol recommendation drops to "pin the dt spelling".)

## 2. Frozen configuration (identical to the campaign's primary case)

`d = 3 mm`, `ρ = 2500 kg/m³`, `D = 15 mm` (`D/d = 5`), silo radius 24 mm,
fill height 36 mm, `μ = 0.50`, `μ_roll = 0.05`, `e = 0.30`, `k_n = 5e3 N/m`,
`k_t/k_n = 0.5`, wall friction 0 (flat plate, Beverloo smooth-wall condition),
0.06 s settle with the outlet plugged, 0.12 s discard, 8 × 0.05 s windows,
`W` = mean of the last 4 windows. Seed 1. `T_c = 3.7353479765792795e-4 s`,
nominal `dt = 1.86767398828964e-05 s` (31054 steps for 0.58 s).

## 3. Runs submitted (5 shards, `run_ulp.py --shard i --nshards 5`)

| shard | bed built at | discharge at | purpose |
|---|---|---|---|
| `A_arch_bed` | `0.05·T_c` (m22885 path) | `0.05·T_c` | archive reproduction (P1) |
| `B_quot_bed` | `T_c/20` (m23114 path) | `T_c/20` | archive reproduction (P1) |
| `C_up1` | `0.05·T_c` (shared bed) | `nextafter(dt, +Inf)` | 1-ULP perturbation up (P2) |
| `D_down1` | shared bed A | `nextafter(dt, −Inf)` = `T_c/20` | 1-ULP perturbation down (P2) |
| `E_finite` | shared bed A | `1.001 · dt` | finite control (P3) |

Shards C, D, E reuse the **bit-identical** settled bed of shard A (deep copy of
the `SiloFlow` state after `build_bed`, `pos`/`vel` sha256 recorded in every
record), so they differ from A in nothing but `dt`.

## 4. Verdict rules (fixed now)

For every record: `W`, the Beverloo relative error, window particle counts, and
`discharge_status` (FLOWING / ARRESTED, `tensordem.silo_flow.beverloo`).

* **P1**: `|W_run − W_archive| / W_archive < 1e-12` for A and B.
* **P2**: `label(C) ≠ label(A)` or `label(D) ≠ label(A)`, and the `W` values of
  the two labels differ by a factor ≥ 5.
* **P3**: `W(E)/W(A) − 1` exceeds the 10 % ledger tolerance (a finite control
  that moves nothing would show the configuration is not marginal).
* The anchor is **NOT decided** by any single record of this family; it may only
  be quoted as an ensemble with the arrest fraction beside it. Nothing here
  changes `C` or `k`, and no amplitude is refitted.

Honest-FAIL rule: a shard that does not produce a record is reported `NOT_RUN`.

## 5. Scope

One frozen configuration (`D/d = 5`, monodisperse, smooth plate, flat bottom).
Nothing here transfers to conical hoppers, wall friction, or polydisperse beds.