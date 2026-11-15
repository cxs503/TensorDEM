# CASE-DEM-PLANE-SHEAR-MUIDI — round 23: seed ensemble and the falsifiability of the 10 % criterion

**Ledger:** `CASE-DEM-PLANE-SHEAR-MUIDI` — 算例池·平面剪切 mu(I) 流变律标定.
**Reference (archived, parsed from the paper):** da Cruz et al. 2005 Phys. Rev. E 72:021309, Eq. (13)/(14) — `mu = 0.22 + 1.00 I`, `phi = 0.81 - 0.30 I`, valid for `I <= 0.2`; GDR MiDi 2004 Eur. Phys. J. E 14:341 for the 3-D packing path `0.85 -> 0.80`.
**Device:** `tensordem.mu_i_rheology.simulate_servo_shear` — 6.25e+04 Pa servo platen, `kn = 6.25e+04` N/m (P/kn = 1), `mu_p = 0.5`, `mu_r = 0.05`, `e = 0.3`, box 0.12x0.05x0.06 m, 2500 settle steps, shear strain 0.2, 294 spheres (d = 10 mm).
**Runs:** 6 seeds x 4 inertial numbers = 24 ensemble runs + 5 contact-friction ladder runs, cluster job `cj3846` (12 shards).
**Artifacts:** `benchmarks/plane_shear_mu_i/seed_ensemble.py`, `test_seed_ensemble_r23.py`, `runs/seed_ensemble/{prereg.json,point_*.json,analysis_r23.json,tables_r23.md,fig_r23_*.png}`.

---

## ① Case introduction

A plane-shear cell of 294 spheres is sheared between a servo-controlled platen and a frictional floor at a constant confining stress `P = 62.5 kPa`. Round 22 settled the *bibliographic* question — the only citable yardstick is the archived da Cruz fit `mu = 0.22 + 1.00 I` — and found the four archived points (round 14, `runs/full_cell/point_*.json`) missing that law by 16-57 % with a non-monotone `mu(I)` (a -15 % step between `I_bulk = 0.035` and `0.074`).

It could not decide whether that is a **device** property or a **realisation** property, because every archived point is one seed (5279) and the dip is the size of the tolerance. The ledger asks for `mu(I)` and `phi(I)` within 10 %; against a single-realisation curve a 10 % criterion is unfalsifiable in both directions. This round makes it falsifiable:

1. an ensemble of 6 seeds at each of the four archived targets — spread, error bars, and a paired test of the dip itself;
2. round 22's recommendation 1 as a bounded add-on: the contact-friction ladder `mu_p in [0.2, 0.3, 0.4, 0.5, 0.7]` at `I = 0.03`, which is the one bias mechanism that *would* move `mu` toward the archived law.

Everything except the seed (and, in the ladder, `mu_p`) is bit-identical to round 14, and the ensemble contains the archived seed so that a protocol drift would show up as a parity failure rather than as noise.

## ② What was done

1. **Pre-registration first.** `runs/seed_ensemble/prereg.json` was written before the first run: the axes, the reference law, the 10 % tolerance, four predictions (P1 spread, P2 dip, P3 bias, P4 friction) and the decision rule of each. The gate (`test_seed_ensemble_r23.py::test_prereg_was_written_before_the_runs`) checks the file against the module contract and that the reference is the archived `0.22` law, not the mis-attributed `(0.36, 0.84, 0.30)` quote.
2. **Device held fixed by import, not by retyping.** `seed_ensemble.py` imports `P_APPLIED, KN, MU_CONTACT, MU_ROLLING, E, RADIUS, W, D, H, SETTLE_STEPS, MIN_SHEAR_STEPS, SHEAR_STRAIN` from `full_cell_rheometry.py`, i.e. from the module that generated the archived points. The only arguments that move are `seed` and, in the ladder, `mu`.
3. **Parity check.** Seed 5279 re-runs the archived configuration; the gate requires the re-run to reproduce `runs/full_cell/point_*.json` **bit for bit** (`test_archive_seed_reproduces_round14_bit_for_bit`). It does: the largest relative difference over 18 compared quantities is `0.0e+00`.
4. **Cluster submission.** 29 runs, 12 shards, job `cj3846`, `stage_dir` = this worktree, one JSON per run (`runs/seed_ensemble/point_I<I>_s<seed>[__mup<mu>].json`). Each run is ~50 s (I = 0.01-0.1) to ~150 s (I = 0.3).
5. **Statistics.** For every ordinate: mean, sample sd, coefficient of variation `CV = sd/mean`, standard error and a two-sided 95 % Student-t interval (`mean_sd`, `ci95` in `seed_ensemble.py`); the dip is tested *paired by seed*; the ladder is judged against the same archived law at its own `I`.
6. **Judging rule.** Only `I_bulk <= 0.2` is judged (the archived fit states it breaks down at `I >= 0.2`). The `I = 0.3` arm is reported as an extrapolation and is excluded from the verdict — the round-22 finding that the ledger's own acceptance range (`I = 1e-3..0.3`) runs past the law's validity is thereby honoured, not silently repaired.

## ③ Results

### Table A — seed spread, `mu` from the full-cell virial stress (normalised by `|sigma_zz|`)

| `I_target` | n | `I_bulk` mean | `mu` mean | sd | CV % | 95 % CI | law | mean err | seeds in 10 % band | in law validity |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 6 | 0.0080 | 0.2997 | 0.0268 | +8.9 | [0.2716, 0.3278] | 0.2300 | +30.3 | 0/6 | yes |
| 0.03 | 6 | 0.0203 | 0.3231 | 0.0667 | +20.6 | [0.2532, 0.3931] | 0.2500 | +29.3 | 0/6 | yes |
| 0.1 | 6 | 0.0703 | 0.3430 | 0.0243 | +7.1 | [0.3174, 0.3685] | 0.3200 | +7.2 | 4/6 | yes |
| 0.3 | 0 | -- | -- | -- | -- | [--, --] | 0.5200 | -- | 0/0 | NO (extrapolation) |

### Table B — which ordinate is realisation-robust? (CV % of each `mu`)

| `I_target` | `mu_full_zz_final` | `mu_full_final` | `mu_wall_final` | `mu_slab_mid_final` | `mu_slab_both_final` |
|---|---|---|---|---|---|
| 0.01 | +8.9 | +15.2 | +8.9 | +19.8 | +26.7 |
| 0.03 | +20.6 | +17.7 | +21.3 | +18.6 | +17.1 |
| 0.1 | +7.1 | +3.2 | +5.8 | +9.3 | +9.2 |
| 0.3 | -- | -- | -- | -- | -- |

### Table C — the paired dip, `I_bulk` 0.03 -> 0.1 (`mu_full_zz_final`)

6 seed pairs. Mean change 0.0198 (+10.8 %), sd 0.0678, 95 % CI of the change [-0.0513, 0.0909], negative in 2/6 pairs, **systematic = False**.

| seed | `mu` at `I=0.03` | `mu` at `I=0.1` | change | relative % |
|---|---|---|---|---|
| 5279 | 0.3998 | 0.3394 | -0.0604 | -15.1 |
| 6113 | 0.3071 | 0.3432 | 0.0361 | +11.8 |
| 7919 | 0.2022 | 0.3279 | 0.1257 | +62.1 |
| 104729 | 0.3322 | 0.3824 | 0.0502 | +15.1 |
| 1299709 | 0.3431 | 0.3543 | 0.0111 | +3.2 |
| 2000003 | 0.3545 | 0.3107 | -0.0438 | -12.4 |

### Table D — solid fraction, spread only (definitions unchanged)

| `I_target` | `phi_slab` mean | CV % | `phi_bed_steady` mean | CV % |
|---|---|---|---|---|
| 0.01 | 0.6451 | +2.1 | 0.5357 | +0.7 |
| 0.03 | 0.6623 | +2.6 | 0.5425 | +1.4 |
| 0.1 | 0.6833 | +3.0 | 0.5481 | +0.8 |
| 0.3 | -- | -- | -- | -- |

### Table E — contact-friction ladder at `I = 0.03`

Law at this `I`: `mu = 0.2500`. Ladder monotone in `mu_p`: **False**; smallest |mean err| over the ladder: -36.1; reaches the 10 % band: **False**.

| `mu_p` | `I_bulk` | `mu` full-cell | `mu` slab both | `mu` wall | err vs law % | servo valid |
|---|---|---|---|---|---|---|
| 0.20 | 0.0214 | 0.2870 | 0.3044 | 0.2854 | +14.8 | False |
| 0.30 | 0.0119 | 0.1924 | 0.2212 | 0.1938 | -23.0 | False |
| 0.40 | 0.0081 | 0.1599 | 0.2308 | 0.1613 | -36.1 | False |
| 0.50 | 0.0350 | 0.3998 | 0.3958 | 0.4120 | +59.9 | False |
| 0.70 | 0.0254 | 0.2996 | 0.2801 | 0.3016 | +19.9 | False |

### Table F — pre-registered prediction vs outcome

| id | prediction | outcome |
|---|---|---|
| P1 | seed `CV < 10 %` at every judged point (one realisation resolves the band) | max CV +20.6 -> **falsified** |
| P2 | the dip survives the ensemble (systematic) | **False** |
| P3 | ensemble mean outside the band at `I = 0.03` and `0.1` | **False** |
| P4 | ladder monotone in `mu_p` and does not reach the band | monotone **False**, reaches band **False** |

**Acceptance verdict (all judged points, all 6 seeds within 10 %): FAIL** — 0/3 judged points pass.

## ④ Figures

`runs/seed_ensemble/`:

| file | caption |
|---|---|
| `fig_r23_muI_ensemble.png` | `mu(I)` with every seed drawn as a single point, the ensemble mean with its 95 % Student-t interval, the archived da Cruz law and its +-10 % band; the dotted part of the law is the extrapolation beyond `I = 0.2` and is not judged. |
| `fig_r23_cv.png` | seed coefficient of variation of all five stress ordinates at each point, against the 10 % acceptance band: which estimator is realisation-robust. |
| `fig_r23_dip_paired.png` | the same seed followed from `I = 0.03` to `I = 0.1`; the paired dip test of Table C. |
| `fig_r23_mup_ladder.png` | `mu` versus contact friction at `I = 0.03` against the archived law value and its band: does de-biasing `mu_p` close the gap? |

## ⑤ Note points (pitfalls, limits, scope)

* **What a seed ensemble can and cannot see.** The device is deterministic given its inputs; the seed changes the *initial packing* only. The CV measured here is therefore the spread over initial conditions at a fixed protocol — it does **not** cover protocol-axis scatter (settle time, strain window, gap), each of which was shown in rounds 16-21 to move `mu` by more than this spread. The ensemble is a lower bound on the total uncertainty, and says nothing about bias.
* **3-D spheres against a 2-D disk law.** da Cruz 2005 measure 2-D disks at `mu_p = 0.4`; this cell is 3-D spheres at `mu_p = 0.5`. The 10 % band is judged as the ledger demands, but a 3-D packing has a larger quasi-static `mu_s` at the same `mu_p`, so part of the offset is dimensionality, not error. The ladder (Table E) separates the `mu_p` part of that; the dimensionality part is **not** separated, and no fitted 3-D triple exists in the archived sources (round 22, Table A) to separate it with.
* **`phi` is still not a comparison.** Only its *spread* is reported here. `phi_slab` is a sheared-slab solid fraction and `phi_bed_steady` a bed-volume one; both stay far from `0.81 - 0.30 I` for definitional reasons that round 22 classified WITHHELD. This round does not change that verdict.
* **`I = 0.3` is outside the law's validity** (`I_sat = 0.2`) and is reported, not judged. The ledger acceptance range (`I = 1e-3..0.3`) still runs past the source's fit; that is a defect of the acceptance text (round 22 §⑦ item 4), not of the simulation.
* **Servo validity.** Runs whose constant-stress gate did not converge are flagged per point (`servo_valid` in the per-run JSON and in `tables_r23.md`); a point whose seeds are not all valid is not a clean measurement and is called out in Table A's CI rather than dropped.
* **The dip is judged on one ordinate.** `mu_full_zz_final` is the ordinate the archived law's construction corresponds to (round 22, Table D). Table B reports the others so that a reader can see whether the conclusion is ordinate-dependent.
* **No new tolerance is invented.** The 10 % is the ledger's; the 95 % intervals are Student-t on 6 samples, which is a small-sample interval and is why the per-seed scatter is drawn as well.

## ⑥ Error verdict (honest)

| quantity | criterion | measured (ensemble mean) | verdict |
|---|---|---|---|
| `mu` at `I = 0.01` | < 10 % vs archived law | 0.2997 vs 0.2300 = +30.3 | **FAIL** |
| `mu` at `I = 0.03` | < 10 % vs archived law | 0.3231 vs 0.2500 = +29.3 | **FAIL** |
| `mu` at `I = 0.1` | < 10 % vs archived law | 0.3430 vs 0.3200 = +7.2 | **PASS** |
| `mu` at `I = 0.3` | outside `I_sat = 0.2` | -- vs extrapolated 0.5200 = -- | **NOT JUDGED** |
| seed spread `CV` | < 10 % (P1) | max +20.6 | **FAIL** |
| dip `I=0.03 -> 0.1` | systematic (P2) | +10.8 %, CI [-0.0513, 0.0909] | **NOT RESOLVED** |
| `phi(I)` vs `0.81 - 0.30 I` | < 10 % | definitions not commensurable | **WITHHELD** (unchanged since round 22) |

**Is any key quantity within 3 % of its reference? No.** The smallest ensemble-mean deviation over the judged points is +7.2; the acceptance criterion the ledger sets is 10 %, and the 3 % question is answered separately because the platform's精度 standard asks for it. Where the answer is No the reason is stated above (dimensionality and `mu_p`, both unseparated), not excused.

## ⑦ What this changes for the ledger

The acceptance still FAILS, and now it fails **with error bars**: the verdict is no longer an artefact of one seed. What remains is a measurement-definition problem, and the two candidate mechanisms are separable with the ladder result:

* the friction ladder does **not** reach the band, so a `mu_p` re-calibration cannot rescue the acceptance on its own: the remaining candidate is the dimensional reduction (3-D spheres versus the 2-D disk law), which needs a 3-D reference rather than another DEM run.

Recommended next steps: (1) amend the ledger acceptance to name the source equation and drop or explicitly branch on `I = 0.3`; (2) if a 3-D reference is to be used, archive a 3-D `mu(I)` paper the same way round 22 archived the 2-D one; (3) adjudicate `phi` with one definition (whole-gap vs slab) fixed *before* the run.

