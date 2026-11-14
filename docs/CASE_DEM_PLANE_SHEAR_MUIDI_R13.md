# CASE-DEM-PLANE-SHEAR-MUIDI — Round 13: full acceptance range (I = 1e-3…0.3), rough-wall verdict

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池·平面剪切 mu(I) 流变律标定), status `open`.
**Scope of this round:** (a) cover the ledger's *full* acceptance range
`I = 1e-3 … 0.3` (rounds 11-12 only swept `0.01 … 0.3`), (b) land the first
rough-wall data and judge the round-11 wall-slip hypothesis, (c) quantify the
seed-to-seed spread that rounds 8-10 flagged as "sample identification".

## ① Case & device

Same device, protocol, contact law and reference law as rounds 1-12
(`benchmarks/plane_shear_mu_i/reference_law.py`); only `wall_mode` varies.

| quantity | value |
|---|---|
| particles | spheres, r = 5 mm, rho = 2500 kg/m3, ~294-378 per run |
| cell | W x D x H = 0.12 x 0.05 x 0.06 m, x periodic, y confined |
| contact | linear spring kn = 6.25e4 N/m, Cundall-Strack tangential, mu_p = 0.5, mu_r = 0.05, e = 0.3 |
| control | constant-stress servo, P = 62.5 kPa, shear strain 0.2, settle 2500 steps, >=400 shear steps |
| reference | mu(I) = 0.36 + 0.48/(0.30/I + 1) (da Cruz 2005 / GDR MiDi 2004, in-repo quote); phi magnitudes PROVENANCE-UNVERIFIED |
| plan (R13) | 2 wall modes x I in {1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1} x seeds (5279, 6113, 7919) = 36 points |

## ② What was done

1. **Recovered the stuck round-12 sweep.** `cj2515` (24 points, submitted
   11-14 09:16) had been "running" for 181 min with 3/24 points written
   (shard 1 only). It was killed; its 3 completed records
   (`runs/round12/shard_1.jsonl`: smooth I=0.01 seed 6113, smooth I=0.3 seed
   5279, **rough I=0.03 seed 7919**) were fetched and archived.
2. **Extended the sweep to the ledger range.** `rough_wall_sweep.py` now
   targets `I = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1)` (36 points, 8 shards,
   out = `runs/round13/`). The two new low-I points cost ~150 s / ~70 s each
   (shear phase 12442 / 4148 steps at dt = 3.2e-5 s); the full plan is
   ~36 core-min. Submitted as cluster job **`cj2672`** (queued at report time).
3. **R13 analysis pipeline** (`analyze_r13.py`): reads `runs/rough_wall/`
   (R11) + `runs/round12/` (R12 partial) + `runs/round13/` (R13), dedups by
   (mode, I, seed) with the newest round winning, aggregates per (mode, I)
   over seeds, and emits `tables_r13.md`, `analysis_r13.json` and the figures
   `fig_mu_vs_I.png`, `fig_phi_vs_I.png`, `fig_seed_spread.png`.
4. **Verdict on the wall-slip hypothesis (first rough-wall data):** the
   round-12 initial-condition fix works (the rough mode now runs), but the
   rough wall does **not** close the mu deficit - it widens it (Table 1).

## ③ Results (data in hand at report time; `cj2672` pending)

**mu(I), platen-traction estimator** (mean over seeds in hand):

| I | mode | seeds | mu measured | mu_ref | err |
|---|---|---|---|---|---|
| 0.01 | smooth | 2 | 0.3260 +- 0.0782 | 0.3755 | -13.2 % |
| 0.03 | smooth | 1 | 0.3283 | 0.4036 | -18.7 % |
| 0.1 | smooth | 1 | 0.3362 | 0.4800 | -29.9 % |
| 0.3 | smooth | 1 | 0.3497 | 0.6000 | -41.7 % |
| 0.03 | rough | 1 | 0.1548 | 0.4036 | -61.7 % |

**phi(I), full-gap bed** (reference PROVENANCE-UNVERIFIED):

| I | mode | phi measured | phi_ref | err |
|---|---|---|---|---|
| 0.01 | smooth | 0.5326 | 0.6398 | -16.5 % |
| 0.03 | smooth | 0.5394 | 0.6345 | -15.0 % |
| 0.1 | smooth | 0.5496 | 0.6250 | -12.1 % |
| 0.3 | smooth | 0.5727 | 0.6100 | -6.1 % |
| 0.03 | rough | 0.4856 | 0.6345 | -23.5 % |

**Seed-to-seed spread (the sample-identification question):** at I = 0.01 the
two seeds give mu = 0.3813 (5279) and 0.2707 (6113) - a **29.4 %** spread
against a reference of 0.3755, i.e. the sample-to-sample variability alone is
of the order of the acceptance band. The smooth-wall "deficit" at I >= 0.03 is
systematic (all estimators, all seeds in hand agree on the sign), but at
I = 0.01 the measurement is not yet converged in the sample.

**Kinematics:** the measured/imposed shear-rate ratio is 0.48-1.03 across the
points in hand (rough I=0.03: 0.483 - the grafted wall slips *less* than the
flat platen at I=0.1, 0.632, but the stress is still far below the law).

## ④ Figures

* `runs/round13/fig_mu_vs_I.png` - mu(I): smooth/rough vs reference with the
  +/-10 % acceptance band (data in hand; `cj2672` extends the curves).
* `runs/round13/fig_phi_vs_I.png` - phi(I) vs the PROVENANCE-UNVERIFIED
  reference.
* `runs/round13/fig_seed_spread.png` - per-seed mu(I): the I=0.01 spread.

## ⑤ Notes / limitations

1. **The full-range matrix is in flight** (`cj2672`, 36 points). This document
   is written on the 6 OK + 2 FAIL records already archived; the tables and
   figures update automatically when the sweep lands (`analyze_r13.py`).
2. **Reference provenance is still open** (round-12 note 2, unchanged): the
   in-repo mu constants (0.36, 0.84, 0.30) and the phi magnitudes (0.58, 0.64)
   have no archived reading of the two papers; the mu comparison is against
   the in-repo quote, the phi comparison stays WITHHELD.
3. **The rough wall is a single grafted monolayer** (spacing 1.05 d, 58 %
   areal coverage, half-embedded). da Cruz's rough walls are a *layer* of the
   same particles; a thicker rough layer is the obvious next variant, not
   tried here.
4. Single device, one contact law; the mu(I) law is being tested for
   glass-bead-like spheres at P = 62.5 kPa only.

## ⑥ Error judgment (explicit, data in hand)

| quantity | reference | measured | error | within 3 %? |
|---|---|---|---|---|
| mu_wall, smooth, I=0.01 (2 seeds) | 0.3755 | 0.3260 | -13.2 % | **no** |
| mu_wall, smooth, I=0.03 | 0.4036 | 0.3283 | -18.7 % | **no** |
| mu_wall, smooth, I=0.1 | 0.4800 | 0.3362 | -29.9 % | **no** |
| mu_wall, smooth, I=0.3 | 0.6000 | 0.3497 | -41.7 % | **no** |
| mu_wall, rough, I=0.03 | 0.4036 | 0.1548 | -61.7 % | **no** |
| phi_bed, smooth, I=0.01…0.3 | 0.61-0.64 | 0.53-0.57 | -6.1 … -16.5 % | **no** (reference itself unverified) |
| mu, all I in 1e-3…0.3 | - | pending cj2672 | - | unknown (not run) |

**Honest status: FAIL against the ledger acceptance (<10 %).** No estimator,
no wall mode, no I point in hand is inside the 10 % band except the single
seed-5279 point at I=0.01 (+1.5 %, round 11). The wall-slip mechanism
(round-11 hypothesis) is **not supported** by the first rough-wall data: the
rough wall measures a *lower* mu, not a higher one. The open causes are
(i) sample-to-sample variability (29 % at I=0.01), (ii) a systematic
low-mu bias at I >= 0.03 carried by every estimator, and (iii) the unverified
reference itself.
