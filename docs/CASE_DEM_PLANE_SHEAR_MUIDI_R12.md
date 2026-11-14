# CASE-DEM-PLANE-SHEAR-MUIDI — Round 12: rough-wall start fixed, kinematic sampling in every mode

**Anchor:** `CASE-DEM-PLANE-SHEAR-MUIDI` (算例池·平面剪切 mu(I) 流变律标定), status `open`.
**Scope of this round:** make the round-11 rough-wall mode *runnable* and the
measurement pipeline *honest in every mode*, then launch the first full
rough-vs-smooth matrix. Verdict on the mu deficit: **pending the sweep**
(cluster job `cj2515`).

## ① Case & device

Same device, protocol, contact law and reference law as rounds 1-11
(`benchmarks/plane_shear_mu_i/reference_law.py`); only `wall_mode` varies.

| quantity | value |
|---|---|
| particles | spheres, r = 5 mm, rho = 2500 kg/m3, ~294-378 per run |
| cell | W x D x H = 0.12 x 0.05 x 0.06 m, x periodic, y confined |
| contact | linear spring kn = 6.25e4 N/m, Cundall-Strack tangential, mu_p = 0.5, mu_r = 0.05, e = 0.3 |
| control | constant-stress servo, P = 62.5 kPa, shear strain 0.2, settle 2500 steps, >=400 shear steps |
| reference | mu(I) = 0.36 + 0.48/(0.30/I + 1) (da Cruz 2005 / GDR MiDi 2004; PROVENANCE-UNVERIFIED constants, unchanged) |
| plan | 2 wall modes x I in {0.01, 0.03, 0.1, 0.3} x seeds (5279, 6113, 7919) = 24 points |

## ② What was done

1. **Rough-wall initial condition (committed m19451, verified this round).**
   The graft monolayers are half-embedded (centres on the wall planes), the bed
   is lifted by one radius onto the floor layer (`seat_bed_on_grafts`), and the
   platen starts one diameter above the top layer
   (`initial_platen_height(..., rough=True)`). Round 11 used the flat-wall
   start, so the bed began interpenetrating the floor grafts by one radius and
   every rough point died with `servo did not reach the target normal stress`.
   Pinned by `test_initial_condition_has_no_graft_prestress` and the
   counter-example `test_flat_wall_initial_condition_would_prestress_the_rough_cell`.
2. **Defect fix (this round): the velocity-profile shear rate is now sampled in
   every mode.** Until now `shear_rate_measured` / `profile_r2` were only
   accumulated inside the `bulk_stress` branch, so the rough-mode smoke test
   (`bulk_stress=False`) had been **red since R11** (merged red in m18512,
   still red at m19451). The profile is a kinematic diagnostic and is never fed
   back into the integration, so the physics of every archived run is
   bit-for-bit unchanged; `test_wall_mode_is_the_only_difference` still pins
   the smooth path.
3. **Analysis pipeline repaired.** `analyze_r12.py` had never been executed:
   the `extra` aggregates were stored as tuples while `tables()` indexed them
   as dicts, and it only read the round-11 directory. Both fixed; it now reads
   `runs/rough_wall/` + `runs/round12/` and dedups by (mode, I, seed).
4. **Full 24-point sweep submitted** to the cluster (`cj2515`, 8 shards,
   `stage_dir` = this worktree, out = `runs/round12/shard_*.jsonl`).

## ③ Results

**Test suite** `tests/test_rough_wall_plane_shear.py`: **15/15 green**
(13 unit/geometry tests ~5 s; `test_rough_mode_runs_and_loads_the_wall`
83.6 s, `test_wall_mode_is_the_only_difference` 85.3 s).

**mu(I) vs reference, platen-traction estimator** (smooth = R11 baseline,
seed 5279; rough = pending `cj2515`):

| I | mode | mu measured | mu_ref | err |
|---|---|---|---|---|
| 0.01 | smooth | 0.3813 +- 0.0000 | 0.3755 | +1.5% |
| 0.03 | smooth | 0.3283 +- 0.0000 | 0.4036 | -18.7% |
| 0.1 | smooth | 0.3362 +- 0.0000 | 0.4800 | -29.9% |
| 0.3 | smooth | 0.3497 +- 0.0000 | 0.6000 | -41.7% |
| 0.01 | rough | n/a (pending cj2515) | | |
| 0.03 | rough | n/a (pending cj2515) | | |
| 0.1 | rough | n/a (pending cj2515) | | |
| 0.3 | rough | n/a (pending cj2515) | | |

Full tables (all five estimators, servo/kinematics, phi(I), acceptance band):
`benchmarks/plane_shear_mu_i/runs/round12/tables.md`.

## ④ Figures

* `runs/round12/fig_mu_vs_I.png` — mu(I): smooth-wall baseline vs the
  da Cruz 2005 / GDR MiDi 2004 reference with the +/-10 % acceptance band.
* `runs/round12/fig_phi_vs_I.png` — phi(I) (full-gap bed) vs the
  PROVENANCE-UNVERIFIED reference band.
* `runs/round12/fig_estimator_bias.png` — per-estimator bias: the deficit is
  carried by *every* estimator, i.e. it is a physics/measurement issue, not an
  estimator artefact.

The rough-wall series will be added to the same figures when `cj2515` lands
(`analyze_r12.py` picks it up automatically).

## ⑤ Notes / limitations

1. **No rough-wall data yet in this document.** The sweep is queued
   (`cj2515`); the round-11 FAIL records are preserved in
   `runs/rough_wall/round11/rough_servo002.jsonl` and listed in the tables.
2. **Reference provenance is still open.** The benchmark uses
   (mu_s, mu_2, I0) = (0.36, 0.84, 0.30) as quoted in rounds 1-5, while
   `tests/test_mu_i_rheology.py` cites the 3-D sphere triple
   (0.379, 0.641, 0.279) from "Table 1 of da Cruz et al. 2005". The two
   curves differ by up to ~15 % in mu at I = 0.3. Neither reading is archived
   in-repo; a provenance round is needed before the ledger acceptance can be
   judged against the *paper* rather than the in-repo quote.
3. Single device, single seed for the baseline; the 3-seed ensemble is part of
   the in-flight sweep.
4. The phi reference magnitudes (0.58, 0.64) remain PROVENANCE-UNVERIFIED
   (see `reference_law.py`); phi acceptance stays WITHHELD.

## ⑥ Error judgment (explicit)

| quantity | reference | measured | error | within 3 %? |
|---|---|---|---|---|
| mu_wall, smooth, I=0.01 | 0.3755 | 0.3813 | +1.5 % | **yes** |
| mu_wall, smooth, I=0.03 | 0.4036 | 0.3283 | -18.7 % | **no** |
| mu_wall, smooth, I=0.1 | 0.4800 | 0.3362 | -30.0 % | **no** |
| mu_wall, smooth, I=0.3 | 0.6000 | 0.3497 | -41.7 % | **no** |
| mu, rough, all I | — | pending cj2515 | — | unknown (not run) |
| phi_bed, smooth, all I | 0.61-0.64 | 0.53-0.57 | -6.1 ... -16.7 % | **no** (and reference itself unverified) |

Honest status: the smooth wall still misses the law at I >= 0.03; whether the
rough wall closes the gap is exactly what `cj2515` answers.
