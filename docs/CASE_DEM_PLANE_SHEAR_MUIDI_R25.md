# CASE-DEM-PLANE-SHEAR-MUIDI — round 25: the returned sweep, judged

**Ledger:** `CASE-DEM-PLANE-SHEAR-MUIDI` — 算例池·平面剪切 mu(I) 流变律标定 (status `open`).
**Round 24 left open:** "drain `cj4011`, run `analyse_2d_r24.py`, issue the round-25 verdict".
This round does exactly that, and the verdict is **FAIL on the pre-registered
predictions P2/P4/P5** — published as such, with no tolerance re-negotiated.
**Reference (read from the archive at run time):** da Cruz et al. 2005 PRE 72:021309,
`mu = 0.22 + 1.00 I`, `phi = 0.81 - 0.30 I`, `I <= 0.2` (`reference_archive.archived_law()`).

---

## ① Case introduction

The ledger question is whether the platform reproduces the plane-shear `mu(I)` / `phi(I)`
law of da Cruz et al. 2005 (2-D frictional disks, `mu_p = 0.4`, constant pressure,
prescribed shear rate). Rounds 1-23 measured it on a **3-D sphere** cell and found a
**+16...+57 % (typically +29 % at I = 0.03)** offset. Round 23 closed the seed hypothesis
(6-seed ensemble, CV up to 20.6 %) and left one candidate mechanism: **dimension** — a 3-D
sphere packing has a larger quasi-static `mu_s` than a 2-D disk packing at the same `mu_p`.
Round 24 therefore built a **2-D disk** device (the law's own dimension) and pre-registered
five predictions (`prereg_r24.json`) before running anything; the production sweep was
still queued when round 24 closed, so no number was published.

Returned artifacts of job **`cj4011`** (succeeded, node n330, fetched to
`benchmarks/plane_shear_mu_i/runs/plane_shear_2d/probe_I*.json`): the **probe-cell**
series — 458 free monodisperse disks (`d = 10 mm`), `W = 0.30 m`, `gap = 0.15 m`,
`P = 1.0e4 Pa`, `mu_p = 0.4`, `mu_rolling = 0`, `kn = 1e5 N/m`, `zeta = 0.2`, one seed,
40 `d/sqrt(P/rho)` of compaction + 0.5 strain transient + 1.0 strain measurement window,
at `I_target = 0.02, 0.05, 0.1, 0.2`. **Not returned:** `I = 0.005, 0.01` (the two most
expensive shards, ~30x the steps of `I = 0.2`) and `I = 0.3`; one seed, not three; and the
1220-disk production cell was never run.

## ② What was done

1. **Drained `cj4011`** and committed its four run artifacts to the worktree (2.6 MB).
2. **Repaired the analysis path so the sweep can actually be read** (`analyse_2d_r24.py`):
   `load_runs()` only globbed `point_I*_s*.json` (the *pre-registered* production name) and
   silently returned nothing for the returned `probe_I*.json` series; and
   `compare_with_3d()` walked a schema the round-23 file does not have (`points` is a dict
   keyed by `I_target`, `I_bulk_mean` is a stats block). Both are fixed; the 2-D/3-D pairing
   now resolves (`vs_3d` is populated, figure 3 exists).
3. **Judged the returned points against the pre-registered rule** (`analyse_2d_r24.py
   --probe` → `runs/plane_shear_2d/analysis_r24_probe.json`, `tables_r24_probe.md`,
   `fig_r24_{muI,phiI,dimensionality}_probe.png`). No tolerance was touched: `mu` 10 %,
   `phi` 10 %, `I <= I_sat = 0.2` judged, `I = 0.2` is in fact *reported outside* the fit at
   the measured `I = 0.246`.
4. **Root-cause instrument added** (`plane_shear_2d.py`, behind flags, round-24 defaults
   unchanged): `hexatic_order()` — the local 6-fold bond-order parameter `psi6` and the mean
   first-shell coordination `Z`, recorded in every run summary — and `--init random`, a
   disordered packing at the *same* area fraction as the round-24 seed (lattice sites
   displaced by 0.4 d, overlaps removed by periodic positional projection). Reason: the
   round-24 cell is seeded from a **triangular lattice jittered by 5 % of a diameter**
   (`_lattice_packing`), i.e. from a crystal, and a crystal shears by basal slip at a far
   lower effective friction than a disordered packing.
5. **Gates:** `pytest -q tests/test_plane_shear_2d_r25.py tests/test_plane_shear_2d_r24.py`
   → **21 passed** (56 s): `psi6` severs lattice from disorder, `_random_packing` is
   admissible/overlap-free/disordered, the round-24 lattice seed is shown to be a crystal,
   and the published verdict is reproduced from the artifacts (no transcription).

## ③ Results (programmatic, from `tables_r24_probe.md`)

**Table A — mu(I) and phi(I), returned sweep (`cj4011`)**

| I_target | I measured | mu_wall | mu law | err % | phi_gap | phi law | err % | judged |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.0261 | 0.0963 | 0.2461 | **-60.9** | 0.873 | 0.802 | **+8.8** | yes |
| 0.05 | 0.0631 | 0.1003 | 0.2831 | **-64.6** | 0.875 | 0.791 | **+10.7** | yes |
| 0.10 | 0.1393 | 0.1238 | 0.3593 | **-65.5** | 0.874 | 0.768 | **+13.8** | yes |
| 0.20 | 0.2464 | 0.1294 | 0.4664 | **-72.3** | 0.875 | 0.736 | **+18.8** | no (I > I_sat at the measured I) |

**Table B — estimator calibration (P1)**

| I_target | p_wall (Pa) | p_bulk (Pa) | sigma_yy/P | p_wall/P | ke_settle (J) | drift_z(mu_wall) |
|---|---|---|---|---|---|---|
| 0.02 | 9695 | 7487 | 1.039 | 0.969 | 4.7e-05 | -6.94 |
| 0.05 | 9937 | 7559 | 1.066 | 0.994 | 4.7e-05 | -13.18 |
| 0.10 | 9093 | 6739 | 0.973 | 0.909 | 4.7e-05 | -2.81 |
| 0.20 | 10104 | 8107 | 1.082 | 1.010 | 4.7e-05 | +3.25 |

**Pre-registered verdict** (`analysis_r24_probe.json`)

| prediction | rule | result |
|---|---|---|
| P1 calibration | `|p_wall/P - 1| <= 10 %` at every judged point | **PASS** (0.91-0.99) |
| P2 mu in band | `|mu_err| <= 10 %`, all judged points | **FAIL** (0/3, worst -65.5 %) |
| P4 phi in band | `|phi_err| <= 10 %`, all judged points | **FAIL** (1/3, worst +13.8 %) |
| P5 stationarity | `|drift_z(mu_wall)| < 2` | **FAIL** (-6.9, -13.2, -2.8, +3.3) |

**Figures** (in `benchmarks/plane_shear_mu_i/runs/plane_shear_2d/`):
`fig_r24_muI_2d_probe.png`, `fig_r24_phiI_2d_probe.png`, `fig_r24_dimensionality_probe.png`.

## ④ What the numbers say

1. **The dimensionality hypothesis is falsified as stated.** P3 required `mu_2D` to sit
   *below* the 3-D value by more than the 3-D seed spread. The 3-D ensemble is
   `mu = 0.2997 +- 0.027` at `I = 0.0080` (r23, n = 6); the 2-D device gives `0.0963` at
   `I = 0.026`, so the 2-D value is **-68 %** below the 3-D one — but the archived law says
   it should be *inside* 10 % of `0.246`, i.e. it is not "a bit lower", it is **on a
   different branch**. A device that reproduces the law's *dimension* but not its *friction*
   cannot adjudicate a +29 % dimensional offset; the offset is left unexplained.
2. **The 2-D cell is on the frictionless branch.** `mu_wall = 0.096...0.129` while
   `mu_p = 0.4`: the archived paper's own frictionless measurement is `mu*_min ~ 0.11`
   (excerpt `frictionless_singular` in `runs/provenance_archive/`). The cell behaves as if
   the contact friction were switched off — which is what a **crystal** does: it shears by
   basal slip plus free rolling (`mu_rolling = 0`), not by a frictional contact network.
3. **The solid fraction corroborates a crystal.** `phi_gap = 0.873...0.875`, flat in `I`,
   i.e. **+8.8...+18.8 % above the archived `nu_max = 0.81`** and within 4 % of the
   hexagonal limit `pi/sqrt(12) = 0.9069`. The round-24 seed is a triangular lattice jittered
   by 0.05 d, and the constant-`P` compaction then closes the gap from `phi = 0.80` to
   `0.874` without ever leaving the crystalline branch. (The same `phi` anomaly is why P4
   fails on the high side, symmetrically to P2's failure on the low side.)
4. **P5 failing is consistent with all of the above,** not an independent defect: on a
   crystal the traction is carried by a handful of slip planes, so the two halves of a 1.0
   strain window differ by many standard errors. The window is too short *for this state*;
   a disordered packing would need re-checking against the same P5 rule.

## ⑤ Note points (pitfalls)

* **A silent empty glob is a wrong answer, not an empty answer.** `load_runs()` matched
  `point_*` while the job wrote `probe_*`; `assemble()` then raised only because the
  directory was empty. Any sweep must be named by the same string in the pre-registration,
  the runner and the analyser — this round adds the gate that the four returned points are
  actually read (`test_returned_sweep_reproduces_the_published_verdict`, `n_points == 4`).
* **`compare_with_3d` had never executed** (no 2-D point had ever existed), so its schema
  drift went unnoticed: the r23 file holds `points` as a dict keyed by `I_target`, with
  `I_bulk_mean` / `mu.mu_full_zz_final` stats blocks. Dead code on the acceptance path is
  the exact place a fabricated verdict would hide; it is now tested.
* **Judging at the measured `I`.** `I_bulk` runs 1.24-1.39x the target (`p_bulk` is 17-24 %
  below `p_wall`), so `I_target = 0.2` lands at `I = 0.246` and is therefore *reported, not
  judged*. Round 24 pre-registered this branch; it is honoured here.
* **`p_bulk < p_wall` is expected, not a defect:** in 2-D the frictional contacts carry a
  contact couple the symmetric Love-Weber virial omits (round 24, ⑤). `mu_wall` is the
  judging ordinate; `mu_bulk = 0.078...0.110` is reported as a lower bound only.
* **Scope of this round.** One seed, four `I` points, the probe cell (458 disks, 15 d gap);
  the 1220-disk production cell and the low-`I` shards (`I = 0.005, 0.01`) were not run.
  The structure test (`run_structure_test.sh`, lattice vs random seed at `I = 0.05, 0.2`)
  returned **locally** before this document was closed; its numbers are in ⑥b. An equivalent
  cluster submission is queued as `cj4129` (same script) for reproducibility.

## ⑥ Error verdict (honest)

| quantity | criterion | measured | versus reference | < 3 %? |
|---|---|---|---|---|
| `mu_wall(I)`, `I = 0.02` | < 10 % (ledger) | 0.0963 | law 0.2461 → **-60.9 %** | **NO** |
| `mu_wall(I)`, `I = 0.05` | < 10 % | 0.1003 | law 0.2831 → **-64.6 %** | **NO** |
| `mu_wall(I)`, `I = 0.10` | < 10 % | 0.1238 | law 0.3593 → **-65.5 %** | **NO** |
| `phi_gap(I)`, `I = 0.02` | < 10 % | 0.873 | law 0.802 → **+8.8 %** | NO (but inside 10 %) |
| `phi_gap(I)`, `I = 0.10` | < 10 % | 0.874 | law 0.768 → **+13.8 %** | **NO** |
| servo calibration `|p_wall/P - 1|` | <= 10 % (P1) | 0.909-1.010 | imposed `P` | no (0.9-9.1 %) |
| device correctness (gates) | 21 tests | 21 passed | — | n/a |

**Does any key quantity come within 3 % of its reference? No.** The ledger tolerance is
10 %, and the primary ordinate misses it by a factor of six (−61 to −66 %). The acceptance
for `CASE-DEM-PLANE-SHEAR-MUIDI` is therefore **not met**, and this round does not claim
otherwise: the 2-D branch is a **FAIL with a diagnosed cause**, not a near miss.

## ⑦ What this changes for the ledger

* **Round 24's hypothesis is retired, not confirmed:** building the law's own dimension did
  not close the +29 % 3-D offset — the 2-D device misses the law by −61...−72 %. The 3-D
  offset remains unexplained and must not be attributed to dimension on this evidence.
* **The blocking defect is named and instrumented:** the round-24 initial condition is a
  crystal, evidenced by `phi_gap = 0.874` (vs `nu_max = 0.81`, hexagonal limit 0.9069) and by
  the frictionless-branch `mu`. `psi6` / `Z` are now measured in every run, and `--init
  random` provides a disordered seed at the same density, so the next round is one command
  away (job `cj4129` already carries it).
* **Ledger acceptance amendment (third request, still outstanding):** the acceptance text
  must name (a) the reference equation, (b) the validity range `I <= 0.2`, (c) the device
  dimension (2-D disks), and (d) the **initial-condition class** (disordered packing, with
  `psi6` reported), because a crystalline cell is not the same measurement as da Cruz's.
* **Recommended next steps:** (1) drain `cj4129`; if `--init random` raises `mu` towards
  `0.22 + I` and drops `phi` towards 0.81, the defect is confirmed and the 2-D branch is
  repairable; (2) if the random seed re-crystallises during compaction, add bidispersity
  (the archived paper states the friction law is preserved up to 50 % polydispersity) —
  `plane_shear_2d.py` already carries per-particle radii; (3) only then run the production
  cell (1220 disks, 3 seeds) and resume the 3-D-vs-2-D comparison.

## ⑥b Structure test: the crystal is produced by compaction, not only by the seed

`run_structure_test.sh` — same probe cell, `I = 0.05` and `0.2`, one seed, round-24 lattice
seed vs the new disordered seed (`--init random`, identical area fraction), end-of-window
structure from `hexatic_order()`:

| I_target | init | mu_wall | phi_gap | psi6 | hexatic frac (psi6 > 0.8) | Z | I measured | p_wall/P |
|---|---|---|---|---|---|---|---|---|
| 0.05 | lattice | 0.1003 | 0.8754 | 0.914 | 0.85 | 4.03 | 0.0631 | 0.994 |
| 0.05 | **random** | **0.0593** | 0.8753 | 0.914 | 0.85 | 4.28 | 0.0669 | 0.967 |
| 0.20 | lattice | 0.1294 | 0.8747 | 0.931 | 0.88 | 4.05 | 0.2464 | 1.010 |
| 0.20 | **random** | **0.1156** | 0.8733 | 0.957 | 0.93 | 3.97 | 0.2383 | 1.139 |

**The repair hypothesis of step 4 is falsified.** A disordered seed does not survive: after
the 40 `d/sqrt(P/rho)` constant-`P` compaction the packing is *more* hexatically ordered than
the lattice seed (`psi6 = 0.914 / 0.957`, hexatic fraction up to 0.93) and `mu` does not rise
— it *falls* (`0.100 -> 0.059` at `I = 0.05`). So the crystal at `phi = 0.874` is what the
compaction of **monodisperse** disks at `P d / kn ~ 1e-3` produces, not a seeding artefact:
slow densification of an equal-sized disk packing crystallises. Breaking it requires a
*physical* symmetry break — polydispersity (the archived paper states the friction law is
preserved up to 50 % polydispersity; `plane_shear_2d.py` already carries per-particle radii
and only `_lattice_packing`/`_random_packing` return a constant radius) — not a different
initial condition. `Z ~ 4.0` in every case is also a defect indicator: an isostatic 2-D
frictional packing sits near `Z = 4` *at jamming*, but these cells are being sheared, where
`Z = 4` with `psi6 ~ 0.93` means the contact network is a near-defect-free lattice.
