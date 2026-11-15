# CASE-DEM-PLANE-SHEAR-MUIDI — round 24: the law's own dimension (2-D disk plane shear)

**Ledger:** `CASE-DEM-PLANE-SHEAR-MUIDI` — 算例池·平面剪切 mu(I) 流变律标定.
**Reference (archived, read from the archive at run time, never retyped):**
da Cruz, Emam, Prochnow, Roux, Chevoir 2005 Phys. Rev. E 72:021309 — `mu = 0.22 + 1.00 I`,
`phi = 0.81 - 0.30 I`, valid for `I <= 0.2` (`reference_archive.archived_law()`).
**New device:** `benchmarks/plane_shear_mu_i/plane_shear_2d.py` (2-D monodisperse disks,
rough grain-glued walls, constant confining stress, prescribed shear rate).
**Pre-registration:** `benchmarks/plane_shear_mu_i/prereg_r24.json` (written before the runs;
five predictions P1-P5 and the decision rule).
**Assembly:** `benchmarks/plane_shear_mu_i/analyse_2d_r24.py` (tables, figures, verdict).
**Gates:** `tests/test_plane_shear_2d_r24.py` — 16 passed.
**Production sweep:** cluster job `cj4011` — **still queued when this document was written**
(the honest status of the measurement is therefore *not judged*: see ⑥).

---

## ① Case introduction

Rounds 1-23 measured `mu(I)` on a **3-D sphere** cell (294 spheres, `0.12 x 0.05 x 0.06` m,
`P = 62.5` kPa, servo platen) and judged it against a **2-D disk** law. Round 23 turned the
verdict into an ensemble verdict (6 seeds per point, `cj3846`): the +29 % offset at
`I = 0.03` is *not* one lucky realisation (`CV` up to 20.6 %), the "dip" between `I = 0.03`
and `0.1` is not systematic, and a contact-friction ladder `mu_p = 0.2 ... 0.7` does not
reach the 10 % band. That left exactly one candidate mechanism for the residual: **the
dimension of the device**.

The physics is unambiguous: the quasi-static effective friction of a frictional packing is
a function of `mu_p` *and* of the packing geometry; a 3-D sphere packing at `mu_p = 0.5`
has a larger `mu_s` than a 2-D disk packing at `mu_p = 0.4`, which is the friction the
reference used. Rounds 1-23 measured one and judged against the other, and no archived
3-D triple exists in the repository to separate the two.

Round 23 recommendation (2) was "archive a 3-D `mu(I)` paper". This round takes the other,
cheaper and falsifiable branch of the same question: **build the device the law was
measured on**. If the 2-D device lands inside the 10 % band, the 3-D offset is
dimensional and the ledger acceptance is passable on a commensurable device; if the 2-D
device reproduces the same +30 % offset, the dimensionality hypothesis is falsified and the
residual sits in the contact model or the estimator.

Parameters (all SI; `thickness = d` so the reported stress is a true Pa and the 2-D
inertial scaling `I = gammadot d / sqrt(P/rho)` is exact):

| quantity | value | source |
|---|---|---|
| free disks | ~1220 monodisperse, `d = 10` mm | this round |
| cell | `W = 0.40` m (40 d) periodic in `x`, gap `0.30` m (30 d) | this round |
| confining stress `P` | `1.0e4` Pa (constant-stress servo) | this round |
| contact stiffness `kn` | `1.0e5` N/m (`P d / kn ~ 1e-3`: rigid-grain regime) | this round |
| particle friction `mu_p` | 0.40 | da Cruz 2005 (2-D disks) |
| rolling resistance | 0 | disks, reference setup |
| restitution/damping | `zeta = 0.2` (engine default) | `tensordem.granular2d.DiskConfig` |
| `I` judged | 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 | ledger range ∩ law validity |
| `I` reported, not judged | 0.30 | outside `I_sat = 0.2` |

## ② What was done

1. **The device** (`plane_shear_2d.py`). Two **pinned rows of disks of the same diameter**,
   spaced `d (1+1e-3)` along `x`: that is what makes the wall *rough* — a free disk can never
   squeeze between wall grains, so wall slip is not a free parameter of the measurement. The
   bottom row moves at `+V/2`, the top row at `-V/2`; the top row is also a **servo in `y`**
   that holds the measured wall normal stress at `P`. The pinned rows are integrated out
   (`advance()` integrates the free degrees of freedom only and re-imposes the wall
   kinematics), which is the exact discrete counterpart of "glued wall grains".
2. **Contact physics is not re-implemented.** The 2-D engine of the repository
   (`tensordem.granular2d`: Hertz/linear normal + Cundall-Strack tangential with one-sided
   history per contact key + optional rolling resistance, `cKDTree` neighbour lists,
   deterministic ordering) does the pair loop; this round only adds the pinned-row
   kinematics, the servo and the measurement.
3. **Protocol fixed before the runs** (`prereg_r24.json`): compaction at constant `P` with the
   rows held still, then a shear transient of 0.5 strain, then a measurement window of 1.0
   strain. Every point is sampled and reduced with a first-half/second-half stationarity
   statistic.
4. **Ordinates, decided before the runs and for a stated reason.** `mu_wall` = `|F_t|/|F_n|`
   summed per row and averaged (the two rows carry equal and opposite tractions). It is the
   primary ordinate because the wall grains are *particles*: no wall load is lost in a
   truncation and no wall-slip law is assumed. `mu_bulk` (symmetric part of the Love-Weber
   tensor) is reported as a consistency check only — in 2-D the friction contacts carry a
   contact couple that the symmetric virial does not contain, so `mu_bulk` is not the
   momentum-consistent traction (this is a real finding of this round, see ⑤).
   `phi_gap` (free-disk area over the surface-to-surface control volume) is the primary
   `phi` and `phi_cell` (centre-to-centre, half wall layers included) the secondary one:
   round 23's WITHHELD `phi` verdict was a definition mismatch, and the definition is now
   fixed *before* the run instead of being chosen after seeing the numbers.
5. **Sixteen gate tests** (`tests/test_plane_shear_2d_r24.py`) pin what the verdict depends
   on: the pinned rows are blocking; the free packing sits inside the gap; the `I` definition
   is invertible and the wall speed reproduces it; the rows do not move and the free `x`
   stays periodic; Newton's third law holds exactly and the rows are loaded in compression;
   overlaps stay inside the rigid-grain bound; the servo closes the gap when `P` is raised;
   the same seed is bit-identical and a different seed is not; the pre-registration matches
   the code; every judged `I` is inside the archived validity and an extrapolation is never
   judgeable; the reference constants come from the archive.
6. **Assembly** (`analyse_2d_r24.py`): per-`I` seed ensemble with sample sd, `CV`, standard
   error and a two-sided 95 % Student-t interval; error against the archived law at the
   *measured* `I`; a paired comparison with the 3-D device at the nearest `I`
   (`runs/seed_ensemble/analysis_r23.json`); tables and three figures.

## ③ Results

### What is complete

| item | status |
|---|---|
| 2-D disk plane-shear device | **delivered** (`plane_shear_2d.py`, 460 lines incl. protocol) |
| pre-registration P1-P5 + decision rule | **delivered** (`prereg_r24.json`), written before the runs |
| gates | **16/16 passed** (`pytest tests/test_plane_shear_2d_r24.py`, 47 s) |
| estimator calibration on a smoke cell (probe, 458 disks, `I = 0.1`) | `p_bulk = 7.9e3` Pa, `sigma_yy/P = 0.90`, `phi_gap = 0.83`, `ke_settle = 5.3e-4` J |
| production sweep (6 `I` x 3 seeds) | cluster job `cj4011` — **queued, not started** at the time of writing |

### What is not complete

The `mu(I)` / `phi(I)` numbers of the production sweep. The job was submitted to the
scheduler (1 core, 2400 s, stage dir = this worktree, artifacts to
`benchmarks/plane_shear_mu_i/runs/plane_shear_2d/probe_I*.json`) and was still at the head
of a 30-deep queue when this document was closed. **No number is invented here for it**: the
tables and figures are produced by `analyse_2d_r24.py` from the run JSONs and are empty
until those files exist.

## ④ Figures

The device-dependent figures (`fig_r24_muI_2d.png`, `fig_r24_phiI_2d.png`,
`fig_r24_dimensionality.png`) are generated by `analyse_2d_r24.py` from the run artifacts;
they cannot be drawn before the sweep returns, and drawing them from a placeholder would be
a fabricated result. The report attached to this round therefore carries **no
result figure**, for the stated reason (no measured data yet), and the three figure
recipes are committed as code so that the round can be completed by re-running one script.

## ⑤ Note points (pitfalls found in this round)

* **A 2-D virial is not a traction.** The first version of the measurement used
  `mu_bulk = 0.5(|sigma_xy| + |sigma_yx|)/p` as the primary ordinate (the round-8 choice in
  3-D). On the probe it gave `mu_bulk = 0.054` where the walls transmitted `0.096`: the
  symmetric part of the Love-Weber tensor omits the contact couple that frictional disks
  carry, so it does not balance the wall traction. The primary ordinate was moved to the
  wall traction *before* the production runs, and `mu_bulk` is kept as a reported
  consistency check. Any future round that judges on a 2-D bulk virial must first add the
  couple-stress term.
* **Row sums are not the cell load.** The two pinned rows carry equal and opposite
  tractions, so `sum over all pinned grains` of `F_y` is (to numerical noise) zero. The wall
  stress is measured per row and averaged; a naive sum over both rows reports noise around
  zero and silently destroys the measurement (observed on the first smoke run).
* **No gravity means the servo is the only load path.** A constant-`P` cell in zero gravity
  needs a compaction stage before the shear: the initial lattice is not in equilibrium and
  the servo has to close the gap first. The protocol therefore starts with a quiescent
  compaction whose length is expressed in `d/sqrt(P/rho)` units, and `ke_settle` is reported
  as the quiescence evidence.
* **Cost is set by the low-`I` end.** The run length in steps is `strain/gammadot/dt` with
  `gammadot prop I`, so `I = 0.005` costs ~30x the steps of `I = 0.2`. The sweep is therefore
  submitted as shards by `I`, not by seed.
* **Scope.** Monodisperse disks, `mu_r = 0`. The 2-D monodisperse system can order
  (crystallise) at high `phi`; the measurement window is 1.0 strain of shear after a 0.5
  strain transient, and the stationarity statistic (`drift_z`) is the pre-registered guard
  against quoting a transient.

## ⑥ Error verdict (honest)

| quantity | criterion | measured | verdict |
|---|---|---|---|
| `mu(I)` vs archived law, `I <= 0.2` | < 10 % (ledger) | production sweep queued (`cj4011`) | **NOT JUDGED — WITHHELD** |
| `phi(I)` vs `0.81 - 0.30 I` | < 10 % | same | **NOT JUDGED — WITHHELD** |
| estimator calibration `|p_wall/P - 1|` | <= 10 % (P1) | probe cell only: 0.80 (`I = 0.1`, pre-registered servo still converging over a short probe) | **WITHHELD** |
| device correctness | 16 gate tests | 16 passed | **PASS** |
| key quantity within 3 % of a reference | < 3 % | no measured `mu` exists yet | **No claim made** |

**Does any key quantity come within 3 % of its reference in this round? No claim is made,
because no production number exists yet.** The device, the pre-registration, the ordinate
choice and the gates are the deliverable of this round; the measurement is one queued job
away, and reporting a number for it now would be fabrication.

## ⑦ What this changes for the ledger

* The dimensionality hypothesis is now **executable**: `plane_shear_2d.py` +
  `analyse_2d_r24.py` produce the 2-D `mu(I)` and `phi(I)` and print the paired 3-D/2-D
  comparison (prediction P3), so the round that follows does not have to build a device, only
  to run one script and read the verdict.
* Two measurement defects of the 3-D line are avoided by construction here: `phi` has a
  single, pre-registered definition, and the wall traction is complete (no slab truncation,
  no slip law).
* The ledger acceptance text still needs the amendment round 23 requested (name the source
  equation, branch on `I = 0.3`); round 24 does not change that, and adds that the *device
  dimension* must also be named in the acceptance, since a 3-D cell and a 2-D law are not
  the same measurement.

Recommended next steps: (1) drain `cj4011`, run `analyse_2d_r24.py`, and issue the round-25
verdict (P1-P5 are already fixed, so no tolerance can be re-negotiated); (2) if P2 fails with
the 3-D-like offset, add the couple-stress term to the 2-D stress estimator before believing
the failure — ⑤ shows the symmetric virial is not the traction; (3) amend the ledger
acceptance to name the reference equation, the validity `I <= 0.2`, and the device dimension.