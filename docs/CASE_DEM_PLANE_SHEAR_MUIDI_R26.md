# CASE-DEM-PLANE-SHEAR-MUIDI — round 26: polydispersity as the physical symmetry break

**Mission** m22550 · **repo** TensorDEM · **branch** `agent/m22550`
**Anchor** `CASE-DEM-PLANE-SHEAR-MUIDI` (status `open`, acceptance: `mu` and `phi` vs
`I` within 10 % of da Cruz et al. 2005 PRE 72:021309 / GDR MiDi 2004, `I = 1e-3 ... 0.3`)

## ① Where round 25 left the case

Round 25 (`docs/CASE_DEM_PLANE_SHEAR_MUIDI_R25.md`, m22440) returned an **honest FAIL**
with a *diagnosed* cause, not a near miss:

* the 2-D disk device measured `mu_wall = 0.096 ... 0.129` against the law's
  `0.246 ... 0.359` (`-61 ... -66 %`), while `phi_gap = 0.874` sat 8 % above the
  archived `nu_max = 0.81` and 4 % below the hexagonal limit `pi/sqrt(12) = 0.9069`;
* the cause was named and instrumented: `psi6 = 0.914`, hexatic fraction 0.85,
  `Z ~ 4.0` — the bed is a **crystal**, and a crystal shears by basal slip on the
  frictionless branch of the law;
* **a disordered seed does not survive compaction.** `--init random` produced an
  equally crystalline bed (`psi6 = 0.914 / 0.957`) and `mu` *fell* (0.100 → 0.059).
  Round 25 therefore falsified the initial-condition hypothesis and concluded that
  breaking the crystal needs a **physical** symmetry break — polydispersity — not a
  different seed. That is round 25 recommendation (2), and it is this round.

## ② What was done

`benchmarks/plane_shear_mu_i/plane_shear_2d.py` gained a size-spread axis, plus three
device defects that only become visible once radii vary were fixed.

### 2.1 The new axis (`--polydispersity w`)

A bounded uniform draw `r_i = radius (1 + w u_i)`, `u_i ~ U(-1, 1)`, **rescaled to its
own arithmetic mean**, so `mean(r) = case.radius` exactly. This matters because the
ordinates are defined through the nominal radius:

| ordinate | definition | consequence of the invariance |
|---|---|---|
| `I` | `gammadot d / sqrt(P/rho)`, `d = 2 radius` | the inertial number keeps its meaning |
| `gap_eff` | `h - 2 radius` | `h_eff`, and hence `phi_gap`, stay comparable |
| `phi_cell` | free-disk area / `W (h_eff + 2 radius)` | idem |

`w = 0.5` is the widest spread for which the archived law is stated to hold
(`r_max/r_min = 3`), and `validate()` refuses anything outside `[0, 0.5]`. Because the
mean radius is fixed, the *area* grows by `<r^2>/r^2 = 1 + w^2/3`; the seed box is
opened by exactly that factor so the seeded state keeps a reachable band fraction
(`PHI_SEED = 0.78`, below the 2-D random close packing ~0.84), and the constant-P
servo — not the seed density — then sets the density under test.

### 2.2 Device defects fixed (all three are real, and none is cosmetic)

1. **`hexatic_order` / `_relax_overlaps` accepted only a scalar radius.** Both now take
   a scalar *or* a per-disk array; the neighbour shell and the contact distance become
   the per-pair `shell (r_i + r_j)` and `r_i + r_j`. With equal radii these reduce
   exactly to the old expressions, and the monodisperse arithmetic is kept bit for bit
   (a dedicated legacy branch), so rounds 24/25 numbers remain reproducible.
2. **The seed acceptance test was blind to periodic images.** `_relax_overlaps` built a
   *non-periodic* k-d tree on the final check while the relaxation itself wrapped `dx`.
   A pair straddling `x = 0` was therefore invisible: the round-25 monodisperse probe
   seed in fact carried **11 overlapping pairs across the periodic boundary** and was
   certified clean, so the engine started from an overlapping state. The wrapped test is
   now the loop's *convergence criterion*, and an unclean seed is an error.
3. **The wall margin assumed a uniform radius.** Free disks were clipped to
   `wall_centre +/- 2r`, which is only correct when every disk has radius `r`. With a
   size spread a large free disk could start in contact with a wall grain. The margin is
   now `r_wall + r_i` (for `w = 0` this is exactly the old `3r` / `gap + r` window).
4. `phi_gap` used `n pi <r>^2`; with variable radii it now sums the actual `pi r_i^2`.

Tests: `tests/test_plane_shear_2d_r26_polydispersity.py` (15 cases) pins the draw
contract (mean invariance, boundedness, size-contrast bound, stream decorrelation), the
`[0, 0.5]` validation, the periodic-image fix, overlap-freeness of the production-size
seed at `w = 0 / 0.25 / 0.5`, and scalar/array equivalence of `hexatic_order`.
**36 tests pass** (15 new + 21 rounds-24/25 regressions).

## ③ Results available so far

### ③a Seed structure — the crystal is gone (measured, no dynamics)

Probe cell (458 disks, `W = 0.30 m`, `gap = 0.15 m`, `P = 10 kPa`, `mu_p = 0.4`,
disordered seed, `seed = 0`), initial configuration only
(`outputs/r26/r26_seed_structure.json`, figure `docs/figures/r26_seed_structure.png`):

| `w` | `r_max/r_min` | band `phi` (seed) | `psi6` | hexatic frac | `Z` |
|---|---|---|---|---|---|
| 0.00 | 1.00 | 0.799 | **0.886** | 0.819 | 2.34 |
| 0.25 | 1.67 | 0.733 | **0.447** | 0.114 | 2.76 |
| 0.50 | 3.00 | 0.734 | **0.390** | 0.079 | 3.14 |

The monodisperse seed is already crystalline at `psi6 = 0.886` (consistent with round
25's post-compaction 0.914), and the size spread destroys it: hexatic fraction
0.819 → 0.114 → 0.079. `Z` here is a *seed* value (the bed is loose, not yet
compacted) and is not the isostatic coordination of the sheared state.

### ③b `mu(I)` and `phi(I)` after compaction — **not_run (queued)**

The 9-point sweep (`I in {0.02, 0.05, 0.2}` x `w in {0, 0.25, 0.5}`, script
`benchmarks/plane_shear_mu_i/run_polydispersity_r26.sh`) was submitted to the cluster as
**`cj4227`** and had not drained when this document closed (queue position 56, ~55 jobs
ahead). `benchmarks/plane_shear_mu_i/analyse_polydispersity_r26.py` reads
`runs/plane_shear_2d/poly_w*_I*.json` and emits the `mu`/`phi` table with
`reference_law.ref_mu/ref_phi` and the relative error per point — one command after the
job drains, no code change.

**This round therefore does not claim the ledger acceptance is met, nor that it is
refuted.** What it establishes is that the mechanism round 25 recommended now exists,
is tested, and — at the seed level — does what it was supposed to do.

## ④ Error / evidence verdict (honest)

| quantity | criterion | this round | verdict |
|---|---|---|---|
| seed overlap-freeness (incl. periodic images) | 0 pairs | 0 pairs, `w = 0/0.25/0.5` | **PASS** (test) |
| mean-radius invariance of the draw | exact | `|mean/r - 1| < 1e-15` | **PASS** (test) |
| size-contrast bound at `w = 0.5` | `<= 3.0` | 2.995 | **PASS** |
| seed disorder (hexatic fraction) | `< 0.5` | 0.079 at `w = 0.5` | **PASS** |
| `mu(I)` within 10 % of the law | < 10 % | **not_run** (`cj4227` queued) | **WITHHELD** |
| `phi(I)` within 10 % of the law | < 10 % | **not_run** (`cj4227` queued) | **WITHHELD** |

**Is any key transport quantity within 3 % of its reference? No claim is made**: the only
quantities measured here are device/seed properties with exact criteria, and the
`mu`/`phi` ordinates were not computed.

## ⑤ Notes, scope, limits

* **The previous revision's certification was wrong in a way that mattered.** The
  monodisperse probe seed passed the old acceptance test with 11 periodic-boundary
  overlaps; rounds 24/25 numbers were produced from such seeds. The compaction (40
  `d/sqrt(P/rho)`) largely absorbs an overlap this small, but the certified-clean claim
  was not true, and the fix changes the monodisperse seed positions slightly.
* **Seed density is not a control variable here.** The polydisperse cell starts looser
  (band fraction 0.78 vs 0.86) by necessity — at fixed mean radius the area grows by
  `1 + w^2/3` and the monodisperse box would exceed the 2-D close packing. What is
  controlled is `P`, through the servo; `I` is reported as *measured*.
* **One seed, one cell, three size spreads.** No ensemble, no 1220-disk production cell,
  and `I = 0.005 / 0.01` (below the probe cell's resolution) are not run. A size trend
  from three points on one seed is a hypothesis, not a calibration.
* **`phi` provenance is still unverified** (`reference_law.PHI_REF_STATUS`); the `phi`
  acceptance of the ledger was already carried as WITHHELD in rounds 13+ and this round
  does not change that.
* The 2-D branch remains a device-level investigation; it does not touch the 3-D
  `CASE-DEM-PLANE-SHEAR-MUIDI` 3-D offset (+29 %), which round 24's hypothesis failed to
  explain and which stays unexplained.

## ⑥ Next step (one command, already queued)

Drain `cj4227`; then
`python3 benchmarks/plane_shear_mu_i/analyse_polydispersity_r26.py`.
If `mu_wall` at `w = 0.5` approaches `0.22 + I` and `phi` drops towards the reference,
the crystal diagnosis is confirmed *and* the 2-D device is repairable in the direction
round 25 predicted; if `mu` stays on the frictionless branch with `psi6 < 0.5`, then the
crystal was **not** the whole story and the wall slip / stress-path terms must be
re-examined before any ledger amendment is requested.

## ⑦ Ledger

Acceptance text still needs the four amendments round 25 requested (reference equation,
validity range `I <= 0.2`, device dimension, initial-condition class) **plus** the size
spread, because a monodisperse and a 50 %-polydisperse cell are different measurements
of the same law.
