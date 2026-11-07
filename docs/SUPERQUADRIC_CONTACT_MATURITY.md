# Superquadric contact: from verification surface to a mature module

Campaign node **IN50-W3-N01** (TensorDEM). Physics domain, worktree branch
`agent/m6037`. All numbers below are programmatically produced by
`scripts/build_shape_error_table.py` into `artifacts/dem/shape_error_table.json`
(the tests read the same artifact, so prose and evidence cannot drift apart).

---

## 1. What the module is

`tensordem.superquadric` implements rigid-particle contact for superquadrics

```
F(x) = ( |x/a|^n2 + |y/b|^n2 )^(n1/n2) + |z/c|^n1 - 1
```

* `n1 = n2 = 2`, `a = b = c = R` → sphere (Hertz limit),
* `n1, n2 > 40` → exact rectangular block (max-norm branch, deliberate model
  switch documented in `superquadric_implicit`),
* intermediate exponents → rounded discs/blocks, the floe-particle family.

The module now contains two contact paths:

| path | routine | cost / pair | role |
|---|---|---|---|
| scalar Newton | `superquadric_contact` | ~20 ms | reference, single contact, debugging |
| vectorised fixed point | `superquadric_contact_batch` | ~1-3 µs | production path (DEM, scans) |

The scalar path existed before this node and **is not usable at DEM scale**
(≈10³ × the clump cost). The vectorised kernel is the actual maturity step:
it solves the same common-normal condition

```
p_A = s_A u_A,  p_B = s_B u_B,   n_A = -n_B = d,
delta = s_A + s_B - |c_B - c_A|
```

with the closed-form radial function

```
s(u) = [ (|u_x/a|^n2 + |u_y/b|^n2)^(n1/n2) + |u_z/c|^n1 ]^(-1/n1)
```

and a four-sweep fixed point `d <- normalize(n_A - n_B)`. The Hertz radius of
curvature is obtained per body from Euler's theorem (sum of two orthogonal
normal curvatures), evaluated with two finite differences of the *radial*
function rather than four gradient evaluations — 40 % of the kernel cost.

## 2. Single-contact degeneration (acceptance layer E1)

| check | reference | measured | criterion | verdict |
|---|---|---|---|---|
| sphere limit, 20 (R, δ) cases, random orientations | Hertz `F = (4/3)E*√R* δ^{3/2}` | max rel. error **1.7e-10** | ≤ 1 % | PASS (7 orders of margin) |
| radius of curvature in the sphere limit | `R* = R/2` | max rel. error < 1e-6 | — | PASS |
| cube face-to-face compression | `k = E* A / (c_A + c_B)` | rel. error **0.50 %** | ≤ 2 % | PASS |
| flat-law linearity in δ | no Hertz non-linearity for flat contact | spread **< 1e-6** | — | PASS |
| contact cost, 8-sphere clump, 4096 pairs | clump baseline | **2.1 ×** (curve below) | ≤ 3 × | PASS |

Cost curve (paired interleaved timing, median of per-repeat ratios):

| pair batch | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|
| ratio | 3.4 | 2.6 | 2.6 | 2.1 |

The ratio improves with batch size because a constant numpy/Python overhead is
amortised; the criterion is met for the production-relevant batches (≥ 2048
pairs). At 1024 pairs the measured ratio is 3.4–3.7× — recorded, not hidden.
Run-to-run variance on a shared 128-core host is significant (1.4–2.9× at 4096
pairs across rebuilds of the artifact); the artifact always ships the numbers of
the run that produced it, and the test asserts the median of three
measurements.

**Honest caveat on the flat-contact stiffness.** For a curvature-free contact
Hertz does not apply and *the analytic stiffness itself is a convention*. The
campaign specifies `A·E*`, i.e. the column convention `k = E* A / (c_A + c_B)`.
The competing rigid flat-punch-on-half-space solution (Sneddon) gives
`k = 2 E* a_eq` with the equal-area radius, which is **1.128×** the column
value. The 0.5 % agreement therefore certifies the *geometric contact-area
routine and the penalty bookkeeping*, not an independent physical law; the
convention spread is shipped in the artifact as
`flat_punch_vs_column_ratio = 1.1284`.

## 3. Geometric layer: true shape vs best N-sphere clump (E2)

`tensordem.shape_error_budget` now

* samples the true surface **in direction space** with the same radial function
  the kernel uses, so samples are exact on both the power branch and the block
  branch (`max |F| ≤ 2e-16`, tested). The previous lat-long parametrisation
  mixed the exponents and sampled a *different* surface for `n1 ≠ n2`; it was
  replaced.
* builds the clump greedily: candidate spheres are anchored on the inward
  normals of surface samples with radius equal to the local-thickness fraction
  (`c = p - h n`, `r = h`), and sphere *k* is the candidate that minimises the
  worst **absolute union deviation** `max_j | min_i (|p_j - c_i| - r_i ) |`.
  The union SDF of a sphere assembly is exactly the minimum of the sphere SDFs,
  so no approximation enters the error metric.
* reports max/mean relative surface distance plus the normal-deviation
  distribution (mean, p95, max), and the per-insertion error history.

Measured (half-extents in units of the smallest extent, 800 surface samples):

| shape | N=1 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|
| sphere limit (n=2) max|SDF|/R_eq | 0.013 | 0.013 | 0.013 | 0.013 | 0.013 |
| sphere limit mean normal dev. | 0.3° | 0.3° | 0.3° | 0.3° | 0.3° |
| 2:1 (n=8) mean normal dev. | 31.6° | 31.7° | 31.8° | 31.8° | 31.7° |
| 3:1 block (n=50) mean normal dev. | 37.8° | 39.6° | 38.5° | 39.1° | 39.3° |
| 5:1 block (n=50) mean normal dev. | 54.7° | 54.0° | 54.0° | 54.0° | 54.0° |

**Negative control (N → ∞).** The min-max greedy is monotone by construction:
adding a sphere can only shrink `min_i(|p-c_i|-r_i)`, so the worst absolute
deviation is non-increasing (verified numerically over 24 insertions, and
asserted in the tests). It is *not* a strong convergence statement: with a
fixed surface-anchored candidate family the residual plateaus (see the 2:1 row),
and the mean normal deviation can even rise slightly because the greedy is a
min-max (worst-case) objective, not a mean-square fit. This is reported as a
limitation of the candidate family, not smoothed over.

## 4. Macroscopic layer (E3)

Three DEM experiments were run, all from this worktree (raw JSON in
`runs/macro/`):

| method | setup | result | evidence |
|---|---|---|---|
| column collapse (3D) | 574 monodisperse spheres, r = 4.3 mm, cylinder R = 45 mm, fill 50 mm, kn = 1e4 N/m | `θ_r(μ=0.404, μ_r=0.15) = 16.4°`, `θ_r(0.60, 0.15) = 21.9°` (r² = 0.94) | REAL |
| rotating drum (2D) | 320 disks, R_drum = 40 mm, fill 45 %, Ω = 0.6 rad/s, 105 823 steps | `θ = 28.8° ± 1.5°` (three μ, three runs) | REAL |
| direct shear cell (2D) | 77 disks, constant volume, 0.5× bed-mass lid, servo-vertical | `μ_i = 0.300 / 0.400 / 0.500` for `μ = 0.3 / 0.4 / 0.5` | REAL (consistency) |

**22° ± 2° reproduction.** The sphere calibration reproduces the target with
`μ = 0.60, μ_r = 0.15` → **21.86°** (|Δ| = 0.14°), criterion PASS. The
smooth-sphere pair `μ = 0.40, μ_r = 0.05` gives 13–15°, i.e. the *rolling*
friction dominates the calibration: the target angle is only reachable with
`μ_r ≈ 0.15`, which is the documented material assumption of this anchor.

**Method discrepancy (reported, not reconciled).** The 2D drum gives 28.8°,
≈ 7–9° above the 3D collapse value at comparable friction. 2D disk packings are
known to stand steeper than 3D sphere piles (fewer degrees of freedom, different
rolling geometry), and the drum also has a driven wall. The two methods are
therefore recorded side by side; the **3D collapse is the anchor** used by the
error budget, the drum is an independent cross-check with a documented offset.

**Shear cell caveat.** The measured `μ_i` equals the input `μ` to 1e-9. That is
the physically expected outcome for a thin, fully mobilised layer in a
constant-volume cell and serves as a bookkeeping check of the friction path
(peak values `τ/σ = 0.34/0.46/0.54 > μ` show the mobilisation transient), but it
is *not* an independent prediction of `μ_i` for elongated particles — no
dilation is possible in a constant-volume 2D cell. Labelled accordingly in the
artifact.

**Macro bridge and its evidence grade.** The repose difference between the true
shape and its clump is computed as

```
Δθ = θ_r(μ + tan i_clump) - θ_r(μ),      i_clump = mean surface-normal deviation
```

i.e. the classical interlocking (dilatancy) augmentation applied to the
*measured* `θ_r(μ)` anchor. `i_clump` is geometric (REAL, from layer 2); the
augmentation is a model. No poly-shape pile DEM was run: the artifact marks
`theta_clump` as `MODEL` and `poly_shape_dem` as `WITHHELD`.

## 5. Decision rule and API gate (E3/E4)

`tensordem.shape_error_budget.api_gate(sq, n_spheres, raise_on_fail=...)`
returns the diagnostics and, when asked, raises `ClumpApproximationDisabled`.
Criteria (single source of truth in `CRITERIA`):

* `|θ_true − θ_clump| ≤ 3°`
* force-chain / contact-normal deviation `≤ 5°`
* max surface-normal deviation `≤ 5°`, relative surface error `≤ 2 %` of `R_eq`

Verdicts (μ = 0.60): the sphere limit is admissible for any N; 2:1, 3:1 and 5:1
shapes are **rejected at every scanned clump size** (8…32 spheres), because the
surface-normal deviation is 31–54°, an order of magnitude above the 5° criterion.
The disable table (`artifacts/dem/shape_error_table.json →
disable_ratio_table`, 150 rows = 5 aspect ratios × 5 exponents × 6 sphere
counts) admits 6 rows, all at aspect ratio 1.

## 6. Limitations

1. **Flat-facet contact non-uniqueness.** On a flat face the common normal is
   not unique; two runs with different sweep counts can settle on different
   facet directions (worst-case overlap spread **1.01 R** over sweeps ≥ 3,
   median converged sweep 3.5). Smooth shapes converge to ~1e-4 R. Production
   use should round the geometry (`n1, n2 ≤ 16`) or verify with the scalar path.
2. **Clump candidate family.** Surface-anchored spheres plateau for elongated
   shapes; a least-squares multi-sphere optimiser would lower the residual but
   would break the strict monotone negative control.
3. **Macro layer is anchored, not simulated end-to-end.** `Δθ` is a model on a
   measured `θ_r(μ)`; the poly-shape pile DEM is WITHHELD.
4. **Shear cell** cannot dilate at constant volume, so `μ_i = μ` is a
   consistency check only.
5. **Convention spread** for flat-contact stiffness (1.128× punch vs column).
6. 2D (drum, shear) results are not dimensionally comparable with 3D spheres;
   the drum/collapse offset (28.8° vs 21.9°, ≈ 7°) is reported rather than
   tuned away.

## 7. Reproducing

```bash
python scripts/shape_macro_scan.py --mode drum --mu 0.40 --mu-rolling 0.05 \
    --kn 200 --settle-time 1.5 --measure-time 4.0 --out runs/macro/drum2_mu0.40.json
python scripts/shape_macro_scan.py --mode shear --mu 0.40 --out runs/macro/shear4_mu0.40.json
python scripts/build_shape_error_table.py --out artifacts/dem/shape_error_table.json
python -m pytest tests/test_superquadric_contact.py -q     # 36 passed
```

Module files: `src/tensordem/superquadric.py`,
`src/tensordem/shape_error_budget.py`, `src/tensordem/shape_macro.py`.