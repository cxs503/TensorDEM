# Non-spherical particle representation: when may a sphere clump replace the true shape?

*Campaign*: P27W2-N07 · *Repository*: TensorDEM · *Node*: analytic contact anchors +
clump convergence + representation verdict
*Artefacts*: `artifacts/shape_representation/*.json`, figures `docs/figures/nonspherical_*.png`
*Reproduce*:

```bash
PYTHONPATH=src python scripts/shape_representation_campaign.py hertz
PYTHONPATH=src python scripts/shape_representation_campaign.py geom
PYTHONPATH=src python scripts/shape_representation_campaign.py dissipation
PYTHONPATH=src python scripts/shape_representation_campaign.py verdict
PYTHONPATH=src python -m pytest tests/test_shape_representation_verdict.py -q
```

---

## 1. Case introduction

A DEM of broken ice, rubble or bulk cargo must choose how to carry a non-spherical
particle. Two representations compete:

* the **true shape** — a superellipsoid `|x/a|^m + |y/b|^m + |z/c|^m = 1`
  (`src/tensordem/superquadric.py`), continuous between the ellipsoid (`m=2`) and
  the rectangular block (`m→∞`);
* the **N-sphere clump** — a rigid union of `N` spheres, cheap because every
  contact is analytic.

The engineering question is quantitative: *at which shape factor, for which use,
and with what error bound is the clump admissible?* Before this node the
repository had a superquadric contact kernel with a curvature-based Hertz radius
and a flat-facet (Boussinesq punch) branch, plus a clump error table — but no
**closed-form stiffness anchor for the ellipsoidal pole**, no measured convergence
of the clump representation in `N`, and no statement about the *contact law* that
a clump implies.

| quantity | symbol | value | note |
|---|---|---|---|
| particle | `a=b=c` | 4.0 mm | superellipsoid half-extents |
| shape exponent | `m` | 2, 3, 4, 8 | 2 = ellipsoid, 8 ≈ rounded cube |
| reduced modulus | `E*` | 1.1 GPa (contact law) / 1.1 MPa (DEM) | DEM value = model material used in the repo campaign |
| density | `ρ` | 2500 kg/m³ | |
| contact | — | Hertz normal, Cundall–Strack tangential, Coulomb rolling | `superquadric_kernel` |
| curvature ratio | `B/A` | 1 … 0.1 | 1 = sphere, →0 = cylinder/line contact |
| clump sizes | `N` | 8, 32, 128, 256 | |

## 2. Method

**(a) Closed-form ellipsoid/plane Hertz stiffness** (`shape_representation/hertz_ellipsoid.py`).

With the pole gap `g(x,y) = A x² + B y²` (`A ≥ B`), the contact ellipse eccentricity
`e` follows from the elliptic-integral relation

```
B/A = (K(e) − E(e)) / (E(e)/(1−e²) − K(e)),        e² = 1 − t²,  t = b/a
```

and then

```
a  = (3P / (4(A+B)E*))^(1/3) · [2 E(e) / (π(1−e²))]^(1/3)
b  = t a,     p0 = 3P/(2π a b),     δ = 3P K(e)/(2π a E*),     k_n = (3/2) P/δ
```

The three coefficients of the interior displacement were **not taken from memory**:
they were measured on the exact half-space quadrature (which is validated by the
sphere limit at 1e-12) as `c_x = −(p0 b/E*)(K−E)/(e²a²)`,
`c_y = −(p0 b/E*)(E/(1−e²)−K)/(e²a²)`, `w(0) = (p0 b/E*)K`, and the semi-axis and
approach formulas above were derived from them.  A first attempt that used the
commonly quoted `(E − (1−e²)K)/(K − E)` relation and a `1/t` factor in the
semi-axis produced a 85 % stiffness error against the quadrature — the memory-based
formula set was **falsified** and replaced by the measured one (§5).

**(b) Independent verification.** The closed-form pressure is substituted into the
elastic half-space integral equation and integrated with a *singularity-free*
polar quadrature around each field point (`dA/r = dr dθ`, and the square-root edge
behaviour removed by `u = u_max(1−s²)`).  For the true solution `w + g` must be
constant over the contact ellipse; the spread of `w + g`, and the stiffness
recomputed from `P = ∬p dA` and `δ = mean(w+g)`, are the verification metrics.  No
elliptic integral enters this check.

**(c) Free-boundary scan.** For a trial contact axis ratio `t` the scale follows
from the x-flatness condition (exact, since `w` is quadratic inside); the remaining
y-residual is minimised over `t` and compared with the elliptic-integral value.

**(d) Superellipsoid pole law.** At the pole of `|x/a|^m+|y/b|^m+|z/c|^m = 1` the
gap is `~ (c/m)(|x/a|^m + |y/b|^m)` for `m ≠ 2`: the curvature vanishes (`m > 2`)
or diverges (`m < 2`), so the contact is a **power-law indenter**, not Hertz.  The
law `P ~ δ^(1+1/m)` is verified by solving Love's axisymmetric integral equation
for a rigid indenter `f(r) = κ r^m` (linear system closed by `p(a) = 0`), the
`m = 2` case being checked against the exact Hertz disk `P = (4/3)E*√R δ^{3/2}`.

**(e) Clump convergence.** A shell-covering multi-sphere fit (surface points pulled
inward along the normal, farthest-point selection, inscribed radii) and the
repository's k-means fit are both measured with
`clump_geometry.geometric_error` (two-sided surface distance and normal deviation).

**(f) Static-bed dissipation (C3).** A cubic bed is relaxed, zeroed, and run on with
no work input; the mechanical-energy loss per step is the spurious dissipation.
The sphere bed is the control (it isolates representation effects from solver
effects).

## 3. Results

### 3.1 C1 — ellipsoid/plane stiffness against the elliptic-integral closed form

Ten curvature ratios from the sphere limit `B/A = 1` to the strip limit `B/A = 0.1`;
worst relative stiffness error **7.97e-13** (criterion 2 %), worst residual spread of
`w+g` **2.9e-12**.  Sphere limit against textbook Hertz: `a`, `δ = a²/R` and
`k = 2E*a` all reproduced to ≤ 1.3e-16.  Free-boundary scan recovers the contact
axis ratio from the integral equation alone to < 2 % (coarse grid, refined locally).
Figure: `docs/figures/nonspherical_hertz_anchor.png`.

### 3.2 C1b — superellipsoid pole is not Hertzian

| `m` | 1 | 1.5 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|---|
| `α` numerical | 2.000000 | 1.666667 | 1.500000 | 1.333333 | 1.250000 | 1.125000 |
| `1+1/m` | 2 | 1.666667 | 1.5 | 1.333333 | 1.25 | 1.125 |
| rel. error | 5.6e-16 | 0 | 1.0e-15 | 1.7e-16 | 3.6e-16 | 2.0e-16 |

`m = 2` against the exact Hertz disk: force error **8.4e-5**.  Figure:
`docs/figures/nonspherical_pole_exponent.png`.

Consequence: a clump is a union of spheres, so its contact law is Hertzian
(`α = 1.5`) for *every* shape; a true superellipsoid pole with `m = 8` obeys
`α = 1.125`.  This exponent gap is an irreducible contact-law error that no
increase of `N` removes.

### 3.3 C2 (geometry half) — clump surface error vs `N`

| `N` | covering fit, `m=4` | covering fit, `m=8` | k-means fit, `m=4` | k-means fit, `m=8` |
|---|---|---|---|---|
| 8 | 0.813 | 0.811 | 0.769 | 0.892 |
| 32 | 0.446 | 0.401 | 0.906 | 0.859 |
| 128 | 0.218 | 0.192 | 0.956 | 0.915 |
| 256 | 0.153 | 0.114 | 0.938 | 0.916 |

(numbers as max two-sided deviation / smallest half-extent; from
`artifacts/shape_representation/geom_convergence.json`)

The covering fit converges with exponent **−0.49** (`m=4`) and **−0.56** (`m=8`),
i.e. the expected surface-covering `N^(−1/2)` law.  Extrapolating the measured law,
5 % surface accuracy needs **N ≈ 2.6e3** (`m=4`) / **1.1e3** (`m=8`), and 1 %
accuracy needs `N ≈ 7e4` / `2e4`.  The repository's k-means fitter does **not**
converge (measured exponents **+0.058** (`m=4`) and **+0.012** (`m=8`); the error
*increases* with `N`, 0.77 → 0.96 for `m=4`) — it must not be used for large-`N`
clumps without a convergence check.
Figure: `docs/figures/nonspherical_clump_convergence.png`,
`docs/figures/nonspherical_verdict_map.png`.

### 3.4 C3 — static-bed spurious dissipation

| bed | drift / step | criterion 1e-6 |
|---|---|---|
| sphere (control) | 1.63e-09 | PASS |
| clump `N=8`, `m=8` | 4.79e-04 | FAIL |
| clump `N=32`, `m=8` | 5.35e-04 | FAIL |

The control proves the measurement is meaningful (the solver alone is clean to
1.6e-9/step).  The clump representation dissipates ~5e-4 per step in a *static*
bed — 300–2000× above the criterion — because the bumpy surface keeps breaking and
re-forming contacts.  More spheres do not help: the excess is set by the bumpiness,
not by `N`.  C3 is therefore an **honest FAIL** for the clump.

### 3.5 C2 (repose half) and C4 — NOT CERTIFIED

The macroscopic channels (angle of repose for sphere / clump / true superquadric,
and its monotonicity in the shape factor inside the published 30–35° sand/gravel
band) are **not certified**.  Four driver configurations were tried and each failed
in a documented, *reproducible* way (§5): a standing column, a monolayer, an
unphysical 1.5 m tower of injected particles, and a still-moving splash with
`v_max = 0.7–3 m/s` at the measurement instant.  No repose number is reported as
measured; the verdict marks C2/C4 `NOT_CERTIFIED` and states the reason.

## 4. Verdict (C5)

`artifacts/shape_representation/shape_representation_verdict.json` answers the
decision question with the measured numbers:

* **contact law**: a clump implies Hertz (`α = 1.5`) for all shapes; the true
  superellipsoid pole obeys `α = 1+1/m` (1.125 at `m=8`).  Only `m ≈ 2` is
  Hertzian, so for angular shapes **no clump reproduces the contact law** — the
  error is in the exponent, not in the geometry.
* **geometry**: the clump surface error decays as `N^(−1/2)`; 5 % accuracy requires
  `N ≈ 10³` per particle, which is not affordable in a 10⁵-particle rubble or ice
  simulation.  Therefore a clump is admissible only for
  (i) near-spherical shapes (`m ≤ 4`) and
  (ii) mass/transport-driven quantities whose tolerance is ≥ 20 % at `N = 128`
  (the measured value);
  it is **not** admissible for interlocking-sensitive quantities (angle of repose,
  shear strength, force-chain anisotropy) at any affordable `N`, and it is **not**
  admissible where a static assembly must not dissipate energy (C3).
* **error bound**: the verdict artefact carries the per-`N` measured surface
  deviation ratio and the fitted exponent, so any caller can quote
  `error(N) = e_8 (N/8)^p` with the measured `(e_8, p)` instead of an assumption.

## 5. Notes — what failed, and why it matters

1. **The remembered Hertz formula set was wrong.**  The classical-looking relation
   `B/A = (E − (1−e²)K)/(K − E)` together with a `1/t` semi-axis factor agrees with
   the exact quadrature to 85 % at `B/A = 0.4` while being *exact* in the sphere
   limit — the sphere limit alone does not validate an ellipsoidal formula.  The
   correct structure (`(K−E)/(E/(1−e²)−K)`, semi-axis factor `2E/(π(1−e²))`) was
   identified from the measured quadratic coefficients of the quadrature field.
   Any future edit must keep the residual test (`verify_closed_form`).
2. **Cancellation near the sphere limit**: `K−E` and `E/(1−e²)−K` are both `O(e²)`
   differences of `π/2`-sized numbers; they are computed from series for `e² < 1e−4`
   (tested against `scipy.special.ellipk/ellipe`).
3. **The k-means clump fit does not converge in `N`** — a fitter property, not a
   solver property; the verdict therefore reports two fitters.
4. **The repose driver is the open gap.**  With 4 mm particles the available
   column-collapse/pour drivers need several hundred particles *and* a squat column
   to make a cone; the combinations that fit the compute budget gave a standing
   column (μ = 0.6, aspect ratio 3), a monolayer (box ≫ heap), or a still-moving
   splash (release height ≫ particle size).  The repository's calibrated repose
   driver (`repose_calibration.py`, lifting cylinder) is sphere-only and therefore
   cannot serve the clump/superquadric comparison.  Closing this needs a
   *trickle-injection* driver with rate control — recorded as the next node.
5. **Applicability**: the C1 anchors are exact elasticity results (any `E*`, any
   aspect ratio, overlap ≪ particle size, frictionless/adhesive-free normal
   contact).  The clump bound applies to the tested superellipsoid family and to
   the covering/k-means fitters; other fitting strategies must be re-measured.

## 6. Error table (key quantities vs their reference)

| quantity | value | reference | error | < 3 %? |
|---|---|---|---|---|
| ellipsoid/plane stiffness (10 curvature ratios) | closed form | half-space integral equation | 7.97e-13 | **yes** |
| residual flatness of `w+g` | — | constant inside contact | 2.9e-12 | **yes** |
| sphere-limit `a`, `δ`, `k` | closed form | textbook Hertz | ≤1.3e-16 | **yes** |
| power-law exponent `α(m)` | numerical (Love) | `1+1/m` | ≤2.0e-16 | **yes** |
| `m=2` indenter force | numerical (Love) | exact Hertz disk | 8.4e-05 | **yes** |
| clump surface error exponent | −0.49 / −0.56 | theory `N^(−1/2)` | 2 % / 12 % | **yes** (`m=4`), 12 % (`m=8`) |
| repose angle (C2/C4) | not measured | experiment 30–35° | — | **no basis — NOT_CERTIFIED** |