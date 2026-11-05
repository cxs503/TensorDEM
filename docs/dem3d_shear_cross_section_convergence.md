# DEM3D direct-shear specimen: cross-section convergence of the shear modulus

Status: **closed** (this page) — the WITHHELD item *"a material shear modulus still
needs a cross-section convergence study or periodic boundaries"* from
`docs/dem3d_shear_static_reference.md` is answered here with a specimen family of
static equilibrium solves, and the answer is conditional (thin specimens yes,
thick specimens no).

All numbers come from
`scripts/audit_dem3d_shear_cross_section_convergence.py`
(machine-readable: `docs/dem3d-shear-cross-section-convergence/report.json`,
table: `summary.md`, figure: `figures/cross_section_convergence.png`).
Reproduce with

```
PYTHONPATH=src python scripts/audit_dem3d_shear_cross_section_convergence.py
PYTHONPATH=src python -m pytest tests/test_dem3d_shear_cross_section_convergence.py -q
```

## 1. Question

`docs/dem3d_shear_specimen.md` measured an "apparent" shear modulus of a bonded
simple-cubic block (bottom layer fixed, top layer displaced laterally, lateral
faces free) and reported values between `0.65` and `1.05 C44`; the rate-free
static reference (`shear_static.py`) explained the scatter with a cross-section
bookkeeping bias plus free-surface relaxation and left the material constant
unidentified. The open question: **does any finite specimen of this test geometry
return the analytic lattice constant `C44 = 7k/(3a)`?**

Because the boundary conditions (clamped bottom, prescribed `u_x` on top, free
lateral faces) do not admit the homogeneous simple-shear field
(`sigma . n != 0` on the lateral faces), the answer can only come from a
systematic size study, not from one run.

## 2. Setup

| quantity | value |
| --- | --- |
| lattice | simple cubic + 12 edge + 8 body diagonals, `a = 2r = 0.05 m` |
| bond stiffness `k` | 2000 N/m (axial only; no tangential/bending stiffness) |
| closed form `C44` | `7k/(3a) = 93 333.3 Pa` |
| applied shear `gamma` | `1e-6` (linear limit; see 3.5) |
| specimen height | `h = (nz-1) a`, thin family `nz = 3` (`h = 2a`) |
| square family | `n = 5, 9, 13, 17, 21, 25, 33, 41` (`5 043` particles at largest) |
| rectangular family | `9x5, 17x9, 25x13, 33x21, 41x25` |
| thick family | `33x33` with `nz = 3, 5, 7, 9` |
| solve | one `scipy.sparse.linalg.spsolve` per specimen, `K` assembled from the solver's own live bond list |

Three measurement definitions are used, all sharing the same bond law
`f = k (n . du) n` that `IceDEM3D.forces` uses:

1. **`G_nom`** — legacy: reaction of the fixed layer over the nominal area
   `nx a * ny a`.
2. **`G_eff`** — reaction over the bond-census area `A_eff`
   (`affine_effective_area`), i.e. effect A of the earlier audit removed.
3. **`G_win`** — the same bond census restricted to an **interior window** whose
   nodes lie at least `margin` lattice sites away from every lateral face, taken
   across a chosen horizontal plane (`layer` argument; default = the fixed
   bottom plane), so numerator and denominator stay consistent under affine
   shear. Under affine shear this measure returns exactly `C44` (machine
   precision, see 3.5) — it is a bookkeeping identity, not a calibration.

## 3. Results

### 3.1 Thin specimens (`h = 2a`): the modulus converges to `C44`

| specimen | N | `A_eff/A_nom` | `G_nom/C44` | `G_eff/C44` | `G_win/C44` (margin 3) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5x5x3 | 75 | 0.7086 | 0.4869 | 0.6872 | 0.7327 (margin 1) |
| 9x9x3 | 243 | 0.8325 | 0.7040 | 0.8457 | 0.9751 |
| 13x13x3 | 507 | 0.8825 | 0.7921 | 0.8976 | 0.9920 |
| 17x17x3 | 867 | 0.9095 | 0.8398 | 0.9234 | 0.9951 |
| 21x21x3 | 1 323 | 0.9265 | 0.8697 | 0.9388 | 0.9964 |
| 25x25x3 | 1 875 | 0.9381 | 0.8902 | 0.9490 | 0.9972 |
| 33x33x3 | 3 267 | 0.9529 | 0.9165 | 0.9618 | 0.9981 |
| 41x41x3 | 5 043 | 0.9620 | 0.9326 | 0.9695 | 0.9985 |
| 9x5x3 | 135 | 0.7873 | 0.6646 | 0.8442 | 0.9029 (margin 2) |
| 17x9x3 | 459 | 0.8814 | 0.8134 | 0.9229 | 0.9955 |
| 25x13x3 | 975 | 0.9178 | 0.8708 | 0.9488 | 0.9974 |
| 33x21x3 | 2 079 | 0.9433 | 0.9072 | 0.9617 | 0.9981 |
| 41x25x3 | 3 075 | 0.9533 | 0.9242 | 0.9694 | 0.9985 |

Two independent closures of the material constant:

* **Size extrapolation** (square family, model `1 - G/C44 = c0 + c1 x + c2 x^2`
  with `x = 1/nx + 1/ny`, i.e. the boundary-layer area fraction):
  `c2`-fit limit `0.99865 C44` (deviation **0.14 %**, `r2 = 0.999994`),
  `c2 = 0` fit limit `1.01474 C44` (**1.47 %**). Two extrapolation orders bracket
  the analytic value from both sides.
* **Interior window** at margin 3 on `33x33x3`: `0.9981 C44` (**0.19 %**), and the
  same margin gives `0.9981 C44` on the rectangular `33x21x3` — the edge
  correction is aspect-ratio independent once the window is fixed in absolute
  sites.

The exponential decay of the window deficit with the margin
(`fit_boundary_layer`) gives a decay length of `0.63 a` (`r2 = 0.952`, five
margins, one dropped as the deficit vanishes). The free-surface relaxation of a
thin slab is therefore a **one-to-two-site layer**, not a bulk softening.

### 3.2 Thick specimens (`33x33`, `h = 2a … 8a`): no material constant

| `nz` | `h/L` | `G_eff/C44` | `G_win/C44` (mid-layer, margin 1/3/5) | mid-layer decay length |
| ---: | ---: | ---: | --- | ---: |
| 3 | 0.06 | 0.9618 | 0.9937 / 1.0016 / 1.0002 | window already at `C44` |
| 5 | 0.12 | 0.9057 | 0.9478 / 0.9891 / 1.0008 | 1.07 a |
| 7 | 0.18 | 0.8484 | 0.8926 / 0.9523 / 0.9815 | 2.28 a |
| 9 | 0.24 | 0.7903 | 0.8338 / 0.9022 / 0.9442 | 3.66 a |

Reading: for `h >= 3a` the shear lag invades the whole cross-section — the fitted
relaxation length grows together with `h/L` (1.07 a at `h/L = 0.12` to 3.66 a at
`h/L = 0.24`) and no interior window reaches `C44`. The measured modulus of a
thick block is a **specimen property**: it depends on `h/L`, not on the bond law.
Consequently the 9x5x3 default specimen of `run_direct_shear` (`h/L = 0.25`) can
only be calibrated against the static reference of the same geometry, never
against `C44`.

### 3.3 Aspect ratio: one perimeter law is not enough

The square family is described to `r2 = 0.99999`; fitting the same
`1/nx + 1/ny` model to square **and** rectangular specimens returns
`r2 = 0.942` and a limit of `0.9783 C44` (2.2 % low). The perimeter-fraction
model is therefore adequate along a fixed aspect ratio, not across aspect ratios:
the long (shear) direction and the height enter differently (3.2). Reported as a
limitation, not silently absorbed into the extrapolation.

### 3.4 Equilibrium and consistency evidence

* `|reaction_solver + reaction_sparse| / |reaction_sparse| <= 1.4e-11` on every
  specimen (the sparse solve is re-evaluated through `IceDEM3D.forces`).
* free-DOF residual / bond-force scale `<= 4 %`; the residual grows with `h` and
  `gamma` because the axial spring law is geometrically nonlinear, and the
  response stays linear in `gamma` (3.5).

### 3.5 Linear regime and affine identity

| `gamma` | `1e-7` | `1e-6` | `1e-5` |
| --- | ---: | ---: | ---: |
| `G_eff/C44` (17x17x3) | 0.9233605 | 0.9233605 | 0.9233605 |
| `G_win/C44` (margin 3) | 0.9950958 | 0.9950958 | 0.9950958 |

Both measures are `gamma`-independent to the last printed digit (relative spread
`< 1e-15`). Under an affine displacement field the window measure returns
`1 + O(1e-16)` for `9x9x3`, `21x21x3` and `33x21x3`, which proves the window
bookkeeping is exact rather than fitted.

## 4. Attention points and scope

* **Only the axial bond law is tested.** There is no tangential or bending
  stiffness in `IceDEM3D`, so `C44 = 7k/(3a)` is the only shear constant the
  model possesses; the convergence statements are about this lattice, not about
  ice.
* **A window margin is a measurement choice.** `G_win` depends on the margin
  (3.1 shows `0.975 … 0.999`); the report quotes the margin with every number.
  On thick specimens the window cannot repair the geometry (3.2).
* **Extrapolation is empirical.** The `1/n` model is motivated by the
  boundary-layer area fraction, and its two orders (linear/quadratic) give 1.47 %
  and 0.14 %; no claim is made that the quadratic intercept is a physical
  observable, only that both bracket `C44` inside 3 %.
* **Rectangular specimens** show that the single-perimeter model is incomplete
  (3.3). A periodic lateral cell would be the clean way to remove the edge
  effect; it is the natural follow-up if a material modulus is ever needed from
  this test.
* Tiny specimens (`5x5`) cannot host a margin-3 window; the margin is clamped and
  recorded (`window_margin_used`).

## 5. Error verdict

| key quantity | reference | measured | relative error | `< 3 %` |
| --- | --- | --- | ---: | --- |
| `G` extrapolated (quadratic in `1/n`, square family) | `C44 = 7k/(3a)` | `0.99865 C44` | 0.14 % | **yes** |
| `G` extrapolated (linear in `1/n`) | `C44` | `1.01474 C44` | 1.47 % | **yes** |
| `G_win` margin 3, `33x33x3` | `C44` | `0.9981 C44` | 0.19 % | **yes** |
| `G_win` margin 3, `33x21x3` | `C44` | `0.9981 C44` | 0.19 % | **yes** |
| `G_win` affine identity (3 specimens) | `C44` | `1.0 + 2e-16` | `2e-16` | **yes** |
| `G_eff` thick `33x33x9` (`h/L = 0.24`) | `C44` | `0.7903 C44` | 21.0 % | **no (specimen property, not an error)** |
| combined square+rect extrapolation | `C44` | `0.9783 C44` | 2.2 % | yes, but model is rejected by `r2 = 0.942` |

Verdict: the material shear modulus of the 26-neighbour central-force lattice is
recovered from the direct-shear specimen to **< 1.5 %** by cross-section
extrapolation and to **< 0.2 %** by an interior window **provided `h/L <= 0.06`**;
for thicker specimens the same test reports a geometry-dependent property with up
to 21 % deviation, which must not be read as a material constant.