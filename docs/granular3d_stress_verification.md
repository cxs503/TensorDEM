# Three-dimensional Virial stress: instrument verification

**Module** `src/tensordem/granular3d/stress.py` · **Tests**
`tests/test_granular3d_stress.py` (16 cases) · **Driver**
`scripts/verify_granular3d_stress.py` · **Artifacts**
`artifacts/granular3d_stress/`

## 1. Case introduction — why a stress *instrument* needs a reference

Two of the open 3-D DEM anchors are decided by a stress ordinate:

* `CASE-DEM-SHEAR-CAMPBELL-ELASTIC-LIMIT`: "does the dimensionless shear stress
  scale with the Bagnold number and with which exponent";
* `CASE-DEM-PLANE-SHEAR-MUIDI`: "do `mu` and `phi` collapse onto `mu(I)`".

Both were carried by private `l (x) f` sums (`elastic_limit_shear`,
`bpm_calibration_feasibility.virial_stress_tensor`), while 2-D had a shared and
documented stress surface (`granular2d.stress`, incl. the discrete balance
identity) and 3-D had only the isotropic pressure inside
`granular3d.jamming.SpherePacking`.  When a private sum is wrong, the case fails
as *physics*; nothing in the repository could tell the two apart.  This module
is the missing 3-D instrument: per-contact tensors, plane fluxes, wall
tractions, the moment identity and the region balance, in float64 NumPy,
independent of any solver.

| item | value |
|---|---|
| convention | tension positive, compression negative |
| contact data | `branch = x_j - x_i`, `force = force on i from j` |
| reductions | volume average, plane flux, wall traction, discrete balance laws |
| environment | python 3.11.6, numpy 2.4.6 |

## 2. Method

* `stress_tensor`: `sigma = (1/V) sum_c l (x) f` (unsymmetrised).
* `plane_traction` / `plane_profile`: only contacts that **cross** the plane
  contribute; the result is the force the upper side exerts on the lower side, so
  the depth profile `sigma_zz(z)` has no bin width and no double counting.
* `wall_stress`: `sigma_ab = (1/A) sum f_a n_b` with `n` the outward normal of the
  particle domain.
* `moment_residual`: `sum_c l(x)f = sum_p x_p (x) F_p^ext` — a rearrangement of
  per-particle equilibrium, hence exact for a balanced configuration.
* `region_balance`: `sum_{p in R} F_p^ext + sum_{cut} f_{on R} = 0`, the
  force form of the divergence theorem.

Reference configurations are **exactly solvable by statics**, not simulated: a
statically supported `m x m` column stack (each contact carries the weight above
it) and the simple-cubic isotropic lattice `p = kn delta / d^2`.  The random
packings are the existing `granular3d.jamming` packings, used as an
*independent-implementation* cross-check of the pressure.

## 3. Results

### 3.1 Sign pin (compressed pair)

| quantity | measured | analytic | abs. error |
|---|---|---|---|
| `sigma_zz` for `branch = d z`, `force = -F z` | -0.3 | -0.3 | 0.00e+00 |
| `p = -tr(sigma)/3` | 0.1 | 0.1 | — |

### 3.2 Static column: plane flux vs statics (`m = 6`, 8 layers)

Max relative error over the 7 planes: **6.04e-16**
(in-plane traction max `|tau|` = 0.00e+00); crossing contacts per
plane = 36 as predicted, measured
36…36.

| z/d | plane flux `sigma_zz` | analytic | abs. diff |
|---|---|---|---|
| 0.50 | -68.670000 | -68.670000 | 2.84e-14 |
| 1.50 | -58.860000 | -58.860000 | 3.55e-14 |
| 2.50 | -49.050000 | -49.050000 | 2.84e-14 |
| 3.50 | -39.240000 | -39.240000 | 0.00e+00 |
| 4.50 | -29.430000 | -29.430000 | 1.78e-14 |
| 5.50 | -19.620000 | -19.620000 | 0.00e+00 |
| 6.50 | -9.810000 | -9.810000 | 0.00e+00 |

![profile](../artifacts/granular3d_stress/fig_profile.png)

### 3.3 Wall traction vs statics

| quantity | measured | analytic | rel. error |
|---|---|---|---|
| floor `sigma_zz` | -78.4800000000 | -78.4800000000 | 0.00e+00 |
| step across the floor (`wall - flux(0.5d)`) | -9.8100000000 | -9.8100000000 | 2.72e-15 |

### 3.4 Pressure vs the independent `SpherePacking` path (random packings, n = 200)

| phi | contacts | `-tr(sigma)/3` (this module) | `SpherePacking.pressure()` | rel. diff | asym. `|sigma-sigma^T|` |
|---|---|---|---|---|
| 0.62 | 430 | 0.0318028923 | 0.0318028923 | 2.18e-16 | 2.59e-17 |
| 0.64 | 455 | 0.0350618195 | 0.0350618195 | 5.94e-16 | 1.95e-17 |
| 0.66 | 470 | 0.0384262912 | 0.0384262912 | 3.61e-16 | 4.99e-18 |
| 0.68 | 492 | 0.0418552078 | 0.0418552078 | 1.66e-16 | 8.03e-18 |

### 3.5 Moment identity, rotation invariance, kinetic term

* `sum l(x)f = sum x(x)F_ext` residual on the column: **2.27e-15**
  (relative, Frobenius).
* rotation: `|R sigma R^T - sigma(Rl, Rf)|` = 3.96e-15;
  pressure change 3.45e-15.
* kinetic stress `(1/V) sum m v(x)v` vs analytic: abs. error
  0.00e+00.

### 3.6 The checks are not vacuous — mutation sensitivity

Each row injects one classic implementation bug and reports the relative
deviation it produces in the *gated* quantity:

| injected bug | relative deviation |
|---|---|
| `force_sign_flipped` | 2.000e+00 |
| `branch_reversed` | 1.857e+00 |
| `wall_normal_sign` | 2.000e+00 |
| `all_contacts_not_crossing` | 3.000e+00 |
| `plane_flux_replaced_by_cell_average` | 8.333e-01 |

![mutation](../artifacts/granular3d_stress/fig_mutation.png)

## 4. Error verdict (explicit, per quantity)

| gated quantity | reference | error | < 3 % ? |
|---|---|---|---|
| compressed-pair `sigma_zz` | closed form `-F d/V` | 0.0e+00 (abs, O(1) value) | **yes** |
| column `sigma_zz(z)` profile | statics `-m g N_above/d^2` | 6.0e-16 | **yes** |
| floor `sigma_zz` | statics `-W_total/A` | 0.0e+00 | **yes** |
| wall-to-flux step across the floor | bottom-layer weight / A | 2.7e-15 | **yes** |
| `-tr(sigma)/3` vs `SpherePacking.pressure()` | independent code path | 5.9e-16 | **yes** |
| moment identity | exact | 2.3e-15 | **yes** |
| rotation invariance | exact | 4.0e-15 | **yes** |

All gated errors are at round-off (`1e-13` or below), i.e. far inside the 3 %
criterion; this is expected because the references are exact statements about
the same contact data, and it is the reason the mutation table above is part of
the evidence: an exact check that cannot fail is worthless.

## 5. Points to note / limitations

* This verifies the **instrument**, not any DEM physics.  It says nothing about
  whether Hertz, rolling resistance or a contact solver is right; it makes the
  stress numbers those solvers feed into `mu(I)` / Campbell-type cases auditable.
* The column and lattice references are *static constructions*: the contact
  forces are computed from statics, not relaxed.  No relaxation or time step is
  involved, so these numbers are reproducible bit-for-bit.
* A plane cut fixes only the `sigma[axis, :]` row (property of a plane cut, not
  a defect); the full tensor comes from the volume average.
* The contact virial excludes the kinetic contribution; `kinetic_stress` is a
  separate return value so the two cannot be silently mixed.
* Non-central contact couples (rolling/drilling, `clump_rolling`) are outside a
  symmetric-stress contract; the unsymmetrised sum is returned, and
  `sigma - sigma^T` is the diagnostic.

## 6. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_granular3d_stress.py -q
PYTHONPATH=src python scripts/verify_granular3d_stress.py
```
Figures written: fig_profile.png, fig_mutation.png.
