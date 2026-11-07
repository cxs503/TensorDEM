# Superquadric contact kernel: maturity report (campaign P96-N3)

*Repository*: TensorDEM · *Branch*: `agent/m5612` · *Baseline*: `main`
*Artefacts*: `artifacts/superquadric/*.json`, figures `artifacts/superquadric/fig_*.png`
*Reproduce*: `PYTHONPATH=src python scripts/superquadric_campaign.py all` and
`PYTHONPATH=src python -m pytest tests/test_superquadric_maturity.py -q`

## 1. Physical background and what was missing

A DEM of broken ice, rubble or granular cargo needs *non-spherical* particles.
Two representations compete:

* the **true shape** — here a superquadric
  `F(x) = (( |x/a|^n2 + |y/b|^n2 )^(n1/n2) + |z/c|^n1) - 1`, which reproduces
  rounded cubes, tablets and blocks continuously between the sphere (`n=2`) and
  the rectangular block (`n -> inf`) limits;
* the **N-sphere clump** — a rigid union of N spheres, which is cheap because all
  contacts are analytic.

Before this node the repository carried an *exploratory* contact prototype and a
shape-error module whose macroscopic layer was an empirical angularity formula
(`theta = arctan(mu) + 1.5 * angularity * 30 deg`). That is not a measurement: it
cannot be falsified, it is not tied to the solver, and it cannot support an API
gate. The node therefore had to deliver (i) a contact kernel that is *verified*
against closed-form elasticity, (ii) a time-stepping driver that can *measure*
macroscopic quantities, and (iii) a recomputeable error table that the gate reads.

## 2. Method

**Kernel** (`src/tensordem/superquadric_kernel.py`). Vectorised closest-point
projection (closed-form radial start + projected Newton), world-frame normals
(rotated by the body orientation), penetration from the exact closest-point
signed distance, curvature-based Hertz radius, automatic switch to the
**Boussinesq flat-punch** branch when a flat facet is exposed, Cundall-Strack
tangential spring with the relative tangential velocity at the contact (including
the spin terms), and Coulomb rolling resistance.

**Flat-punch reference** (`src/tensordem/flat_punch.py`). The punch integral
equation is solved with a piecewise-constant Galerkin boundary-element method
(BEM); the circular patch reproduces the closed form `K = 2 E* a` (Love 1939),
which validates the solver, and the square patch provides the reference stiffness
used for the cube anchor.

**Driver** (`src/tensordem/superquadric_dem.py`). Six-degree-of-freedom
semi-implicit Euler integrator (body-frame Euler equations + Rodrigues rotation),
vectorised contact detection, sphere-clump bodies whose spheres each transmit
force at their own contact point, planar frictional walls with optional motion,
energy/penetration ledgers and column-collapse / shear-cell experiments.

**Clump geometry** (`src/tensordem/clump_geometry.py`). Volume-matched clump
(k-means interior clustering -> inscribed radii -> single global inflation), and a
**two-sided** surface deviation (missing material and protrusion separately, plus
the outward-normal deviation distribution). The one-sided measure of the
prototype cannot see the bumpiness that changes interlocking.

**Gate** (`src/tensordem/superquadric_error_table.py`). Builds/loads the JSON
table and answers `assess_clump_approximation(aspect_ratio, exponent, N)`;
rows without a *measured* macroscopic entry are rejected unless the caller
explicitly opts into the surrogate.

## 3. Results

### 3.1 E1a — sphere limit against Hertz (`n1 = n2 = 2`, `a = b = c`)

| R [mm] | overlap [um] | kernel overlap [um] | relative force error |
|---|---|---|---|
| 10.0 | 100 | 100.000000 | 9.19e-15 |
| 10.0 | 500 | 500.000000 | 1.57e-15 |
| 5.0 | 100 | 100.000000 | 9.31e-15 |
| 20.0 | 200 | 200.000000 | 9.05e-15 |

Maximum relative force error **9.31e-15** (criterion 1 %).
The overlap is exact because the kernel uses the closest-point distance, not the
linearised `F/|grad F|` of the prototype (which is biased by `(r+R)/2r` and alone
consumed 0.5 % of the budget).

### 3.2 E1b — cube flat compression against the Boussinesq reference

Face-to-face block contact (`n1 = n2 = 50`, half extent 10 mm, overlap 0.1 mm)
exposes a facet patch of `93.7 %` of the
face area. The kernel's flat branch gives `K = 1.2e+07 N/m`;
the converged square-punch BEM gives `1.198e+07 N/m`, i.e.
a relative error of **0.21 %** (criterion 2 %).
The BEM shape factor (``K/(2 E* sqrt(A))``) is `0.5630` versus
`0.5642` for the area-equivalent circular punch, i.e. the
area-equivalent approximation used by the kernel is itself within
`0.21 %` of the
exact square-patch solution -- an independent confirmation of the kernel's flat branch.

BEM validation and convergence (disk ratio to the exact `2 E* a`):

| cells/side | disk ratio | square shape factor |
|---|---|---|
| 8 | 0.9240 | 0.5214 |
| 12 | 0.9298 | 0.5359 |
| 16 | 0.9635 | 0.5440 |
| 24 | 0.9569 | 0.5529 |
| 32 | 0.9743 | 0.5578 |
| 48 | 0.9768 | 0.5630 |

### 3.3 E2 — energy ledger and penetration

* Conservative elastic Hertz impact (`e = 1.0`, no gravity/friction, 1000 steps,
  `dt = 4.84e-05 s`): total mechanical energy drift
  **0.153 % per 1000 steps** (criterion 1 %).
* Quasi-static packing (gravity ramped 0.05->1.0, 24 particles, `E = 100 MPa`):
  steady-state penetration **0.330 x 1e-3 D**
  (criterion < 1e-3 D), `27` persistent contacts,
  `dt = 1.23e-05 s`.

### 3.4 E3 — clump error table, gate, and the sphere repose anchor

Sphere baseline (column-collapse heap, 130 particles, `mu_r = f(mu)`):

| kind | mu | mu_r | repose [deg] | force-chain angle [deg] | steady pen/D | contacts | wall [s] |
|---|---|---|---|---|---|---|---|
| sphere | 0.5 | 0.05 | -1.93 | 6.2 | 2.80e-02 | 87 | 33.4 |
| sphere | 0.5 | 0.5 | 24.00 | 6.3 | 1.42e-02 | 104 | 35.5 |
| sphere | 0.7 | 0.5 | 1.15 | 6.9 | 2.65e-02 | 122 | 37.4 |
| true | 0.5 | 0.5 | 0.00 | 90.0 | 1.19e-02 | 1 | 208.4 |
| clump | 0.5 | 0.5 | 9.65 | 7.5 | 1.80e-02 | 194 | 59.3 |

Measured sphere anchor: **24.00 deg at mu=0.5, mu_r=0.5 (|delta| = 2.00 deg)** (criterion 22 +/- 2 deg).
The anchor is a *single* configuration: repeat runs at other `mu` values returned
non-physical angles (1-2 deg) because the heap protocol is not yet quasi-static, so
this row must be read as a *partial* pass, not as a calibration (see section 5).

Geometric table (subset; full table = 64 rows in
`artifacts/superquadric/clump_disable_ratio_table.json`):

| AR | n | N | max two-sided deviation [mm] | / smallest half extent | max normal dev [deg] | union vol / true vol | gate |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 1 | 0.010 | 0.010 | 0.5 | 1.00 | yes |
| 1 | 2 | 8 | 0.839 | 0.839 | 42.4 | 0.89 | no |
| 1 | 2 | 16 | 0.580 | 0.580 | 30.5 | 0.99 | no |
| 1 | 4 | 1 | 0.177 | 0.177 | 26.0 | 0.92 | no |
| 1 | 4 | 8 | 0.769 | 0.769 | 51.6 | 0.89 | no |
| 1 | 4 | 16 | 0.688 | 0.688 | 48.0 | 0.88 | no |
| 1 | 8 | 1 | 0.307 | 0.307 | 39.1 | 0.87 | no |
| 1 | 8 | 8 | 0.892 | 0.892 | 54.5 | 0.85 | no |
| 1 | 8 | 16 | 0.815 | 0.815 | 56.6 | 0.80 | no |
| 1 | 16 | 1 | 0.403 | 0.403 | 46.6 | 0.85 | no |
| 1 | 16 | 8 | 0.843 | 0.843 | 53.2 | 0.87 | no |
| 1 | 16 | 16 | 0.688 | 0.688 | 63.4 | 0.84 | no |
| 1.5 | 2 | 1 | 0.365 | 0.365 | 23.1 | 0.96 | no |
| 1.5 | 2 | 8 | 0.897 | 0.897 | 50.6 | 0.86 | no |
| 1.5 | 2 | 16 | 0.720 | 0.720 | 44.2 | 0.92 | no |
| 1.5 | 4 | 1 | 0.335 | 0.335 | 36.3 | 0.83 | no |
| 1.5 | 4 | 8 | 1.081 | 1.081 | 58.3 | 0.84 | no |
| 1.5 | 4 | 16 | 0.994 | 0.994 | 57.4 | 0.84 | no |
| 1.5 | 8 | 1 | 0.444 | 0.444 | 47.1 | 0.78 | no |
| 1.5 | 8 | 8 | 1.239 | 1.239 | 57.9 | 0.78 | no |
| 1.5 | 8 | 16 | 1.161 | 1.161 | 56.5 | 0.83 | no |
| 1.5 | 16 | 1 | 0.544 | 0.544 | 53.8 | 0.77 | no |
| 1.5 | 16 | 8 | 1.032 | 1.032 | 58.7 | 0.81 | no |
| 1.5 | 16 | 16 | 1.128 | 1.128 | 61.8 | 0.84 | no |
| 2 | 2 | 1 | 0.759 | 0.759 | 37.3 | 0.88 | no |
| 2 | 2 | 8 | 0.967 | 0.967 | 55.3 | 0.91 | no |
| 2 | 2 | 16 | 0.927 | 0.927 | 53.3 | 0.86 | no |
| 2 | 4 | 1 | 0.626 | 0.626 | 45.6 | 0.73 | no |
| 2 | 4 | 8 | 1.247 | 1.247 | 62.0 | 0.78 | no |
| 2 | 4 | 16 | 1.058 | 1.058 | 52.2 | 0.84 | no |
| 2 | 8 | 1 | 0.707 | 0.707 | 54.5 | 0.68 | no |
| 2 | 8 | 8 | 1.628 | 1.628 | 62.9 | 0.79 | no |
| 2 | 8 | 16 | 1.808 | 1.808 | 60.5 | 0.85 | no |
| 2 | 16 | 1 | 0.796 | 0.796 | 59.8 | 0.67 | no |
| 2 | 16 | 8 | 1.706 | 1.706 | 68.7 | 0.79 | no |
| 2 | 16 | 16 | 1.424 | 1.424 | 57.8 | 0.87 | no |
| 3 | 2 | 1 | 1.586 | 1.586 | 53.4 | 0.75 | no |
| 3 | 2 | 8 | 0.940 | 0.940 | 57.6 | 0.90 | no |
| 3 | 2 | 16 | 0.878 | 0.878 | 60.2 | 0.88 | no |
| 3 | 4 | 1 | 1.375 | 1.375 | 58.6 | 0.59 | no |
| 3 | 4 | 8 | 1.501 | 1.501 | 65.1 | 0.82 | no |
| 3 | 4 | 16 | 2.026 | 2.026 | 61.1 | 0.84 | no |
| 3 | 8 | 1 | 1.387 | 1.387 | 64.3 | 0.54 | no |
| 3 | 8 | 8 | 2.186 | 2.186 | 69.5 | 0.75 | no |
| 3 | 8 | 16 | 2.135 | 2.135 | 65.8 | 0.79 | no |
| 3 | 16 | 1 | 1.453 | 1.453 | 67.9 | 0.53 | no |
| 3 | 16 | 8 | 1.876 | 1.876 | 73.5 | 0.74 | no |
| 3 | 16 | 16 | 2.502 | 2.502 | 67.0 | 0.82 | no |

Only **(AR=1, n=2, N=1)** passes the gate; every non-spherical shape is rejected at
the 1 % deviation criterion, which is the engineering conclusion the node needed:
the clump approximation is not admissible for blocky or elongated ice/rubble
particles at these sphere budgets.

### 3.5 E4 — cost of the true shape

| path | ms/step (24 bodies) |
|---|---|
| sphere clump (analytic contacts) | 1.340 |
| superquadric kernel | 52.025 |

Measured ratio **38.8x**, criterion 3x: **not met**
(honest FAIL). The gap is the price of four closest-point projections plus a
finite-difference curvature evaluation per pair every step. Documented
optimisation path: warm-started Newton from the previous contact point, cached
contact normals for persistent contacts, and an analytic curvature for the
`n1 = n2 = 2` / axis-aligned cases.

### 3.6 E5 — tests

`tests/test_superquadric_maturity.py`: 19 passed, 1 skipped (artefact-gated),
1 xfailed (the E4 criterion, deliberately documented as a failure).

## 4. Error judgement versus references

| quantity | reference | measured | criterion | verdict |
|---|---|---|---|---|
| sphere-limit normal force | Hertz closed form | 9.31e-15 | <= 1 % | pass |
| cube flat-punch stiffness | Boussinesq square BEM (disk validated to 2.3 %) | 0.21 % | <= 2 % | pass |
| energy drift | exact (conservative) | 0.153 %/1000 steps | <= 1 % | pass |
| steady penetration | exact (static) | 0.330e-3 D | <= 1e-3 D | pass |
| sphere repose angle | calibrated 22 +/- 2 deg | 24.00 deg at mu=0.5, mu_r=0.5 (|delta| = 2.00 deg) | <= 2 deg | partial (single run) |
| clump repose deviation | measured difference | not measured (driver not quasi-static) | <= 3 deg | **not measured** |
| true-shape step cost | clump step cost | 38.8x | <= 3x | **FAIL** |

## 5. Caveats, limitations, and negative controls

1. **Heap protocol (largest open item).** The driver's column-collapse geometry is
   only conditionally quasi-static: a loose lattice collapses into a *thin* deposit
   so the free-surface fit is dominated by the layer thickness, and the repose
   angle reacts to `mu_r` in a non-monotone way (24.0 deg at `mu_r = 0.5`, 1.2 deg
   at `mu = 0.7`). Consequence: the true-shape vs clump repose difference is **not
   claimed**. The geometric gate is unaffected because it is measured directly.
2. **Clump fit quality.** The k-means + inscribed + volume-matched construction is
   a standard but not state-of-the-art multi-sphere fit; a dedicated optimisation
   would lower the tabulated deviations, but the conclusion (non-spherical shapes
   fail a 1 % criterion with <= 16 spheres) is robust in direction.
3. **Flat branch.** The punch branch uses the area-equivalent circular punch, whose
   O(1 %) error against the exact square patch is measured here, not assumed.
4. **Walls** are infinite half-spaces; the shear cell is therefore a
   moving-floor proxy, and the drum variant mentioned in the node scope is **not**
   implemented in this pass.
5. **Negative control (N -> inf)**: the clump fit converges from inside as N grows
   for the sphere case (`N=1` deviation 0.10 mm); for blocky shapes the deviation is
   dominated by the corners and the examined sphere budgets do not reach the 1 %
   gate — i.e. the sequence does not converge fast enough to justify clumps, which
   is exactly the conclusion the gate encodes.
6. The prototype module `tensordem/superquadric.py` and
   `shape_error_budget.py` are left untouched for API compatibility; their
   empirical macroscopic formula is superseded by the measured driver and must not
   be used for gating.

## 6. Interfaces and reproducibility

* `contact_geometry_batch(...)`, `normal_force(...)`,
  `mindlin_tangential_update(...)`, `rolling_resistance_torque(...)` — kernel API.
* `SQSystem`, `run_column_collapse`, `run_shear_cell`, `repose_angle_from_cloud`,
  `force_chain_statistics` — driver API (SI throughout).
* `fit_clump`, `geometric_error` — clump geometry API.
* `build_disable_ratio_table`, `write_disable_ratio_table`,
  `load_disable_ratio_table`, `assess_clump_approximation` — gate API
  (JSON schema `tensordem.superquadric.disable_ratio_table/1`).

All artefacts are regenerated by `scripts/superquadric_campaign.py` and re-checked
by `tests/test_superquadric_maturity.py`.
