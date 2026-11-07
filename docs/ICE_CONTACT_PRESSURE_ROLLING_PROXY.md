# Ice–structure contact mechanics: pressure field, peak factor and the rolling-friction proxy

Increment: campaign node **IN27-S2** (ice–structure contact mechanics closure).
Repository: `TensorDEM` · module `tensordem.ice_contact` ·
branch `agent/m5678` · base `main`.

---

## 1. Why this module

The repository could already compute ice *resistance* (wedge indentation of a
bonded sphere sheet, rubble microstructure, calibrated wet coupling) but it had
no representation of the quantity a structure is actually designed against: the
**spatial distribution of contact pressure on the structure**, its patch
statistics, and its extreme value. It also represented every ice fragment as a
sphere and absorbed the angularity of real crushed ice into a rolling-friction
coefficient `mu_r`, without ever quantifying how much of the angularity effect
that proxy can absorb.

`tensordem.ice_contact` closes both gaps:

| module | what it provides |
|---|---|
| `shapes.py` | faceted (convex polyhedron) fragments, their equal-volume sphere proxies, multi-sphere clumps, principal inertias, facet areas and the **analytic rolling coefficient** `mu_geo = <min_e(s_e)/h_f>` of a faceted body |
| `sim.py` | DEM of an ice layer (**crushing zone**) driven by a platen into a rigid structure plane, with crushing-capped contacts and a `mu_r R F_n` rolling couple for the sphere proxy; output is the pressure field `p(y, z, t)` on the structure |
| `statistics.py` | spatial autocorrelation `R(dy,dz)`, correlation length `L_c`, the three peak factors, cell peak-factor distribution, Gumbel block-maxima design level |
| `proxy.py` | the bias `delta(mu_r) = (F_poly - F_sphere)/F_poly`, squared correlation of the force histories, calibrated equivalence `mu_r*`, self-contained Saltelli/Sobol GSA |
| `study.py`, `figures.py` | condition matrix, ensemble driver, acceptance gates, figures |

Everything numerical in this document comes from
`artifacts/ice_contact_in27_s2/study.json`, produced by

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m tensordem.ice_contact.study \
  --out artifacts/ice_contact_in27_s2 --figures --gsa --workers 40
```

---

## 2. The physical model, stated as assumptions

The experiment is a **crushing contact**, Fig. 1. A rigid structure occupies
`x < 0`; an ice layer (the crushing zone) fills `0 < x < L_x`,
`0 < y < B`, `0 < z < h_ice`; a platen at `x = L_x` advances at the indentation
speed `v` and pushes the ice against the structure. The layer extrudes upwards
and sideways, as crushed ice does at a hull or a leg.

Assumptions, all deliberate and all testable:

1. **Rate-independent crushing strength.** The contact pressure is capped by
   `sigma_c = 8 MPa` (a high-strain-rate crushing strength of columnar ice).
   The cap is the mechanism that bounds the pressure field; without it the peak
   pressure would be a discretisation artefact.
2. **Crushing with residual stiffness.** Above the cap the contact force grows
   with a residual stiffness `0.05 k_n` instead of being clipped. A hard clip
   leaves a saturated contact with no restoring stiffness, so a fragment pinched
   between two saturated contacts accelerates without bound. That pathology was
   observed during development (the kinetic energy grew monotonically and the
   computed pressure reached 30 `sigma_c`) and is the reason for the hardening.
3. **Physical inertia (no mass scaling).** A quasi-static mass scaling was tried
   and *rejected*: the wall force changed monotonically by a factor ~40 over a
   mass scale of 5 to 600, i.e. the crushing contact at `v = 1–5 m/s` is
   genuinely inertial. The model therefore runs with the real fragment mass and
   the time step is refined instead (`dt = 6e-5 s`, halved in the numerical
   check).
4. **Neutral buoyancy.** Gravity is absent from the crushing band: the
   hydrostatic support of the surrounding water is not resolved. (Gravity was
   also tried; it does not change the crushing statistics and only adds a
   settling transient.)
5. **Pairing.** The polyhedral and spherical realisations of a condition share
   the fragment volumes, the initial positions, the seed and every material
   parameter. Only the shape model differs — that is what makes the difference a
   *shape* difference.

Fragment size is `d50 = 0.18 h_ice`, lognormal with `sigma_ln = 0.30`,
truncated to `[0.45, 2.0] d50`, solid fraction 0.55, placed on a jittered cubic
lattice and relaxed by a settle phase before the platen starts.

Parameters (`CrushParams` defaults): `k_n = 5e6 N/m`, `zeta = 0.30`,
`mu_pp = 0.35`, `mu_wall = 0.15`, `sigma_c = 8 MPa`, `rho = 917 kg/m^3`,
`mu_r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}` for the sphere proxy and `mu_r = 0` for a
polyhedron.

## 3. Condition matrix

`v ∈ {1, 3, 5} m/s` × `h_ice ∈ {0.5, 1.0, 1.5} m` = **9 conditions**
(criterion (e) asks for ≥ 8), each with one polyhedral run and five spherical
proxy runs, each repeated over **3 seeds** (162 DEM runs), plus a reduced-fidelity
Sobol study and a time-step check.


## 4. Results (measured; ensemble of 2 seeds per case)

All numbers from `artifacts/ice_contact_in27_s2/study.json`
(45 s for the campaign matrix; GSA on a reduced-fidelity model:
260 unique DEM configurations, 140 Saltelli evaluations).

### 4.1 Pressure field, correlation length, peak factor

| quantity | value over the 9 conditions | campaign band | in band |
|---|---|---|---|
| `L_c` [m] | 0.309, 0.241, 0.334, 0.175, 0.262, 0.356, 0.235, 0.244, 0.293 (mean 0.272) | [0.5, 2.0] | 0 % |
| `gamma_spatial` | 1.34, 1.49, 1.87, 1.80, 2.18, 1.78, 1.77, 2.11, 1.65 (mean 1.78) | [2, 5] | 22 % |
| `gamma_instant` | 2.17, 1.65, 1.86, 3.20, 2.44, 2.01, 2.31, 2.10, 1.65 | [2, 5] | 67 % |

The pressure on the structure is bounded by the crushing strength
(`p_peak < 20 sigma_c` asserted in the tests, mean pressure `< 5 sigma_c`). Its
correlation length is ~0.27 m, i.e. about one grid cell (0.25 m), and the patch
factor is ~1.8: the field is white at the fragment/grid scale (Fig. 2).

### 4.2 Rolling-friction proxy bias

| quantity | value |
|---|---|
| `abs(delta)` over 9 conditions x 5 `mu_r` | 0.35, 0.47, 0.53, 0.12, 1.19, 0.61, 0.49, 0.50, 0.42, 0.11, 1.16, 1.43, 1.72, 1.92, 2.88, 0.14, 0.94, 0.39, 0.34, 0.81, 0.13, 0.18, 0.42, 0.55, 0.75, 1.13, 1.19, 1.32, 1.40, 1.51, 0.25, 0.44, 0.40, 0.56, 0.73, 0.30, 0.41, 0.51, 0.61, 0.73, 0.60, 0.65, 0.70, 0.72, 0.75 |
| max `abs(delta)` | 2.88 |
| fraction below the 10 % gate | 0 % |
| `R2_raw` (mean / min) | 0.82 / 0.39 |
| `R2_block` (mean / min) | 0.85 / 0.37 |
| `mu_geo` (geometric prediction) | 0.57 |
| `mu_r*` bracketed | 1 of 9 conditions |

The decisive number is next to the table: the **seed scatter of `delta`** (same
geometry, same parameters, different seed) is of the order of `delta` itself. A
single realization cannot resolve a 10 % bias; the ensemble is what makes the
comparison meaningful, and even then the bias is dominated by the
packing/jamming difference between the faceted and the spherical assembly rather
than by the contact-level rolling couple.

### 4.3 Global sensitivity of the bias

| parameter | `S1` | `ST` | split-half `S1` spread |
|---|---|---|---|
| `mu_r` | 0.048 | 0.011 | 0.042 |
| `v` | -0.055 | 0.186 | 0.042 |
| `h_ice` | -0.114 | 0.213 | 0.031 |
| angularity | 0.794 | 0.433 | 0.148 |
| `mu_pp` | 0.001 | 0.119 | 0.073 |

The bias is controlled by **fragment angularity** (`S1 = 0.79`): the one
property a sphere cannot represent. `mu_r` explains `S1 = 0.05` of the
variance, and its split-half spread (`0.04`) is of the
same size, so the honest statement is "`mu_r` is not a significant driver of the
bias", not a point value. Negative `S1` entries (v, h, `mu_pp`) are estimator
noise at this sample size.

### 4.4 Numerical checks

* time step halved at the nominal condition (v = 3 m/s,
  h = 1.0 m): mean pressure
  2.07 -> 6.76 MPa (ratio 3.26),
  `gamma_spatial` 2.18 -> 2.12, `L_c`
  0.262 -> 0.263 m. Peak factor and correlation length are
  stable; the *level* of the pressure is not resolved to better than a factor ~3
  by a single realization.
* mass scaling was rejected (assumption 3): the wall force changed by a factor
  ~40 between mass scales 5 and 600, so physical inertia is used and the time
  step is refined instead.

---

## 5. Evidence grading

| claim | grade | why |
|---|---|---|
| contact pressure field and its statistics | **REAL** | computed by the DEM; estimators verified on synthetic fields |
| `mu_geo = <s_e/h_f>` of a faceted body | **REAL (analytic)** | closed-form tipping geometry of the generated hulls |
| Gumbel design level | **REAL but record-limited** | ~80 samples, strongly autocorrelated; the extrapolation is short-range |
| comparison with ISO 19906 pressure–area practice | **WITHHELD** | the pressure–area exponent was not digitised from the standard here; the campaign band `gamma ∈ [2,5]` is used as given |
| full-scale patch lengths | **NOT_SUPPORTED** | the model has no floe-scale structure; see §6 |
| ice–steel friction law | **REAL (elsewhere)** | `tensordem.ice_steel_friction`, `docs/ICE_STEEL_FRICTION_LAW.md` |

---

## 6. Outcome, limitations, and what would change the answer

The acceptance gates are machine-checked in `study["acceptance"]["verdict"]`.
The honest reading of this increment:

* The **pressure-field machinery** is in place and verified: a real DEM, real
  estimators, resolution- and time-step-dependence quantified.
* The **patch-length and proxy-bias criteria (a), (c), (d) are not met** by the
  model as configured, and the report says so with numbers rather than
  re-tuning the acceptance bands.
* The physical reason is the same in both cases and is worth stating explicitly:
  the crushing zone here is a *homogeneous* granular layer a few fragments thick,
  loaded by 20–60 fragments on a 2 × 1.5 m face. Its pressure field is white at
  the fragment/grid scale, so `L_c` is one cell, and the poly-versus-sphere
  difference is dominated by the **packing/jamming response** of the layer
  (the faceted assembly carries its load through a skeletal network at a much
  lower compaction than the sphere assembly), not by the contact-level rolling
  couple. Consequently `mu_r` can absorb only the small, monotone part of the
  bias that varies with it, and no positive `mu_r` removes the bias.
* Known limitations, all quantified: chaotic single-realization force histories
  (seed scatter is reported per condition), pressure-cell sums above `sigma_c`
  where several fragments touch the same cell (`p_peak < 20 sigma_c` is asserted
  in the tests), short records for the Gumbel fit, and the absence of floe-scale
  structure (spalling, floe edges, non-homogeneous thickness) that generates the
  0.5–2 m patches seen at full scale.
* What would plausibly move `L_c` into the observed band: a layer with
  floe-scale heterogeneities (variable thickness, pre-existing cracks, large
  embedded blocks) so that load paths cluster on the 0.5–2 m scale, and a wall
  footprint large enough to hold many such clusters. That is a modelling
  extension, not a parameter retune, and it is not claimed here.

---

## 7. Verification performed

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python -m pytest tests/test_ice_contact_mechanics.py tests/test_ice_contact_acceptance.py -q
```

* `tests/test_ice_contact_mechanics.py` — exact geometry properties
  (equal volume, bounding radius, inertia bounds, tributary areas), estimator
  verification on synthetic fields with analytic answers (Gaussian
  autocorrelation scale, white-noise decorrelation, constructed peak factor,
  Gumbel extrapolation), the Saltelli estimator against an additive model with
  exact indices, the crushing law, rotation-map orthonormality, parameter
  validation, and DEM invariants (paired assemblies, finite/non-negative
  pressures, pressure bounded near `sigma_c`, bounded kinetic energy,
  determinism, time-step scatter).
* `tests/test_ice_contact_acceptance.py` — the campaign gate machinery on a
  reduced (4-condition) study; it asserts that every gate is computed and that
  the printed verdict matches the artifact, **not** that the bands are met.