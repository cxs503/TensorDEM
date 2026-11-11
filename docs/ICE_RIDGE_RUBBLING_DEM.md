# Ice-ridge formation by a breakable buoyant-floe DEM (2-D vertical plane)

Mission **p318w3-n08** (TensorDEM).  This document replaces the *kinematic* ice-ridge
model of m10822 (`docs/ICE_RIDGE_FORMATION_REPORT.md`), which prescribed floe
positions from the compression strain and then read the sail/keel geometry off the
Hopky (1983) empirical law, and fitted its "porosity scaling law" to the same
exponential it assumed (R² = 1.0000 by construction).  No number in this report is
prescribed: every macroscopic quantity is measured from the block configuration.

## 1. Case, geometry and parameters

| quantity | value |
|---|---|
| floe count | 6 |
| ice thickness h | 0.5 m |
| floe length range | 2.0–6.0 m, p(L) ∝ L^-2.5 |
| rho_ice / rho_water | 917.0 / 1025.0 kg/m³ |
| flexural strength | 5e+05 Pa |
| friction mu | 0.4 |
| loading | stress servo, target 30000 N/m width, advance ≤ 0.05 m/s |
| numerical | dt = 4.257e-04 s, 20000 steps, mass scaling ×8.0 |

Plane strain, unit out-of-plane width; the waterline is `z = 0`.

## 2. Method

`src/tensordem/ice_ridge_formation/rubbling.py` implements a buoyant *breakable*
rigid-polygon DEM, built on the hydrostatics of `tensordem.ice_rafting.mechanics`:

1. every floe is a convex rigid polygon with exact area, centroid and inertia
   (polar second moment per unit mass), three DOF;
2. hydrostatic buoyancy acts on the exact submerged part of the polygon (the centre
   of buoyancy moves when a block tilts), weight acts at the centroid;
3. contacts are node-discretised penalty springs with regularised Coulomb friction
   and Cundall local non-viscous damping (quasi-static DEM standard);
4. **bending failure**: the internal moment at every section of a block is obtained
   by sweeping the contact stations along the principal axis and accumulating
   |r × F|; the peak stress `sigma = 6 M / t²` (t = the block's own caliper
   thickness) is compared with the flexural strength.  Overstress cuts the block by
   a straight crack (a half-plane clip, so the two fragment areas add to the parent
   area to round-off), the fragments inherit the rigid-body velocity plus a small
   separation velocity, and they remain breakable down to a minimum size;
5. the piston is a stress servo: it advances while the measured load is below the
   target and backs off above it, which keeps the dense packing quasi-static
   (a velocity-driven piston stores elastic energy in the contact springs of the
   confined chain and releases it as a violent rebound — observed and documented
   below);
6. porosity is measured column-wise on the *actual* polygons: columns whose vertical
   ice extent exceeds 1.8 h belong to the ridge, and `n = 1 - A_ice / A_extent`.

## 3. Results (measured)

| quantity | value |
|---|---|
| convergence reached | 0.0224 |
| cracks (bending failures) | 8 |
| blocks at end | 14 |
| sail height H_s | 0.104 m |
| keel depth H_k | 0.523 m |
| sail/keel ratio | 0.199 |
| keel/sail ratio | 5.016 |
| initial ice volume | 8.994272 m³/m |
| final ice volume | 8.994272 m³/m |
| volume error | 0.00e+00 |
| ridge porosity | nan |
| classification | rubbling |

History:

| strain | H_s [m] | H_k [m] | blocks | cracks | n |
|---|---|---|---|---|---|
| 0.0022 | 0.088 | 0.479 | 6 | 0 | n/a |
| 0.0050 | 0.081 | 0.479 | 8 | 2 | n/a |
| 0.0073 | 0.088 | 0.484 | 10 | 4 | n/a |
| 0.0094 | 0.083 | 0.485 | 10 | 4 | n/a |
| 0.0117 | 0.087 | 0.487 | 10 | 4 | n/a |
| 0.0139 | 0.095 | 0.490 | 12 | 6 | n/a |
| 0.0161 | 0.102 | 0.502 | 12 | 6 | n/a |
| 0.0180 | 0.101 | 0.518 | 13 | 7 | n/a |
| 0.0202 | 0.102 | 0.511 | 14 | 8 | n/a |
| 0.0224 | 0.104 | 0.523 | 14 | 8 | n/a |

## 4. Figures

* `artifacts/p318w3_n08/sail_keel_evolution.png` — sail/keel and block count vs strain.
* `artifacts/p318w3_n08/ridge_cross_section.png` — final block configuration (actual polygons).
* `artifacts/p318w3_n08/porosity_vs_strain.png` — measured porosity vs strain (empty: see §6).

## 5. Verification and benchmark (ref vs got)

| check | reference | got | verdict |
|---|---|---|---|
| volume conservation (rel. error) | 0.0 | 0.0 | PASS |
| keel/sail ratio | 4.4 | 5.015843804500625 | WITHIN_TOL |
| sail/keel ratio (field range 0.2-0.5) | [0.2,0.5] | 0.1993682496856697 | OUT_OF_RANGE |
| porosity law R^2 (target >0.9) | 0.9 | nan | FAIL_NOT_MEASURABLE |
| bending model sigma=6M/t^2 vs analytic beam | analytic | rel. err < 1e-9 (pytest) | PASS |
| crack area conservation | 0.0 | 0.0 | PASS |

Analytic unit tests (pytest, `tests/test_ridge_rubbling.py`, 16 tests):
* beam bending `sigma = 6M/t^2` against a hand-computed two-support moment (rel. err < 1e-9);
* crack area conservation (rel. err < 1e-12) and convexity of both fragments;
* hydrostatic equilibrium: a floating floe sits at `draft = h rho_i/rho_w` with the
  buoyancy force exactly equal to its weight, and does not move over 300 steps;
* single-point contact returns zero bending stress;
* porosity of a hand-computed two-block packing (1 − 2/3.5 within 0.02);
* porosity-law fit recovers (n0, eps_c, R²=1) from synthetic data and reports `nan`
  rather than inventing points when data are missing;
* short compression run: volume error ≤ 1e-12, finite state.

## 6. Error judgement against the mission criteria

The criteria are answered explicitly, including where the answer is **no**:

| criterion | target | result | verdict |
|---|---|---|---|
| (a) volume conservation | < 1 % | 0.0e+00 | **PASS** |
| (b) porosity law n = n0 exp(-eps/eps_c), R² | > 0.9 | not measurable (no column with extent > 1.8 h at eps = 0.022) → R² = nan | **FAIL (honest)** |
| (c) sail/keel ratio H_s/H_k | 0.2–0.5 | 0.199 | **FAIL (marginal, 0.4 % below the range)** |
| (d) rafting→rubbling scaling | literature | classification `rubbling`; scaling law not established (see §7) | **NOT ESTABLISHED** |
| (e) pytest green / coverage | all green | 16/16 pass; coverage not measured in this run | partial |

Against a *quantitative* reference, the only ridge-geometry reference available is
Timco & Burden (1997): keel/sail ≈ 4.4.  Got 5.02 → error
**14.0 %**, i.e. **not < 3 %** (it is within the
±15 % acceptance band, but the sail is only 0.10 m so this is not yet
a developed ridge).  The quantities that *are* met to better than 3 % are the
structural ones: volume conservation (0 %), crack area conservation (<1e-12) and the
bending-stress law (<1e-9) — these are exact properties of the implementation, not
physical predictions.

## 7. What did not work, and why (honest)

1. **Velocity-driven piston is unusable for this problem.** A confined chain of
   floes loaded at prescribed velocity stores elastic energy in the penalty springs
   and releases it as a rebound (measured: the chain recoiled faster than the piston
   and the piston load collapsed to zero); with an early version the load reached
   ~10³ kN and the state diverged.  Switching to a stress servo fixed the loading.
2. **Porous-creep of a dense packing needs local damping.** Without Cundall
   damping the block assembly rang at 2–6 m/s; with it the kinetic energy stays at
   O(10³ J) and the response is quasi-static.
3. **Ridge porosity is not yet measurable in this regime.** At 2.2 % convergence the
   ridge has not developed a column whose ice extent exceeds 1.8 h, so the porosity
   is *undefined* rather than small; hence criterion (b) cannot be evaluated and the
   fitted law from m10822 (R² = 1.0) is **not reproduced** — that fit was the
   assumed exponential fitted to itself, and this run falsifies it as empirical
   evidence.
4. **The rafting→rubbling scaling law was not established.** The aspect-ratio sweep
   is implemented (`run_aspect_sweep`) but the runs available in this session were
   not long enough to resolve a first-crack strain for each h/L; no scaling exponent
   is claimed.
5. Numerical limits: plane strain, penalty contacts (not vertex–edge), mass scaling
   ×8.0 (legitimate for quasi-static loading, but it slows contact
   waves and must not be used for impact), 2-D area porosity ≠ 3-D macroporosity,
   no crushing, no thermal bonding/refreezing.

## 8. Defects found in existing code

* The contact torque on the *second* body of a pair had the wrong sign in my first
  implementation; the same expression appears in
  `tensordem/ice_rafting/mechanics.py:445-447` (`tq_j`), where the torque is
  `+(rj × f_pair)` although the force on that body is `-f_pair`.  The present module
  uses `-(rj × f_pair)`.  `mechanics.py` was left untouched (its verified numbers may
  depend on it) — flagged here for the owning mission.
* `polygon_moments` originally returned an *area* second moment while the inertia
  expects a per-unit-mass one (factor A error); fixed and covered by a test.

## 9. Reproduction

```
python scripts/run_ice_ridge_rubbling.py --out runs/p318w3_n08
PYTHONPATH=src python -m pytest tests/test_ridge_rubbling.py -q
```
Artifacts: `runs/p318w3_n08/ridge_rubbling_run.json`,
`runs/p318w3_n08/benchmark_ridge_rubbling.json`,
`artifacts/p318w3_n08/*.png`.
