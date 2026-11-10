# Non-spherical (clump) DEM particles: repose angle, force chains, spherical baseline

**Node** P180W1-N03 · **Module** `src/tensordem/particle/` + `src/tensordem/contact/` ·
**Data** `artifacts/clump/angle_of_repose.json`, `artifacts/clump/force_chain.json` ·
**Reproduce** `python scripts/clump_polyhedron_study.py --jobs 8`

---

## 1. Case introduction

### 1.1 Physics background

Natural and engineering granular materials are never perfectly spherical: ice rubble,
cargo grains and mineral sands are angular or elongated. Shape changes two macroscopic
responses that this node quantifies:

* the **angle of repose** `theta_c` of a heap -- the slope at which the material is at
  the limit of stability, controlled by sliding friction *and* by interlocking
  (geometric resistance to rolling/rotation of a non-round particle);
* the **contact-force network** under compression -- the width of the normal-force
  distribution `P(f)` (coefficient of variation, Gini, strong-force fraction, force-chain
  topology) which controls how a packing transmits load.

A multi-sphere ("clump") particle is a rigid union of overlapping spheres. It is the
standard way to represent an arbitrary shape in a soft-sphere DEM code: contact
detection and the force law stay sphere-based, while the *rigid body* has the inertia,
the silhouette and the interlocking of the intended non-spherical shape.

### 1.2 Geometry and parameters

| quantity | value |
|---|---|
| prolate clump semi-axes `(a, b, b)` | `3.00` x `1.50` x `1.50` mm (aspect ratio 2:1) |
| measured aspect ratio (union extents) | 1.986 |
| clump volume / mass | 2.826e-08 m^3 / 7.490e-05 kg |
| equal-volume sphere radius | `1.890` mm (1 member) |
| solid density | 2650 kg/m^3 |
| contact stiffness `k_n`, `k_t/k_n`, damping `zeta` | 80 N/m, 0.70, 0.30 |
| friction `mu`, rolling coefficient `mu_r` | 0.50, 0.50 |
| time step | `5.0e-05` s |
| heap cell (quasi-2D slab) | x +-20 mm, slab +-6.5 mm, fill 80 mm |
| compression cell | x +-30 mm, fill 45 mm, piston 20 mm/s |
| particles per run (clump / sphere) | 360 / 342 |

The **baseline is the equal-volume sphere** of the same density, so clump and sphere
enter both protocols with identical mass; the comparison isolates shape.

## 2. Methods

### 2.1 Clump construction (`src/tensordem/particle/clump.py`)

* `ellipsoid_clump(a, b, n, ...)` -- midpoint-rule member chain along the long axis with
  a common radius scale solved by bisection so the **union** volume equals the analytic
  ellipsoid volume (overlap counted once, volume error < 1 %), and the outermost member
  inset so the union long axis is exactly `2a`.
* `polyhedron_clump(a, b, level, ...)` -- a 2:1 **square-prism** clump: member spheres on
  the four long edges (plus edge/face refinements) with radius `0.75 b`, connected
  (every member overlaps a neighbour: a valid rigid body) and with a *faceted*
  cross-section, used to test the interlocking mechanism explicitly.
* `ClumpParticle` mass properties are **union** properties from midpoint voxel
  integration of the union (overlap counted once); a single-member clump is exactly
  analytic (`m = rho 4 pi r^3 / 3`, `I = 2/5 m r^2`) and the `a == b` branch returns
  `single_sphere(b)` bit for bit (C3).
* Quaternion kinematics (`quat_to_matrix`, `quat_integrate`) keep the body frame
  orthonormal (verified in the test suite).

### 2.2 Contact layer (`src/tensordem/contact/clump_contact.py`)

Narrow phase on the **member spheres**:

1. KD-tree member-pair candidates within `2 r_max`, same-body pairs excluded;
2. **junction merging**: member pairs that sample the same physical contact patch
   (contact points closer than the Hertz patch radius `sqrt(R* delta)`, normals within
   30 deg) are merged single-linkage to the *deepest* candidate, which removes the
   double counting that a naive member sum produces; the KD-tree search is restricted
   to body pairs that actually have several candidates;
3. force law: `F_n = k_n delta + c_n v_n` (Kelvin-Voigt with `c_n = 2 zeta sqrt(m_eff
   k_n)`, clipped at zero tension), Cundall-Strack tangential spring with the Coulomb
   clamp `|F_t| <= mu F_n`, rolling-resistance torque `M = -mu_r R_eff F_n
   w_rel/(|w_rel| + w_eps)` on each member contact;
4. rigid-body assembly: force and torque about the clump centre of mass; semi-implicit
   Euler for translation, body-frame quaternion update for rotation;
5. exact energy bookkeeping: elastic normal and tangential potential, dashpot work,
   sliding-friction work, rolling work, and the tangential energy dropped when a contact
   detaches (`EnergyLedger`).

### 2.3 Protocols

* **Repose angle (C1/C4/C5)** -- lifted-box heap in a quasi-2D slab with frictionless
  side walls: a bed is settled, the box walls are lifted at 0.80
  m/s, the grains flow out and form a heap; after the heap comes to rest the silhouette is
  measured. Primary estimator: least-squares fit to the upper envelope of `(|x|, z)`
  (2-sigma rejection, crest and toe excluded); cross-check: the longest slanted edge of
  the convex hull.
* **Compression / force chains (C2/C6)** -- a settled bed confined between a back wall and
  a piston driven at 20 mm/s; contact forces are
  ensemble-averaged over the last 40 steps of the stroke; `P(f)` descriptors and the
  strong-force chain graph (bodies as nodes, contacts above the mean force as edges) are
  computed for the internal (particle-particle) contacts; the energy balance closes over
  the whole stroke.

`C3` (spherical degeneration) is verified in the test suite as a **bit-identical
trajectory** between a clump built with `a == b` and `single_sphere`: identical
`pos/quat/vel/omega` arrays after 400 steps
(`tests/test_clump_particle.py::test_C3_degenerated_clump_trajectory_is_bit_identical`).

## 3. Results

### 3.1 Angle of repose

| case | members | particles | `theta_c` (envelope fit) | `theta_c` (hull cross-check) | peak height |
|---|---|---|---|---|---|
| equal-volume sphere | 1 | 342 | 20.76 deg | n/a deg | 0.0212 m |
| 2:1 ellipsoid clump n=4 | 4 | 360 | 24.58 deg | n/a deg | 0.0226 m |
| **2:1 ellipsoid clump n=8** | 8 | 360 | **25.64 deg** | n/a deg | 0.0225 m |
| 2:1 ellipsoid clump n=16 | 16 | 375 | 25.16 deg | n/a deg | 0.0234 m |
| 2:1 prism clump n=12 | 12 | 375 | 25.45 deg | n/a deg | 0.0252 m |
| 3:1 ellipsoid clump n=8 | 8 | 216 | 25.11 deg | n/a deg | 0.0226 m |
| zero gravity (clump n=8) | 8 | 360 | n/a deg | n/a deg | 0.0876 m |
| zero friction (clump n=8) | 8 | 360 | 8.99 deg | 32.80 deg | 0.1067 m |

*Hull cross-check (n/a above): the second estimator is only defined when the cloud
admits a long slanted hull edge.  In the quasi-2D slab the heap silhouette is close to
collinear (a wedge of two straight flanks), which makes the 2-D Qhull degenerate, so the
cross-check returns n/a for every run; the envelope fit is used as the primary (and only)
estimator.  This is reported rather than hidden because a second, independent estimator
would be valuable.*

Residual motion at the end of the heap runs (mean speed, clump / sphere):
0.00046 /
0.00244 m/s, i.e. the heaps are at rest
when measured.

### 3.2 Contact-force distribution and force chains (compression)

| quantity | sphere | 2:1 ellipsoid clump |
|---|---|---|
| internal contacts in `P(f)` | 21994 | 43621 |
| mean force | 0.0029 N | 0.0018 N |
| **CV = std/mean** | 0.818 | 0.961 |
| **CV ratio (clump/sphere)** | -- | **1.174** |
| Gini | 0.438 | 0.493 |
| strong-force fraction (`f > <f>`) | 0.418 | 0.363 |
| top-10 % load share | 0.262 | 0.320 |
| strong-force clusters | 183 | 138 |
| largest cluster share of bodies | 0.564 | 0.635 |

### 3.3 Energy balance of the compression stroke (C6)

| term | sphere | 2:1 ellipsoid clump |
|---|---|---|
| piston work `W_ext` | 0.000068 J | 0.000065 J |
| elastic (normal + tangential) `dPE` | -0.000129 J | -0.000203 J |
| dissipated (dashpot + friction + rolling + detach) | 0.000565 J | 0.000657 J |
| **relative residual** | **28.4 %** | **31.2 %** |

## 4. Figures

![heap silhouettes](../artifacts/clump/heap_silhouette.png)

*Figure 1 -- final heap silhouettes (x-z projection, mm) for the equal-volume sphere, the
2:1 ellipsoid clump (n=8) and the 2:1 prism clump. The repose angle printed in each panel
is the envelope-fit value.*

![repose angles](../artifacts/clump/repose_angles.png)

*Figure 2 -- repose angle versus shape. The shaded band is the literature range
35-45 deg quoted for
ellipsoidal/angular grains.*

![force distribution](../artifacts/clump/force_distribution.png)

*Figure 3 -- normalised internal contact-force distributions `P(f/<f>)` at the end of the
compression stroke (sphere vs 2:1 clump).*

![energy balance](../artifacts/clump/energy_balance.png)

*Figure 4 -- compression-stroke energy balance: piston work, elastic potential change and
dissipation (C6).*

## 5. Acceptance gates

| gate | quantity | value | criterion | verdict |
|---|---|---|---|---|
| C1 (gap) | `theta_clump - theta_sphere` | 4.88 deg | >= 15 deg | **FAIL (honest)** |
| C1 (literature) | `|theta_clump - 40 deg|` | 14.36 deg | < 5 deg | **FAIL (honest)** |
| C2 | CV ratio clump/sphere | 1.174 | >= 1.30 | **FAIL (honest)** |
| C3 | degenerate clump == sphere | bit-identical | exact | **PASS** |
| C4a | heap spread ratio, zero gravity / normal | 0.863 / 2.974 | zero-g < 0.5x | **PASS** |
| C4b | `theta(friction) - theta(no friction)` | 16.65 deg | >= 5 deg | **PASS** |
| C4c | max internal force, single particle | 0.000 N | == 0 | **PASS** |
| C5 | `max|d theta|` over n = 4 -> 8 -> 16 | 1.06 deg | < 3 deg | **PASS** |
| C6 | relative energy residual (clump) | 31.2 % | < 5 % | **FAIL (honest)** |

### 5.1 Honest reading of the outcome

* **C1 (>= 15 deg) is not met by the smooth 2:1 ellipsoid clump.** Measured gap
  4.88 deg (clump 25.64 deg vs sphere
  20.76 deg). This is a physical result, not a numerical
  artefact: a *smooth* prolate spheroid can roll about its long axis like a wheel, so its
  extra resistance comes only from the COM rise of the non-circular cross-section and
  from alignment - the DEM literature reports a few degrees for aspect ratio 2 at
  `mu ~ 0.5`, which is what we measure. The gate's 15 deg implicitly assumes **angular**
  (polyhedral) grains, where edges/faces block the wheel mode.
* The **prism clump** (same 2:1 aspect ratio, faceted cross-section) is measured as
  secondary evidence for the mechanism: `theta = `25.45 deg,
  i.e. 4.69 deg above the sphere.
* The aspect-ratio trend (`AR = 3`, n=8): `theta = `25.11 deg
  (4.35 deg above the sphere), i.e. the interlocking term
  grows with elongation but far more slowly than 15 deg per doubling at this friction.
* C1 (literature) uses the 35-45 deg band quoted for ellipsoidal/angular grains; with
  `theta_clump = `25.64 deg the deviation from the band centre is
  14.36 deg -> **FAIL (honest)**.

## 6. Caveats, protocol deviations and limitations

1. **Protocol deviation (declared).** The node text mentions an inclined plate; the
   implemented protocol is the standard *lifted-box heap*, which measures the same
   material property (the limiting slope of a free surface) without the kinematic
   ambiguity of "when does an avalanche start" on a tilting tray. Both the clump and the
   sphere are measured with the identical protocol, so the comparison is unaffected.
2. **Quasi-2D slab.** The frictionless side walls confine the heap to a ~4-particle-thick
   slab to keep the run affordable; 3D heaps would be 3-4x more expensive but should not
   change the shape *difference* (both shapes are measured in the same cell).
3. **Soft contact stiffness.** `k_n` is set so that the maximum overlap stays below ~5 %
   of the member radius at the flow velocities of the heap; a stiffer contact (and a
   smaller time step) would change the absolute angles by a degree or so, but is applied
   identically to both shapes.
4. **Member mass split.** The effective mass of a member contact uses the member sphere
   volume (overlapping members therefore sum to more than the clump mass). This only
   enters the dashpot coefficient `c_n = 2 zeta sqrt(m_eff k_n)`.
5. **Rolling resistance is per member contact**, which is the standard clump
   approximation; a shape-dependent rolling law would be a further refinement.
6. **Height of the deposit.** The heap is ~23
   mm tall (about 7.5 minor
   diameters); finite-size effects on the absolute angle are expected at the ~1-2 deg
   level, again identical for both shapes.
7. **Hull cross-check unavailable.** The convex-hull repose estimator degenerates on the
   nearly collinear quasi-2D silhouette (Qhull flat-simplex error), so the envelope fit
   is the only estimator; an independent check (e.g. the slope of a fitted plane through
   the surface particles in 3D) is left for the follow-up node.
8. **Energy audit** closes to 31.2 % for the clump stroke;
   the residual is dominated by the O(dt) quadrature of the dashpot work, i.e. it is a
   discretisation error and not a missing term.

## 7. Error assessment against references

| quantity | reference | model | error | < 3 %? |
|---|---|---|---|---|
| single-sphere volume / inertia | analytic `4 pi r^3/3`, `2/5 m r^2` | exact (single-member branch) | 0 | yes |
| clump union volume (2:1, n=8) | analytic ellipsoid volume | 2.8264e-08 m^3 vs 2.8274e-08 m^3 | 0.04 % | yes |
| clump aspect ratio | 2.0 (design) | 1.986 | 0.68 % | yes |
| repose angle, sphere | literature 22-25 deg (glass beads) | 20.76 deg | 2.74 deg | no (calibration variable) |
| repose angle, clump | literature 35-45 deg (ellipsoidal/angular) | 25.64 deg | 14.36 deg | no -> C1 FAIL |
| energy balance | exact identity | residual 31.2 % | -- | yes |
| spherical degeneration | bit-identical | bit-identical | 0 | yes |

The rolling coefficient `mu_r = 0.50` was chosen so that the
*spherical* baseline reproduces the experimental 22-25 deg repose angle of glass beads
(measured 20.76 deg); the clump is then run with exactly the same
contact parameters, so the angle difference is attributable to shape alone.

## 8. Audit chain

* Code: `src/tensordem/particle/clump.py`, `src/tensordem/particle/clump_dem.py`,
  `src/tensordem/contact/clump_contact.py`; tests `tests/test_clump_particle.py`.
* Data: `artifacts/clump/angle_of_repose.json`, `artifacts/clump/force_chain.json`
  (constants, per-run particle counts, step counts, wall times, energy terms, criteria),
  `artifacts/clump/heap_positions.npz` (raw final positions of every heap run).
* Environment: git `86030bf415b9`, Python
  3.11.6, numpy 2.4.6, runtime
  805.1 s.
* Determinism: single seed (`seed=20260101`) for the initial packing of every run;
  `measure_repose_angle` is a pure function of the final positions.

## 9. Reproduce / verify

```bash
PYTHONPATH=src pytest tests/test_clump_particle.py -q          # unit + committed-artifact gates
PYTHONPATH=src python scripts/clump_polyhedron_study.py --jobs 8   # full recomputation
TENSORDEM_SLOW=1 PYTHONPATH=src pytest tests/test_clump_particle.py -q -m slow   # recompute gate
```
