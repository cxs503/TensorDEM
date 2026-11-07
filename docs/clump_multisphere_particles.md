# Multi-sphere (clump) particles in TensorDEM

Status: **implemented and verified** for the two-sphere / general multi-sphere
rigid clump with member-summed contacts (campaign node IN14-D3, mission m4950).
Convex polyhedral particles are **not** implemented: they are declared on the
roadmap at the end of this document (see [Roadmap](#roadmap-polyhedral-particles)).

| capability | evidence level | module |
|---|---|---|
| clump geometry, validation, union mass properties | REAL | `src/tensordem/clump.py` |
| closed-form double-sphere reference, voxel cross-check | REAL | `src/tensordem/clump.py` |
| rigid-body clump dynamics (6 DOF), clump-wall contact | REAL | `src/tensordem/clump_dynamics.py` |
| clump-sphere contact (sum over member spheres) | REAL | `src/tensordem/clump_dynamics.py` |
| clump-clump contact (narrow phase between clumps) | NOT_SUPPORTED | — |
| convex polyhedron particle (hull, SAT/clipping) | NOT_SUPPORTED (roadmap) | — |
| rolling resistance / torsion friction for clumps | REAL (m5052) | `src/tensordem/clump_rolling.py`, `docs/clump_rolling_resistance_calibration.md` |

## 1. Why clumps belong in TensorDEM

The bonded-sphere engines (`dem.py`, `dem3d.py`) treat ice as *deformable*
aggregates: every particle is an independent body and the bonds carry the
mechanics. Many DEM problems need the complementary representation: a **rigid
clump**, a set of overlapping spheres glued into one non-deformable body. Clumps
give DEM access to non-spherical shapes (dumbbells, rubble, rounded polyhedra)
while keeping sphere-sphere contact detection — the cheapest and most robust
narrow phase available — because a clump contact is nothing but a *sum of
member-sphere contacts*, each contributing a force at its own contact point and
therefore a torque about the clump centre of mass.

The physics is therefore: (i) correct **mass properties of the union** (overlap
counted once, as it must be for a rigid body), (ii) correct **rigid-body
kinematics** (contact-point velocity includes `omega x (x_contact - x_com)`), and
(iii) an **explicit, verifiable integrator** (velocity Verlet + body-frame
quaternion update). Everything in this note is about making those three things
measurable.

## 2. Formulation

### 2.1 Union mass properties

For members `i` with centres `c_i`, radii `r_i` and a uniform density `rho`, the
union volume, first moment and second moment tensor are

```text
V = sum_i V_i - sum_{i<j} V_ij
S = sum_i S_i - sum_{i<j} S_ij
Q = sum_i Q_i - sum_{i<j} Q_ij          S = int x dV,  Q = int x x^T dV
```

with `V_i = 4/3 pi r_i^3`, `S_i = V_i c_i`, `Q_i = V_i c_i c_i^T + (V_i r_i^2/5) I`
exact for a ball, and `(V_ij, S_ij, Q_ij)` the moments of the **lens** (the
intersection of two balls). The inertia tensor about the centre of mass follows
from `I_O = tr(Q) I - Q` and the parallel-axis (Steiner) shift.

The lens moments are integrated **exactly**, not approximately. Sweeping the lens
along the line of centres, the cross-section at height `z` is a disc of radius

```text
a(z)^2 = R^2 - (z - z_c)^2      (the smaller of the two spheres; they switch at
                                 the radical plane z* = (R1^2 - R2^2 + L^2)/(2L))
```

so `a^2` (and `a^4`) are polynomials of degree <= 4 in `z`. Splitting `[z_lo, z_hi]`
at `z*` and applying an 8-node Gauss-Legendre rule (exact to degree 15) yields the
lens volume and second moments to machine precision. Cross moments vanish by
axisymmetry. For two *equal* spheres the implementation reproduces the textbook
lens volume `pi (2r - d)^2 (4r + d) / 12` to `1e-15` relative.

Inclusion-exclusion is exact only when the overlap regions are **exclusive**: if
three members share a common point the triple intersection is subtracted twice and
added back zero times. That precondition is checked geometrically (one centre
inside all three balls, or the intersection circle of a pair meeting the third)
and the constructor raises `ClumpTopologyError` when it is violated. The
`inertia_model="member_sum"` path (naive parallel-axis sum of whole members,
overlap double-counted) is kept for comparison and for codes that use that
convention; it is *not* the default.

### 2.2 Validation of the clump itself

The constructor rejects, with distinct exception types and messages:

| rejection | exception |
|---|---|
| zero / negative / non-finite radius | `ClumpGeometryError` |
| non-positive or non-finite density | `ClumpGeometryError` |
| empty member list | `ClumpGeometryError` |
| member penetration above `max_overlap_fraction` (default 25 % of the smaller radius) | `ClumpOverlapError` |
| coincident member centres | `ClumpOverlapError` |
| disconnected member graph (not one rigid body) | `ClumpTopologyError` |
| triple (non-exclusive) member overlap | `ClumpTopologyError` |

### 2.3 Member-summed contact and rigid-body dynamics

For every member `i` that penetrates a wall (half-space `(x - p) . n >= 0`) or a
spherical obstacle:

```text
delta_i   = overlap along the contact normal
x_c,i     = c_i - r_i n_i                     (contact point)
a_i       = x_c,i - x_com                      (torque arm)
v_c,i     = v + omega x a_i                    (contact-point velocity)
F_n       = max(k_n delta_i - c_n (v_c,i . n_i), 0)
xi        = xi + v_t dt                        (Cundall-Strack, body frame)
F_t       = -k_t xi,  |F_t| <= mu F_n
F        += F_n n_i + F_t
T        += a_i x (F_n n_i + F_t)
```

with `c_n = 2 zeta sqrt(k_n m_eff)` and the *contact-point* effective mass

```text
1/m_eff = 1/m + n . [ (I^-1 (a x n)) x a ],
```

so that the damping is calibrated against the local dynamic compliance rather
than the crude total mass. Rotation follows Euler's equations in the body frame,
`I_b domega_b/dt = tau_b - omega_b x (I_b omega_b)`, integrated together with the
quaternion; translation uses velocity Verlet. Tangential history is advanced
**once per step** (on the second force evaluation), the same convention
`dem3d.forces(update_history=False)` uses to avoid a phantom `v_t dt` increment.

## 3. Results

All numbers below are produced by `scripts/m4950_clump_validation.py`, which
writes `artifacts/m4950/clump_validation.json` (the report embeds the same
JSON). Reference clump: two spheres, `r = 0.05 m`, `rho = 900 kg/m^3`,
`m = 0.94247780 kg`.

### 3.1 Criterion 1 — inertia vs the closed-form dumbbell

Analytic reference (`I_xx` along the line of centres):
`I_xx = 2 (2/5) m_s r^2`, `I_yy = I_zz = 2 (2/5 m_s r^2 + m_s (d/2)^2)`.

| d / r | I_xx (computed) | I_xx (analytic) | I_yy (computed) | I_yy (analytic) | rel. error |
|---|---|---|---|---|---|
| 2.0 (touching) | 9.424777960769e-04 | 9.424777960769e-04 | 3.298672286269e-03 | 3.298672286269e-03 | 1.3e-16 |
| 2.2 | 9.424777960769e-04 | same | 3.793473129210e-03 | 3.793473129210e-03 | 2.3e-16 |
| 3.0 | 9.424777960769e-04 | same | 6.243915399010e-03 | 6.243915399010e-03 | 1.4e-16 |

**Worst relative error 2.3e-16 — machine precision, target `< 1e-10` met.**
Mass matches to `1.2e-16`.

### 3.2 Independent numerical cross-check of the union geometry

`mass_properties_by_voxel` integrates the union geometry by midpoint quadrature
on a uniform grid, counting a voxel once if it lies in *any* member. It shares no
code with the exact moments, so it catches exactly the class of error that blinds
analytical self-checks (see §5.1).

| d / r | overlap volume fraction | member-sum inertia overestimate | voxel inertia error (48 / 96 / 192 / 256 cells) |
|---|---|---|---|
| 2.0 | 0.0 % | 0.0 % | 3.0e-3 / 1.6e-4 / 3.0e-5 / 4.1e-5 |
| 1.5 | 4.3 % | 0.38 % | 3.7e-3 / 7.1e-4 / 1.1e-4 / 3.9e-5 |
| 1.3 | 8.1 % | 1.20 % | 1.9e-3 / 3.5e-4 / 6.9e-5 / 5.8e-5 |

The voxel error contracts with resolution towards the exact value, and the naive
member sum over-predicts the transverse inertia by 0.4-1.2 % for the overlaps
shown (larger for deeper overlap) — the error is real but second-order in the
overlap, which is why the 25 % overlap limit is a defensible default.

### 3.3 Criterion 2a — free fall, no rotation

`dt = 1e-3 s`, 2000 steps (2 s), tilted initial orientation.

| quantity | value |
|---|---|
| energy drift, relative | **1.35e-12** |
| energy drift, absolute | 2.50e-11 J (E = 18.49 J) |
| position error vs `z0 - g t^2/2` | 1.03e-12 m (5.9e-14 relative) |
| max `|omega|` | 0.0 (exactly) |
| angular momentum increment | 0.0 (exactly) |
| quaternion drift | 0.0 (exactly) |
| final velocity vs analytic | -19.620000000 vs -19.62 m/s |

Conservation is at round-off level: velocity Verlet is exact for constant
acceleration, and the rotational state is frozen because no torque acts.

### 3.4 Criterion 2b — inclined-plane rolling (directionality band)

"Rolling-pin" dumbbell: the member axis is transverse to the slope, so both
members stay in contact while the clump rolls about its own axis. The predicted
acceleration is the rolling-without-slip value

```text
a = g sin(theta) / (1 + I_roll/(m r^2)),  I_roll = 0.4 m r^2  ->  a = 5/7 g sin(theta)
```

for `theta = 10 deg`, `mu = 0.6`, `dt = 1e-4 s`, 0.5 s.

| quantity | value |
|---|---|
| analytic `a` (rolling) | 1.216777587795 m/s^2 |
| measured `a` | **1.216777535 m/s^2** (relative error 4.4e-8) |
| frictionless-slide band upper bound | 1.703488622913 m/s^2 |
| no-slip residual `v - r omega` | -9.3e-8 m/s (1.5e-7 relative) |
| normal offset | 0.9991 r (on the plane) |
| mirrored tilt (`-10 deg`) | 1.216777535 m/s^2, difference 0.0, `omega` sign flips |

The measured acceleration sits inside the admissible band
`[a_roll, a_slide]` and agrees with the rolling value to 4e-8 — the friction
spring and the union inertia are consistent with the analytic rolling constraint.

### 3.5 Contact bookkeeping

| check | value |
|---|---|
| static two-member contact carries the weight | 4.622853589757 + 4.622853589757 = 9.245707179515 N vs `m g` = 9.245707179515 N (0.0 relative error) |
| analytic penetration `m g /(2 k_n)` | 4.622853589757e-05 m (4.6e-14 relative error) |
| clump-sphere third-law closure `F_obstacle = -sum_i F_i` | residual **0.0 N** over the whole contact |
| off-centre impact on a spherical obstacle | approach -1.734 m/s, after contact -0.102 m/s (repelled, torqued) |
| measured restitution vs target (log-decrement `zeta`) | 0.137/0.315/0.468/0.619/0.777/0.886 for 0.2/0.35/0.5/0.65/0.8/0.9 (max abs error 0.063) |

### 3.6 Test suite

`python -m pytest tests/test_clump_multisphere.py -q` → **36 passed** (plus the
full repository suite still collects 1745 tests with no collection errors). The
suite contains 11 negative controls (invalid radius, over-limit overlap,
coincident members, empty list, invalid density, disconnected members, triple
overlap, unknown inertia model, and illegal restitution/damping parameters) and
the quantitative criteria above.

## 4. Figures

* `artifacts/m4950/fig1_clump_geometry_inertia.png` — union geometry and the
  member-sum vs union inertia.
* `artifacts/m4950/fig2_voxel_convergence.png` — voxel integration converging to
  the exact union inertia.
* `artifacts/m4950/fig3_free_fall.png` — free-fall trajectory vs the closed-form
  parabola and the energy error.
* `artifacts/m4950/fig4_inclined_rolling.png` — rolling displacement against the
  analytic parabola and the `r omega = v` closure.
* `artifacts/m4950/fig5_contact_summation.png` — net obstacle reaction equals the
  sum of the member-sphere contact forces.

## 5. Notes, pitfalls and limitations

### 5.1 Two defects this work caught (and why they matter)

1. **Missing transport terms in the lens moments.** The lens second moments were
   first assembled in the line-of-centres frame without transporting them from
   the slice origin to the global origin. Volume was right to `1e-4`, but the
   transverse inertia of an overlapping pair was **systematically wrong by
   2.5 %** — and refinement of the voxel grid did not reduce it, which is the
   signature of a modelling bug rather than a discretisation error. Fixed by
   applying `Q_global = V c c^T + c s^T + s c^T + R Q_local R^T`. Routed out only
   because the numerical cross-check was independent of the analytic path.
2. **Contact-point velocity from the member-centre arm.** Using
   `v + omega x (c_member - x_com)` instead of
   `v + omega x (x_contact - x_com)` drops the `omega x (-r n)` term. For a
   single-sphere clump the contact point then looks stationary while the body
   spins up: the sphere rotated in place at 170 rad/s with zero translation and
   an unreachable "no-slip" state. Fixed by `ClumpBody.point_velocity`.

### 5.2 Limits of the model as implemented

* **No clump-clump contact.** Only clump-wall and clump-sphere (an infinite-mass
  spherical obstacle) are implemented. Two clumps require the reverse
  member-vs-member loop plus the corresponding reaction on the second body's
  torque; the contact bookkeeping is written to make that extension local.
* **Triple member overlaps are rejected**, not silently mis-integrated. Clumps
  whose members form a densely overlapping core (common for rounded polyhedra)
  need inclusion-exclusion to higher order or a voxel/hull-based mass property;
  for such shapes use `inertia_model="member_sum"` knowingly, or wait for the
  polyhedron path.
* **Restitution is approximate**: the linear spring-dashpot undershoots the
  requested coefficient by 0.014-0.063 absolute over `e = 0.2 ... 0.9`
  (measured, §3.5). This is the documented behaviour of the log-decrement
  calibration; calibrate `restitution` against a measured impact instead of
  trusting the parameter when the coefficient matters.
* **Tumbling geometries lock in Cundall-Strack.** A dumbbell whose axis points
  *down* the slope must tumble end over end; the two member contacts then
  alternately separate, the tangential springs reset, and the model settles into
  a spurious static state (no rolling). This is a known limitation of
  history-based Coulomb friction with alternating contact, not a bug in the
  integrator: it is why the verification case uses the "rolling-pin"
  configuration, where both members stay in contact. Rolling resistance
  (`T_r = mu_r R |F_n|`) is not implemented and would not remove the artifact.
* **No rotational damping or torsion friction**, so a clump can spin freely about
  a contact normal.
* **Mass properties assume uniform density.** Layered/heterogeneous clumps are
  not supported.

## 6. Reproduction

```bash
python -m pytest tests/test_clump_multisphere.py -q      # 36 passed
python scripts/m4950_clump_validation.py                 # writes artifacts/m4950/
```

## 7. Roadmap: polyhedral particles

Status: **NOT_SUPPORTED** — declared here, not implemented. The staged plan keeps
every step independently verifiable, and reuses the mass-property machinery above.

**Stage P1 — convex hull geometry (no contact yet).**
`ConvexPolyhedron` from a vertex set: half-space representation via
`scipy.spatial.ConvexHull`, exact volume, centroid and inertia tensor by
decomposing the hull into tetrahedra `(origin, v_i, v_j, v_k)` and summing the
closed-form tetrahedron moments. Verification: cube/icosahedron/sphere-limit
comparison against `Clump` with a dense member set (relative inertia error must
decrease as the member count grows), plus divergence-free tetrahedralisation
(`sum det = 6V`).

**Stage P2 — polyhedron vs plane and polyhedron vs sphere.**
Plane: support-function penetration (`min over vertices of (v - p) . n`), one
contact point at the deepest vertex plus a face-clip fallback. Sphere: closest
point on the hull (vertex/edge/face cases). Verification: rest on a plane carries
the weight; centre-of-mass torque vanishes for a symmetric hull; contact point
continuity across vertex-edge-face transitions.

**Stage P3 — polyhedron vs polyhedron narrow phase.**
Minkowski-difference GJK for the overlap test plus a contact-manifold builder
(SAT for the separating axis, clipping for the penetration patch). Verification:
two cubes in face-face, edge-edge and vertex-face configurations against
analytically known penetration depths and contact areas; symmetry
`contact(A,B) == mirrored contact(B,A)`.

**Stage P4 — coupling.** Polyhedra as clump *templates* (a clump of members
approximating a hull, reusing the member-summed contacts here) vs true hull
contact, compared on the same impact and packing cases to quantify where the
multi-sphere approximation is acceptable (`< 5 %` peak-force and penetration
error is the working target).

Nothing in stage P1-P4 is implemented in this mission; the tables above are the
baseline that a polyhedron implementation must beat or match.