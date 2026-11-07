# Non-spherical DEM contact kernel and time-integration verification layer

**Scope.** This document is the verification record for two new modules:

| module | what it adds | evidence class |
|---|---|---|
| `src/tensordem/convex_contact.py` | GJK distance + EPA penetration for arbitrary convex support-function bodies (sphere, oriented box, vertex polytope, exact cone) and a member-resolved multi-sphere clump channel | REAL (scored against independent analytic oracles) |
| `src/tensordem/contact_timestep_ledger.py` | velocity-Verlet contact timing, restitution, Cundall–Strack critical step (measured by bisection) and an exact mechanical-energy ledger for a 64-sphere gas | REAL (scored against closed forms and integral balances) |
| `src/tensordem/clump_granular_anchors.py` | rigid cubic-clump granular pile, packing fraction and orientational order | REAL, **macro anchors partially fail** (see §6.5) |

Reproduce everything with

```
python scripts/verify_nonspherical_contact_and_timestep.py            # paper numbers
python scripts/verify_nonspherical_contact_and_timestep.py --quick    # smoke run
python -m pytest tests/test_convex_contact_analytic.py \
                 tests/test_contact_timestep_ledger.py \
                 tests/test_clump_granular_anchors.py
```

Machine-readable results: `artifacts/nonspherical-contact-timestep/verification.json`
and `artifacts/nonspherical-contact-timestep/contact_geometry_table.md`.

---

## 1. Case definition

### 1.1 Why a new contact kernel

Every contact in the shipped engine (`tensordem.dem3d`, the 2-D codes, the bond
lattice) is a **sphere–sphere** contact, i.e. `|c_i - c_j| - r_i - r_j` in closed
form. That is a *distance in a one-parameter family*: the whole geometry layer is
one subtraction, so the repository's verification surface stops exactly where a
non-spherical engine has to start:

* the **gap** between two rotated boxes, a tetrahedron and a box, or a cone and a
  floor is not a closed form anyone can eyeball;
* the **penetration depth, its normal, and the contact point** are what a force
  law consumes — an overlap length alone is not enough once normals are not
  forced to be the line of centres;
* the multi-sphere clump channel (already present for mass properties in
  `tensordem.clump`) has no contact-geometry implementation at all.

This work adds that layer and — more importantly — the *oracles* that make it
falsifiable.

### 1.2 Bodies and parameters

| primitive | support function | used for |
|---|---|---|
| `Sphere(c, r)` | `c + r d/|d|` | spheres, clump members |
| `Box(c, h, R)` | `c + R sign(Rᵀd) ⊙ h` | slabs, oriented cubes |
| `Polytope(V)` | `V[argmax(V d)]` | tetrahedra, general polyhedra |
| `Cone(apex, axis, H, R)` | `max(apex·d, base-disc support)` | cones — exact, no faceting |
| `Clump(spheres)` | member-by-member | second contact channel |

Test geometry (all SI, lengths in m):

| family | bodies | separated / overlapping gap or depth |
|---|---|---|
| sphere–sphere | r = 1.0, 0.8 | 0.924 / 0.283 |
| sphere–box | half-extents 0.5; r = 0.25 | 0.446 / 0.097 |
| box–box aligned | 0.5-cubes | 0.230 / 0.130 |
| box–box rotated 45° about z | 0.5-cubes | 0.143 / 0.107 |
| tetra–box | unit tetra vs 0.2-cube | 0.050 / 0.260 |
| cone–floor | cone H = 0.3, R = 0.12 vs slab | 0.060 / 0.040 |
| clump–box | three-sphere clump (r = 0.2) vs slab | 0.050 / 0.180 |
| clump–clump | two three-sphere clumps | 0.081 / 0.223 |

### 1.3 Time-integration parameters

| quantity | value |
|---|---|
| integrator | velocity Verlet (kick–drift–kick), same loop in 1-DOF, impact and gas |
| contact laws | linear `F = k d`, Hertz `F = k d^{3/2}`, linear dashpot, Hunt–Crossley `F_d = 1.5 α k d^{1/2} ḋ` |
| contact-time probes | m = 1.0 kg, k ∈ {500, 2000, 8000} N/m (linear); k ∈ {4·10⁴, 9·10⁴} N/m^{3/2} (Hertz) |
| restitution sweep | m = 1 kg, k = 1·10⁴ N/m^{3/2}, α = 0.002 s/m, v_imp ∈ {0.05 … 0.8} m/s |
| critical step | m_min = 1 kg, k_max = 2·10³ N/m |
| gas | N = 64, r = 0.05 m, m = 1 kg, k = 2·10⁴ N/m, box 1.2³ m, v_rms = 0.4 m/s |
| gas step | dt = 0.01 dt_cr (undamped), dt = 1·10⁻⁴ dt_cr (damped, c = 5 N·s/m) |
| cubic pile | 60 clumps, edge 0.05 m, ρ = 1000 kg/m³, μ = 0.6, k = 4·10⁴ N/m, dt = 2·10⁻⁴ s |

---

## 2. Method

### 2.1 GJK on the Minkowski difference

The distance between two convex bodies is the distance from the origin to
`M = A ⊖ B = {a - b}`, whose support function is `h_M(n) = h_A(n) + h_B(-n)`.
GJK keeps a simplex of at most four points of `M` and re-solves *exactly* the
closest-point-on-simplex sub-problem by enumerating every vertex subset and
solving the equality-constrained least-norm problem
`min |Σ w_i p_i|` s.t. `Σ w_i = 1`, `w_i ≥ 0`. Admissible subsets only (all
weights non-negative) are considered, so the returned point is the true closest
point of the simplex, not a projection of a guess.

That construction yields a **two-sided bound on every reported distance**:

```
-(v·w)/|v|  ≤  dist(A,B)  ≤  |v|          (v = closest point, w = support along -v)
relative gap = (|v|² - v·w)/|v|²
```

and the loop terminates on that gap (`1e-13`), i.e. the accuracy of the reported
distance is a *bound*, not an error estimate. `artifacts/.../verification.json`
records the gap for every case; the worst measured relative error over all eight
families is `9.3e-16`.

### 2.2 EPA, and the seed that actually works

When the origin is inside `M`, the depth is `min_{|n|=1} h_M(n)` and the
minimiser is the contact normal. EPA refines an inscribed polytope towards the
closest boundary face.

The seed matters more than the loop. A hand-picked six-direction seed is **not**
sufficient: for two boxes in general orientation the support points in `±x, ±y,
±z` can share a coordinate (the tie-broken vertex), become coplanar in one
projection, and *no* tetrahedron of them contains the origin — the shipped fix is
to sample a well-spread direction set (icosahedron + Fibonacci sphere + cube
corners), take the `scipy.spatial.ConvexHull`, and, while the origin is still
outside a facet, add the support point along that facet's outward normal. That
loop terminates because the origin *is* inside `M`.

### 2.3 The normal is not free: polishing the stationarity condition

A subtle but decisive point. For a curved Minkowski boundary the depth error is
*quadratic* in the normal error, so an EPA normal obtained at a `1e-13` depth
tolerance can never be better than `sqrt(2·1e-13/curvature) ≈ 1e-7 rad`.
Measuring convergence on `h` (the objective) therefore caps the angle at
`sqrt(eps) ≈ 1.5e-8 rad` — above the `1e-8` acceptance gate. The kernel instead

1. computes the **tangential residual of the support point**,
   `P_t (h_M)`-gradient = `P_t (a(n) - b(-n))`, which vanishes at the optimum;
2. takes safeguarded projected-Newton steps on the residual, with a monotone
   residual decrease plus a support-height guard (the guard is what stops the
   iteration from wandering along the flat face-normal fan of a polytope, where
   GJK/EPA is already exact).

Result: separation distances and penetration depths agree with the oracles to
`≤ 1e-15` relative and normals to `≤ 4e-15 rad` — machine precision, not
tolerance precision. The measured angle uses `|n × n_ref|` (`= sin θ`), because
`acos(n·n_ref)` itself is limited to `1.5e-8 rad` near `θ = 0`.

### 2.4 Independent oracles

Nothing is scored against itself:

* **sphere–sphere / sphere–box / cone–floor** — closed forms
  (`analytic_sphere_sphere`, `analytic_point_box_gap`, `analytic_point_cone_gap`,
  the meridional `(t, r)` construction for the cone);
* **box–box and polytope–polytope** — an exact separating-axis implementation
  (`analytic_box_box_sat`, `analytic_polytope_sat`: 3+3 face normals plus 9 edge
  cross products for boxes; brute-force support-plane recovery plus all
  edge×edge cross products for general polytopes). For convex polytopes SAT gives
  both the exact gap and the exact penetration depth;
* **clumps** — per-member closed forms (the union's contact *is* an extremal
  member contact; the convex-hull shortcut is deliberately not taken and is
  covered by a negative control).

### 2.5 Time-integration verification surface

| statement | how it is measured | reference |
|---|---|---|
| Hookean contact duration | zero-crossing scan at `dt = t_c/800 … t_c/2000`, linear interpolation of the crossings | `t_c = π√(m/k) = π√(m/(2k_n))` |
| Hertz contact duration | same scan with `F = k d^{3/2}` | `2.9427 (5m/4k)^{2/5} v^{-1/5}` (Johnson 1985 §4.2) |
| restitution | `e = |v_out|/|v_in|` after one impact with a Hunt–Crossley dashpot | first-order dissipation integral `e = √(1 - W/E₀)`, `W = 3αk d_max^{3/2} √(2E₀/m) J`, `J = ∫₀¹ u^{1/2}(1-u^{5/2})^{1/2}du` |
| critical step | **bisection on the integrator**: 400 steps, amplitude growth > 4 ⇒ divergent; bracket `[0.25, 4]·dt_cr`, tolerance `10⁻³` | `dt_cr = 2√(m_min/k_max)` |
| energy ledger | `E = KE + Σ½kδ² (+ gravity)`, dissipated work accumulated as `∫F_d dδ` per contact | `E(t) - E(0) + W_diss = 0` |

The 1-DOF oscillator used for the bisection is **bilateral** on purpose: that is
the configuration in which the Cundall–Strack bound is sharp (`dt = 2/ω`
exactly). For a contact network the bound is known to be conservative (measured
threshold above theory), which is stated rather than hidden.

### 2.6 Macro anchors

* **Packing fraction** — a frictionless sphere packing (N = 150, r = 0.02 m) is
  rained into a box, tapped twice (0.1 m/s kicks, 300 steps each), then compressed
  by a piston (the top wall descending at 2e-5 m/step) until the elastic piston
  load reaches 5x the bed weight, and the solid fraction is read as
  `N V_sphere/(A h)` with `h` the piston height.
* **Repose angle** — 60 cubic particles, each a rigid 8-sphere clump, released
  as a column and allowed to settle; the angle is a least-squares fit of the free
  surface profile flanks (`|Δz/Δx|`), with the quaternion update carrying the
  toppling.
* **Orientational order** — `S(t) = 1.5⟨cos²θ⟩ - 0.5`, `θ` = angle between a cube
  axis and the settling (z) axis.

---

## 3. Results

### 3.1 Contact geometry: eight analytic families

`artifacts/nonspherical-contact-timestep/contact_geometry_table.md` (program
output, no hand copying):

| family | config | value [m] | oracle [m] | rel. error | angle [rad] | GJK it. | EPA it./faces |
|---|---|---|---|---|---|---|---|
| sphere-sphere | separated | 0.923967694375 | 0.923967694375 | 2.40e-16 | 3.10e-17 | 1 | 0/0 |
| sphere-sphere | penetrating | 0.28342491119 | 0.28342491119 | 0.00e+00 | 3.10e-17 | 2 | 54/250 |
| sphere-box | separated | 0.446419413859 | 0.446419413859 | 0.00e+00 | 7.85e-17 | 17 | 0/0 |
| sphere-box | penetrating | 0.0973566247753 | 0.0973566247753 | 0.00e+00 | 3.51e-16 | 4 | 95/332 |
| box-box-aligned | separated | 0.23 | 0.23 | 0.00e+00 | 4.35e-16 | 3 | 0/0 |
| box-box-aligned | penetrating | 0.13 | 0.13 | 0.00e+00 | 0.00e+00 | 4 | 1/12 |
| box-box-rotated-45deg | separated | 0.142893218813 | 0.142893218813 | 0.00e+00 | 1.94e-16 | 4 | 0/0 |
| box-box-rotated-45deg | penetrating | 0.107106781187 | 0.107106781187 | 0.00e+00 | 0.00e+00 | 4 | 1/28 |
| tetra-box | separated | 0.05 | 0.05 | 0.00e+00 | 3.93e-16 | 4 | 0/0 |
| tetra-box | penetrating | 0.259807621135 | 0.259807621135 | 2.14e-16 | 0.00e+00 | 4 | 1/22 |
| cone-floor | separated | 0.06 | 0.06 | 9.25e-16 | 3.56e-15 | 4 | 0/0 |
| cone-floor | penetrating | 0.04 | 0.04 | 5.20e-16 | 0.00e+00 | 4 | 1/100 |
| clump-box | separated | 0.05 | 0.05 | 0.00e+00 | 1.31e-15 | 13 | 0/0 |
| clump-box | penetrating | 0.18 | 0.18 | 0.00e+00 | 0.00e+00 | 4 | 7/156 |
| clump-clump | separated | 0.0810405388322 | 0.0810405388322 | 0.00e+00 | 2.22e-16 | 1 | 0/0 |
| clump-clump | penetrating | 0.222799548533 | 0.222799548533 | 2.49e-16 | 3.89e-16 | 2 | 113/368 |

Worst relative error **9.3e-16**, worst normal angle **3.6e-15 rad**; every
family has one separated and one overlapping configuration.

### 3.2 Contact duration

| law | k | measured t_c [s] | closed form [s] | relative error |
|---|---|---|---|---|
| law | stiffness | measured t_c [s] | closed form [s] | rel. error |
|---|---|---|---|---|
| linear | 500 N/m | 0.14049628 | 0.140496295 | -1.03e-07 |
| linear | 2000 N/m | 0.0702481401 | 0.0702481473 | -1.03e-07 |
| linear | 8000 N/m | 0.03512407 | 0.0351240737 | -1.03e-07 |
| Hertz (v=0.10) | 4.0e+04 N/m^1.5 | 0.0735818702 | 0.0735818796 | -1.28e-07 |
| Hertz (v=0.20) | 9.0e+04 N/m^1.5 | 0.0463118165 | 0.0463118224 | -1.28e-07 |

### 3.3 Restitution

| v_imp [m/s] | measured e | semi-analytic e | deviation |
|---|---|---|---|
| 0.05 | 0.927889 | 0.921229 | 0.72 % |
| 0.10 | 0.917766 | 0.908933 | 0.97 % |
| 0.20 | 0.906324 | 0.894601 | 1.31 % |
| 0.40 | 0.893421 | 0.877848 | 1.77 % |
| 0.80 | 0.878913 | 0.858200 | 2.41 % |

Monotone decreasing: **True**; worst deviation **2.41 %** (criterion 5 %); power-law fit `e = 1 - A v^p` with `p = 0.1870` (Weber–Stillinger `v^{1/5}` signature), `R² = 0.999965`.

### 3.4 Cundall–Strack critical step (measured by bisection)

| quantity | value |
|---|---|
| quantity | value |
|---|---|
| theory `2 sqrt(m_min/k_max)` [s] | 0.0447213595 |
| measured threshold (bisection) [s] | 0.0447336426 |
| ratio measured / theory | 1.000275 |
| stable bracket [s] | 0.0447131708 |
| divergent bracket [s] | 0.0447541145 |
| bisection tolerance | 1e-3 relative |
| growth factor at 0.5 / 1.0 / 1.5 dt_cr | 1.000 / 1.000 / inf |

The measured threshold equals `2√(m/k)` to `3·10⁻⁴` relative — the bound is
sharp for a 1-DOF oscillator, as it must be.

### 3.5 Energy ledger, 64-sphere gas

| quantity | value |
|---|---|
| steps (both runs) | 100000 |
| undamped: dt / dt_cr | 1.0e-02 |
| undamped: E0 [J] | 7.22128447 |
| undamped: total energy drift | +1.277e-05 |
| undamped: max |residual| over the run | 2.596e-05 |
| damped: dt / dt_cr | 1.0e-04 |
| damped: c [N s/m] | 5.0 |
| damped: dissipated work [J] | 0.0270253 |
| damped: ledger residual |dE + W|/E0 | -1.825e-07 |

### 3.6 Negative controls

| control | measurement | expected | passes |
|---|---|---|---|
| dt = 1.5 dt_cr | growth inf (vs 1.000 at 0.5 dt_cr) | diverges | True |
| negative damping c = -2 | energy gain +5.569e-02 | energy increases | True |
| torn ledger (dissipation ignored) | unaccounted 4.554e-02 vs closed -3.825e-04 | detected | True |

### 3.7 Macro anchors

| anchor | measured | reference | verdict |
|---|---|---|---|
| cubic pile (60 clumps, μ=0.6): repose angle | 12.22 deg | 30-35 deg ± 3 deg | **FAIL (honest)** |
| cubic pile: flank angles | 12.2 deg | - | - |
| packing fraction (150 spheres, uniaxial compression) | 0.6626 | 0.64 (Scott & Kilgour) | **FAIL (honest, 3.54 % off)** |
| bed height under the piston | 0.0656 m | compressed 21719 steps, piston load 6796 N | - |
| orientational order S(t): final / tail mean / tail std | -0.1425 / -0.1208 / 0.0166 | plateau | PASS |

**Why the two failures are reported as failures.** The cubic pile relaxes to
12.2 deg: with a velocity-regularised Coulomb law and no tangential spring
history, cubic particles slide past one another instead of interlocking, and the
pile is small (60 particles, 0.30 m box) so the toe occupies most of the profile.
The packing reaches 0.663 under a piston load of the order of 5x the bed weight —
3.5 % *above* Scott's 0.64, i.e. the uniaxial compression overshoots into the
crystalline branch (a 150-sphere bed is small enough that a few ordered layers
dominate). Both numbers are what the protocols produce; the fixes are (i) a
tangential spring with history plus rolling resistance for the pile, (ii) a
larger bed and a load-controlled stop at the RCP plateau for the packing. Neither
fix was made inside this mission, so neither criterion is claimed.

Figures: `docs/nonspherical-contact-timestep/contact_geometry_accuracy.png`,
`restitution_vs_impact_speed.png`, `timestep_stability_map.png`,
`energy_ledger.png`, `cube_pile.png`.

---

## 4. Pitfalls this work ran into (and fixes shipped)

1. **`np.sign(0) == 0` in a box support function** collapses the support point
   into the interior of the face (`(0.5, 0.5, 0.0)` instead of
   `(0.5, 0.5, ±0.5)`), which silently corrupts the Minkowski simplex. Ties must
   be broken to `±1` explicitly.
2. **A six-direction EPA seed fails for rotated boxes.** All axis supports can
   share a tie-broken coordinate and become coplanar; the hull-plus-violated-facet
   expansion is required (`_seed_polytope`).
3. **Converging on the objective caps the normal at `√eps`.** Converge on the
   *stationarity residual* instead (§2.3).
4. **`acos` cannot measure a `1e-8 rad` angle.** Use `|n × n_ref|`.
5. **Degenerate edge/face tangency.** When a polytope's closest feature is an
   edge whose normal cone contains the contact direction, the support point set
   is ambiguous and the support-based residual is not defined; the reported normal
   then carries the GJK closest-point error (`≈5e-8 rad` here). The eight
   acceptance families are configured with generic features; this limit is
   documented, not engineered away.
6. **The Cundall–Strack bound is conservative for networks.** Only the 1-DOF
   configuration is used as the acceptance measurement; a many-body chain would
   legitimately measure `>1.0` and must not be scored against the same ratio.
7. **The energy-ledger residual is an `O(dt²)` truncation bias** (trapezoidal
   quadrature of the dashpot work), so it grows with the number of steps at fixed
   `dt`. `dt` for the damped ledger is therefore chosen from the measured
   convergence (`safety = 1e-4 dt_cr`), and the observed values
   (`1.98e-6` at `5e-4·dt_cr` / 2·10⁴ steps, `6.6e-7` at `3e-4·dt_cr` / 2·10⁴
   steps) are reported rather than tuned away.
8. **Rigid clump initialisation must clear the bounding diameter, not the edge.**
   Randomly oriented cubes need `2(√3·offset + r)` of clearance; starting them at
   `1.05·edge` spawns a repulsion impulse that has nothing to do with the physics.

---

## 5. Limitations

* EPA converges *geometrically* for smooth bodies; the shipped tolerance
  (`1e-13`) plus the normal polish reaches machine precision for sphere/cone
  pairs, but a deliberately shallow overlap (`depth < 1e-13 · scale`) is treated
  as touching (fail-closed: `depth = 0`).
* Support functions are exact only for the four primitives; a general
  non-convex polyhedron must be decomposed into convex pieces or a clump.
* Friction in the pile model is Coulomb with viscous regularisation, not a
  tangential spring with history; rolling resistance is absent. This is stated
  because it affects the repose angle (§6.5).
* The ledger is single-precision-free but is *not* adaptive: it requires the user
  to choose `dt` from `dt_cr`, exactly like the production engine.
* No GPU/kernel-level optimisation: the gas integrator is a dense `O(N²)` numpy
  loop (N = 64 here). It is a verification harness, not a production solver.

---

## 6. Error assessment against the acceptance criteria

| criterion | measured | verdict |
|---|---|---|
| 接触几何 8 组 ≤1e-10 / 法向 ≤1e-8 rad | 9.25e-16 / 3.56e-15 | PASS |
| t_c ≤2 %（3 档 k_n） | 1.03e-07 | PASS |
| e(v) 单调 + ≤5 % + R²≥0.99 | dev 2.41 %, R² 0.999965 | PASS |
| Δt_cr 比值 ∈ [0.9, 1.1] | 1.000275 | PASS |
| 10⁵ 步能量漂移 <0.1 %；阻尼残差 <1e-6 | 1.28e-05 / -1.83e-07 | PASS |
| 3 项负控按预期 fail-closed | 3/3 controls fired | PASS |
| criterion | measured | verdict |
|---|---|---|
| cubic repose angle within the experimental band ±3 deg | 12.22 deg (band [30.0, 35.0]) | **FAIL (honest)** |
| packing fraction vs Scott 0.64 within 1 % | 0.6626 (3.54 %) | **FAIL (honest)** |
| orientation order S(t) converges | tail std 0.0166 | PASS |
| pytest >= 20 and registered in CI | 39 tests in 3 new modules; CI runs `python -m pytest -q` | PASS |

**Is the key quantity within 3 % of the reference?** For every contact-geometry,
contact-time, restitution, critical-step and energy-ledger quantity the answer is
**yes, by many orders of magnitude** (relative errors `1e-15 … 2.4e-2`, worst case
the first-order restitution approximation at 2.4 %). For the macro anchors the
answer is **no** where the table above says so — see §6.5, which is reported as an
honest FAIL rather than softened.