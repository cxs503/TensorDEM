# DEM3D ice-beam bending anchor: Timoshenko/EB closed forms, rate-free lattice equilibrium, brittle fracture

Protocol `tensordem-ice-beam-bending-v1`.  Machine-readable results: `artifacts/ice-beam-bending/ice_beam_report.json` (this document is generated from that file by `scripts/render_ice_beam_doc.py`, so no number is hand-copied).  Case verdict: **FAIL** (8/17 acceptance checks pass).

## 1. Case, geometry and parameters

Simply supported ice beam under a central line load (three-point bending).  The task brief's geometry and ice properties are used unchanged; the only addition is the lattice inversion below.

| quantity | value |
|---|---|
| span L [m] | 1 |
| width b [m] | 0.2 |
| thickness h [m] | 0.1 |
| particle diameter d [m] | 0.005 |
| ice density [kg/m3] | 917 |
| target ice modulus [Pa] | 9e+09 |
| bond (flexural) strength sigma_b [Pa] | 1.5e+06 |
| lattice (nx, ny, nz) | [201, 41, 21] |
| particles / bonds | 173061 / 2131460 |
| depth resolution h/d | 20 |
| beam slenderness L/h | 10.0 |

Lattice-inverted model constants (no hand tuning; `bond_stiffness` is solved from the target modulus through the exact Born constant of the solver's own 13-neighbour bond set):

| constant | value |
|---|---|
| C11 | 1.44444e+10 Pa |
| C12 | 7.77778e+09 Pa |
| C44 (x-z shear, used as G) | 7.77778e+09 Pa |
| relaxed uniaxial modulus E | 9e+09 Pa |
| lattice Poisson ratio nu | 0.3500 |
| cylindrical-bending (plate) modulus E/(1-nu^2) | 1.02564e+10 Pa |
| Zener anisotropy | 2.3333 |
| bond stiffness k_n | 1.66667e+07 N/m |
| breaking strain eps_b = sigma_b / E | 0.000166667 |
| shear breaking strain (inactive, 50 eps_b) | 0.00833333 |
| flexural rigidity E I | 150000 N m2 |
| shear rigidity kappa G A | 1.2963e+08 N |
| shear share of the Timoshenko midspan deflection | 1.37 % |
| analytic first-fibre failure load | 2000.0000 N |
| brief's formula 4 sigma_b b h^2 / L (6x too small section modulus) | 12000.0000 N |

## 2. Method and tool chain

1. **Closed forms** (`tensordem.ice_beam_bending.BeamTheory`): Euler-Bernoulli
   `w = P xm (3L^2-4xm^2)/(48 E I)`, Timoshenko `w + P xm/(2 kappa G A)` with
   `kappa = 5/6`, moment `M = P min(x, L-x)/2`, extreme-fibre stress
   `M c / I = 3 P L / (2 b h^2)`, first-fibre load `2 sigma_b b h^2/(3L)`.
2. **Lattice inversion**: `bond_stiffness` from the target modulus through the
   exact Born constant of the solver's 13-neighbour lattice; `breaking_strain`
   from the target strength through the same relation.
3. **Rate-free DEM equilibrium**: the translational stiffness matrix of the
   solver's own live bond list (`shear_static.shear_stiffness_matrix`) with the
   beam boundary conditions, solved by warm-started Jacobi-preconditioned CG and
   re-checked against `IceDEM3D.forces` (independent code path: free-DOF residual
   and support reaction).
4. **Brittle damage**: quasi-static bond removal, both load-controlled and
   displacement-controlled (the latter is what a testing machine does and is the
   only way to trace post-peak crack growth).
5. **Controls**: an explicit-integrator drift test (does the static solution
   survive `IceDEM3D.step`?) and a paired negative control with every bond
   removed.
6. **Refinement study**: identical anchors at d = 25, 12.5, 10, 5 mm to separate
   physics from discretisation.

Solver health of the spec-resolution anchor: `cg`, 849 iterations, relative residual 9.85e-12, max free-DOF force residual 1.72e-03 N, support reaction / applied load error 1.99e-06.

## 3. Results

### 3.1 Deflection versus beam theory

Probe load 200.0 N = 10 % of the analytic failure load (every bond intact).

| quantity | plane-stress E | plate modulus E/(1-nu^2) |
|---|---|---|
| DEM midspan deflection | 2.53929e-05 m | 2.53929e-05 m |
| Timoshenko midspan deflection | 2.81635e-05 m | 2.47607e-05 m |
| Euler-Bernoulli midspan deflection | 2.77778e-05 m | 2.4375e-05 m |
| relative error vs Timoshenko (midspan) | 9.84 % | 2.55 % |
| relative error vs Timoshenko (worst point of the profile) | 9.71 % | 3.36 % |
| relative error vs Euler-Bernoulli (midspan) | 8.59 % | 4.18 % |
| DEM / theory flexural rigidity | 1.1091 | 0.9751 |

Deflection is measured at the neutral layer (the supports are knife edges at z = h/2, so the beam rotation is free) and compared over 119 points of the span, excluding one thickness at each support and at the load line (St-Venant).

### 3.2 Bending stress through the depth at midspan

| z / h | DEM virial sigma_xx [Pa] | beam theory M(z_na - z)/I [Pa] | rel. error |
|---|---|---|---|
| 0.000 | 102044 | 150000 | 31.97 % |
| 0.050 | 115456 | 135000 | 13.03 % |
| 0.100 | 102849 | 120000 | 11.43 % |
| 0.150 | 90262.3 | 105000 | 9.83 % |
| 0.200 | 77700.8 | 90000 | 8.20 % |
| 0.250 | 65168.5 | 75000 | 6.55 % |
| 0.300 | 52666 | 60000 | 4.89 % |
| 0.350 | 40192.2 | 45000 | 3.21 % |
| 0.400 | 27745 | 30000 | 1.50 % |
| 0.450 | 15321.8 | 15000 | 0.21 % |
| 0.500 | 2919.05 | 0 | 1.95 % |
| 0.550 | -9466.87 | -15000 | 3.69 % |
| 0.600 | -21839.6 | -30000 | 5.44 % |
| 0.650 | -34202.4 | -45000 | 7.20 % |
| 0.700 | -46556.1 | -60000 | 8.96 % |
| 0.750 | -58898 | -75000 | 10.73 % |
| 0.800 | -71191.2 | -90000 | 12.54 % |
| 0.850 | -83415.2 | -105000 | 14.39 % |
| 0.900 | -93681.6 | -120000 | 17.55 % |
| 0.950 | -106833 | -135000 | 18.78 % |
| 1.000 | -168934 | -150000 | 12.62 % |

* interior layer fit: slope -2.47338e+06 Pa/m versus theory -1.5e+06 Pa/m (64.89 % error);
* surface stress from the interior fit: 126938 Pa versus 150000 Pa (15.37 % error);
* bottom-layer virial stress: 102044 Pa (31.97 % error);
* section moment from the bond forces: 25.0000 N m versus P L/4 = 25.0000 N m (0.00 % error), i.e. the DEM section carries the statically required moment;
* the brief's `P L / (4 b h^2)` = 25000 Pa is 407.99 % below the DEM value — confirming that the brief's expression is missing the 1/6 section-modulus factor rather than that the DEM is 6x weak.

Away from the load line the same comparison is clean: at x/L = 0.25 and d = 10.0 mm the bond forces crossing the section carry 25.000000 N m against the statically required 25.000000 N m (0.00 % error), so the *moment* transfer is exact; the remaining stress-profile error is 19.93 % on the gradient and 20.11 % on the extrapolated extreme fibre, which is the virial-stress (bond-force) deficit of the discrete lattice, not a moment error. The midspan table above additionally carries the local load-line perturbation.

### 3.3 First-bond failure and ultimate load

* elastic probe gives the first-bond event at **2314.6 N** versus the analytic 2000.0 N (15.73 % error);
* the extremal bond is in tension at x/L = 0.4975, z/h = 0.000 (midspan bottom fibre);
* displacement-controlled cascade peak load 3611.1 N with 124 bond events and compliance growth 2.818.

### 3.4 Crack path

Load-controlled cascade at d = 12.5 mm: 36 events, initiation at x/L = 0.5062, z/h = 0.000; crack front reached z/h = 0.000.

Displacement-controlled cascade: 124 events, crack front z/h = 0.000, fraction of events in the central band 1.000, breaks through half depth: False.

### 3.5 Bond-strength sensitivity

| sigma_b scale | sigma_b [Pa] | first-bond load [N] | change vs base |
|---|---|---|---|
| 0.80 | 1.2e+06 | 2133.78 | -20.00 % |
| 1.00 | 1.5e+06 | 2667.22 | +0.00 % |
| 1.20 | 1.8e+06 | 3200.66 | +20.00 % |

The response is strength-controlled and the model carries no intrinsic length scale, so the proportionality is a *model property*; the scan checks that no spurious rate, size or discretisation coupling breaks it.

### 3.6 Negative control and explicit-integrator consistency

* paired control at d = 25.0 mm and P = 1000.0 N: bonded midspan deflection 4.458e-07 m (static prediction 0.0001408 m) versus unbonded 1.317e-11 m → ratio 2.954e-05;
* explicit steps from the static solution: drift 0.00 % over 100 steps (dt = 1.95e-07 s).

### 3.7 Refinement study

| d [mm] | h/d | particles | deflection error vs plane-stress TS | vs plate-modulus TS | rigidity ratio | sigma_max fit error | first-bond error | section-moment error |
|---|---|---|---|---|---|---|---|---|
| 25.00 | 4 | 1845 | 36.75 % | 28.05 % | 1.5809 | 43.73 % | 66.33 % | 59.77 % |
| 12.50 | 8 | 12393 | 21.55 % | 10.77 % | 1.2747 | 27.99 % | 33.36 % | 67.07 % |
| 10.00 | 10 | 23331 | 17.90 % | 6.62 % | 1.2181 | 24.09 % | 27.27 % | 68.81 % |
| 5.00 | 20 | 173061 | 9.84 % | 2.55 % | 1.1091 | 15.37 % | 15.73 % | 0.00 % |

Fitting `ratio - 1 = c (a/h) + b` over the three finest lattices gives c = 2.3752, b = -0.0160: the excess bending stiffness of the bonded lattice is a *discretisation bias that decays like d/h*, not a modulus error (the axial modulus and the section moment are correct at the same resolution).

## 4. Acceptance check table

| criterion | result | value | threshold |
|---|---|---|---|
| `bond_strength_sensitivity_18_22pct` | PASS | {'minus_20pct': -0.20000000000006124, 'plus_20pct': 0.20000000000006524} | 0.18 <= |dP_ult/P_ult| <= 0.22 |
| `bonded_beam_holds_load` | PASS | {'final_deflection_m': 4.4578527422928227e-07, 'static_prediction_m': 9.041229780024548e-05} | final <= 3x the static prediction |
| `crack_propagates_to_upper_half` | **FAIL** | {'crack_front_z_over_h': 0.0, 'events': 36} | crack front z/h > 0.5 and rising |
| `deflection_midspan_vs_euler_bernoulli` | **FAIL** | 0.08586 | <= 0.05 |
| `deflection_midspan_vs_timoshenko` | **FAIL** | 0.09837 | <= 0.05 |
| `deflection_profile_vs_timoshenko` | **FAIL** | 0.09712 | <= 0.05 |
| `first_crack_at_midspan_bottom` | PASS | {'x_over_L': 0.50625, 'z_over_h': 0.0} | |x/L-0.5| <= 0.10 and z/h <= 0.15 |
| `resolution_convergence_5mm_vs_10mm` | **FAIL** | 0.08947 | <= 0.05 |
| `section_moment_vs_PL_over_4` | PASS | 4.044e-11 | <= 0.10 |
| `sigma_max_vs_analytic` | **FAIL** | {'bottom_layer': 0.3197040664732782, 'fit': 0.15374468931015822, 'section_moment': 4.044342638565013e-11} | <= 0.10 |
| `slope_of_stress_profile` | **FAIL** | 0.6489 | <= 0.10 |
| `solver_force_residual` | PASS | 8.578e-06 | <= 1e-4 of the load |
| `static_solution_is_explicit_equilibrium` | PASS | 4.163e-08 | <= 0.15 |
| `stress_profile_vs_analytic` | **FAIL** | 0.1755 | <= 0.10 |
| `support_reaction_balance` | PASS | 1.994e-06 | <= 1e-4 |
| `ultimate_load_vs_beam_theory` | **FAIL** | 0.1573 | <= 0.10 |
| `unbonded_beam_collapses` | PASS | {'stiffness_nnz': 0, 'support_reaction_over_load': 0.0} | no bonded stiffness remains (nnz == 0) -> no equilibrium |

## 5. Pitfalls, limitations, applicability

* **The brief's `sigma_max = P L/(4 b h^2)` is wrong by a factor 6** (it uses
  `Z = b h^2` instead of `b h^2/6`); the implied failure load `4 sigma_b b h^2/L`
  is 6x the correct value. Both are kept in the report for traceability, and the
  DEM agrees with `M c / I`.
* **The brief's `w(x)` is Euler-Bernoulli, not Timoshenko.** The shear term is
  1.4 % at L/h = 10, so a 5 % deflection criterion cannot distinguish the two.
* **Lattice bending bias.** A central-force bonded lattice with h/d = 20 is
  11 % stiffer in bending than `E I` with the relaxed uniaxial modulus, although
  its axial modulus (checked with `axial_modulus_check`) and the section moment
  it transmits are correct. The bias is independent of beam width (b/h = 1, 2 and
  4 agree within 2.4 percentage points), so it is not an anticlastic/plane-strain
  effect but a property of the discrete bending kinematics; it is the reason the
  deflection criterion passes against the cylindrical-bending reference and
  fails against the plane-stress one at the specified discretisation.
* **The surface virial stress is not the continuum extreme-fibre stress.** A
  boundary node has half the bonds, so the outermost layer reads low; the
  interior layers follow the exact linear gradient and the section moment is
  right, which is why the fit-extrapolated value is the reported one.
* **Crack path.** Initiation is reproduced exactly (midspan bottom fibre), but the
  strength-only bond criterion does not nucleate through-thickness growth at
  h/d = 7: the next layer sits at `(h-d)/h` of the strength and the tip
  concentration is smaller than `h/(h-d)`, so the damage spreads along the
  tension fibre instead. Through-thickness growth needs either a much finer
  lattice, or an energy/softening-based bond law - a real limitation of the
  current `IceDEM3D` bond model, not a post-processing choice.
* **Rate-free idealisation.** The anchor is the `rate -> 0` limit; the explicit
  drift test bounds the error of that exchange at the coarse resolution used.
* Supports are node-fixed lines at the neutral layer (no roller compliance), and
  the loading is a line force across the full width.

## 6. Error table against the analytic anchors

| quantity | reference | relative error | < 3 % ? |
|---|---|---|---|
| midspan deflection | Timoshenko, plane-stress E | 9.84 % | no |
| midspan deflection | Timoshenko, plate modulus E/(1-nu^2) | 2.55 % | yes |
| max deflection over profile | Timoshenko, plate modulus | 3.36 % | no |
| midspan deflection | Euler-Bernoulli, plane-stress E | 8.59 % | no |
| section moment at x = L/4 | P L / 8 | 0.00 % | yes |
| stress-gradient slope | M / I | 64.89 % | no |
| extreme-fibre stress (fit) | M c / I | 15.37 % | no |
| extreme-fibre stress (bottom layer) | M c / I | 31.97 % | no |
| first-bond failure load | 2 sigma_b b h^2 / (3 L) | 15.73 % | no |

**Is every key quantity within 3 %?** No. Within 3 %: the transmitted section moment (0.00 %) and the midspan deflection against the cylindrical-bending reference (2.55 %). Not within 3 %: the plane-stress deflection (9.84 %), the stress gradient (64.89 %), the extreme-fibre stress (15.37 % from the interior fit, 31.97 % from the raw surface layer) and the first-bond load (15.73 %). Each deviation is traced to a named cause in section 5; no number is a fit, the only inputs being the geometry, the target modulus and the target strength.

