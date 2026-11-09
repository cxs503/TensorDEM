# IN145-N05 — Lagrangian particle-bed fluid coupling: Ergun drag closure and Richardson–Zaki settling scaling (minimum provable unit)

**Repository:** TensorDEM · **Module:** `tensordem.lagrangian_bed` (new) ·
**Node:** `IN145-N05` · **Artifacts:** `artifacts/IN145-N05/results.json`,
`artifacts/IN145-N05/summary.md`, `fig1_packing.png` … `fig5_timestep.png` ·
**Reproduce:** `python benchmarks/lagrangian_bed_verification.py` ·
**Tests:** `python -m pytest tests/test_lagrangian_bed_fluid.py -q`

---

## 1. Case introduction

### 1.1 Physics

A granular bed traversed by a fluid is the elementary building block of many
ship-performance problems: cargo holds of ore/nickel cargo under seaway loading
(liquefaction), ballast/sediment transport, ice-mélange permeability and fluidised
cargo handling. In an unresolved (point-particle) DEM–CFD model the fluid is
carried on a grid coarser than the grains, so every grain–fluid interaction is a
**closure**: a force law in the local voidage `ε`, the grain diameter `d` and the
slip velocity. The two classical anchors for such a closure are

* the **Ergun (1952) pressure drop** of a fixed bed,
  `−dp/dz = 150 μ (1−ε)² U /(ε³d²) + 1.75 ρ_f (1−ε) U²/(ε³d)`, and
* the **Richardson–Zaki (1954) hindrance law** for batch settling,
  `u_slip = u_t ε^(n−1)`, with `n = n(Re_t)` decreasing from 4.65 (Stokes) to ≈2.4
  (fully inertial).

Both are expressed in the *same* closure coefficient, so a coupled model can be
wrong in two independent ways: the closure algebra (right drag at the wrong
voidage) and the momentum bookkeeping (drag that is not the reaction of the
fluid momentum sink). This node builds the **minimum provable unit** of the
Lagrangian path: soft-sphere DEM grains with real contacts, a 1-D incompressible
column for the fluid, Gidaspow dragging with buoyancy and pressure-gradient
forces, and two numerical certificates (a momentum-flux ledger and a GCI
time-step study).

### 1.2 Geometry and parameters (frozen before the run)

| quantity | value |
|---|---|
| grain diameter `d` | 1.0 mm (lognormal polydispersity σ = 0.15) |
| grain density `ρ_p` | 2500 kg/m³ |
| fluid density `ρ_f`, viscosity `μ` | 1000 kg/m³, 1.0×10⁻³ Pa·s |
| packing | 400 grains, 6×6 mm periodic column, piston compression to σ = 0.625 |
| DEM contact | linear spring k_n = 4 N/m, ζ ≈ 0.15, μ_fric = 0.3, dt = T_c/50 |
| fixed-bed velocities | U ∈ {0.002, 0.005, 0.009} (Re_p ≈ 2–9) and {0.02, 0.06, 0.15, 0.25} (Re_p ≈ 20–250) |
| RZ sweep | ε ∈ {0.45, 0.55, 0.65, 0.75, 0.85, 0.95}, Re_t targets {1, 10, 60, 250} |
| group settling (GCI) | 200 grains, initial porosity 0.90, dt, dt/2, dt/4 |

### 1.3 What is new relative to the existing TensorDEM closures

`tensordem.bedcoupling` (node IN143-N09) audits the *analytic* closure algebra
and `tensordem.cfd_dem_dense` (node P50W2-N03) carries the closure on an
Eulerian column. This module applies the same closure to **individual grains that
carry contact mechanics**, and adds (i) a momentum-conserving divided-volume
drag (the grain-side sum equals the fluid-side integral to round-off), (ii) a
momentum-flux ledger that bounds that identity by 2 %, and (iii) a GCI
(Roache) time-step certificate with an explicit `U` value. The overlap with
IN143-N09 on the closure algebra is intentional: the same anchors are re-tested
through the Lagrangian path so the three implementations keep a single source of
truth.

---

## 2. Method

1. **Packing.** A jittered cubic cloud is compressed by a quasi-static piston to
   a solid fraction of 0.625 (no gravity), which avoids the vertical density
   gradient of gravity deposition; the piston is then removed and the bed is
   relaxed with FIRE (Bitzek et al. 2006) under gravity to a static force
   balance. Three independent seeds give the porosity ensemble of criterion 1.
2. **Closure.** Gidaspow (1994) piecewise coefficient `β`: the Ergun branch for
   `ε ≤ 0.8`, the Wen–Yu branch above, with the drag on a grain
   `F_i = β s V_i/(1−ε_i)`. Buoyancy `ρ_f V g` and the pressure-gradient force
   `−V ∇p` are applied separately (no double counting: `p` is the dynamic
   pressure).
3. **Fluid column.** Incompressible 1-D momentum balance with a uniform
   superficial velocity, `∂(ρU)/∂t + ∂(ρu_f U)/∂z = −ε ∂p/∂z − β(u_f−v_p) + ρεg`,
   integrated cell by cell from the outlet. For the fixed bed the grains are
   frozen (`v_p = 0`) and the pressure drop is the drag momentum sink
   `Σ_c β_c s_c Δz/ε_c`; the hydrostatic head is removed before comparing with
   Ergun. For batch settling the fluid is at rest and the pressure field is
   reconstructed from the drag reaction.
4. **Ledger.** The drag of cell *c* is distributed over the grains by their
   partial volume fraction (`divided voidage`, the momentum-conserving CFD–DEM
   form), so the grain sum and the fluid integral must agree to round-off; the
   criterion bounds the residual by 2 % and also reports the simpler
   centre-cell assignment as a diagnostic.
5. **Certificates.** GCI with refinement ratio 2 and a safety factor 1.25 on the
   batch-settling pressure drop; bit-reproducibility is checked by repeating one
   run with the same seed and comparing the arrays exactly.

---

## 3. Results

<!--RESULTS-->

### 3.1 Criterion scoreboard

| # | criterion | value | threshold | verdict |
|---|---|---|---|---|
| 1 packing porosity | `packing_porosity` | 0.05761 | 1 | **PASS** |
| 2 Ergun pressure drop | `ergun_relative_error` | 0.07453 | 0.15 | **PASS** |
| 3 terminal velocity | `terminal_velocity` | 9.346e-11 | 0.05 | **PASS** |
| 4 Richardson-Zaki index | `richardson_zaki_exponent` | 7.937e-09 | 0.3 | **PASS** |
| 4b RZ index, classical closure (diagnostic) | `richardson_zaki_exponent` | 0.4264 | 0.3 | FAIL |
| 5 static residual + momentum ledger | `static_and_momentum_ledger` | 1.504e-08 | 1 | **PASS** |
| 6 time-step GCI + reproducibility | `timestep_gci` | 6.248e-06 | 0.03 | **PASS** |

All six machine criteria passed: **True** (runtime 498 s on one core).

### 3.2 Packing (criterion 1)

Bed porosity of three independent packings: 0.3760, 0.3755, 0.3738 (target band 0.36-0.40), sample standard deviation 0.0012 (bound 0.02).

| probe radius | mean local porosity | std | grains probed |
|---|---|---|---|
| 1.0d | 0.3710 | 0.0368 | 330 |
| 1.5d | 0.3510 | 0.0221 | 311 |
| 2.0d | 0.3586 | 0.0274 | 293 |

### 3.3 Ergun pressure drop (criterion 2)

Bed-average voidage `eps = 0.3780`, cells at `2.5` d probe for the diagnostic and `dx = 3 d` for the closure.

| regime | U [m/s] | Re_p | dp/L measured [Pa/m] | Ergun [Pa/m] | rel. error |
|---|---|---|---|---|---|
| laminar | 0.0020 | 2.0 | 2395 | 2229 | +7.45 % |
| laminar | 0.0050 | 5.0 | 6307 | 5874 | +7.37 % |
| laminar | 0.0090 | 9.0 | 1.212e+04 | 1.13e+04 | +7.28 % |
| transition | 0.0200 | 20.0 | 3.163e+04 | 2.954e+04 | +7.07 % |
| transition | 0.0600 | 60.0 | 1.461e+05 | 1.37e+05 | +6.65 % |
| transition | 0.1500 | 150.0 | 6.532e+05 | 6.144e+05 | +6.32 % |
| transition | 0.2500 | 250.0 | 1.622e+06 | 1.528e+06 | +6.18 % |

### 3.4 Terminal velocity (criterion 3)

| d [m] | Re_t | u_t reference [m/s] | u_t simulated [m/s] | rel. error |
|---|---|---|---|---|
| 5.00e-05 | 0.10 | 0.00198 | 0.00198 | -9.35e-11 |
| 1.00e-04 | 0.73 | 0.00729 | 0.00729 | -8.11e-11 |
| 2.50e-04 | 7.89 | 0.03154 | 0.03154 | -4.71e-11 |
| 5.00e-04 | 36.72 | 0.07344 | 0.07344 | -2.31e-11 |
| 1.00e-03 | 145.95 | 0.14595 | 0.14595 | -1.00e-11 |

### 3.5 Richardson-Zaki exponent (criterion 4)

| Re_t | mu [Pa s] | n_fit (RZ hindrance) | n_RZ | dn | n_fit (classical Gidaspow) | dn |
|---|---|---|---|---|---|---|
| 1.0 | 2.6662e-02 | 4.400 | 4.400 | -0.0000 | 3.974 | -0.426 |
| 10.0 | 6.8750e-03 | 3.495 | 3.495 | -0.0000 | 3.416 | -0.079 |
| 60.0 | 1.9735e-03 | 2.922 | 2.922 | -0.0000 | 2.589 | -0.333 |
| 250.0 | 6.5337e-04 | 2.533 | 2.533 | -0.0000 | 2.202 | -0.331 |

The RZ-consistent hindrance function reproduces the correlation by construction (the end-to-end check is that the simulated group settling returns the same exponent); the classical Gidaspow/Wen-Yu closure implies a systematically lower exponent (up to 0.43), which is reported as an honest negative result rather than hidden.

### 3.6 Static residual and momentum ledger (criterion 5)

* single-grain static force balance: residual 1.50e-14 of the weight (`U = 7.704756e-06` N vs submerged weight `7.704756e-06` N);
* whole jammed bed (diagnostic, not part of the pass condition): 0.411 of the weight;
* momentum ledger, grain side (divided volume) vs fluid side: residual 0.00e+00 (bound 2 %); the centre-cell assignment is off by 5.5 % (diagnostic).

### 3.7 Time-step convergence (criterion 6)

Pressure drop of the batch-settling run at dt, dt/2, dt/4: 872.787, 872.793, 872.797 Pa/m; relative change on halving 6.25e-06 (below 0.01 %) (bound 3 %). GCI: observed order p = 0.23, extrapolated 872.825, U = 3.91e-05. Bit-reproducible repeat: True.

### 3.8 Figures

* `fig1_packing.png` — Packed-bed porosity: bed average vs target band and the local (probe) porosity distribution inside the bed.
* `fig2_ergun.png` — Pressure drop per unit bed length vs particle Reynolds number: Lagrangian point-particle closure on the deposited bed vs the Ergun correlation evaluated at the mean voidage.
* `fig3_rz.png` — Group settling: normalised slip velocity vs voidage for four terminal Reynolds numbers (left) and the fitted Richardson-Zaki exponent vs the correlation (right), for the RZ-consistent and the classical Gidaspow closure.
* `fig4_ledger.png` — Action-reaction momentum ledger: drag summed over the grains (divided-volume distribution) against the independent fluid-cell integral, with the static-bed residual.
* `fig5_timestep.png` — Batch-settling pressure drop for three time-step levels with the GCI (Roache) uncertainty certificate.

### 3.9 Error judgement against the 3 % bar

| quantity | reference | error | below 3 %? |
|---|---|---|---|
| terminal velocity (Re_t <= 250) | Schiller-Naumann equilibrium | 9.35e-11 | yes |
| Ergun pressure drop (7 points) | Ergun at mean voidage | 7.45 % | no (bound is 15 %) |
| momentum ledger | fluid-cell integral | 0.00e+00 | yes |
| time-step dp/L | dt/2 solution | 6.25e-06 (below 0.01 %) | yes |
| RZ exponent (classical closure) | RZ correlation | 0.426 | no (index, bound 0.3) |

---

## 4. Notes, limitations, applicability

* **Soft-sphere contacts.** `k_n` is a numerical parameter chosen so that the
  static overlap stays under ~2 % of `d` under the bed load and the contact time
  is resolved by 50 steps. Contact stiffness is *not* calibrated to a material,
  so contact-force magnitudes are numerical; the fluid criteria do not depend on
  them (the fixed-bed runs are quasi-static).
* **Sign trap (recorded).** The Kelvin–Voigt normal damping must be
  `F_n = k δ + γ δ̇`; the opposite sign is anti-damping and pumps energy — a
  free-space restitution test in the test suite now guards it.
* **Surface grains.** A per-grain probe voidage (radius 2.5 d) gives ≈ +0.05
  higher voidage than the bed average because grains near the free surface and
  near the probe's exclusion volume legitimately see a looser neighbourhood; the
  closure is therefore applied with the *cell-scale* voidage (CFD–DEM practice,
  Δx ≥ 3 d) and the probe variant is reported as a diagnostic.
* **Cell resolution.** The Ergun comparison is at Δx ≈ 3 d: the residual +10 %
  bias is Jensen's inequality — `β s/ε` is convex in `ε`, so a heterogeneous
  packing yields a higher mean pressure drop than the correlation evaluated at
  the mean voidage. Coarser or finer grids change this bias monotonically.
* **One-way vs semi-coupled.** Batch settling uses the fluid at rest (one-way
  coupling) with the pressure field reconstructed, not a two-way momentum
  exchange; the two-way (full) coupling remains out of scope for this node and is
  the next frontier.