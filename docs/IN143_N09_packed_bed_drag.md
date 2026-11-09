# IN143-N09 — Ergun pressure-drop drag closure and Richardson–Zaki settling scaling

**Repository:** TensorDEM · **Module:** `tensordem.bedcoupling` (new) ·
**Node:** IN143-N09 (DAG level: minimum verifiable unit for dense particle–fluid coupling) ·
**Artifacts:** `artifacts/IN143-N09/results.json`, `artifacts/IN143-N09/fig1..4*.png` ·
**Reproduce:** `python scripts/in143_n09_study.py` (~100 s single core)

---

## 1. Case introduction

An unresolved (point-particle) DEM–CFD coupling needs a *drag closure*: the fluid
force on a particle as a function of the slip velocity `u = u_f − u_p`, diameter
`d` and **local** voidage `ε`. This node implements and audits the classical
Gidaspow blend — Ergun branch for `ε < 0.8`, Wen–Yu branch above — together with
the buoyancy/pressure-gradient force, and it exercises the pair of limiting
relations the closure must satisfy: the **Ergun pressure drop** of a dense packed
bed and the **Richardson–Zaki hindering law** of a settling suspension.

Nothing in the module carries a ship, ice or hull constant: it is a generic
granular–fluid coupling layer, which is why it belongs to TensorDEM and not to
the platform or to the ship repository.

| Quantity | Packed-bed case | Settling case | Unit |
|---|---|---|---|
| particle diameter `d` | 1.0 | 1.0 | mm |
| particle density `ρ_p` | 2650 | 2650 | kg/m³ |
| fluid density `ρ_f` | 998 | 1200 | kg/m³ |
| dynamic viscosity `μ` | 1.0·10⁻³ | 5.0·10⁻² | Pa·s |
| gravity `g` | 9.80665 | 9.80665 | m/s² |
| packing | random close packing, `N` = 1500, `ε` = 0.380 | jittered cubic arrays, `ε` = 0.80/0.86/0.92/0.96 | — |
| Reynolds number `Re_t` | 155 (U = 0.05 m/s) | 0.353 | — |

Closures, with their validity ranges (all in `tensordem.bedcoupling.closures`):

* **Ergun (1952)** — per-particle form used by the assembly
  `F = V_p u [150 μ(1−ε)/(ε³d²) + 1.75 ρ_f|u|/(ε³d)]`, whose bed average is
  `ΣF/V_bed = Δp/L` exactly (asserted in `tests/test_bedcoupling_bed.py` to 1e-12).
* **Wen & Yu (1966) / Gidaspow (1994)** —
  `F = ½ρ_f C_D(εRe)|u|u (πd²/4) ε^−1.65`, `C_D = 24/Re(1+0.15Re^0.687)`,
  `Re = ερ_f|u|d/μ`; Stokes limit `F = 3πμdu ε^−2.65`.
* **Richardson & Zaki (1954)** — `n` from the tabulated piecewise correlation
  (`4.65`; `4.4Re^−0.03`; `4.4Re^−0.1`; `2.4`), used only as the yardstick.
* **Buoyancy** — not a separate term: `F = −V_p∇p`. Two pressure fields are
  implemented, and the choice is the physics of the node (§5).

## 2. How it was done

1. **Packing.** A dilute random configuration on a periodic cubic cell is
   quasi-statically compressed in ≤1 % linear steps, each followed by a
   soft-sphere pair-separation relaxation with a Verlet neighbour list
   (`SKIN = 0.2d`, list rebuilt every 40 sweeps). The cell is finally set to the
   exact target volume and relaxed until the residual overlap falls below
   `1e-3 d`. Porosity is therefore exact by volume balance; the *fluctuating*
   quantity is the local voidage.
2. **Local voidage.** `ε_i = 1 − Σ_j V(R_probe ∩ V_j)/V(R_probe)` with the exact
   sphere–sphere intersection volume, minimum-image periodic, probe radius
   `R = 3d` (sensitivity to `R` is reported).
3. **Fixed bed.** Particles held at rest in a uniform superficial velocity `U`
   along `+z`; `Δp/L = Σ F_drag,z / V_bed`. Two variants: per-particle probe
   voidage (the unresolved-coupling value) and the bed-mean voidage.
4. **Static bed.** Fluid at rest: the only fluid force is `−V_p∇p`; the assembled
   net force is compared with the effective submerged weight per particle and in
   aggregate, normalised by `Σm_ig`.
5. **Settling.** Semi-coupled single-fluid batch settling in a closed periodic
   cell: one fluid degree of freedom, `u_f = −(1−ε)/ε ⟨u_p⟩` from zero net volume
   flux; each particle feels gravity + pressure gradient + closure drag,
   integrated explicitly with `Δt = 0.1 τ` (`τ = m/c`). The runs are contact-free
   (minimum separation ≥ d, asserted). `n` is fitted by log–log regression of
   `u_p(ε)`.
6. **Time step.** The same run is repeated at `Δt` and `Δt/2`; the drag pressure
   drop is compared at steady state and at a common physical time `t = 2τ`.

## 3. Results

### 3.1 Criterion table (all numbers injected from `results.json`)

| # | Criterion | Measured | Threshold | Pass |
|---|---|---|---|---|
| 1 | packing porosity in [0.36, 0.40], spread ≤ 0.02 | `ε̄` = 0.37858 (balance 0.380), `σ` = 0.00469, max overlap 4.24·10⁻⁴ d | 0.02 | ✅ |
| 2 | `Δp/L` vs Ergun, laminar & transition | +1.47 % (U = 2 mm/s, inertial 3.6 %), +1.36 % (U = 50 mm/s, inertial 48.4 %) | 15 % | ✅ |
| 3 | terminal velocity vs Schiller–Naumann | −0.0129 % (0.155280 vs 0.155300 m/s) | 5 % | ✅ |
| 4 | settling index vs Richardson–Zaki | `n_sim` = 4.505 vs `n_RZ` = 4.540, Δ = −0.035 | 0.3 | ✅ |
| 5 | static-bed force residual | 0.0 (aggregate float64 sum), 1.24·10⁻¹⁶ per particle | 10⁻⁶ | ✅ |
| 6 | `Δp/L` change on halving `Δt` | 2.0·10⁻¹⁶ (steady), 1.21 % (transient, `t = 2τ`) | 3 % | ✅ |

Packing geometry (evidence that the configuration is admissible, not just dense):
`N` = 1500, `L` = 10.820 mm (10.82 d), solids fraction 0.620000, minimum
separation 0.99958 d, maximum overlap 4.24·10⁻⁴ d, 2.25·10⁶ candidate pairs
checked.

Local-voidage statistics versus probe radius (the pass is *not* obtained by
tuning the kernel; `R = 3d` was fixed before the run):

| `R/d` | mean | σ | min | max |
|---|---|---|---|---|
| 2.0 | 0.37632 | 0.00991 | 0.34356 | 0.40856 |
| 2.5 | 0.38275 | 0.00625 | 0.36333 | 0.40467 |
| **3.0** | **0.37858** | **0.00469** | 0.36555 | 0.39338 |
| 3.5 | 0.38051 | 0.00365 | 0.37072 | 0.39130 |

### 3.2 Ergun sweep and the branch control

| `U` [m/s] | regime | inertial fraction | Ergun `Δp/L` [Pa/m] | probe-voidage assembly | mean-voidage assembly | Wen–Yu branch (control) |
|---|---|---|---|---|---|---|
| 0.0005 | laminar | 0.009 | 530.34 | +1.48 % | −2.6·10⁻¹⁴ % | −85.7 % |
| 0.002 | laminar | 0.036 | 2180.55 | +1.47 % | +3.3·10⁻¹⁴ % | −85.1 % |
| 0.01 | transition | 0.158 | 12481.5 | +1.44 % | −2.6·10⁻¹⁴ % | −84.0 % |
| 0.05 | transition | 0.484 | 101874.8 | +1.36 % | +4.7·10⁻¹⁴ % | −84.8 % |
| 0.08 | transition | 0.600 | 210360.7 | +1.33 % | +1.7·10⁻¹⁴ % | −85.9 % |

Two statements are separated deliberately: the mean-voidage assembly reproduces
Ergun to round-off (a *consistency* identity), while the probe-voidage assembly
sits ~1.4 % **high** — the voidage-variance bias `½ Var(ε)·f″/f` of a
heterogeneous bed evaluated with a nonlinear closure. That bias is a physical
prediction of the model, not a residual error, and it is an order of magnitude
below the 15 % acceptance band.

### 3.3 Richardson–Zaki scaling

Acceptance case: glass beads in glycerin–water (`Re_t` = 0.353, `u_t` = 0.014719 m/s
from the Schiller–Naumann balance).

| `ε` | `u_p` [m/s] (mixture buoyancy) | `Δp/L` [Pa/m] |
|---|---|---|
| 0.80 | 0.0053988 | 2275.1 |
| 0.86 | 0.0074899 | 1712.0 |
| 0.92 | 0.0101450 | 1046.6 |
| 0.96 | 0.0122730 | 546.0 |

Fit: `u_p = 0.014762 ε^4.5047`, `R² = 0.999994`, max |log residual| = 9.1·10⁻⁴;
the prefactor recovers the independently computed `u_t` to **+0.29 %**.

Where does the Gidaspow blend stop tracking Richardson–Zaki? (`n_sim − n_RZ`,
both buoyancy conventions; segmentation is a data property, not a tolerance)

| case | `Re_t` | `n_RZ` | `n_sim` (mixture) | Δ (mixture) | `n_sim` (hydrostatic) | Δ (hydrostatic) |
|---|---|---|---|---|---|---|
| glycerin–water 30 P | 0.0105 | 4.65 | 4.636 | **−0.014** | 3.638 | −1.012 |
| glycerin–water 5 P | 0.353 | 4.540 | 4.505 | **−0.035** | 3.527 | −1.013 |
| sand–water | 145.9 | 2.673 | 3.032 | +0.358 | 2.369 | −0.305 |
| steel–water | 378.2 | 2.430 | 2.909 | +0.478 | 2.277 | −0.153 |
| sand–air | 455.8 | 2.385 | 2.892 | +0.506 | 2.264 | −0.121 |

The blend is RZ-consistent in the creeping-flow regime (`Re_t ≲ 1`) and departs
by +0.36…+0.51 in the transitional regime — an honest limitation of the
Gidaspow/Wen–Yu voidage function, reported rather than tuned (§5).
The Gidaspow blend itself is discontinuous at its own switch voidage:
at `ε = 0.8`, `u = 0.05 m/s` the Wen–Yu coefficient is **−59.0 %** of the Ergun
branch, so a run whose voidage crosses 0.8 sees a step in the drag. That jump is
measured by `gidaspow_branch_jump` and pinned by a test.

## 4. Figures

* `fig1_packing.png` — one-diameter slab of the random packing coloured by local
  voidage (field map), the voidage histogram with `ε̄`, `σ` and the volume-balance
  value, and the probe-radius sensitivity.
* `fig2_ergun.png` — `Δp/L` vs `U` (log–log): Ergun, probe-voidage assembly,
  mean-voidage assembly, Wen–Yu branch; right panel the relative error with the
  ±15 % band and the two acceptance regimes marked.
* `fig3_richardson_zaki.png` — `u_p` vs `ε` (log–log) for both buoyancy
  conventions with the fitted power laws and the Richardson–Zaki line; right
  panel `n_sim − n_RZ` versus `Re_t` for the five fluids with the ±0.3 band.
* `fig4_timestep.png` — coupled `Δp/L(t)` at `Δt` and `Δt/2` (time in units of the
  drag relaxation time) and the two time-step error measures against the 3 % line.

## 5. Notes, traps and limitations

1. **Buoyancy is not a second force.** `F = −V_p∇p` *is* the Archimedes force in a
   hydrostatic field; adding a separate `ρ_f V_p g` doubles it. The static-bed
   audit is exactly the guard: a double count shows up as an O(1) residual (the
   buoyancy term is 37.6 % of the total gravity in the test bed, test
   `test_double_counting_buoyancy_would_be_caught`).
2. **One power of voidage moves the RZ index by one unit.** Two pressure fields
   are defensible: the pure-fluid hydrostatic one (`∇p = −ρ_f g`, net body force
   `V_p(ρ_p−ρ_f)g`) and the mixture one that actually supports a suspension
   (`∇p = −[ερ_f+(1−ε)ρ_p]g`, net body force `V_p ε(ρ_p−ρ_f)g`). Only the mixture
   form is consistent with the Anderson–Jackson inter-phase momentum balance that
   the Gidaspow β is written for, and only it reproduces RZ (4.505 vs 4.540);
   the hydrostatic form loses exactly one exponent unit (3.527). Both are
   implemented (`body_force_per_particle(buoyancy_model=...)`) and both are
   reported, because this is the classic one-exponent discrepancy between
   Ergun-type and RZ-type correlations.
3. **The Ergun criterion is a closure-consistency audit, not an independent
   validation of Ergun.** The drag assembly *uses* an Ergun-derived branch, so
   agreement to 1e-12 with the mean voidage is by construction. What is
   non-trivial is (i) the voidage-bias value (~1.4 %) that the heterogeneous
   probe field produces, and (ii) the −85 % control: using the dilute branch in a
   dense bed fails the same check by five standard deviations of the acceptance
   band, so no unit slip can hide behind the identity.
4. **The fluid has one degree of freedom.** The settling model carries a uniform
   velocity fixed by zero net volume flux; there are no wakes, no clustering and
   no resolved pressure field. Consequently the measured `n` is a property of the
   closure verified end-to-end through the DEM integrator and the continuity
   constraint — not an emergent two-phase flow phenomenon.
5. **The runs are contact-free by construction** (jittered arrays with
   separation ≥ d, monodisperse packing with residual overlap < 10⁻³ d). Contact
   mechanics, fluidisation onset and wall effects are outside this node.
6. **The steady-state time-step criterion is trivially satisfied** (2·10⁻¹⁶):
   the overdamped steady state does not depend on `Δt`. The informative number is
   the transient one (1.21 % at `t = 2τ`), which is what an adaptive-step
   production run actually experiences.
7. **RZ tabulation spread.** Published tabulations of `n` differ by ~0.1 at low
   `Re`; the acceptance margin (0.035) is unaffected, and even the extreme
   `n = 4.65` value leaves Δ = −0.145 < 0.3.

## 6. Error assessment against the acceptance bands

| key quantity | value | reference | error | < 3 %? |
|---|---|---|---|---|
| local-voidage mean ε̄ | 0.37858 | 0.380 (volume balance) | −0.37 % | **yes** |
| `Δp/L`, laminar (U = 2 mm/s) | 2212.71 Pa/m | 2180.55 Pa/m (Ergun) | +1.47 % | **yes** |
| `Δp/L`, transition (U = 50 mm/s) | 103258.3 Pa/m | 101874.8 Pa/m (Ergun) | +1.36 % | **yes** |
| terminal velocity | 0.155280 m/s | 0.155300 m/s (Schiller–Naumann) | −0.013 % | **yes** |
| RZ prefactor (fitted `u_t`) | 0.014762 m/s | 0.014719 m/s (Schiller–Naumann) | +0.29 % | **yes** |
| static-bed net force | 0.01272391 N | 0.01272391 N (analytic submerged weight) | 0.0 % (1.2·10⁻¹⁶ per particle) | **yes** |
| `Δp/L` on `Δt → Δt/2` | 2275.1428 Pa/m | 2275.1428 Pa/m | 2.0·10⁻¹⁶ % (steady), 1.21 % (transient) | **yes** |
| RZ index | 4.5047 | 4.5395 | Δ = −0.035 (index units, threshold 0.3) | **yes** (not a % quantity) |
| Gidaspow blend vs RZ, `Re_t ≈ 146` | 3.032 | 2.673 | Δ = +0.358 | **no — out of band, reported as a limitation** |

## 7. Code map and tests

| file | contents |
|---|---|
| `src/tensordem/bedcoupling/closures.py` | Ergun, Wen–Yu/Gidaspow blend, drag coefficient, buoyancy/pressure gradient, Schiller–Naumann terminal velocity, RZ correlations |
| `src/tensordem/bedcoupling/packing.py` | random close packing, jittered arrays, exact sphere–sphere overlap, kernel local voidage, packing statistics |
| `src/tensordem/bedcoupling/bed.py` | bed force assembly, fixed-bed `Δp/L`, static-balance audit, settling dynamics, power-law fit, settling sweep |
| `src/tensordem/bedcoupling/study.py` | the six-criterion acceptance study, JSON writer |
| `tests/test_bedcoupling_closures.py` | 17 analytic anchors (Stokes limit, Ergun split, branch switch/jump, RZ piecewise, terminal-velocity balance) |
| `tests/test_bedcoupling_packing.py` | 8 tests (overlap volume vs Monte-Carlo, analytic single-particle voidage, packing window/determinism) |
| `tests/test_bedcoupling_bed.py` | 12 tests (Ergun identity to 1e-12, voidage-bias band, negative control, static residual, buoyancy conventions, terminal velocity, RZ fit, DEM-vs-algebra exponent) |
| `tests/test_bedcoupling_acceptance.py` | 8 tests (end-to-end reduced-size study, all six criteria) |

Verification commands:

```
python -m pytest tests/test_bedcoupling_closures.py tests/test_bedcoupling_packing.py \
                 tests/test_bedcoupling_bed.py tests/test_bedcoupling_acceptance.py -q
python scripts/in143_n09_study.py --out-dir artifacts/IN143-N09
```

No pre-existing module was modified; the node is additive and its public surface
is exported from `tensordem.bedcoupling`.
