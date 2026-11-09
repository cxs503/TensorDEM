# P50W2-N03 — Dense particle–fluid coupling: Richardson–Zaki / Ergun / minimum-fluidisation anchors with added mass and pressure-gradient force

**Repository:** TensorDEM · **Module:** `tensordem.cfd_dem_dense` (new) ·
**Node:** `p50w2-arb7250-n03-dense-dem-cfd` · **Attempt:** 2 ·
**Artifacts:** `artifacts/cfd_dem_dense/study.json`, `artifacts/cfd_dem_dense/summary.md`, `fig1..fig5*.png` ·
**Preregistration:** `prereg/P50W2_N03_dense_coupling_prereg.json` (thresholds frozen before the sweep) ·
**Reproduce:** `python benchmarks/dense_coupling_verification.py` (49 s, single core) ·
**Tests:** `python -m pytest tests/test_dense_dem_cfd.py -q` (35 passed, 98 s)

---

## 1. Case introduction

### 1.1 What is being verified

An unresolved (point-particle) DEM–CFD coupling represents a dense bed by
carrying the fluid on a grid coarser than the grains; every grain–fluid
interaction is therefore a **closure** — a force law in the slip velocity, the
diameter and the *local* voidage. A dense bed is where closures are hardest and
where errors are least visible: the bed does not move, a wrong pressure drop
looks like a wrong permeability, and a wrong voidage exponent looks like a
wrong bed height.

This node builds the verification surface for the dense closures in TensorDEM:
five drag/force channels that can be switched on and off individually
(Schiller–Naumann, Wen–Yu voidage correction, Di Felice, Ergun/Gidaspow dense
branch, pressure-gradient force, Maxey–Riley added mass), and three classical
**anchors** that a dense closure must satisfy:

* **Richardson–Zaki (1954) bed expansion** — the voidage exponent `n` implied by
  the closure must agree with the tabulated correlation at the same `Re_t`;
* **Ergun (1952) pressure drop** — the fluid momentum balance across a static bed
  must reproduce the porous-media pressure drop at both ends of the Re range;
* **minimum fluidisation** — the force balance that holds a static bed must agree
  with the closed-form Ergun inversion.

Plus the numerical backbone that makes any of it trustworthy: action–reaction
momentum conservation, grid convergence, mass/energy conservation, and three
negative controls that must fail.

### 1.2 Relation to the existing `bedcoupling` node

`tensordem.bedcoupling` (node IN143-N09) owns a packing-level study of the
Gidaspow blend with kernel voidage and a batch-settling run. This node is a
different surface: it adds the **Di Felice** branch, the **added-mass** force and
a fully switchable **pressure-gradient** channel, and it replaces settling-curve
fitting with *dynamic inversion* against pre-declared analytic anchors. The two
nodes share one Richardson–Zaki tabulation (`rz_correlation_exponent` mirrors
`bedcoupling.closures.rz_exponent`) so the programme quotes a single correlation.

### 1.3 Geometry and parameters

The verification object is a **periodic homogeneous bed element**: `N`
representative spheres in a 1-D column of length `L` and unit cross-section at a
prescribed solid fraction `φ`, with the fluid driven at a prescribed superficial
velocity `u_s`. There is no free surface and no contact network; bed *expansion*
is reported through the exact mass identity `H/H0 = φ0/φ`.

| Quantity | Symbol | RZ anchor | Ergun anchor | Unit |
|---|---|---|---|---|
| particle diameter | `d` | 100 | 200 | µm |
| particle density | `ρ_p` | 2500 | 2500 | kg/m³ |
| fluid density | `ρ_f` | 1000 | 1000 | kg/m³ |
| dynamic viscosity | `μ` | 1.0·10⁻³ | 1.0·10⁻³ | Pa·s |
| gravity | `g` | 9.80665 | 9.80665 | m/s² |
| solid fraction | `φ` | 0.42 (packed), ε swept 0.55–0.95 | 0.50 | — |
| bed length | `L` | 0.02 | 0.20 | m |
| superficial velocity | `u_s` | inverted per point | 0.01 / 0.30 | m/s |
| terminal Reynolds | `Re_t` | 0.73 | — | — |
| grid / grains | `n_cells` / `N` | 64 / 64 | 64 / 64 | — |

`Re_t = 0.73` is deliberate: the Richardson–Zaki correlation was established on
fine-particle data, and this is the window where the Wen–Yu dilute branch is
calibrated. Higher-`Re_t` deviations are *reported*, not tuned away (§5.3).

---

## 2. How it was done

### 2.1 Model A two-fluid balances

The package works with bed-volume force densities `f` [N/m³] and per-particle
forces `F = f V_p/φ`. The two-fluid equations (quasi-steady fluid, no fluid
inertia, `z` up) are

```
fluid:  0           = -ε dp/dz - ε ρ_f g - f
solid:  ρ_p φ dv/dt =  f - φ dp/dz - φ ρ_p g - dσ_c/dz
```

Eliminating `dp/dz` gives the combined solid force density

```
ρ_p φ dv/dt = f/ε - φ (ρ_p - ρ_f) g - dσ_c/dz .
```

Two consequences drive the whole design:

1. **The pressure-gradient force is not optional** with the Ergun/Wen–Yu family:
   `f` already carries the voidage (buoyancy) correction, and adding a separate
   Archimedes term double counts it. `ForceModel.pressure_gradient=False` exists
   precisely to expose that failure (NC3).
2. **The ε bookkeeping is where bugs live.** The classic Ergun form is a
   *porous-media* relation; the per-particle drag needs one extra factor `ε`.
   The pressure-drop-consistent grouping is

   ```
   β_ModelA = 150 φ² μ/(ε² d²) + 1.75 φ ρ_f |u| /(ε² d)   (= ε · classic)
   ```

   so that `f = β u` gives `-dp/dz = classic·u` and the Model-A balance
   reproduces the textbook `u_mf` at *every* Re. The widely quoted Gidaspow (1994)
   form puts `ε d` in the inertial term, which differs by a factor `ε` and biases
   a fully inertial `u_mf` by ~20%: that difference is a documented finding of
   this node (the test
   `test_ergun_beta_reproduces_classic_porous_media_form` pins the choice).

### 2.2 Switchable closure panel

`ForceModel` (frozen dataclass) exposes: `drag_model` ∈ {`ergun`, `wen_yu`,
`gidaspow`, `di_felice`, `schiller_naumann`}, a `voidage_exponent` override,
`pressure_gradient`, `added_mass` (+ `C_am = 0.5`), `fluid_reaction`,
`cd_scale`, `cd_model` ∈ {`closure`, `newton`}. `describe()` is written into the
study JSON so every number carries its switch state.

* **Schiller–Naumann** `C_D = 24/Re (1 + 0.15 Re^0.687)` (valid `Re < 800`;
  within 12% of tabulated sphere drag over `Re` 1–1000).
* **Wen–Yu** voidage correction `ε^-2.65` on the dilute drag; Stokes limit
  `F = 3πμ d u ε^-2.65`.
* **Di Felice** `χ = 3.7 - 0.65 exp[-(1.5 - log10 Re)²/2]`, `F ∝ ε^-χ`.
* **Ergun/Gidaspow** Ergun group for `ε < 0.8`, Wen–Yu above; the measured branch
  jump at `ε = 0.8` is **−28.9%** (`β` drops from 1.066·10⁶ to 7.577·10⁵ at
  `u = 1 mm/s`) — it is *not* smoothed, it is reported.
* **Pressure-gradient** `dp/dz = -(ε ρ_f g + f)/ε`; `F = -V_p ∇p`.
* **Added mass** Maxey–Riley `C_am ρ_f V_p (Du/Dt - dv/dt)`, carried on the
  effective mass `m + m_a` of the integrator (`m_a/m = 0.2` here).

### 2.3 Dynamic inversion (why the anchors are not algebraic identities)

Each anchor is computed **twice**:

* *dynamic route* — the coupled particle equation of motion is integrated with
  RK4 (`t_end = 3 ms`, `dt = 50 µs`, effective mass including added mass) and the
  zero-drift superficial velocity is found by bisection on `u_s`. 15 bisections
  per point, 9 points for the RZ sweep;
* *algebraic route* — the Model-A equilibrium `f = ε φ (ρ_p - ρ_f) g` solved by
  bisection on `u`.

The two routes agree to `dynamic_vs_algebraic_relative < 1e-3` (table in
`study.json`), which is what makes the dynamic numbers meaningful and the
algebraic ones cheap.

### 2.4 Grid criterion: the voidage the CFD grid actually sees

In an unresolved coupling the "grid" is the resolution of the volume-fraction
field: the drag closure is evaluated at the *filtered* voidage. The graded-bed
case uses

```
φ(z) = 0.55 sin²(π z / L) ,     L = 0.20 m
```

sampled on a 20 001-point reference grid, filtered with a normalised hat kernel
of half width `2Δx` (exactly the operation an unresolved code performs), and the
pressure drop is integrated cell by cell. Refining the grid refines **both** the
quadrature and the filter — two `O(Δx²)` error sources — and the reference is a
600-point Gauss–Legendre quadrature of the exact integrand. Errors at
`n_cells = 20, 40, 80, 160, 320` are all non-zero and decrease.

### 2.5 Conservation identities with teeth

* **Momentum (criterion 1)**: per-particle forces are deposited as reactions on
  the fluid grid with normalised hat weights; the residual is normalised by
  `Σ|F|` (not by `|ΣF|`, which cancels at equilibrium and would let a bug hide).
  Particle velocities are heterogeneous (50% spread) so the forces differ; the
  same audit is re-run with `fluid_reaction=False`, which must fail at O(1).
* **Mass (criterion 6)**: conservative FV upwind transport of the solid fraction
  with a smooth velocity field `w(z) = w₀(1 + 0.5 sin 2πz/L)`; the same test is
  re-run in *advective* (non-conservative) form, which must drift at % level.
* **Energy (criterion 6)**: kinetic energy is taken as `½(m + m_a)v²` (the added
  mass carries the kinetic energy of the entrained fluid — omitting it produces a
  spurious 21% mechanical-energy residual, which is exactly how the bookkeeping
  error was caught). Reported channels: inter-phase power pair (exact, tests the
  superficial vs interstitial velocity convention), the discrete RK4 ledger
  identity (exact), the potential-energy/work identity, and the *independent*
  trapezoidal mechanical closure, which is quadrature-limited and therefore
  reported with its own Δt-convergence ratio instead of being claimed at 1e-10.

### 2.6 Toolchain

Python 3.11 / numpy 2.4.6, no GPU, single core, 49 s for the whole study; figures
with matplotlib (Agg). Every number in `study.json` is produced by the code that
the test-suite exercises — nothing is transcribed.

---

## 3. Results

### 3.1 Criterion table (generated by `study.summary_markdown`, verbatim)

| # | criterion | measured | threshold | verdict |
|---|---|---|---|---|
| 1 | momentum conservation | 1.803e-16 (reaction off: 1.000) | $\leq$ 1e-12 | PASS |
| 2 | Richardson--Zaki index | n = 4.5011 vs 4.4419 -> 1.333%; monotone = True | $\leq$ 5% | PASS |
| 3 | Ergun pressure drop | 0.0957% (viscous, inertial) | $\leq$ 5% | PASS |
| 4 | $u_{mf}$ force balance | 1.253531e-04 vs Ergun 1.252366e-04 -> 0.0930% | $\leq$ 8% | PASS |
| 5 | grid convergence | ratios 3.83, 3.96, 3.99, 4.00 | $\geq$ 2.5 | PASS |
| 6 | conservation | mass 3.78e-16, power pair 7.25e-17, ledger 8.99e-16 | $\leq$ 1e-10 | PASS |
| 7 | negative controls | NC1 73.55%, NC2 0.0000%, NC3 296.50% | >20% / $\leq$2% / >20% | PASS |

`all_criteria_passed = True` (wall clock 48.7 s, Python 3.11.6 / numpy 2.4.6).

### 3.2 Richardson–Zaki expansion (fig. 1)

Nine dynamic inversions over `ε ∈ [0.55, 0.95]`:

| ε | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|
| `u_s` [m/s] (dynamic) | 5.0065e-4 | 1.0769e-3 | 2.0591e-3 | 3.5992e-3 | 5.8517e-3 |

Least-squares `ln u_s = ln u_t + n ln ε`: **n = 4.5011**, `R² = 0.99992`,
monotone increasing, largest log residual 0.0118, fitted `u_t = 7.459e-3 m/s`
vs the single-sphere terminal velocity `7.292e-3 m/s` (`Re_t = 0.73`), i.e. 2.3%. Tabulated correlation at `Re_t = 0.73`:
**n = 4.4419** → deviation **1.333%** (threshold 5%). The algebraic route gives
`n = 4.5016` (deviation 1.344%), i.e. the two independent routes agree to 1e-4
relative — the anchor is a property of the coupled force assembly, not of a
rearranged formula.

### 3.3 Ergun pressure drop at two Re ends (fig. 2)

| `u_s` [m/s] | `Re_s` | inertial fraction | `Δp` measured [Pa] | `Δp` reference [Pa] | uniform error | graded error (160 cells) |
|---|---|---|---|---|---|---|
| 0.01 | 2 | 4.5% | 1.7661e4 | 1.7661e4 | 0.0 | 8.75e-4 |
| 0.30 | 60 | 58.3% | 1.08196e6 | 1.08196e6 | 2.2e-16 | 9.57e-4 |

The uniform-bed column is algebraically exact (which is the *point*: it pins the
`ε` factors), the graded column carries a real discretisation error below 0.1% at
160 cells. The reference is the physical statement that the pressure supports the
total bed weight: `ρ_f g H + classic Ergun Δp`, and at `u = u_mf` the measured
`Δp` equals `(φρ_p + ερ_f) g H` to 0.2%
(`test_fixed_bed_pressure_drop_equals_bed_weight`).

### 3.4 Minimum fluidisation (fig. 5, right)

`ε_mf = 0.42`, `d = 100 µm`, Gidaspow closure:

| quantity | value |
|---|---|
| `u_mf` dynamic (coupled ODE root) | 1.25353e-4 m/s |
| `u_mf` Ergun closed form | 1.25237e-4 m/s |
| deviation | **0.093%** (threshold 8%) |
| `Re_mf` | 0.0125 |
| drift velocity at `0.90 u_mf` | −2.10e-5 m/s (settles) |
| drift velocity at `1.15 u_mf` | +3.16e-5 m/s (entrained) |

The two-sided behaviour is the independent consistency check: the same integrator
that reproduces the closed form also flips the sign of the drift on either side.

### 3.5 Grid convergence (fig. 4)

| `n_cells` | 20 | 40 | 80 | 160 | 320 |
|---|---|---|---|---|---|
| `Δx` [m] | 1.0e-2 | 5.0e-3 | 2.5e-3 | 1.25e-3 | 6.25e-4 |
| relative error | 1.69e-2 | 4.41e-3 | 1.11e-3 | 2.79e-4 | 6.98e-5 |
| ratio (halving) | — | 3.83 | 3.96 | 3.99 | 4.00 |

Observed order `p = log2(ratio) → 2.00`: the voidage filter + quadrature are
second order, as designed. Minimum ratio **3.83 ≥ 2.5**.

### 3.6 Conservation (fig. 3)

| identity | measured | threshold |
|---|---|---|
| solid mass drift (flux form) | 3.78e-16 | 1e-10 |
| non-conservative variant (teeth) | 1.51e-1 | must fail |
| inter-phase power pair | 7.25e-17 | 1e-10 |
| discrete RK4 ledger identity | 8.99e-16 | 1e-10 |
| potential energy vs gravity work | 1.4e-12 | 1e-10 |
| independent trapezoid closure | 1.06e-6 at `Δt = 2.5 µs` | reported (quadrature-limited) |
| trapezoid-closure Δt convergence | ratio 4.0 per halving (2nd order) | — |

Action–reaction (criterion 1): residual **1.80e-16** at `ε_s = 0.3` and exactly
0 at `ε_s = 0.5`, with `Σ|F| = 6.0e-4 N` and 2.4e-3 N respectively (the residual
is not a 0/0). The drag-density/pressure-drop cross-check agrees to 3.2e-16.
With the reaction disabled the audit returns **1.000**, so the diagnostic has
teeth.

### 3.7 Negative controls (fig. 5, left)

| control | expected | measured | verdict |
|---|---|---|---|
| **NC1** particle→fluid reaction off | `Δp` deviation > 20% | **73.55%** (196.1 Pa vs 741.4 Pa) | PASS |
| **NC2** `C_D × 2`, Newton regime | `u_t` ratio = 2^−0.5 ± 2% | ratio 0.70710678, deviation **1.2e-14** | PASS |
| **NC3** pressure-gradient force off | `u_mf` deviation > 20% | **296.5%** (4.97e-4 vs 1.25e-4 m/s) | PASS |

NC2's Stokes-regime cross-check is reported, not hidden: with the Re-dependent
Schiller–Naumann coefficient the ratio is **0.5203** (26% from 2^−0.5), because
the Stokes drag scales as `1/u`, not `1/u²`. The ≤2% criterion is only
meaningful in the inertial regime, and that is where it is applied.

---

## 4. Figures

| figure | content |
|---|---|
| `fig1_richardson_zaki.png` | left: `u_s(ε)` from the dynamic inversion with the fitted `n = 4.501` line and the `n = 4.442` correlation line (log–log); right: mass-conserving expansion `H/H0 = φ0/φ` |
| `fig2_ergun_pressure_drop.png` | left: measured vs reference `Δp` at the two Re ends; right: relative error for the uniform and graded beds with the 5% threshold |
| `fig3_conservation.png` | left: action–reaction residual at `ε_s = 0.3/0.5` against the 1e-12 threshold; right: the three energy channels vs `Δt` with the 1e-10 line, showing the exact identities flat and the trapezoid closure converging at second order |
| `fig4_grid_convergence.png` | graded-bed error vs `Δx` with the `∝ Δx²` reference line and each measured ratio annotated |
| `fig5_negative_controls.png` | left: NC1/NC2/NC3 deviations (log scale) against their acceptance thresholds; right: `u_mf` dynamic vs Ergun closed form |

---

## 5. Notes, pitfalls and scope

### 5.1 Pitfalls found (and how they were caught)

1. **Buoyancy double counting.** Adding `-V_p ∇p` on top of an Ergun-type drag
   double counts the voidage correction. The Model-A derivation in §2.1 is the
   guard; NC3 (296% `u_mf` shift) quantifies what the wrong choice costs.
2. **The `ε` in the Ergun inertial term.** Gidaspow's textbook grouping (`ε d`)
   is *not* the pressure-drop-consistent one (`ε² d`). Left as `ε d`, the
   inertial pressure drop comes out a factor `ε` low (29% error at `ε = 0.5`),
   which the criterion-3 test catches immediately.
3. **Added mass and the energy ledger.** Kinetic energy must be `½(m + m_a)v²`.
   Using `½ m v²` leaves a clean, Δt-independent **21%** mechanical-energy
   residual that no amount of time-step refinement removes — a very characteristic
   signature that the ledger, not the integrator, is wrong.
4. **The trapezoid vs the integrator.** The mechanical closure only reaches 1e-6
   at `Δt = 2.5 µs`; it is quadrature-limited and reported as such. The 1e-10
   threshold is applied to identities that are algebraic given the scheme (mass
   flux form, action–reaction, power pair, RK4 ledger), all of which close at
   1e-16 and each of which has a demonstrated failure mode.
5. **Attempt-1 lessons.** The previous attempt reported "all criteria passed"
   while (i) comparing the fitted RZ index against a hand-set `[2.5, 5.0]` band
   instead of the correlation at the case `Re_t`, (ii) calling the grid study
   "conceptual" with three *identical* errors and an infinite convergence rate,
   (iii) claiming energy conservation "verified through momentum conservation",
   and (iv) leaving the negative controls as bare booleans. This attempt replaces
   each of those with a measured number and a stated failure threshold; the two
   dynamic/algebraic routes and the "teeth" re-runs are there so a criterion
   cannot be satisfied by construction.

### 5.2 Branch jump at `ε = 0.8`

The Gidaspow blend is discontinuous: at `u = 1 mm/s` the coefficient drops by
**28.9%** across the switch. Any dense-bed simulation that hovers near `ε = 0.8`
(bubbling beds are exactly there) sees a spurious force discontinuity. This node
reports it rather than smoothing it; smoothing (e.g. a 0.05-wide blend) is a
design decision that should be made with a user, not silently.

### 5.3 Closure sensitivity — what the anchors actually pin down

| closure | `n_fit` (RZ) | deviation | `u_mf` [m/s] | deviation vs Ergun |
|---|---|---|---|---|
| `wen_yu` (anchor) | 4.5016 | **1.34%** | 1.4389e-4 | 14.9% |
| `di_felice` | 5.5667 | 25.3% | 5.8365e-5 | 53.4% |
| `gidaspow` (blend) | 5.3041 | 19.4% | 1.2535e-4 | **0.09%** |
| `schiller_naumann` (no voidage factor) | 1.8770 | 57.7% | 1.4111e-3 | 1027% |

Reading: (i) the RZ anchor is a *dilute-branch* property — only the Wen–Yu
exponent reproduces the correlation, while Di Felice's `χ(Re)` implies `n = 2+χ`
and overshoots by 25%; (ii) the `u_mf` anchor is a *dense-branch* property — only
the Ergun/Gidaspow branch reproduces the closed form, and the dilute closures are
off by 15% (Wen–Yu) to a factor of 11 (no voidage correction); (iii) the Gidaspow
blend passes `u_mf` because the packed case sits on its Ergun branch, but its RZ
index is 19% high because the sweep crosses the `ε = 0.8` switch. Applying the
*right* anchor to the *right* regime is the whole content of this node.

### 5.4 Limitations

* Homogeneous bed element: no particle contacts, no contact stress, no free
  surface, no fluid inertia or turbulence at the particle scale. The value of `n`
  is therefore a closure property; a free-surface bed (attempt-1 prototype) adds
  an `O(Δx)` surface-position bias that is *not* quantified here.
* Monodisperse spheres; polydispersity, sphericity and clustering are outside the
  scope (each is its own closure question).
* The fluid is quasi-steady. `Re_superficial ≤ 60` in the Ergun case and `≤ 0.02`
  in the `u_mf` case, so fluid inertia in the interphase force is negligible, but
  that is an assumption.
* The grid criterion measures the resolution of the *voidage field*
  (the unresolved-coupling grid), not wall-normal resolution of a resolved
  simulation.
* The `u_mf` criterion compares two expressions built on the same closure; it
  certifies the *implementation* of the force balance (factor-of-ε bookkeeping,
  pressure-gradient channel, integrator) and not the closure's physical
  accuracy. The closure's accuracy is what the sensitivity table in §5.3 and the
  RZ anchor speak to.

---

## 6. Error judgement against references

| key quantity | reference | measured deviation | **< 3%?** |
|---|---|---|---|
| RZ index `n` | tabulated RZ correlation at `Re_t = 0.73` (4.4419) | **1.333%** | **yes** |
| Ergun `Δp`, viscous end | classic Ergun + hydrostatic | 0.0% (uniform) / 8.8e-4 (graded) | **yes** |
| Ergun `Δp`, inertial end | classic Ergun + hydrostatic | 2.2e-16 (uniform) / 9.6e-4 (graded) | **yes** |
| `u_mf` | closed-form Ergun quadratic | **0.093%** | **yes** |
| `Δp` at `u_mf` | bed weight `(φρ_p + ερ_f) g H` | 0.2% | **yes** |
| fitted `u_t` in the RZ fit | single-sphere terminal velocity (7.292e-3 m/s) | 2.3% (7.459e-3 m/s) | **yes** |
| action–reaction residual | 0 (exact) | 1.8e-16 | **yes** |
| grid order | 2 (design) | observed 1.99–2.00 | **yes** |
| NC2 terminal-velocity ratio | `2^−0.5` | 1.2e-14 | **yes** (inertial regime) |
| NC2 ratio, Stokes regime | `2^−0.5` | 26.4% | **no** — expected: Stokes drag scales as `1/u`, so the ratio is 1/2; reported as a cross-check |

Nothing above is a hidden failure: the two "no" rows are the *expected* limits of
the respective laws and are stated as such. No acceptance threshold was changed
after the sweep; the study JSON records the thresholds next to the measurements.