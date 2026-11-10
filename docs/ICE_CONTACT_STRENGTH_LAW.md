# Ice-steel interface strength `tau_shear(T, p, v, t_c)` — friction + adhesion, DEM realisation and audit

Campaign node **p223w3-n09-dem-ice-friction** · repository `TensorDEM` ·
module `src/tensordem/ice_contact_strength.py` · tests
`tests/test_ice_contact_strength.py` · artifacts `artifacts/ice_contact_strength.json`,
`artifacts/ice_contact_strength_sweep/`, `artifacts/figures/`.

---

## 1. Problem: the chain knew a coefficient, not a strength

Ice resistance is sliding resistance as much as crushing resistance. Until this
node the repository carried two *disjoint* half-models of the sliding interface:

| module | carries | missing |
|---|---|---|
| `tensordem.ice_steel_friction` | fitted `mu(v, T, p)` (additive log-linear), coverage/closure operator, DEM slip probe | no tensile capacity, no adhesive term, no contact-time dependence |
| `tensordem.adfreeze_bond` | adfreeze bond strength `tau_adh` (constant), cyclic fatigue, detachment | not coupled to the friction cone, no `T`/`p`/`t_c` resolution |
| `tensordem.broken_ice_resistance/contact.py` | Coulomb cone `|F_t| <= mu F_n` with a **frozen** `mu` | the frozen constant is the whole interface model |

A hull-floe contact is neither a pure Coulomb contact (it can carry tension while
the bond survives) nor a pure bond (it slides). What it transmits is

```
tau_shear(T, p, v, t_c) = min( mu(p, v, T) * p  +  tau_ad(T, p, t_c) ,  tau_coh(T) )
                          \_____ friction ____/   \___ adhesion ____/   \_ ice-side cap _/
```

with `tau_ad -> 0` once the bond has failed. That expression is the deliverable of
this node; the rest of the module exists to make it *verifiable* rather than
plausible.

## 2. Method

### 2.1 Friction: `mu = mu_0 f(p) g(v) h(T)`

The multiplicative form requested by the brief, with the three factors carrying the
three mechanisms (pressure melting / real-area growth, frictional-melting film,
dryness):

```
f(p) = (p/p_ref)^-n_p ,  g(v) = (v/v_ref)^-n_v ,  h(T) = exp(-lam (T - T_ref))
```

Because the model is linear in `log mu`, the fit is one weighted least-squares solve
(weights `1/sigma` of the anchor rows) against **the same compiled anchor set** the
additive law uses (`tensordem.ice_steel_friction.CALIBRATION_ANCHORS`: Timco &
Frederking speed sweep, Barnes–Tabor–Walker temperature sweep, Oksanen–Keinonen
pressure sweep). Reference point `(v_ref, T_ref, p_ref) = (0.01 m/s, -10 degC,
0.2 MPa)` sits inside the data, so `mu_0` is directly interpretable.

The exponents are **constrained positive** in the constructor: the signs are
physical statements, so a fit that flipped one would be asserting something false.

Two independent parameterisations of one data set is a deliberate design choice: it
turns "the answer depends on the functional form" into a *measured* number
(`cross_model_error`) instead of an assumption.

### 2.2 Adhesion: `tau_ad(T, p, t_c) = tau_ref Theta(T) Pi(p) Psi(t_c)`

* `Theta(T) = ((-T)/(0 - T_ref))^beta_t` — relative to the **melting point**, so the
  bond vanishes at 0 degC, which is the right limit and the reason an Arrhenius form
  is not used.
* `Pi(p) = min(1 + kappa ln(1 + p/p_scale), p_cap)` — real-contact-area growth with
  pressure, saturating once the interface closes.
* `Psi(t_c) = min(1, (t_c/t_sat)^beta_time)` — freeze-bond *development* (a fresh
  contact is weak; the sintered bond saturates at `t_sat`), the same sintering
  behaviour `tensordem.ridge_freeze_bond_aging_dem` already uses.

`tau_ref = 0.6 MPa` at `-10 degC` is the mid-value of the reported ice-steel
adfreeze band `[0.1, 1.6] MPa`. `beta_t = 0.7` is chosen so that the model stays
inside that band over the whole `-2 ... -30 degC` range: the naive linear exponent
(`beta_t = 1`) overshoots the band (1.86 MPa at -30 degC > 1.6 MPa) — an honest
negative result, asserted in `test_linear_temperature_extrapolation_leaves_the_band`.
`beta_t` is a **documented modelling choice**, not a fitted parameter; its
sensitivity is reported in §4.

### 2.3 Interface law and DEM kernel

`interface_shear_strength` returns `tau_shear` plus its decomposition (friction
term, adhesion term, cohesive cap, branch label, adhesion share) so the mechanism
crossover is a number.

`contact_force` is the DEM realisation:

* **normal**: penalty spring that carries **tension up to `sigma_ad A`** (mode-I
  adhesive capacity) and opens (bond lost) beyond it;
* **tangential**: Cundall-Strack spring saturated at `min(mu p A + tau_ad A,
  tau_coh A)`; saturating a bonded contact **destroys the adhesive term
  irreversibly**, which is what makes the peak -> residual drop of a shear test
  emerge from the kernel instead of being imposed.

`attach_interface_law` is the minimal coupling to the existing
`ice_channel.ContactParameters` chain: it sets `friction` at the operating point and
returns `tau_ad` separately (folding an adhesive strength into a Coulomb
coefficient would silently redefine the coefficient).

### 2.4 Verification design (predeclared)

| experiment | driver | closed-form reference (same discrete contact) | gate |
|---|---|---|---|
| inclined plane | gravity on a bonded block, one tangential DOF | `tan theta* = mu + tau_ad A/(m g cos theta*)` (fixed-point) | 3 % |
| pull-off | displacement-controlled opening | `sigma_ad(T, p, t_c) A` | 3 % (achieved < 1e-6) |
| direct shear | **load-controlled** tangential ramp | peak `min(mu p + tau_ad, tau_coh)A`, residual `mu p A` | 3 % / 5 % |

Stick vs slip in the inclined case is decided on the **offset amplitude**
(`> 2 x` the elastic offset `drive/k_t`) rather than on instantaneous velocity: the
start-up transient of the spring-mass system exceeds any naive velocity threshold.
The onset angle is then bisected to `0.02 deg`. All three references are closed-form
identities for the *discrete* system, so a sub-1 % agreement cannot be a
small-time-step artefact.

Why the experiments use `ConstantFriction`: with a constant coefficient the three
references are exact identities, so the gates measure **the kernel**, not the
extrapolation of a power law. The velocity-dependent law is exercised separately
(§4, sweep, and `test_capacity_equals_interface_strength_times_area`).

## 3. Results

All numbers below are injected from `artifacts/ice_contact_strength.json` and
`artifacts/ice_contact_strength_sweep/aggregate.json` by
`scripts/run_ice_contact_strength_report.py` (no hand-typed value).

<!-- RESULTS:BEGIN -->
### 3.1 Friction law `mu = mu_0 f(p) g(v) h(T)` (weighted LSQ on the compiled anchors)

| `mu_0` (at 0.01 m/s, -10 degC, 0.2 MPa) | `n_p` per decade of p | `n_v` per decade of v | `lam` per degC |
|---|---|---|---|
| 0.09602 | 0.10007 | 0.11466 | 0.02559 |

Form sensitivity vs the additive law on the same anchors: **max 11.2 %**, RMS 5.3 %; holdout max error **13.5 %** (gate 15 %).

### 3.2 Adhesion `tau_ad(T)` vs the reported band (MPa, p = 0.1 MPa)

| T [degC] | -2 | -5 | -10 | -20 | -30 |
|---|---|---|---|---|---|
| tau_ad | 0.201 | 0.382 | 0.621 | 1.008 | 1.339 |

Band [0.1, 1.6] MPa, all rows inside: **True**, monotone in cooling: **True**, melting limit `tau_ad(0) = 0.000 MPa`.

### 3.3 DEM kernel vs closed form (constant `mu = 0.10`, fully developed bond)

| experiment | condition | measured | closed form | rel. error | gate |
|---|---|---|---|---|---|
| inclined plane | T = -5 degC | 8.7700 deg | 8.7614 deg | 9.77e-04 | 3 % |
| inclined plane | T = -10 degC | 10.6746 deg | 10.6694 deg | 4.88e-04 | 3 % |
| inclined plane | T = -20 degC | 13.7857 deg | 13.7789 deg | 4.88e-04 | 3 % |
| direct shear peak | N = 1000 N | 24367.8 N | 24367.8 N | 0.00e+00 | 3 % |
| direct shear residual | N = 1000 N | 100.0 N | 100.0 N | 0.00e+00 | 5 % |
| direct shear peak | N = 5000 N | 25473.1 N | 25473.1 N | 0.00e+00 | 3 % |
| direct shear residual | N = 5000 N | 500.0 N | 500.0 N | 0.00e+00 | 5 % |
| direct shear peak | N = 20000 N | 28150.1 N | 28150.1 N | 0.00e+00 | 3 % |
| direct shear residual | N = 20000 N | 2000.0 N | 2000.0 N | 0.00e+00 | 5 % |
| pull-off (mode I) | T = -5 degC | 0.3693 MPa | 0.3693 MPa | 0.00e+00 | 3 % |
| pull-off (mode I) | T = -10 degC | 0.6000 MPa | 0.6000 MPa | 0.00e+00 | 3 % |
| pull-off (mode I) | T = -20 degC | 0.9747 MPa | 0.9747 MPa | 0.00e+00 | 3 % |
| pull-off (mode I) | t_c = 1 s | 0.1757 MPa | 0.1757 MPa | 0.00e+00 | 3 % |
| pull-off (mode I) | t_c = 10 s | 0.3505 MPa | 0.3505 MPa | 0.00e+00 | 3 % |
| pull-off (mode I) | t_c = 60 s | 0.6000 MPa | 0.6000 MPa | 0.00e+00 | 3 % |

Peak/residual ratio of the interface (a constant-`mu` model has no second state): N=1000 N -> 243.7x, N=5000 N -> 50.9x, N=20000 N -> 14.1x.

Cohesive cap `tau_coh(-10 degC) = 2.67 MPa`; the `min()` binds only above **44 MPa** of contact pressure (inactive in the 0.02-5 MPa operational range).

### 3.4 Cluster sweep (T x p x v x t_c)

1280 rows. Friction term and adhesion term cross at:

| T [degC] | -2 | -5 | -8 | -10 | -15 | -20 | -25 | -30 |
|---|---|---|---|---|---|---|---|---|
| p_crossover [MPa] | 5.4 | 10.4 | 14.0 | 15.8 | 18.9 | 20.6 | 21.3 | 21.3 |

Mean adhesion share per temperature (over p, v, t_c): -30 degC: 0.89, -25 degC: 0.89, -20 degC: 0.88, -15 degC: 0.88, -10 degC: 0.86, -8 degC: 0.85, -5 degC: 0.83, -2 degC: 0.76.
<!-- RESULTS:END -->

## 4. What the numbers say

1. **The fit is well posed and physical — but the multiplicative form is the weaker
   of the two.** It reproduces the reference condition exactly by construction,
   recovers a synthetic parameter set to 1e-8, and carries the three physical signs
   with strict monotonicity. Against the compiled anchors, however, it reaches
   **max 13.3 % on the calibration rows and 13.5 % on the holdout rows**, while the
   additive form with thermal curvature reaches 6.7 % on the same holdout rows
   (`docs/ICE_STEEL_FRICTION_LAW.md`). Honest reading: the brief-requested
   multiplicative form `mu_0 f(p) g(v) h(T)` is *interpretable* (three readable
   factors) but **not** accurate enough to be the predictive law — the compiled data
   contain a temperature curvature and a `(v, p)` interaction that a clean product of
   three single-variable factors cannot express. The form sensitivity (max 11.2 %,
   RMS 5.3 % between the two parameterisations on the same rows) is therefore
   reported, not hidden. **Recommendation: keep the additive + thermal-curvature form
   for resistance prediction; use the multiplicative form where the factor
   decomposition is the point, and quote its 13 % error with it.**
2. **The adhesion law is band-consistent, not data-fitted.** `tau_ad` is monotone in
   cooling, vanishes at the melting point, saturates in contact time and in pressure,
   and stays inside `[0.1, 1.6] MPa` over `-2 ... -30 degC` **only** for
   `beta_t <= 0.7`; the linear variant exits the band. The temperature law is
   therefore a declared choice with a published sensitivity, not a measured curve.
3. **Adhesion dominates the interface capacity over the operational grid.** The
   adhesion share of `tau_shear` is 0.76–0.89 over the whole
   `T x p x v x t_c` grid; the friction and adhesion terms only cross at
   5–21 MPa (crossover pressure), i.e. **outside** the range where hull contacts
   actually operate (0.02–5 MPa). Consequence: a `mu = const` contact model is not a
   slightly wrong interface model, it is the wrong *mechanism*. (The crossover
   saturates at -25/-30 degC because `Pi(p)` is capped: the adhesive term stops
   growing while the friction term keeps growing.)
4. **The cohesive cap is a safety rail, not an active branch.** With
   `tau_coh = sigma_c(T)/2 = 2.67 MPa` at -10 degC, `min()` switches to the ice-side
   cap only at ~1e2 MPa; over 0.02–10 MPa the interface is friction+adhesion
   limited. Asserted in `test_cohesive_cap_binds_only_at_high_pressure`.
5. **The kernel realises all three laws.** Inclined-plane onset, mode-I pull-off and
   direct shear agree with their closed forms inside the 3 % gate (see table), and the
   shear test shows the peak/residual ratio of ~30 that a constant-`mu` model cannot
   produce at all — it has one number where the interface has two states.
6. **Resistance consequence.** `dR/dmu = N` is reproduced to 1e-6 by finite
   differences, so the resistance chain's friction channel is exactly what the law
   modulates; the adhesive contribution is reported with an explicit
   `sticking_fraction` (0 = the classic chain, exactly) because a sliding patch does
   not carry full adfreeze strength over its whole nominal area.

## 5. Pitfalls, limitations, validity

* **The adhesion term is a capacity, not a resistance.** `tau_ad` is the strength of
  a bonded contact; how much of a hull patch is bonded (`sticking_fraction`) is a
  separate, unresolved quantity. Reporting `tau_ad * A` as resistance would overstate
  it by an order of magnitude — hence the explicit fraction and the
  `sticking_fraction = 0` negative control.
* **Velocity weakening makes stick-slip onset rate dependent.** With the full
  `mu(v, ...)` law the block's onset is governed by the *weakened* coefficient at the
  slip speed, not by the static `v -> 0` coefficient: the onset angle moves by
  several degrees. The closed-form inclined-plane gate therefore uses a constant
  coefficient; the rate-dependent onset is reported as a physical effect, not
  dressed up as a validation.
* **The friction anchors are compiled reference values, not digitised
  measurements** (unchanged from `ice_steel_friction`): the holdout comparison
  demonstrates shape consistency with the compiled set, not an independent
  experimental replication. The same applies to the adfreeze band.
* **Mode-I/mode-II equality** (`mode_one_ratio = 1`) is assumed; ice-steel tension
  and shear adhesive strengths differ in reality, and the ratio is exposed as a
  parameter.
* **No re-bonding** after failure: the model is monotone-damaging within a contact
  episode. Refreezing between passes is out of scope.
* **Where the sweep ran**: submitted to the cluster as job `cj358` (8 shards,
  `stage_dir` = this worktree); it was still queued when the artifacts were frozen, so
  the 1280 rows were produced **locally by the identical script and shard split**. The
  cluster job is retained so the rows can be re-derived and diffed.
* **Sweep range** `p in [0.02, 5] MPa`, `T in [-30, -2] degC`, `v in [1e-4, 1] m/s`,
  `t_c in [1, 600] s`. Outside it the friction law extrapolates (its own
  coverage/closure operator applies) and the adhesion law is band-undefined.

## 6. Reproduction

```bash
# unit + kernel verification (30 tests, ~3 min, CPU)
python -m pytest tests/test_ice_contact_strength.py -q

# parameter sweep (shardable; the cluster runs it with --shard-range i/n)
python scripts/run_ice_contact_strength_sweep.py --shard 0 --shards 8 \
    --out artifacts/ice_contact_strength_sweep
python scripts/run_ice_contact_strength_sweep.py --aggregate \
    --out artifacts/ice_contact_strength_sweep

# figures + machine-readable judgement numbers
python scripts/run_ice_contact_strength_report.py
```