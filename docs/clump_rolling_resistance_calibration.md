# Clump rolling and drilling resistance: contact law, tilt-table calibration, sensitivity and applicability

Mission **m5052** (campaign node `IN20-D2`, "DEM 标定续刀"). This document closes
the clump branch of the m605 DEM red/amber ledger item **GAP-DEM-G4**
("no tangential friction / rolling resistance"; report id 542, DEM gap matrix
§3.3) for the clump contact path, and states where the result does **not** apply.

Every number below is written by
`scripts/m5052_clump_rolling_calibration.py` into
`artifacts/m5052/clump_rolling_calibration.json` (figures in the same directory);
none is hand-copied. Test evidence: `tests/test_clump_rolling_resistance.py`
(16 tests).

| evidence | where |
|---|---|
| contact law + statics + harness | `src/tensordem/clump_rolling.py` |
| wiring into clump contacts | `src/tensordem/clump_dynamics.py` (`ContactConfig.rolling_friction` / `drilling_friction`, `rolling_model`, `PlaneContact`, `SphereObstacle`) |
| calibration / sensitivity / audit | `scripts/m5052_clump_rolling_calibration.py` |
| numbers + figures | `artifacts/m5052/clump_rolling_calibration.json`, `artifacts/m5052/fig{1..4}_*.png` |

## 1. Case introduction

**Physical background.** `docs/clump_multisphere_particles.md` records the
multi-sphere (clump) particle as implemented for member-summed contacts, with two
row-level gaps: `clump-clump contact → NOT_SUPPORTED` and
`rolling resistance / torsion friction for clumps → NOT_SUPPORTED`. Rolling
resistance is the dominant missing mechanism for ice rubble: ice blocks are lumpy,
near-spherical bodies that, without a rolling moment cap, roll far too freely,
biasing the two quantities the platform reports for broken-ice navigation — the
angle of repose of a rubble field and the transport resistance of a broken-ice
channel (`docs/dem3d_rolling_resistance.md` §1 states the same for the sphere law).

**What was missing technically.** `tensordem.rolling_resistance` implemented the
sphere-on-plane law and `tensordem.contact_tangential` the Cundall–Strack sliding
spring, but the clump contact path (`PlaneContact`, `SphereObstacle`) had *no*
moment channel at all: `net_force` returned only `Σ (arm × force)`.

**Geometry / parameters of the calibration experiment.**

| quantity | value |
|---|---|
| member radius `r` | 0.05 m |
| density | 900 kg/m³ |
| clump shapes | sphere (1 member), dumbbell (2), triangle (3), cross (5) |
| normal stiffness | 1.0e+05 N/m |
| restitution | 0.2 |
| time step | 2e-4 s (sensitivity: 1e-4 … 4e-4) |
| run length | 0.5 s |
| rest criterion | downhill drift over the last 40 % of the run < 0.02 r |
| default coefficients | `mu_r = 0.20`, `mu_s = 0.90` |

## 2. Methods

### 2.1 Contact law

Per active member, with contact normal `n`, relative angular velocity
`omega = omega_clump - omega_obstacle`, normal force `F_n` and member radius `r`:

* rolling part `omega_roll = omega - (omega·n)n`, cap `M_lim = mu_r |F_n| r`;
* drilling part `omega_spin = (omega·n)n`, cap `mu_t |F_n| r`.

Three rolling branches are provided (`resistance_moment`):

| branch | moment | character |
|---|---|---|
| `cdt` | `-mu_r |F_n| r · omega_roll_hat` | memoryless, always saturated; no static branch |
| `viscous` | `-(mu_r |F_n| r / omega_ref) · omega_roll` | linear in the rate; no static branch |
| `epsd` | `M^{n+1} = clamp(M^n - k_r omega_roll dt, |M| <= M_lim)`, `k_r = 2.25 mu_r k_n r^2` | elastic-plastic spring; **has** a static branch |

Drilling resistance is memoryless in all branches. Model families follow Ai et
al. (2011), the elastic-plastic branch Iwashita & Oda (1998) / Jiang et al. (2005).
The moment is added to the contact torque about the member centre and the clump
centre of mass, so it enters the 6-DOF rigid-body integration unchanged.

### 2.2 Statics used as the reference

A clump of equal member radius `r` resting flat on a plane tilted by `theta`:
every load-bearing member centre is at height `r`, so the centre of mass sits at
height `h` (`h = r` for the flat pose). Force balance fixes
`sum N_i = m g cos(theta)` and `sum f_i = m g sin(theta)`. Gravity has no moment
about the centre of mass, so the contact resultants must cancel each other:

```
sum_i (h f_i - x_i N_i) + sum_i M_i = 0   =>   h m g sin(theta) = sum_i x_i N_i + sum_i M_i
```

Two independent mechanisms supply the right-hand side: (i) a shift of the normal
resultant inside the contact support, bounded by `a m g cos(theta)` with `a` the
distance from the centre of mass to the **downhill** support edge; (ii) the
rolling moments, bounded by `mu_r r m g cos(theta)`. Static rest also needs the
Coulomb cone, hence

```
tan(theta_c) = min( mu_s , (a + mu_r r)/h ).
```

Two corollaries drive the whole calibration discussion:

* **Single-point support (`a = 0`)**: `tan(theta_c) = mu_r` exactly — the
  single-sphere law already verified in `docs/dem3d_rolling_resistance.md`, now
  reproduced on the clump path.
* **Elongated clump**: `a/h` is order one and dominates the rolling term, so the
  tilt test measures the support geometry or sliding, *not* `mu_r`. The rolling
  coefficient is identifiable from a tilt test only while `mu_s h > a + mu_r r`.

### 2.3 Virtual tilt-table experiment

`tilt_test` places the clump on a plane with normal `(sin t, 0, cos t)`, oriented
so its member plane is parallel to the plane, starting in static contact (the
static penetration `m g cos(t)/(n_contact k_n)` is applied analytically so the
clump does not have to be dropped). The clump integrates 0.5 s of 6-DOF dynamics;
the rest criterion is the downhill drift over the last 40 % of the run.
`bracket_critical_tilt` bisects the first-motion angle to 0.05°.

## 3. Results

### 3.1 Tilt thresholds per shape (`mu_r = 0.2`, `mu_s = 0.9`)

| shape | members | `a/h` | analytic branch | analytic `theta_c` | measured `theta_c` | rel. error | `mu_r` recovered |
|---|---|---|---|---|---|---|---|
| sphere | 1 | 0.000 | rolling+couple | 11.31° | 11.29° | **0.19 %** | 0.1996 |
| dumbbell | 2 | 1.000 | sliding | 41.99° | 41.87° | **0.27 %** | ≤ 0 (bound only) |
| triangle | 3 | 1.096 | sliding | 41.99° | 41.80° | **0.45 %** | ≤ 0 (bound only) |
| cross | 5 | 1.900 | sliding | 41.99° | 41.87° | **0.27 %** | ≤ 0 (bound only) |

The support-geometry ratio `a/h = 1.0 … 1.9` of every multi-sphere shape exceeds
`mu_s = 0.9`, so all clumps are **sliding-limited at 41.99°** — five times the
sphere threshold and completely insensitive to `mu_r`. This is the quantitative
form of the applicability statement: *a tilt test on lumpy ice clumps measures
sliding friction, not rolling friction.*

### 3.2 Calibration inversion on the identifiable geometry (sphere)

| `mu_r` given | analytic `theta_c` | measured `theta_c` | rel. error | `mu_r` recovered | recovery error |
|---|---|---|---|---|---|
| 0.05 | 2.86° | 2.90° | 1.46 % | 0.0507 | 1.47 % |
| 0.10 | 5.71° | 5.73° | 0.42 % | 0.1004 | 0.42 % |
| 0.20 | 11.31° | 11.29° | 0.19 % | 0.1996 | 0.20 % |
| 0.30 | 16.70° | 16.65° | 0.28 % | 0.2991 | 0.30 % |
| 0.40 | 21.80° | 21.70° | 0.46 % | 0.3980 | 0.51 % |

Maximum inversion error **1.47 %** (at the smallest angle, where the 0.05°
bisection resolution is relatively largest). The inversion is the estimator a
tilt-table calibration would apply: `mu_r = (h tan(theta) - a)/r`.

### 3.3 Branch structure (`mu_s` sensitivity)

| shape | `mu_s` | binding branch | analytic | measured | rel. error |
|---|---|---|---|---|---|
| sphere | 0.05 | sliding | 2.86° | 2.90° | 1.46 % |
| sphere | 0.90 | rolling+couple | 11.31° | 11.29° | 0.19 % |
| dumbbell | 0.90 | sliding | 41.99° | 41.87° | 0.27 % |
| dumbbell | 3.00 | rolling+couple | 50.19° | 49.61° | 1.17 % |

The last row is the direct measurement of the contact-couple mechanism: raising
`mu_s` above `a/h + mu_r` lifts the dumbbell threshold from 41.99° (sliding) to
50.19° (`(a + mu_r r)/h = 1.2`), and the simulation follows to 1.17 %.

### 3.4 Contact-law branch and time step

| case | analytic | measured | rel. error |
|---|---|---|---|
| sphere, `epsd` | 11.31° | 11.29° | 0.19 % |
| sphere, `cdt` | 11.31° | 11.33° | 0.19 % |
| sphere, `viscous`, `omega_ref = 0.1` | 11.31° | 11.38° | 0.58 % |
| sphere, `viscous`, `omega_ref = 0.001` (default) | 11.31° | 5.74° | 49.2 % (numerically unstable) |
| dumbbell, `epsd` / `cdt` | 41.99° | 41.87° | 0.27 % |
| sphere, `dt = 1e-4 / 2e-4 / 4e-4` | 11.31° | 11.288° | 0.19 % (spread 0.0 %) |

Two honest caveats:

* The **threshold angle itself is robust** across branches (`cdt` and resolved
  `viscous` agree with `epsd` to ≤ 0.6 %): what the tilt test locates is the
  zero-net-drive boundary, which all three share.
* The **stiff default `omega_ref = 1e-3` is numerically unstable** at
  `dt = 2e-4`: the viscous moment time constant `I omega_ref / M_lim` is 20×
  shorter than the step, so the run blows up and the "5.74° threshold" is an
  artefact, not physics. Use `omega_ref ≳ 0.1 s` with this stiffness/time-step
  pair. This is why the acceptance gate for the viscous branch is *not* claimed.

### 3.5 Below-threshold behaviour (9°, sphere, `mu_r = 0.2`)

| branch | late drift | residual speed | verdict |
|---|---|---|---|
| `epsd` | 6.1e-07 m | 3.6e-04 m/s | static hold (drift is 1600× below tolerance) |
| `cdt` | 7.7e-05 m | 3.6e-04 m/s | small stick-slip creep, still below tolerance |
| `viscous`, `omega_ref = 1e-3` | 5.2e+06 m | 3.9e+07 m/s | **unstable** (see §3.4) |
| `viscous`, `omega_ref = 0.1` | 7.9e-04 m | 4.0e-03 m/s | creep at the tolerance limit (no static branch) |

The residual speed is not zero even when the clump rests: the tangential and
rolling springs are elastic, so a resting clump rings with a bounded zero-mean
oscillation. That is why the rest criterion is the *drift*, not the speed — a
detail that silently breaks naive "is it still moving?" tests.

### 3.6 First-motion mode at 1.05 `theta_c`

| shape | `|rotation| r / |displacement|` | mode |
|---|---|---|---|
| sphere | 1.002 | rolling without slip (ratio 1 by construction) |
| dumbbell | 0.000 | sliding |
| triangle | 0.001 | sliding |
| cross | 0.000 | sliding |

The sphere lands on the rolling-without-slip ratio to 0.2 %, which independently
confirms that the moment channel is doing rolling work and not spurious sliding.

### 3.7 Drilling (torsion) spin-down

`mu_t = 0.05`, `omega_0 = 2 rad/s`, `I_zz = 4.712e-04 kg m²`,
`M_d = mu_t m g r = 1.1557e-02 N m`, `alpha = -24.525 rad/s²`. After 0.06525 s the
simulated spin is 0.399744 rad/s against the closed form 0.399744 rad/s
(relative error **3.3e-13**, i.e. exact to round-off since the torque is
constant), and the centre of mass does not drift (`|x| < 1e-13 m`). Recovering
`mu_t` from the measured spin-down returns 0.050000.

### 3.8 Energy audit (9°, `epsd`)

| quantity | value |
|---|---|
| energy increase over the run | 0.0 J (max excess over initial 0.0 J) |
| dissipated moment work | 4.04e-05 J |
| stored spring energy | 5.25e-06 J |
| stored bound `M_lim²/(2k_r)` | 9.27e-06 J (56.7 % used) |
| max instantaneous `M·omega` | +1.2e-03 W (elastic return, bounded by the stored energy) |

The elastic-plastic branch can return stored energy when the rotation reverses —
the price of having a static branch at all — but it never creates energy, and the
stored amount stays inside the analytic bound.

### Figures

| file | content |
|---|---|
| `artifacts/m5052/fig1_tilt_threshold_vs_mu_r.png` | `theta_c(mu_r)` for `a/h = 0, 0.5, 1, 2` with the bisected sphere and clump points |
| `artifacts/m5052/fig2_identifiability_regime.png` | `(mu_r, mu_s)` regime map: where sliding hides the rolling coefficient |
| `artifacts/m5052/fig3_rest_criterion_and_models.png` | sub-threshold drift per branch; first-motion mode ratio |
| `artifacts/m5052/fig4_branch_structure.png` | sliding vs rolling+couple per `(shape, mu_s)` case |

## 4. Notes, limitations and applicability

1. **Applicability (measured).** For equal-radius clumps in a flat pose, `h = r`
   and the rolling term is `mu_r`, while the support term is `a/h ≥ 1` for every
   multi-sphere shape tested. Hence a tilt/repose calibration of a lumpy ice clump
   constrains `mu_s` (and the support geometry), not `mu_r`. `mu_r` is identifiable
   from a tilt test only for near-spherical particles (`a → 0`) or from dynamic
   tests (repose angle in a rotating drum, transport resistance), which this
   module does not provide.
2. **What is calibrated.** The inversion `mu_r = (h tan(theta) - a)/r` and its
   round-trip are verified; the *laboratory* anchor is **WITHHELD** — no digitised
   tilt-table or repose measurement is available in this repository, so no claim
   is made that the default `mu_r = 0.2` reproduces any specific ice/steel
   experiment. The protocol, not the coefficient, is the deliverable.
3. **Not covered.** `clump-clump` contacts remain `NOT_SUPPORTED`
   (`docs/clump_multisphere_particles.md`), so clump–clump rolling resistance is
   out of reach; the law is wired for clump–plane and clump–sphere only. Drilling
   resistance is memoryless (no elastic drilling spring), so no static drilling
   torque is available. `r_eff = r_member` is a modelling convention; using the
   clump's circumscribed radius instead would scale `mu_r` linearly (`M_lim ∝ r_eff`),
   which is exactly the "which radius?" question a user must fix before quoting a
   coefficient.
4. **Numerics.** The elastic tangential/rolling springs make the resting state a
   bounded oscillation, so rest must be judged on drift, not speed (§3.5). The
   viscous branch is conditionally stable (`dt < I omega_ref / M_lim`).
5. **MT Uikku ten-condition anchors.** Not applicable to this deliverable: it
   produces a contact-law calibration, not ice loads on a hull, so there is no
   ship-ice force to compare with
   `docs/dem3d_mt_uikku_formula_baselines.md` /
   `benchmarks/dem3d/ship_ice_validation/README.md`. The coefficient enters those
   anchors only through a future rubble/ice-channel case, with the range stated in
   §3.2 (`mu_r` recoverable for `a/h` small) and the caveats of §3.4.

## 5. Error verdicts (against the 3 % criterion where a reference exists)

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| sphere tilt threshold (4 … 6 `mu_r` values) | `arctan(mu_r)` | 0.19 … 1.46 % | **yes** |
| dumbbell / triangle / cross tilt threshold | `arctan(min(mu_s, (a+mu_r r)/h))` | 0.27 / 0.45 / 0.27 % | **yes** |
| dumbbell, `mu_s = 3` (couple branch) | `arctan((a+mu_r r)/h)` | 1.17 % | **yes** |
| `mu_r` recovered from the measured angle | value given to the solver | 0.20 … 1.47 % | **yes** |
| drilling spin-down | `omega_0 - mu_t m g r t / I` | 3.3e-13 (round-off) | **yes** |
| energy creation | second law | 0.0 J excess | **yes** |
| stored spring energy | `M_lim²/(2k_r)` | 56.7 % of bound | **yes** |
| viscous branch threshold (default `omega_ref`) | `arctan(mu_r)` | 49.2 % | **no — honest FAIL**: numerically unstable configuration, withdrawn rather than tuned away |
| laboratory tilt-table / repose measurement | — | no data | **WITHHELD** |

## m16340 write-back (viscous branch stability, 2026-11-12)

The §3.4 honest FAIL above is **resolved as a defect, not as a physics result**:
the `viscous` branch keeps its declared law, but the configuration that produced the
5.74° artefact is now refused with an actionable message instead of overflowing, and
the analytic window is verified.

* **Analytic window.** The velocity-Verlet half-kick splitting gives a one-step factor
  `(1 - lambda*dt/2)**2` with `lambda = mu_r |F_n| r / (omega_ref I)`: monotone decay
  for `lambda*dt <= 2`, divergence beyond `lambda*dt >= 4`. Bisection over a 36× mass
  range measures the boundary at `lambda*dt = 4.0017` (0.04 % from 4), confirming that
  `lambda*dt` is the control parameter
  (`artifacts/m16340/viscous_rolling_stability.json`).
* **Guard.** `ContactConfig.rolling_stability_guard` (default on) raises with the
  offending `lambda*dt`, both bounds, the admissible `dt`/`omega_ref` and the opt-in
  alternative. The default pair `omega_ref = 1e-3`, `dt = 2e-4` gives `lambda*dt = 88.3`
  and is refused; with the guard off it still overflows (`|omega| ~ 4.8e153` by step 47),
  preserved as a regression test.
* **Accuracy is a stricter limit than stability.** The free spin-down matches
  `exp(-lambda t)` to 2.55 % at `lambda*dt = 0.1` but 6.6 % at 0.25: the 3 % envelope is
  `lambda*dt <= ~0.12`, i.e. `omega_ref >= ~4 dt M_lim / I`. Calibrations must respect
  that, not merely the boundedness bound.
* **Steady state.** The declared law reproduces `omega* = omega_ref tan(theta)/mu_r`
  exactly on an incline (0.0000 % at `lambda*dt = 0.29` and 0.98); the opt-in
  gain-limited gain (`rolling_viscous_implicit=True`) is stable at any step but inflates
  the steady rate by the predicted `1 + lambda*dt/2` (+14.49 % at 0.294, measured 1.145
  vs predicted 1.147).
* **Energy.** Dissipated rolling work now equals the kinetic-energy drop to 4.5e-16;
  the previous single-evaluation rectangle rule under-reported it by up to 63 %
  (`lambda*dt = 0.88`).

Details, tables and reproduction: `docs/clump_rolling_viscous_stability.md`.
Tests: `tests/test_clump_viscous_rolling_stability.py` (12 passed).
