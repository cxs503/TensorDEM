# Ball-ball rolling-friction contact mechanics: analytic verification (P40-N04)

**Repository**: TensorDEM · **Module**: `tensordem.ball_ball_rolling`,
`tensordem.ball_ball_rolling_verification` · **Driver**:
`scripts/verify_ball_ball_rolling_contact.py` · **Tests**:
`tests/test_ball_ball_rolling_contact.py` (35 tests)

**Artifact**: `artifacts/ball_ball_rolling_contact/results.json`
(sha256 `n/a`), figures in
`docs/figures/ball_ball_rolling_contact/`.

---

## 1. Case introduction (physics, geometry, parameters)

The repository already verified a rolling-resistance couple for a *single*
contact against a fixed body (ball-plane, `docs/dem3d_rolling_resistance_verification.md`).
This work extends the verification surface to the **two-body** case: two elastic
spheres in point contact where the normal, tangential (Cundall-Strack), rolling
and rotational degrees of freedom of **both** particles are active.  Six
physical faces are covered:

1. **Cundall-Strack tangential spring**: `F_t = -k_t delta_t`, cone
   `|F_t| <= mu_s F_n`, integrated from the exact contact slip
   `s = v_rel - (r_i omega_i + r_j omega_j) x n`.
2. **Rolling resistance moment**: `M_r = mu_r r_eff F_n` with
   `r_eff = r_i r_j / (r_i + r_j)` (Ai et al. 2011), applied as a couple
   resisting the rolling (non-spinning) part of `omega_rel = omega_j - omega_i`.
3. **Rotational damping**: `M = -c_r omega_roll`, closed form
   `omega(t) = omega_0 exp(-c_r t / I)`.
4. **Rolling-sliding transition**: sliding stops when the two material contact
   points co-move, `|s| = 0`; the impulse balance gives
   `omega*/omega_0 = 1/(1 + m r^2/I) = 2/7` (solid sphere).
5. **Inclined two-ball stack stability**: rolling resistance caps the moment a
   contact can transmit (`mu_r r_eff F_n`); the assembly holds while the
   required static moments stay inside the caps.  In the plane limit
   `r_b/r_a -> 0` this reduces to `mu_r >= tan(theta_cr)`.
6. **Energy conservation / dissipation ledger**: for a collision, the sum of the
   explicitly dissipated works plus the residual kinetic energy must equal the
   initial kinetic energy.

| quantity | symbol | unit | value / range |
|---|---|---|---|
| density (steel/ice-like) | $\rho = 2500$ | kg/m^3 | solid sphere |
| radii | $r_i, r_j$ | m | 0.05 / 0.05 unless stated |
| normal stiffness | $k_n$ | N/m | 1.0e5 (collision face) |
| tangential stiffness | $k_t$ | N/m | 2.0e4 .. 5.0e4 |
| Coulomb friction | $\mu_s$ | - | 0.10 .. 0.60 |
| rolling resistance | $\mu_r$ | - | 0.05 (moment face), 0.10 (creep) |
| rotational damping | $c_r$ | N m s | $I/\tau$, $\tau = 0.05$ s |
| gravity (incline faces) | $g$ | m/s^2 | 9.81 |

## 2. Method and tool chain

* Contact law (`src/tensordem/ball_ball_rolling.py`): piecewise tangential
  spring (elastic increment, tangent-plane projection, Coulomb plastic
  projection), rolling couple with a saturating viscous branch
  `min(c_r |omega_roll|, mu_r r_eff F_n)` and an elastic-plastic (EPSD) branch,
  plus the exact contact-point slip kinematics for two free spheres.
* Integration: semi-implicit (symplectic) Euler for translation and rotation;
  the normal coordinate is prescribed where the face isolates a tangential or
  rotational mechanism.
* Analytical yardsticks: closed forms above, plus the pre-existing
  `tensordem.rolling_resistance` (ball-plane) law used as an independent
  cross-check of the rolling couple (`legacy_rel_err`).
* Driver `scripts/verify_ball_ball_rolling_contact.py` evaluates all faces,
  writes `results.json` + `results.sha256`, and renders the figures; the test
  suite re-derives the same numbers through module-scoped fixtures.
* Environment: Python 3.11, torch 2.14.1 CPU, single thread
  (`torch.set_num_threads(1)`); no cluster run needed (total driver runtime
  ~45 s).

Commands:

```
PYTHONPATH=src python scripts/verify_ball_ball_rolling_contact.py
PYTHONPATH=src python -m pytest tests/test_ball_ball_rolling_contact.py -q
PYTHONPATH=src python scripts/report_ball_ball_rolling_contact.py
```

## 3. Results

### 3.1 (a) Tangential stiffness recovery

Kinematic tangential drive (`|s| = 1e-4` m/s) with the peak force kept at ~10 %
of the cone; `|F_t|` is regressed on `|delta_t|` through the origin.

| k_t input [N/m] | k_t recovered [N/m] | rel. error | dt [s] | steps | elastic? |
|---|---|---|---|---|---|
| 2.00000e+04 | 2.00000e+04 | 3.64e-16 | 6.25e-04 | 200 | yes |
| 5.00000e+04 | 5.00000e+04 | 1.46e-16 | 0.0012 | 200 | yes |
| 8000 | 8000 | 1.14e-16 | 5.00e-04 | 200 | yes |

Worst relative error **3.64e-16** (machine precision: the
law is linear by construction, so the test is a strict binding of the
implementation to the analytic ramp, not an accuracy claim about DEM itself).

### 3.2 (b) Coulomb cone, 10 independent cases

Each case is driven until the elastic demand exceeds the cone by >= 3x.

| mu_s | F_n [N] | cap mu_s F_n [N] | |F_t| [N] | rel. error | cos(F_t, -s) |
|---|---|---|---|---|---|
| 0.3 | 5 | 1.5 | 1.5 | 1.48e-16 | 1 |
| 0.5 | 5 | 2.5 | 2.5 | 1.78e-16 | 1 |
| 0.1 | 5 | 0.5 | 0.5 | 0 | 1 |
| 0.4 | 12 | 4.8 | 4.8 | 0 | 1 |
| 0.25 | 0.5 | 0.125 | 0.125 | 0 | 1 |
| 0.6 | 50 | 30 | 30 | 1.18e-16 | 1 |
| 0.35 | 20 | 7 | 7 | 0 | 1 |
| 0.45 | 8 | 3.6 | 3.6 | 0 | 1 |
| 0.2 | 2 | 0.4 | 0.4 | 0 | 1 |
| 0.55 | 30 | 16.5 | 16.5 | 0 | 1 |

Worst cap error **1.78e-16**; the force direction is
anti-parallel to slip to machine precision in all 10 cases.

### 3.3 (c) Rolling-resistance moment, 9 cases

| r_i [m] | r_j [m] | r_eff [m] | F_n [N] | |M_r| [N m] | mu_r r_eff F_n | rel. error |
|---|---|---|---|---|---|---|
| 0.05 | 0.05 | 0.025 | 1 | 0.00125 | 0.00125 | 0 |
| 0.05 | 0.05 | 0.025 | 10 | 0.0125 | 0.0125 | 0 |
| 0.05 | 0.05 | 0.025 | 100 | 0.125 | 0.125 | 0 |
| 0.05 | 0.1 | 0.0333333 | 1 | 0.00166667 | 0.00166667 | 0 |
| 0.05 | 0.1 | 0.0333333 | 10 | 0.0166667 | 0.0166667 | 2.08e-16 |
| 0.05 | 0.1 | 0.0333333 | 100 | 0.166667 | 0.166667 | 0 |
| 0.02 | 0.2 | 0.0181818 | 1 | 9.09091e-04 | 9.09091e-04 | 1.19e-16 |
| 0.02 | 0.2 | 0.0181818 | 10 | 0.00909091 | 0.00909091 | 0 |
| 0.02 | 0.2 | 0.0181818 | 100 | 0.0909091 | 0.0909091 | 0 |

Worst error **2.08e-16**; agreement with the
pre-existing ball-plane law (same `r_eff`, `cdt` branch) is
1.67e-16.

### 3.4 (d) Rotational damping

`c_r = I/tau` with `I = 0.001309` kg m^2, `tau = 0.05` s,
`dt = 2.50e-05` s, `4000` steps, decay over
0.8688 decades.

* `R^2` (against `omega_0 exp(-t/tau)`) = **0.999999898** (criterion >= 0.999)
* fitted time constant `tau = 0.0499875` s, relative error
  2.50e-04
* maximum relative deviation over the record 5.00e-04

### 3.5 (e) Rolling-sliding transition

The spun sphere is spun up (or down) by friction under a prescribed normal load;
the first zero of the contact slip is the transition to pure rolling.

| case | r_i [m] | omega*/omega_0 | t* [s] | v/(omega r_j) | v/(omega r_eff) | spin-ratio err | time err |
|---|---|---|---|---|---|---|---|
| plane_limit | 1.000e+09 | 0.285522 | 0.187 | 1.00094 | 1.00094 | 6.74e-04 | 1.40e-13 |
| equal_spheres | 0.05 | 0.285522 | 0.187 | 1.00094 | 2.00189 | 6.74e-04 | 1.40e-13 |
| ratio_20 | 1 | 0.285522 | 0.187 | 1.00094 | 1.05099 | 6.74e-04 | 1.40e-13 |

* analytic spin ratio 2/7 = 0.285714; measured
  0.285522 (error
  6.74e-04)
* analytic transition time
  0.187 s vs measured
  0.187 s
* the `r_eff` convention of the task statement is the `r_i -> inf` limit: at
  `r_i = 1e9 r_j` the relative error is 9.45e-04;
  for **equal spheres** the exact kinematic rolling radius is `r_j = 2 r_eff`,
  so `v/(omega r_eff) = 2.00189`
  (see section 5).
* driven-rolling force balance (steady rolling against rolling resistance):
  `|F_t| r_j` vs `M_r` error 0.00143,
  moment at the plastic limit error 2.14e-14,
  acceleration on the balance point 1.33e-06 m/s^2
  (under-drive -0.04677, over-drive
  0.04677 m/s^2: the negative controls move the
  expected way).

### 3.6 (f) Inclined two-ball stack stability

Static equilibrium of the stack is solved as a 4x4 linear system in
`[f_plane, f_pair, M_plane, M_pair]`; `theta_cr` is the angle at which the
maximum utilisation `|M| / (mu_r r_eff F_n)` reaches one.  Analytic reference
(min of the two contact caps):
`tan(theta_cr) = min[ mu_r r_eff/r_b , mu_r (m_a+m_b) r_a / ((m_a+2m_b) r_a + m_b r_b) ]`,
which equals `mu_r` for `r_b/r_a -> 0` and `mu_r/2` for equal spheres
(`mu_r = 0.2`).

| case | r_a [m] | r_b [m] | r_b/r_a | theta_cr measured [deg] | theta_cr analytic [deg] | err vs analytic | err vs plane limit |
|---|---|---|---|---|---|---|---|
| equal_spheres | 0.05 | 0.05 | 1 | 5.71059 | 5.71059 | 3.11e-16 | 0.495 |
| ratio_2 | 0.1 | 0.05 | 0.5 | 7.59464 | 7.59464 | 0 | 0.328 |
| ratio_20 | 1 | 0.05 | 0.05 | 10.7843 | 10.7843 | 1.65e-16 | 0.0465 |
| ratio_50 | 2.5 | 0.05 | 0.02 | 11.0937 | 11.0937 | 0 | 0.0191 |

Dynamic confirm (1-DOF pure rolling with an elastic-plastic rolling spring and a
rolling dashpot, `mu_r = 0.2`): at `0.9 theta_cr` the ball comes
to rest (tail velocity 3.26e-11 m/s), at
`1.1 theta_cr` it keeps rolling away
(0.00282 m/s and growing).

### 3.7 (g) Collision energy ledger

| case | e | K_final/K_0 | dashpot | tangential | rolling | released | closure |
|---|---|---|---|---|---|---|---|
| elastic_limit | 1 | 1 | 0 | 0 | 0 | 0 | 2.34e-08 |
| oblique_frictional | 0.6 | 0.81757 | 0.111663 | 0.0143787 | 0.0481413 | 0.00823681 | 1.00e-05 |
| fully_dissipative | 0.005 | 1.07111e-05 | 0.996819 | 0 | 0 | 0.00316984 | 7.20e-08 |

* elastic limit (`e = 1`, no friction): energy conserved to
  2.34e-08 of `K_0`;
* overdamped limit (`e ~ 0`, `gamma = 10 gamma_crit`): kinetic energy left is
  1.07e-05 of `K_0`, i.e. the total
  dissipation accounts for 99.6819 %
  of the initial kinetic energy (the remainder is the elastic energy booked at the
  tension cut-off, 0.00317);
* oblique frictional impact: ledger closure 1.00e-05;
  measured rebound ratio 0.629964 vs
  input `e = 0.6`.

### 3.8 Acceptance criteria

| id | criterion | measured | threshold | verdict |
|---|---|---|---|---|
| (a) | tangential stiffness recovery | 3.64e-16 | <= 1e-2 | PASS |
| (b) | Coulomb cone, 10 cases | 1.78e-16 | <= 1e-12 | PASS |
| (c) | rolling moment, 9 cases | 2.08e-16 | <= 2e-2 | PASS |
| (d) | rotational decay R^2 | 0.999999898 | >= 0.999 | PASS |
| (e) | rolling condition (exact) | 9.45e-04 | <= 3e-2 | PASS |
| (e) | rolling condition (r_eff, plane limit) | 9.45e-04 | <= 3e-2 | PASS |
| (f) | stack stability, plane limit | 0.0191 | <= 5e-2 | PASS |
| (g) | energy ledger closure | 1.00e-05 | <= 1e-4 | PASS |
| (h) | pytest suite | 35 passed | >= 18 green | PASS |

ALL PASSED = **True**

## 4. Figures

| figure | content |
|---|---|
| `fig1_tangential_law.png` | tangential spring ramp, analytic overlay, Coulomb cap |
| `fig2_rotational_decay.png` | semi-log spin-down vs closed form |
| `fig3_roll_slip_transition.png` | slip decay and `omega/omega_0` through the transition |
| `fig4_stack_stability.png` | `theta_cr` vs radius ratio, closed form and plane limit |
| `fig5_energy_ledger.png` | fractional energy split per collision case |

## 5. Notes, limitations, traps

* **The `r_eff` rolling condition is a plane-limit convention.** The task
  statement prescribes `v_t = omega R_eff`.  Exact rigid-body kinematics of two
  free spheres give `v_t = (r_i omega_i + r_j omega_j) x n`; for a sphere rolling
  on a *fixed* sphere this is `v_t = r_j omega`, which coincides with the `r_eff`
  form only when `r_i -> inf` (relative gap `r_j/r_eff - 1 = r_j/r_i`).  We
  therefore verify **both**: the exact rolling condition (error
  9.45e-04, all three radius ratios) and the
  `r_eff` convention in its domain of validity (error
  9.45e-04 at `r_i/r_j = 2e10`, 1.9 % error at
  `r_i/r_j = 50`).  For equal spheres the two differ by a factor 2; this is
  reported explicitly rather than hidden.
* **Same for the incline criterion.** `mu_r >= tan(theta_cr)` is the
  `r_b/r_a -> 0` limit; the exact geometric factor `r_eff/r_b` is derived and
  verified instead (equal spheres: `tan(theta_cr) = mu_r/2`,
  `mu_r/tan = 2`).
* **Tension cut-off in the normal contact.** The linear spring-dashpot is
  non-adhesive; when the total normal force would become tensile the contact is
  declared lost and the stored elastic energy is released
  (`u_n_release`).  This is why the measured restitution of the oblique impact
  (0.63) is ~5 % above the
  input `e` (0.6), and why the fully
  dissipative case cannot reach exactly 100 % dissipation.
* **Coulomb projection is incremental (Cundall-Strack).** The tangential force is
  history dependent; the *stiffness* face therefore uses a monotone drive to stay
  inside the cone, while the *transition* face stays saturated during the
  sliding phase (elastic demand >> cone), which is what makes the 2/7 spin ratio
  an integral of the motion.
* **Scope.** Verified: contact law level and reduced-DOF dynamics of a single
  pair. Not verified here: multi-contact assemblies, polydispersity effects on
  the `r_eff` convention beyond the tested ratios, adhesion (JKR) coupling,
  arbitrary particle orientation (the tangential state is stored in a fixed
  frame; no contact-history rotation), rolling resistance in 3-D tumbling, and
  cluster-scale performance.  Ice-specific parameter calibration is out of scope
  for this node.
* **Stack stability is a static capacity result** (equilibrium feasibility with
  rolling-moment caps) plus a 1-DOF dynamic confirm in the plane limit; transient
  creep onset of the full two-ball stack is not simulated.

## 6. Error verdict (per quantity, explicit < 3 % answer)

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| `k_t` recovery | input stiffness | 3.64e-16 | yes |
| Coulomb cap | `mu_s F_n` | 1.78e-16 | yes |
| `M_r` | `mu_r r_eff F_n` | 2.08e-16 | yes |
| spin-down `tau` | `I/c_r` | 2.50e-04 | yes |
| transition spin ratio | 2/7 | 6.74e-04 | yes |
| transition time | impulse balance | 1.40e-13 | yes |
| rolling condition (exact) | `v = omega r_j` | 9.45e-04 | yes |
| rolling condition (`r_eff`, plane limit) | `v = omega r_eff` | 9.45e-04 | yes |
| steady-rolling force balance | `M_r / r_j` | 0.00143 | yes |
| stack `theta_cr` (plane limit) | `arctan mu_r` | 0.0191 | yes |
| stack `theta_cr` (exact factor) | closed form | 3.11e-16 | yes |
| stack `theta_cr`, equal spheres vs `arctan mu_r` | plane-limit form | 49.5082 % | **no — by design**, the plane-limit criterion does not apply at `r_b/r_a = 1` (documented in section 5) |
| energy ledger closure | `K_0` | 1.00e-05 | yes |
| elastic energy conservation | `K_0` | 2.34e-08 | yes |
| oblique restitution | input `e` | 0.0499 (5.0 %) | **no — 5.0 %**, attributable to the non-adhesive tension cut-off; reported as a model limitation, not a numerical error |
| rotational decay | `exp(-t/tau)` | `R^2` = 0.999999898 (max rel. dev. 5.00e-04) | yes |

Honest summary: all six mandated faces pass their stated thresholds; the two
non-< 3 % entries above are **convention/model** differences that are derived
analytically and reported rather than tuned away.

## 7. Audit chain

* `artifacts/ball_ball_rolling_contact/results.json` — all numbers in this
  report, generated by `scripts/verify_ball_ball_rolling_contact.py`;
  sha256 in `results.sha256`
  (`n/a`).
* `artifacts/ball_ball_rolling_contact/figures.json` — figure manifest
  (script-written PNGs).
* `tests/test_ball_ball_rolling_contact.py` — 35 tests; the numerical faces are
  recomputed in module-scoped fixtures, so the tests cannot pass on stale data.
* Regression: `tests/test_rolling_resistance.py`,
  `tests/test_contact_tangential.py`, `tests/test_clump_rolling_resistance.py`,
  `tests/test_dem3d_energy_friction_rolling.py` — 77 tests still green.
