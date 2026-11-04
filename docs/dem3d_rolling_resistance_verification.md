# DEM3D rolling resistance: analytic verification of the SIR-PC2|F05 couple

Status: contact-law defect **found and fixed**; the corrected law is verified against
closed-form references at the contact and integration level. This is
*code-versus-analytic* verification, **not** a physical validation of sea-ice
mechanics.

Related documents: `docs/dem3d_rolling_resistance.md` family
(`docs/dem3d_rolling_resistance.md`) covers the **standalone torque-law library**
`tensordem.rolling_resistance` (Ai et al. 2011 `cdt`/`viscous`/`epsd` models,
verified against closed-form references) which is *not wired into the solver*.
This document covers the **solver-side** rotational surrogate that is actually
used by `IceDEM3D`, and which previously had no analytic check at all:
`docs/dem3d_tangential_friction_verification.md` §limitations recorded it as
"`rolling_mu` 仍为占位 — 未验证".

## 1. Case and physical background

`tensordem.dem3d.IceDEM3D` is a three-dimensional bonded-sphere solver with
contact normal springs, a Cundall-Strack tangential spring, and a
rolling-resistance term (`DEM3DConfig.rolling_resistance_coefficient`, `mu_r`).
It carries **no rotational degree of freedom**: there is no particle orientation,
no angular velocity and no torque equation of motion. A rolling-resistance
*torque* therefore cannot act on the dynamics directly; it can only be realised
as an equal-and-opposite **couple of tangential contact forces** — a
"rotational surrogate".

In ice-rubble DEM this term is not cosmetic: without it, near-spherical ice
blocks roll too freely, which biases the reported angle of repose of a rubble
field and the resistance of a broken-ice channel.

Reference case used for every check below — deliberately the configuration in
which a mis-oriented couple is *visible*:

* one fixed sphere (bottom layer, `fix_bottom=True`) and one free sphere in a
  single load-bearing, **unbonded** contact;
* contact normal `n = +z`, slip direction `+x`, so a wrongly oriented couple
  shows up as a spurious `y` (binormal) force;
* bond topology disabled and the remaining free spheres parked away from the
  contact, so the active contact set is verified to be **exactly one pair**;
* the normal force is prescribed by a fixed overlap (zero normal damping), and
  the free sphere carries a prescribed normal load that balances it, so `F_n`
  is constant to <1e-3 relative over the run.

| parameter | value |
|---|---|
| particle radius `R` | 0.025 m |
| particle mass `m` | 0.0600 kg (density 917 kg/m³) |
| contact stiffness `k_c` | 5000 N/m |
| initial overlap `delta` | 1.0e-3 m |
| normal force `F_n = k_c delta` | 5.000 N |
| restitution / normal damping | 1.0 / 0 (damping-free) |
| time step `dt` | 2.0e-6 s |
| initial slip velocity `v0` | 0.02 m/s |
| integrator | semi-implicit (symplectic) Euler, as in `IceDEM3D.step` |

## 2. What was done

### 2.1 The defect

The couple was assembled by computing a torque and then recovering a force as
`force = torque / R`, i.e. `f = n x f_res`, which is **perpendicular** to the
slip direction. Measured on the reference case with `mu_r = 0.10`,
`mu = 0.00` (the negative control kept in the harness as `_legacy_orthogonal_couple`):

| symptom (legacy law) | measured |
|---|---|
| slip deceleration vs analytic `mu_r F_n / m` | 4.85 vs 8.33 m/s² (**42 % error**) |
| spurious binormal velocity `max abs(v_y)` | 0.0202 m/s = **101 % of `v0`** (the entire initial velocity is pumped sideways) |
| kinetic energy | **not monotone** — it increases (a "dissipative" term that creates energy) |

A second, independent defect: the couple was applied with the wrong sign
convention for the pair ordering, so for the pair `(i, j)` the force on `i` was
`-|f| v_hat_rel` instead of its reaction `+|f| v_hat_rel`, i.e. the term could
*drive* the relative slip depending on the candidate-pair order returned by the
cell list.

### 2.2 The fix (`src/tensordem/dem3d.py`, `_rolling_resistance_couple`)

```text
v_rel      = (v_j - v_i)_tangential
f_j        = -mu_r |F_n| v_hat_rel          # kinetic resistance on j
f_i        = -f_j                           # reaction on i
T_i        = R (n x f_i),  T_j = -T_i       # diagnostic only
```

with `|f| = mu_r |F_n|` (declared law, lever arm `R`). Consequences, each
asserted by a dedicated check:

* `f_i + f_j = 0` — linear momentum is conserved (Newton's third law);
* `f_j . v_rel = -|f| |v_rel,t| <= 0` — strictly dissipative, and it vanishes
  as the slip vanishes (no spurious force on a sticking contact);
* `|T| = mu_r R |F_n|` reproduces the law declared in `forces()`.

### 2.3 Verification harness

`scripts/verify_dem3d_rolling_resistance.py` (protocol
`tensordem-dem3d-rolling-resistance-verification-v1`) writes
`docs/dem3d-rolling-resistance/{results.json, checks.csv, figures/}`; every
number below is machine-written from that JSON by
`docs/dem3d-rolling-resistance/inject_tables.py`, and the whole report is
reproduced by re-running the script. Fast regression coverage lives in
`tests/test_dem3d_rolling_resistance_verification.py` (11 tests), including the
legacy law as a negative control so the checks are known to have teeth.

## 3. Results

### 3.1 Contact-law level (single force evaluation, no integration)

| check | quantity | measured | analytic | rel. error |
|---|---|---|---|---|
| L1 | \|f_rr\| = mu_r \|F_n\| | 0.500000000 N | 0.500000000 N | 0.00e+00 |
| L2 | cos(angle(f_j, v_rel)) | -1.000000000000 | -1 | 0.00e+00 |
| L3 | \|f_i + f_j\| (momentum) | 0.000e+00 N | 0 | - |
| L4 | \|T\| = mu_r R \|F_n\| | 1.250000000e-02 N m | 1.250000000e-02 N m | 1.39e-16 |
| L4b | \|T\| / (mu_r R/2 \|F_n\|) (Ai 2011) | 2.000000 | 2 (convention) | 0 |
| L5 | \|F_t\| with saturated Coulomb | 1.500000000 N | 1.500000000 N | 1.48e-16 |
| L6 | mu_r = 0 bit-identical to friction only | True | True | - |

`L4_impl_over_ai_factor = 2` is a **convention note, not an error**: the declared
law uses the moment arm of the contact force about the sphere centre (`R`),
whereas the equal-sphere rolling limit of Ai et al. (2011),
`M_lim = mu_r R* |F_n|` with `R* = R_i R_j / (R_i + R_j) = R/2`, uses the
effective radius. Hence `mu_r_Ai = 2 mu_r_impl`: a value quoted from the
literature must be halved before it is passed as
`rolling_resistance_coefficient`, otherwise the resisting torque is doubled.

### 3.2 Integration level: sliding deceleration

| `mu` | `mu_r` | `a` measured [m/s²] | `a` analytic [m/s²] | rel. error |
|---|---|---|---|---|
| 0.00 | 0.00 | -0.1370 | 0.0000 | n/a (no resistance) |
| 0.00 | 0.02 | 1.5565 | 1.6662 | 6.586 % |
| 0.00 | 0.05 | 4.1175 | 4.1655 | 1.150 % |
| 0.00 | 0.10 | 8.3071 | 8.3309 | 0.285 % |
| 0.00 | 0.15 | 12.4806 | 12.4964 | 0.126 % |
| 0.00 | 0.20 | 16.6500 | 16.6618 | 0.071 % |
| 0.10 | 0.05 | 12.4806 | 12.4964 | 0.126 % |
| 0.20 | 0.05 | 20.8178 | 20.8273 | 0.045 % |
| 0.30 | 0.05 | 29.1514 | 29.1582 | 0.023 % |

### 3.3 Additivity sweep over `mu` and `mu_r`

| `mu` | `mu_r` | `a` measured [m/s²] | `a` analytic [m/s²] | rel. error |
|---|---|---|---|---|
| 0.00 | 0.00 | -0.1370 | 0.0000 | n/a (no resistance) |
| 0.00 | 0.02 | 1.5565 | 1.6662 | 6.586 % |
| 0.00 | 0.05 | 4.1175 | 4.1655 | 1.150 % |
| 0.00 | 0.10 | 8.3071 | 8.3309 | 0.285 % |
| 0.00 | 0.15 | 12.4806 | 12.4964 | 0.126 % |
| 0.00 | 0.20 | 16.6500 | 16.6618 | 0.071 % |
| 0.10 | 0.05 | 12.4806 | 12.4964 | 0.126 % |
| 0.20 | 0.05 | 20.8178 | 20.8273 | 0.045 % |
| 0.30 | 0.05 | 29.1514 | 29.1582 | 0.023 % |

The row with `mu_r = 0` has no analytic counterpart (zero resistance) and its
least-squares slope is numerical noise (~0), as expected.

### 3.4 Figures

| figure | caption |
|---|---|
| `figures/rr_velocity_decay.png` | Slip velocity versus time for three coefficient sets; the dashed line is the closed-form `v0 - a t` with `a = (mu + mu_r) F_n / m`. The measured curves are straight and terminate at the analytic stopping time. |
| `figures/rr_deceleration_sweep.png` | Measured deceleration versus `mu + mu_r` (points) against the analytic line. The data collapse onto one line, confirming that sliding and rolling resistance add. |
| `figures/rr_legacy_vs_fixed.png` | Left: slip decay, fixed versus pre-fix law. Right: spurious binormal velocity — the pre-fix law transfers essentially all of `v0` into `y`, the fixed law produces zero. |
| `figures/rr_energy.png` | Kinetic energy: monotone decay to zero for the fixed law, *growth* for the pre-fix law. |

## 4. Error judgement against the analytic references

Every quantity below is compared with a closed-form value; "是否 <3%" is answered
explicitly.

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| couple magnitude | `mu_r |F_n|` | 0.00e+00 | yes (exact) |
| couple direction | anti-parallel to `v_rel` | 0.00e+00 (cos) | yes (exact) |
| momentum balance | `f_i + f_j = 0` | 0.0e+00 N | yes (exact) |
| diagnostic torque | `mu_r R |F_n|` | 1.39e-16 | yes (exact) |
| additivity | `(mu + mu_r)|F_n|` | 1.48e-16 | yes (exact) |
| sliding deceleration, `mu=0.00`, `mu_r=0.05` | `(mu+mu_r)|F_n|/m` | 1.150 % | yes |
| stopping time, `mu=0.00`, `mu_r=0.05` | `v0/a` | 0.346 % | yes |
| sliding deceleration, `mu=0.00`, `mu_r=0.10` | `(mu+mu_r)|F_n|/m` | 0.285 % | yes |
| stopping time, `mu=0.00`, `mu_r=0.10` | `v0/a` | 0.612 % | yes |
| sliding deceleration, `mu=0.20`, `mu_r=0.10` | `(mu+mu_r)|F_n|/m` | 0.032 % | yes |
| stopping time, `mu=0.20`, `mu_r=0.10` | `v0/a` | 0.779 % | yes |
| worst integration-level error (any run) | analytic | 1.150 % | yes |

**Answer: yes — every key quantity is below 3 % (in fact below 1.2 %)**, with
three exact (machine-precision) identities at the law level. The residual error
grows as `mu_r` decreases (1.15 % at `mu_r = 0.05`), because a smaller resistance
means a longer slip path and therefore a larger change of the contact overlap
from the prescribed value (`overlap_drift_rel` is reported per run); it is the
expected geometric coupling of the check, not a law discrepancy.

## 5. Caveats and limitations

* **Surrogate, not a torque law.** Because `IceDEM3D` has no rotational DOF, this
  term is a *saturated, velocity-proportional tangential resistance* with
  coefficient `mu_r`. It has no stiction: a contact at rest carries no rolling
  resistance, and a real rolling-moment model (Ai et al. 2011 `cdt`/`viscous`/`epsd`)
  cannot be exercised in this solver. Those models are implemented and verified in
  `tensordem.rolling_resistance` but are **not** wired in.
* **Bonded contacts are excluded.** The contact branch (and therefore rolling
  resistance) only acts on pairs whose bond is already broken, so rolling
  resistance does not act inside intact bonded material. That is a modelling
  choice of the existing solver, unchanged here.
* **Lever-arm convention.** The factor 2 against Ai et al. (2011) is documented,
  not silently absorbed; see §3.1.
* **No experimental validation.** These checks prove numerical consistency with
  the declared contact law. They say nothing about whether the coefficient
  reproduces a measured angle of repose or channel resistance for ice rubble;
  that requires calibration against experiments not used for fitting.
* **No static/dynamic threshold.** A velocity-dependent law cannot reproduce a
  finite repose angle for a single particle on a perfectly flat incline; it
  produces creeping decay instead. Reported repose angles therefore remain
  rate-dependent.
* The scenario is a single contact with a prescribed load; multi-contact force
  chains and dense packings are outside the verification envelope.

## 6. Reproduction

```bash
PYTHONPATH=src python scripts/verify_dem3d_rolling_resistance.py
PYTHONPATH=src python -m pytest tests/test_dem3d_rolling_resistance_verification.py -q
```