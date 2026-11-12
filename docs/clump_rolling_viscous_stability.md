# Viscous rolling resistance on the clump path: stability window, guard, and energy audit (m16340)

**Status of the claim it closes.** `docs/clump_rolling_resistance_calibration.md` §3.4
carried an honest FAIL: the `"viscous"` rolling-resistance branch overflows at its own
default `rolling_omega_ref = 1e-3` rad/s with `dt = 2e-4` s, so the 5.74° "tilt
threshold" it produced was a numerical artefact and was withdrawn. This note replaces
the observation with a verified, machine-checkable surface: an analytic stability
window, a guard that refuses the configuration instead of overflowing, and an energy
audit that is exact to round-off. **Evidence:** `artifacts/m16340/viscous_rolling_stability.json`,
figure `artifacts/m16340/viscous_rolling_stability.png`, tests
`tests/test_clump_viscous_rolling_stability.py` (12 tests), harness
`scripts/m16340_viscous_rolling_stability.py`.

## 1. Case, geometry and parameters

| item | value |
|---|---|
| body | one-sphere clump (member radius `r = 0.05` m, `rho = 900` kg/m³) |
| mass / inertia | `m = 0.47124` kg, `I_xx = 4.71239e-4` kg m² |
| contact | `PlaneContact`, `k_n = 1e5` N/m, `restitution = 0.5`, `dt = 2e-4` s |
| rolling law | `M = -(mu_r |F_n| r / omega_ref) * omega_roll` (viscous branch of `clump_rolling.resistance_moment`) |
| coefficients | `mu_r = 0.9` (spin-down), `mu_r = 0.3 … 0.5` (incline), `omega_ref = 1e-3 … 1` rad/s |
| reference cases | (a) free spin-down `omega(t) = omega0 exp(-lambda t)`; (b) incline terminal rolling `omega* = omega_ref tan(theta)/mu_r` |

The rate constant is `lambda = mu_r |F_n| r / (omega_ref I)`; `F_n ≈ m g` at
equilibrium. The velocity-Verlet step of `ClumpBody` applies two half kicks, so each
half step multiplies the rolling rate by `1 - lambda*dt/2` and one step by
`(1 - lambda*dt/2)**2`. Hence the two bounds used everywhere below:

* `lambda*dt <= 2` — **monotone** decay (factor `>= 0`, no sign reversal);
* `lambda*dt < 4` — **bounded** (`|1 - lambda*dt/2| < 1`); beyond it the run diverges.

## 2. Method

1. Reproduce the recorded failure at the default pair with the guard disabled
   (`rolling_stability_guard=False`): the run overflows within tens of steps.
2. Bisect on `lambda*dt` with a divergence detector (`|omega| > 2` or a numerical
   exception within 400 steps) for three geometries, to test that `lambda*dt` — not
   mass, radius or stiffness — is the control parameter.
3. Measure first-order accuracy of the free decay against the exact solution inside
   the window, for both the declared (explicit) branch and the optional gain-limited
   (implicit) branch.
4. Measure the steady rolling rate on an incline at `theta = 5°, 10°`, where the
   closed form `omega* = omega_ref tan(theta)/mu_r` is exact and independent of mass
   and inertia; this separates "stable" from "still the declared law".
5. Compare the dissipated rolling work with the kinetic-energy drop, and compare the
   three quadrature rules analytically.

## 3. Results

### 3.1 Stability window and guard

| `omega_ref` [rad/s] | `dt` [s] | `lambda*dt` | guard |
|---|---|---|---|
| 1e-3 (default) | 2e-4 | **88.3** | refused |
| 1e-3 | 1e-4 | 44.1 | refused |
| 1e-1 | 2e-4 | 0.883 | ran |
| 1e-1 | 1e-3 | 4.41 | refused |
| 1 | 2e-4 | 0.0883 | ran |

The refusal message quotes the offending `lambda*dt`, both bounds, the admissible
`dt`, the `omega_ref` that would be admissible, and the gain-limited alternative:

```
ValueError: viscous rolling law is outside its stability window: lambda*dt =
dt*M_lim/(I*omega_ref) = 88.29 > 4 (divergence bound 4, monotone bound 2).
dt <= 4.531e-06 s keeps it bounded, dt <= 2.265e-06 s keeps the decay monotone; ...
```

With `rolling_stability_guard=False` the default pair still diverges
(`|omega| → 4.8e153` within ~47 steps, then `math domain error`), i.e. the recorded
honest FAIL is preserved verbatim as a regression test
(`test_guard_off_reproduces_the_recorded_blow_up`).

Bisection on the divergence boundary:

| mass [kg] | radius [m] | density [kg/m³] | measured `lambda*dt` boundary |
|---|---|---|---|
| 0.4712 | 0.05 | 900 | **4.0017** |
| 0.03016 | 0.02 | 900 | **4.0017** |
| 1.0996 | 0.05 | 2100 | **4.0017** |

A 36× mass range gives the same boundary to 0.04 % — the criterion `lambda*dt` is
confirmed as the control parameter, and the analytic bound 4 is verified.

### 3.2 Free spin-down against the exact solution (first order)

| `lambda*dt` | explicit rel. error | gain-limited rel. error |
|---|---|---|
| 0.01 | −0.263 % | +0.262 % |
| 0.05 | −1.326 % | +1.299 % |
| 0.10 | −2.553 % | +2.449 % |
| 0.25 | −6.597 % | +5.943 % |
| 0.50 | −13.99 % | +11.34 % |
| 0.90 | −25.60 % | +16.99 % |

### 3.3 Incline: steady state separates the two branches

| `theta` | `mu_r` | `omega_ref` | `dt` | `lambda*dt` | explicit | gain-limited |
|---|---|---|---|---|---|---|
| 10° | 0.3 | 0.1 | 2e-4 | 0.294 | **0.0000 %** | +14.49 % |
| 5° | 0.5 | 0.05 | 2e-4 | 0.981 | **−0.0000 %** | +48.86 % |
| 10° | 0.3 | 0.1 | 2e-3 | 2.94 | +632.1 % (in the overshoot band) | +144.9 % |
| 10° | 0.3 | 1e-3 | 2e-4 | 29.4 | refused by the guard | +1449 % |

The declared law reproduces its closed-form steady state **exactly** (the fixed point
of an explicit scheme with constant forcing is `T/gamma`, independent of `dt`), while
the gain-limited branch inflates it by the predicted `1 + lambda*dt/2`
(measured 1.145 vs predicted 1.147 at `lambda*dt = 0.294`). This is why the
gain-limited gain is **opt-in**: it buys unconditional stability at a quantified DC
cost, and it is not the default.

### 3.4 Energy audit

`ClumpContact._resistance` now records the dissipated work of the rate law from the
model's own one-step factor (`0.5 I |omega_roll|² (1 - F²)`, `F = (1 - lambda*dt/2)²`)
instead of a single-evaluation rectangle rule. Measured agreement with the
kinetic-energy drop:

| `lambda*dt` | work [J] | ΔKE [J] | rel. error |
|---|---|---|---|
| 0.01 | 2.0691792e-4 | 2.0691792e-4 | −2.3e-15 |
| 0.10 | 2.0533949e-4 | 2.0533949e-4 | 0 |
| 0.50 | 2.1203090e-4 | 2.1203090e-4 | +4.4e-16 |

Analytic error of the three quadrature rules on a pure exponential decay (exact
`W = 0.5 I omega0² (1 - e^{-2 lambda dt})`):

| `lambda*dt` | rectangle (legacy) | trapezoid | model factor (shipped) |
|---|---|---|---|
| 0.01 | −1.00 % | +0.003 % | +0.25 % |
| 0.10 | −9.67 % | +0.33 % | +2.33 % |
| 0.50 | −41.8 % | +8.20 % | +8.14 % |
| 0.88 | −63.4 % | +24.6 % | +8.90 % |

The legacy rule under-reported the dissipation by up to 63 % in the regime it was
being used in; the shipped rule is exact by construction wherever the factor is a
contraction, and the trapezoid is retained as the fallback outside it.

## 4. Error judgement (vs analytic references)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| divergence boundary `lambda*dt` | 4 (two half kicks) | +0.04 % | **yes** |
| steady rolling rate, explicit, `lambda*dt = 0.294` | `omega_ref tan(theta)/mu_r` | 0.0000 % | **yes** |
| steady rolling rate, explicit, `lambda*dt = 0.981` | same | 0.0000 % | **yes** |
| free decay, both branches, `lambda*dt <= 0.10` | `exp(-lambda t)` | ≤ 2.55 % | **yes** |
| free decay, `lambda*dt = 0.25` | same | 6.6 % | **no** — outside the accuracy envelope (see §5) |
| dissipated work vs ΔKE | `0.5 I (omega0² - omega²)` | ≤ 4.5e-16 | **yes** |
| gain-limited steady rate inflation | `1 + lambda*dt/2` | 0.2 % | **yes** |
| default `omega_ref = 1e-3`, `dt = 2e-4` | stable run | *diverges by 4 decades × 40* | **no** — refused by design |

## 5. Notes, limits and the things this does **not** fix

* **Stability is not accuracy.** The bounded window is `lambda*dt < 4`, but the 3 %
  accuracy criterion is only met for `lambda*dt <= ~0.12`. The guard refuses
  divergence, not inaccuracy: at `lambda*dt = 2.94` the incline steady state is
  +632 % wrong while the run stays finite. Users calibrating `mu_r` must stay inside
  `lambda*dt <= 0.1`; `viscous_stability_dt(...)["dt_safe"]` is the conservative value.
* **The guard checks the instantaneous normal force** at each member evaluation, so a
  configuration can pass at equilibrium and be refused later in a strongly loaded
  contact — that is intended (the load sets the stiffness of the rate law), but it
  means "it ran the first step" is not a validation.
* **The gain-limited branch is not a substitute for the declared law.** It preserves
  first-order accuracy of the transient and unconditional stability, but shifts the
  steady state by `1 + lambda*dt/2` (14 % at `lambda*dt = 0.29`, 1450 % at 29). It is
  exposed only as `ContactConfig(rolling_viscous_implicit=True)`.
* **Still WITHHELD:** the laboratory identification of `mu_r` from a tilt table or a
  repose test (see `docs/clump_rolling_resistance_calibration.md` §5). Nothing here
  validates `mu_r` against experiment; it validates the integrator that consumes it.
* **Not covered:** the `cdt` and `epsd` branches (no rate law, no such stiffness
  limit), clump–clump contacts, and the 3-D fluid coupling gaps (`GAP-DEM-G1`).
* `scripts/m5052_clump_rolling_calibration.py` still configures a `viscous` row at
  `omega_ref = 1e-3`; with the guard in place that row now stops with the actionable
  message instead of writing an artefact. The historical 5.74° value stays withdrawn;
  reproducing it requires `ContactConfig(rolling_stability_guard=False)` deliberately.

## 6. Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_clump_viscous_rolling_stability.py -q
PYTHONPATH=src python scripts/m16340_viscous_rolling_stability.py          # JSON + figure
PYTHONPATH=src python scripts/m16340_viscous_rolling_stability.py --figures-only
```

## 7. Implementation summary

| file | change |
|---|---|
| `src/tensordem/clump_rolling.py` | `viscous_damping_gain` (gain-limited, contraction-proved), `viscous_stability_dt` (monotone/divergence bounds), `inertial_moment_about`; `resistance_moment(..., inertia=)` |
| `src/tensordem/clump_dynamics.py` | `ContactConfig.rolling_stability_guard` (default on), `rolling_viscous_implicit` (default off), per-member axis inertia, refusal message, model-exact work accounting |
| `tests/test_clump_viscous_rolling_stability.py` | 12 tests: guard, window scaling, gain properties, decay order, steady state, energy |
| `scripts/m16340_viscous_rolling_stability.py`, `artifacts/m16340/` | harness, JSON, figure |