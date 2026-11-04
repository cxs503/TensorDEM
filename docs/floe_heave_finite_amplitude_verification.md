# Finite-amplitude (nonlinear) heave of the 2-D floe: exact reference and verification

Status: **closes the open item recorded in `docs/ice_floe_physics_validation.md` §6**
("the period anchor uses `2e-3 R`; the nonlinear (finite-amplitude) heave period is not
verified and is expected to differ").
Evidence grade for "floe heave period vs analytic reference": **REAL** — an independently
coded closed-form reference, a shipped-solver measurement, converging quadrature, and
negative controls that provably break if the code under test degrades.

Module: `src/tensordem/floe_heave_nonlinear.py`.
Tests: `tests/test_floe_heave_nonlinear.py` (23 tests).
Benchmark: `benchmarks/floe_heave_finite_amplitude.py` →
`artifacts/floe_heave_nonlinear/{report.json,figure.png,tables.md}`.

## 1. What was unverified, and why it matters

`floe_buoyancy` verifies the *linear* heave anchors of the 2-D extruded-disk buoyancy
model: equilibrium draught (`f_sub = rho_i/rho_w`), waterplane stiffness
`K = rho_w g t sum_i c(h_i)`, period `T0 = 2 pi sqrt(m/K)`, and the damping ratio for
linear wet drag — all at an amplitude of `2e-3 R`. That anchor cannot see the nonlinearity
of Archimedes restoring: `A(h)`, the submerged area of a disk whose centre stands `h` above
the free surface, is *not* linear in `h` (only its slope at the waterline, the chord
`c(h) = 2 sqrt(R^2-h^2)`, enters `K`). A floe that heaves by a fraction of its own
thickness therefore runs on a different period than `T0`, and nothing in the tree measured
by how much or whether the shipped integrator reproduces it.

This is not academic bookkeeping: slush/rubble-scale heave of an ice floe model is quoted
in ice-load screening, and the model's own amplitude dependence must be known before a
period is compared with anything physical.

## 2. Model and parameters

Rigid-heave oscillator of a discrete floe made of `N` tangent extruded disks (hex lattice,
`hex_lattice_floe`), the same per-particle Archimedes kernels the 3-D/2-D DEM uses:

```
m y'' = rho_w g t * sum_i A(y + o_i - water_level) - m g,     m = rho_i pi R^2 t N
```

| parameter | value |
|---|---|
| particle radius `R` | 0.100 m |
| extrusion thickness `t` | 1.0 m |
| disk density `rho_i` | 917 kg/m^3 |
| water density `rho_w` | 1025 kg/m^3 |
| gravity `g` | 9.81 m/s^2 |
| particles `N` | 19 (hex lattice, tangent, non-overlapping) |
| floe mass `m` | 547.3597 kg |
| equilibrium height `y_eq` | -0.3198110 m (centroid below the surface, as `rho_i < rho_w`) |
| linear stiffness `k1` | 5815.8076 N/m |
| quadratic coefficient `k2` | -8323.70 N/m^2 |
| cubic coefficient `k3` | -112252.27 N/m^3 |
| linear period `T0` | 1.9275749 s |

`k2 != 0` makes the well **asymmetric**, so the symmetric-oscillator shortcut
`T = 4 int_0^{x_max} dx/sqrt(...)` is invalid here and both turning points must be found.

## 3. Method: the independent reference

Geometry is built from the chord antiderivative rather than from the production
`(theta - sin theta)` fraction:

```
P(u) = (u sqrt(R^2-u^2) + R^2 asin(u/R)) / 2
A(h) = pi R^2/2 - 2 P(h),                    |h| <= R
I_A(h) = int_{-R}^{h} A ds = 2[Q(h) + R(P(h)+pi R^2/4)] + 2(h+R)(pi R^2/4 - P(h)),
Q(u) = -(R^2-u^2)^{3/2}/3
```

with the branch behaviour that matters for a deeply submerged floe:
`I_A(h) = pi R^3` for `h >= R` (dry, `A = 0`) but `I_A(h) = pi R^2 (h + R)` for `h <= -R`
(fully wet, `A = pi R^2`). Saturating both ends — the obvious mistake — flattens the
potential below full submergence and is caught by the derivative test
(`test_saturated_antiderivative_control_fails`).

The exact period of the conservative oscillator released from rest at `x_hi = A` follows
from energy. The sweep from one turning point to the other is a **half** period, so

```
T(E) = sqrt(2 m) * int_{x_lo}^{x_hi} dx / sqrt(E - U(x)),   E = U(x_hi),  U(x_lo) = E
```

(dropping the factor 2 is a factor-of-two error — see §5). The integral is evaluated by
Gauss-Legendre quadrature on the analytic integrand; the two panels that touch a turning
point use `x = x_turn -/+ (1-u)^2`, which removes the `1/sqrt` endpoint behaviour, and the
panels are split at every height where a particle enters or leaves the water (the potential
is only `C^1` there).

**Numerical point that decides whether the check is meaningful.** `E - U(x)` written as a
difference of aggregate potentials loses about `1e-13 J` to cancellation (the potentials are
`O(500 J)`), which is larger than the true gap near a turning point; the first version of
this module produced a 24 % period error from exactly that. The implementation therefore
never subtracts potentials:

```
E - U(x) = (x_hi - x) * [ m g - rho_w g t * sum_i mean_A(h_i over [x, x_hi]) ]
```

with `mean_A` from the midpoint value plus the `A'' d^2/24` curvature term for short spans
and from `(I_A(h_hi) - I_A(h_lo))/d` otherwise. The bracket stays accurate at `~1e-9 N`
down to spans of `1e-11 m`, which is what makes the `1e-4 R` linear-limit check possible
(agreement there is `7.7e-8` relative).

Two independent small-amplitude cross-checks on the reference itself:

* `A -> 0`: `T(A)/T0 - 1 = 8.1e-6` at `A = 0.01 R`, `7.7e-8` at `A = 1e-4 R`;
* Lindstedt-Poincaré for `F = -k1 x - k2 x^2 - k3 x^3`,
  `T/T0 = 1 - (3k3/(8k1) - 5k2^2/(12k1^2)) A^2`: matches the quadrature to `1.8e-10` at
  `A = 0.01 R` and departs as `O(A^4)` above it (`1.5e-4` at `0.5 R`), as it should.

## 4. Results

Measured side: the shipped `RigidHeaveFloe` (velocity Verlet) driving the shipped
`compute_buoyancy_force`, `dt = 1e-3 s`, 4 cycles, first cycle discarded; period from
linear-interpolated upward crossings of `y_eq` averaged over the remaining cycles.

| A/R | amplitude [m] | T exact [s] | T measured [s] | rel. error meas-vs-exact | linear T [s] | linear error |
|---|---|---|---|---|---|---|
| 0.05 | 0.00500 | 1.927963 | 1.927962 | -4.42e-07 | 1.927575 | -2.01e-04 |
| 0.10 | 0.01000 | 1.929126 | 1.929125 | -4.42e-07 | 1.927575 | -8.04e-04 |
| 0.20 | 0.02000 | 1.933793 | 1.933792 | -4.40e-07 | 1.927575 | -3.22e-03 |
| 0.30 | 0.03000 | 1.941713 | 1.941712 | -4.36e-07 | 1.927575 | -7.28e-03 |
| 0.50 | 0.05000 | 1.966280 | 1.966279 | -4.26e-07 | 1.927575 | -1.97e-02 |

| dt [s] | T measured [s] | rel. error vs exact |
|---|---|---|
| 0.004 | 1.933780 | -7.04e-06 |
| 0.002 | 1.933790 | -1.76e-06 |
| 0.001 | 1.933792 | -4.40e-07 |

Cycle-to-cycle spread of the measured periods is `<= 1e-9 s` (`<= 5e-10` relative), i.e.
the measured oscillation is genuinely periodic rather than a noisy crossing sequence.

Time-step sensitivity is second order, as velocity Verlet must be: halving `dt` cuts the
period error by a factor `4.0` (table above). At the working step `dt = 1e-3 s` the
integrator contribution (`4.4e-7`) is already below the quadrature's own
`~1e-8`-level uncertainty scaled by the residual `dt` error; the reported `4.3e-7 ... 4.4e-7`
is therefore the agreement floor of the two independent computations, not a tuned number.

Figure (`artifacts/floe_heave_nonlinear/figure.png`), four panels:

1. `T(A)/T0` — measured, exact quadrature, Lindstedt-Poincaré series and the constant
   linear-theory level: the period **grows** with amplitude (softening: at large upward
   excursion the waterplane area shrinks faster than the disk deepens).
2. `|measured/exact - 1|` against `|linear/exact - 1|` on a log axis: the solver sits at
   `~4e-7` while the linear anchor drifts to `2e-2` at `A = 0.5 R`.
3. Simulated `y(t)` at `A = 0.5 R` with the exact turning points `x_lo, x_hi` drawn: the
   measured envelope touches both analytic bounds and shows no drift.
4. Per-particle submerged fraction at the lower turning point, equilibrium and upper
   turning point: the floe partially emerges (`f_sub -> 0` for the top particles), which is
   precisely the regime the linear anchor never entered.

## 5. Pitfalls recorded while doing this

* **Half period vs full period.** `sqrt(m/2)` gives half the true period; the correct
  prefactor is `sqrt(2m)`. Caught immediately by the linear-limit test (`T/T0 = 0.5`
  exactly), which is why that test exists.
* **Saturated antiderivative.** `I_A` must grow linearly below `-R`; clamping it there
  produced "turning point not bracketed" and would have silently biased every period. The
  finite-difference derivative test covers `[-1.6R, 1.6R]` for that reason.
* **Cancellation near turning points** (§3): the mean-force form is not cosmetic, it is
  what makes the small-amplitude limit checkable.
* **Asymmetric well.** `k2 != 0` for any floe whose particles straddle the waterline
  unevenly; `x_lo/x_hi` differ by up to 4× at large amplitude.
* **Amplitude definition.** "Amplitude" here means the release displacement from
  equilibrium with zero velocity, not half the peak-to-peak excursion (the well is
  asymmetric, so those differ).

## 6. Error judgement (explicit)

The reference is the exact quadrature above; the verified quantity is the heave period of
the shipped model.

| amplitude `A/R` | measured vs exact | < 3 % ? | linear anchor vs exact | < 3 % ? |
|---|---|---|---|---|
| 0.05 | 4.4e-7 | yes | 2.0e-4 | yes |
| 0.10 | 4.4e-7 | yes | 8.0e-4 | yes |
| 0.20 | 4.4e-7 | yes | 3.2e-3 | yes |
| 0.30 | 4.4e-7 | yes | 7.3e-3 | yes |
| 0.50 | 4.3e-7 | yes | 2.0e-2 | yes (marginal) |

Conclusion: the shipped 2-D floe heave dynamics reproduces the exact nonlinear period to
`4.4e-7` relative over `A/R in [0.05, 0.5]` — far inside the 3 % acceptance band — and the
integrator error is second order in `dt`. The *old* linear anchor is still within 3 % at
`A = 0.5 R` but has consumed two thirds of that budget (`2.0e-2` vs `3e-2`) purely from the
amplitude it never mentioned, so quoting `T0` as "the" floe heave period is only safe for
`A <~ 0.6 R`; beyond that it must be replaced by the nonlinear reference.

## 7. Scope and limitations (not verified here)

* **Model scope inherited.** Per-particle Archimedes on extruded disks: no added mass, no
  free-surface deformation, no wave damping, no rotation, no horizontal drift, no contact.
  Added mass alone would lengthen the physical period by `sqrt(1+Ca)`; the comparison is
  therefore a check of the *model's* amplitude dependence, not of a real floe's heave
  period, and no number here should be quoted as a physical prediction.
* **2-D only.** `HeaveOscillator.from_config` rejects `cfg.model == "3d"`; the spherical-cap
  analogue (submerged volume `pi h^2 (3R-h)/3`, waterplane area `pi (R^2-h^2)`) is a
  straightforward but separate increment.
* **Unlumped lattice.** The floe is a rigid kinematic lattice: internal bonds, fracture and
  particle rotation are irrelevant to this anchor by construction.
* **Single floe size.** Only `N = 19`, `R = 0.1 m`; the lattice-resolution dependence of
  `k1` is covered by the existing `discrete_stiffness` slab study, not here.
* **No damping.** Wet drag is forced to zero for the measurement, so the anchor says nothing
  about the linear-drag damping ratio (already covered by the small-amplitude anchors).