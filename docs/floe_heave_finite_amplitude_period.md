# Finite-amplitude heave period of a floating floe: nonlinear verification

**Status:** verified. The open item recorded in `docs/ice_floe_physics_validation.md`
§6 -- *"the (finite-amplitude) heave period is not verified and is expected to
differ"* -- is closed for the model's own single-degree-of-freedom heave
oscillator: the production integrator reproduces an independent nonlinear
reference to **4.18e-07** relative over the whole bounded-amplitude range.

This is *code-versus-analytic* verification of the flow-force bookkeeping, **not**
a physical validation of floe heave (no added mass, no free surface, no waves).

## 1. Case

A floating ice floe represented as one circular disk (the 2-D extruded buoyancy
model of `tensordem.buoyancy_wet`: per-particle Archimedes force on a disk of
radius `R` and extrusion thickness `t`). The body is a single heave degree of
freedom: rigid vertical translation, no rotation, no drag.

| parameter | value |
|---|---|
| disk radius `R` | 0.5 m |
| extrusion thickness `t` | 1.0 m |
| ice density `rho_i` | 917.0 kg/m^3 |
| water density `rho_w` | 1025.0 kg/m^3 |
| gravity `g` | 9.81 m/s^2 |
| floe mass `m = rho_i pi R^2 t` | 720.210 kg |
| equilibrium centre height `h_eq / R` | -0.675534 (frozen in `rho_i/rho_w` draft) |
| linearised stiffness `K = rho_w g t chord(h_eq)` | 7414.024 N/m |
| linear period `T_0 = 2 pi sqrt(m/K)` | 1.958315 s |
| bounded-well (escape) amplitude `A_esc / R` | 0.280746 |
| production time step `dt` | 9.792e-04 s (`T_0/2000`) |

**Why the linear anchor is not enough.** The hydrostatic restoring force of a
disk is `F_b(h) = rho_w g t A(h)` with `A(h)` the submerged circular segment.
Its derivative -- the waterplane chord `2 sqrt(R^2 - h^2)` -- changes sign of
curvature at the waterline, so the oscillator is genuinely nonlinear. The well
is also **bounded**: below `h = -R` the disk is fully submerged (constant
upward force) and above `h = +R` it is dry (free fall), so only amplitudes
`A < A_esc = 0.281 R` give a periodic orbit.

## 2. Method

1. **Independent geometry.** `submerged_disk_area` uses the circular-segment
   closed form; `submerged_disk_area_quadrature` re-derives the same area from
   Gauss-Legendre column quadrature of the chord, splitting the `phi` range at
   the chord line where the integrand has a C0 kink. The two agree to machine
   precision (worst 1.9e-12 relative at `|h| -> R`) and both reproduce the
   engine's own `acos` kernel, so the reference does not inherit engine bugs.
2. **Exact potential.** `I(h) = int A dh` has a closed-form antiderivative, so
   `U(h) = m g (h - h_eq) - rho_w g t [I(h) - I(h_eq)]` is exact (checked by
   finite differencing `dI/dh = A(h)`).
3. **Nonlinear period by energy integral.**
   `T(A) = 2 int_{y_lo}^{y_hi} dy sqrt(m / (2 (E - U)))` with `E = U(h_eq + A)`,
   the lower turning point from a bisection on `U`, and the endpoint `1/sqrt`
   singularity removed by `y = y_m + a sin(theta)` (Gauss-Legendre, 400 nodes).
4. **Unrelated cross-check.** RK4 integration of the same ODE at `T_0/20000`
   with crossing-based period extraction; agreement with the quadrature is
   better than 1e-5 relative up to the largest amplitude.
5. **Production model.** `tensordem.floe_buoyancy.RigidHeaveFloe` (velocity
   Verlet on the engine's per-particle buoyancy kernel) is released from rest at
   `h_eq + A`; the period is the mean of consecutive **upward crossings of the
   equilibrium height**, i.e. one full orbit -- the same definition as the
   reference. Equilibrium is bisected on the engine's own force kernel, and the
   energy ledger is audited on a thinned sample (relative drift
   < 4.0e-08).

## 3. Results

| `A/R` | `A/A_esc` | `T` model [s] | `T` reference [s] | rel. error | `T_0` [s] | `T_0` bias | `T/T_0` model | energy drift |
|---|---|---|---|---|---|---|---|---|
| 0.020 | 0.071 | 1.958608 | 1.958609 | -4.18e-07 | 1.958315 | -0.015% | 1.000150 | 2.6e-10 |
| 0.020 | 0.070 | 1.958598 | 1.958599 | -4.10e-07 | 1.958315 | -0.015% | 1.000145 | 2.5e-10 |
| 0.084 | 0.300 | 1.963779 | 1.963780 | -4.10e-07 | 1.958315 | -0.278% | 1.002790 | 4.8e-09 |
| 0.168 | 0.600 | 1.982626 | 1.982627 | -4.07e-07 | 1.958315 | -1.226% | 1.012414 | 2.0e-08 |
| 0.239 | 0.850 | 2.015429 | 2.015430 | -4.06e-07 | 1.958315 | -2.834% | 1.029165 | 4.0e-08 |

* Peak relative error of the production model against the nonlinear reference:
  **4.18e-07** (gate 5e-3, and the task-level 3 % gate is met by four orders
  of magnitude).
* The amplitude correction is real and *hardening*: `T/T_0 = 1.0292`
  at `A = 0.85 A_esc`, so a linear anchor quoted there
  is biased by **-2.83%** -- inside 3 % at the top
  of the sweep but already -1.23% at
  `0.60 A_esc`. The `T_0` anchor is thus usable only
  for `A/R` below roughly 0.1.
* Timestep independence: doubling `dt` changes the period by
  1.2e-06 relative.

**Negative control (anti-tautology).** Replacing the buoyancy force by the
linear spring `K y` makes the period amplitude-flat
(`growth = 1.000000`, i.e. 2.2e-16
relative change over `0.07 A_esc -> 0.85 A_esc`), whereas the verified model grows
by `1.0290`. The check therefore *can* detect a
missing or mis-scaled nonlinear term; it is not an identity.

## 4. Lattice floe (production geometry)

For the 7-disk hex-lattice floe (`a = 0.125 m`) the same
harness gives `T = 1.591127 s` at `A = 0.02 R` and
`T = 1.835745 s` at `A = 0.25 R`, i.e. a growth of
`1.1537` -- the same hardening sign and order as the continuum
disk (1.0330).

## 5. Files

* `src/tensordem/floe_heave_amplitude.py` -- reference + model driver;
* `tests/test_floe_heave_finite_amplitude.py` -- gates;
* `scripts/verify_floe_heave_finite_amplitude.py` -- this campaign;
* `artifacts/floe_heave_finite_amplitude/report.json`, `artifacts/floe_heave_finite_amplitude/amplitude_sweep.csv`, `artifacts/floe_heave_finite_amplitude/heave_finite_amplitude.png`.

## 6. Notes, limitations, open items

* **Not a physical validation.** The reference is the model's own 1-DOF
  oscillator. Added mass (which would lengthen the physical period by
  `sqrt(1 + Ca)`), free-surface deformation, wave radiation damping and rotation
  are all absent, so no statement here is a prediction of a real floe's heave
  period.
* **Small-amplitude floor.** Below `A ~ 1e-3 R` the energy integral loses
  accuracy: `E = O(A^2)` while `U` is a difference of two `O(m g A)` terms, and
  the Gauss-Legendre nodes nearest a turning point see `E - U` at the
  cancellation floor. The floor is documented rather than hidden; the linear
  limit is checked at `A = 1e-3 R` instead of `1e-4 R`.
* **Lattice growth law not quantitatively verified.** The waterline of a 7-disk
  cluster cuts through particles, so its well geometry scales with the cluster
  half-height (1.5 a), not with `R`; the measured lattice growth
  (`1.1537`) therefore cannot be compared one-to-one with the
  disk value. Only the sign and order of magnitude are claimed. A lattice-specific
  potential (per-particle analytic potentials already exist as
  `floe_buoyancy.discrete_buoyancy_potential`) is the natural follow-up.
* **Energy ledger cost.** `RigidHeaveFloe.total_energy` evaluates a 2048-slice
  Simpson integral per particle, so the ledger is audited on a thinned sample
  (~200 points) instead of every step.
* **Runtime.** Full campaign ~1 min on one CPU thread; the pytest module ~1 min.
