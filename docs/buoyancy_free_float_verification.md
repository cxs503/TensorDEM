# 2-D buoyancy / wet-drag force law: analytic verification campaign

Protocol `tensordem-buoyancy-free-float-verification-v1`.  Report artifact: `artifacts/buoyancy_free_float/verification.json`
(regenerate with the command in section 8).  Figure:
`docs/figures/buoyancy_free_float_verification.png`.

## 1. What is verified, and what is not

`tensordem.buoyancy_wet` supplies a buoyancy force built from the submerged area of a
circular cross-section, plus a linear wet drag weighted by that submerged fraction.
The existing unit tests (`tests/test_buoyancy_wet.py`) check force *values* at
hand-picked positions.  They cannot detect a wrong gradient, a wrong equilibrium
under time integration, or a drag law that is right at `t = 0` but wrong in its decay.
This campaign supplies those missing analytic references.

Verified: the geometry, the restoring gradient, the Archimedes draft reached by free
float, the damped heave frequency / envelope / waveform against the linearised
oscillator, the pure-drag exponential decay, timestep convergence of every
measurement, and the O(amplitude) character of the residual.

**Not** verified: physical qualification of ice-floe motion.  The model is 2-D (per
unit thickness) and has no hydrodynamic **added mass**, no free-surface waves, no
particle rotation and no quadratic form drag.  The reported `omega_n` is therefore a
property of the code model and must not be compared with a measured floe heave
period; only code-versus-analytic consistency is claimed.

## 2. Fixture

Single disk, no contacts, no other particles, SI units.

| parameter | value |
|---|---:|
| Disk radius `R` | 0.5 m |
| Extrusion thickness `t` | 1.0 m |
| Ice density $\rho_i$ | 917.0 kg/m3 |
| Water density $\rho_w$ | 1025.0 kg/m3 |
| Gravity `g` | 9.81 m/s2 |
| Particle mass $m=\rho_i\pi R^2 t$ | 720.210116 kg |
| Wet drag $C_wet$ | 516.4 N s/m |
| Buoyancy stiffness $k(h_eq)$ | 7414.023684 N/m |
| Natural frequency $\omega_n=\sqrt{k/m}$ | 3.208465516 rad/s |
| Damping ratio $\zeta=C_wet f_eq/(2m\omega_n)$ | 0.099964343 |
| Damped frequency $\omega_d=\omega_n\sqrt{1-\zeta^2}$ | 3.192394377 rad/s |
| Heave period $2\pi/\omega_d$ | 1.968173 s |
| Target submerged fraction $f_eq=\rho_i/\rho_w$ | 0.894634146341 |
| Analytic equilibrium centre height $h_eq$ | -0.337767114588 m |
| Analytic draft $R-h_eq$ | 0.837767114588 m |
| Excitation amplitude `A` | 0.005 m |

The equilibrium condition follows from the force law itself:

    rho_w * g * pi * R^2 * t * f_sub(h_eq) = rho_i * pi * R^2 * t * g
    =>  f_sub(h_eq) = rho_i / rho_w = 0.894634146341

with the circular-segment area fraction `f = (theta - sin theta)/(2 pi)`,
`theta = 2 acos(h/R)`.  Its derivative is the water-plane chord width,

    d f_sub / d h = -2 sqrt(R^2 - h^2) / (pi R^2)
    d F_b / d h   = -rho_w * g * t * 2 sqrt(R^2 - h^2)  =: -k(h)

so the restoring stiffness at the draft is `k = 7414.023684 N/m` and the
linearised heave about the draft is a damped oscillator with
`omega_n = 3.208466 rad/s`, `zeta = 0.099964`,
`omega_d = 3.192394 rad/s`.

## 3. How it was done

1. **The force law is not re-implemented.**  Every right-hand side is evaluated by
   calling `compute_buoyancy_force` and `compute_wet_drag_force` from
   `tensordem.buoyancy_wet` on the instantaneous state; weight is `-m g`.  The
   verification module only adds analytic references, the integrator and the
   measurement logic.
2. **Independent reference geometry.**  `submerged_fraction_quadrature` integrates
   the chord width after the substitution `y = h + R sin(phi)`, giving
   `dA = 2 R^2 cos^2(phi) d phi` with `phi in [-pi/2, asin(-h/R)]`.  This route never
   evaluates the closed-form segment formula, so a transcription error in
   `(theta - sin theta)/(2 pi)` cannot survive.
3. **Equilibrium reference.**  `h_eq` is found by bisection on the *quadrature*, so
   the reference geometry — not the code — supplies the draft; the code then has to
   reproduce it.
4. **Dynamics.**  Fixed-step classical RK4 (`Delta t` = 0.01 s for the accuracy
   runs) on the 1-DOF motion, released from rest at `h_eq + A`.
5. **Machine measurement, no eyeballing.**  `omega_d` from the mean spacing of
   interpolated zero crossings of `y - h_eq`; `zeta*omega_n` from a least-squares fit
   of `ln|extremum|` against extremum time; the waveform against
   `y-y_h = A exp(-zeta omega_n t)[cos(omega_d t) + (zeta omega_n/omega_d) sin(omega_d t)]`.
6. **Acceptance gate.** 3% relative error for every dynamical quantity (C3-C7).  The
   geometric and gradient identities are exact mathematics and are gated at 1e-9 and
   1e-6 respectively.

## 4. Results

| check | quantity | measured vs reference | sample | relative error | < 3%? |
|---|---|---|---|---:|---|
| C1 geometry | closed-form vs angle-substituted Simpson | max abs 3.531e-14 | 61 heights | 3.946e-12% | YES |
| C2 restoring stiffness | central-difference $dF_b/dh$ vs $-\rho_w g t W(h)$ | max rel 1.223e-07 | 4 heights | 1.223e-05% | YES |
| C3 Archimedes draft | settled draft vs $f_{sub}=\rho_i/\rho_w$ | 0.837764838 vs 0.837767115 m | settling run | 2.718e-04% | YES |
| C4 heave frequency | measured $\omega_d$ vs linearised | 3.191492734 vs 3.192394377 rad/s | 6 zero crossings | 2.824e-02% | YES |
| C5 envelope decay | measured $\zeta\omega_n$ vs analytic | 0.320954403 vs 0.320732147 1/s | envelope fit | 6.930e-02% | YES |
| C6 waveform | max `abs(y_sim - y_lin)/A` over 3 periods | - | 601 samples | 4.470e-01% | YES |
| C7 pure drag decay | fitted rate vs $C_wet/m$ | 0.641464293 vs 0.641464294 1/s | 100 samples | 2.282e-08% | YES |
| C8 timestep | observed order 3.750, $\omega_d$ spread | dts 0.08, 0.04, 0.02 s | 3 runs | 4.626e-03% | YES |

The exact identities (C1, C2) reproduce the closed form to numerical precision; the
dynamical checks (C3-C7) sit two to six orders of magnitude inside the 3% gate.  The
settled submerged fraction differs from `rho_i/rho_w` by 2.14e-06
(absolute), i.e. the Archimedes condition is met to force-residual level
(`|v_final| = 1.25e-05 m/s`).

### 4.1 Restoring gradient vs chord-width identity (C2)

| h [m] | dF_b/dh (central difference) [N/m] | analytic -rho_w g t W(h) [N/m] | rel. error |
|---|---:|---:|---:|
| -0.437767 | 4858.254745 | 4858.255339 | 1.223e-07 |
| -0.337767 | 7414.023516 | 7414.023684 | 2.256e-08 |
| -0.237767 | 8845.575362 | 8845.575460 | 1.113e-08 |
| 0.000000 | 10055.249933 | 10055.250000 | 6.666e-09 |

### 4.2 Timestep convergence (C8)

`y(t_end = 6.0 s)` for a release from `h_eq + A`:

| Delta t [s] | y(t_end) [m] | omega_d [rad/s] | successive difference [m] |
|---:|---:|---:|---:|
| 0.08 | -0.337047601090 | 3.191620105 | 2.291e-07 |
| 0.04 | -0.337047830167 | 3.191472476 | 1.703e-08 |
| 0.02 | -0.337047847194 | 3.191488284 | - |

Observed order 3.750 (RK4 is fourth order; the measured value is limited
by rounding of the state differences).  The measured `omega_d` varies by only
4.63e-03% across the three step sizes, so the C4-C6 residuals are not
integrator artefacts.

### 4.3 Linearisation limit (C9)

| amplitude A [m] | max normalised waveform error | omega_d rel. error |
|---:|---:|---:|
| 0.01 | 8.861e-01% | 5.831e-02% |
| 0.001 | 9.005e-02% | 5.510e-03% |

Reducing the amplitude 10x cuts the waveform residual by 9.8x, i.e. the residual is
the O(amplitude) curvature of the exact circular-segment law rather than an
implementation defect.  Consequence for use: the linearised reference is a valid
3%-gate benchmark **only for small excitations**; this is stated as a limitation and
not tuned away.

## 5. Figure

![buoyancy free-float verification](figures/buoyancy_free_float_verification.png)

(a) Free-float heave history (RK4, dt = 0.01 s) against the analytic damped oscillator and
its envelope `±A exp(-zeta omega_n t)`.
(b) RK4 timestep refinement of `y(t_end)`, successive differences
2.29e-07 -> 1.70e-08 m.
(c) Normalised waveform residual against the 3% gate as the excitation amplitude is
reduced.

## 6. Pitfall found while building the reference (recorded)

The first version of the reference quadrature integrated the chord width in the
height variable with 64 Simpson panels.  The chord width has **square-root
endpoints**, so that rule converges only as `O(n^-1.5)` and was biased by `3.1e-4` in
`f_sub`.  Bisecting on that biased reference moved `h_eq` by `5.7e-4 m`; the force law
then appeared unbalanced by `3.4e-3 m/s^2` and the free-float comparison showed a
spurious **11%** waveform error — an apparent solver defect that was entirely an
artefact of an under-resolved *reference*.  The angle substitution removes the
singularity, and `test_naive_height_quadrature_is_biased` keeps the biased form in the
suite as a documented counter-example.  Lesson: gate the reference
(`test_quadrature_panel_independence`) before gating the code.

## 7. Limitations and applicability

* 2-D per-unit-thickness law; a 3-D floe needs the spherical-cap analogue, which is
  neither implemented nor verified here.
* No added mass, no radiation damping, no free-surface waves: `zeta` and `omega_d` are
  model parameters, not measured ice dynamics.
* Drag is linear in velocity and `f_sub`-weighted; real wet drag is quadratic plus
  history effects.  The verification is exact for the law as coded and says nothing
  about its physical sufficiency.
* Single particle, no contacts, collisions or rotation; the coupled
  `CalibratedIceDEM` path is not exercised, and still has no free surface or buoyancy
  (see `docs/calibrated-wet/README.md`).
* Constant density, no wave field, no current.

## 8. Reproduction

    PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python scripts/verify_buoyancy_free_float.py
    OMP_NUM_THREADS=1 PYTHONPATH=src \
      python -m pytest -q tests/test_buoyancy_free_float_verification.py

The campaign is deterministic (no random numbers, fixed step sizes) and takes ~17 s on
one core; `--quick` shortens the settling window for CI (~8 s) with unchanged
tolerances.
