# DEM3D contact-impact analytic validation

Anchors: `scripts/benchmark_dem3d_contact_impact.py`,
`tests/test_dem3d_contact_impact.py`, `src/tensordem/contact_impact.py`,
artifacts in `artifacts/dem3d_contact_impact/`.
All numbers below are rendered from `summary.json`; do not edit by hand.

## 1. Physics background and case definition

The DEM3D normal contact is the linear spring-dashpot (Hookean) model truncated at
zero force:

    F_n = max(0, k_n * overlap - c_n * v_n)          (pair, tool and platen branches)

`overlap = 2R - |x_j - x_i|`, `v_n` is the relative normal velocity (negative while
closing), `k_n = contact_stiffness`, `c_n = contact_damping`.

Validation case: two equal free particles collide head-on in free flight
(`drag = 0`, all lattice bonds broken before impact, all other lattice sites parked
and fixed). This isolates one contact pair, so the reference is a 1-DOF problem and
the analytic solution is exact.

| parameter | value |
|---|---|
| particle radius R | 0.05 m |
| density | 900 kg/m^3 |
| particle mass m | 0.4712 kg (sphere, 4/3 pi R^3 rho) |
| effective mass m_eff | m/2 = 0.2356 kg |
| normal stiffness k_n | 5.0e4 N/m |
| damping c_n | from the restitution mapping under test |
| impact speed v0 | 0.5 m/s (each particle), relative speed 1.0 m/s |
| time step | solver `recommended_dt` x scale |

## 2. Method

Two independent analytic references are used, because they answer different
questions.

**Unclamped (free oscillator).** For `m_eff*delta'' + c*delta' + k*delta = 0` with
`delta(0)=0`, `delta'(0)=v0`, `zeta = c/(2*sqrt(k*m_eff))`, `omega_n = sqrt(k/m_eff)`:

    delta(t) = (v0/omega_d) exp(-zeta*omega_n*t) sin(omega_d*t),  omega_d = omega_n sqrt(1-zeta^2)
    e = exp(-pi*zeta/sqrt(1-zeta^2)),   t_contact = pi/omega_d

This is the identity inverted by `contact_tangential.restitution_to_damping`.

**Clamped (what the solver actually integrates).** The force vanishes when
`k*delta + c*delta' = 0`, i.e. at `theta* = omega_d*t*` satisfying
`(1-2*zeta^2) sin(theta*) + 2*zeta*sqrt(1-zeta^2) cos(theta*) = 0`:

    theta*   = atan2(2*zeta*sqrt(1-zeta^2), 2*zeta^2 - 1)      (pi/2 < theta* <= pi for zeta < 1/sqrt(2))
    e_clamped = -delta'(t*)/v0
    t_force   = theta*/omega_d
    t_overlap = t_force + delta(t*)/(e_clamped*v0)             (the pair keeps overlapping after the force clamps)

The closed form is cross-checked against an independent RK4 integration of the
clamped ODE over the whole parameter range (maximum absolute difference in
`e`: 1.174e-11).

Solver protocol: `IceDEM3D.step()` (semi-implicit Euler) until the pair separates;
the realised restitution is `e = -v_after/v0` of the lower particle, momentum drift is
`sum(v_z)` before vs after, and the assembled normal force is compared with the
documented law at a prescribed overlap.

## 3. Results

### 3.1 Restitution versus both analytic references

| e_target | e_measured | e_clamped_analytic | e_unclamped_analytic | err_vs_clamped_pct | err_vs_unclamped_pct |
|---|---|---|---|---|---|
| 0.95 | 0.9516 | 0.9505 | 0.95 | 0.1104 | 0.1638 |
| 0.9 | 0.9037 | 0.902 | 0.9 | 0.1835 | 0.409 |
| 0.8 | 0.811 | 0.8081 | 0.8 | 0.3609 | 1.377 |
| 0.7 | 0.7221 | 0.7182 | 0.7 | 0.5433 | 3.158 |
| 0.6 | 0.6366 | 0.6323 | 0.6 | 0.6885 | 6.107 |
| 0.5 | 0.5543 | 0.5503 | 0.5 | 0.7332 | 10.86 |
| 0.4 | 0.4751 | 0.472 | 0.4 | 0.6496 | 18.77 |
| 0.3 | 0.3994 | 0.3971 | 0.3 | 0.5642 | 33.13 |

### 3.2 Contact duration (two distinct definitions)

| e_target | contact_time_force_ms | contact_time_force_analytic_ms | contact_time_force_err_pct | contact_time_geometric_ms | contact_time_geometric_analytic_ms | contact_time_geometric_err_pct |
|---|---|---|---|---|---|---|
| 0.95 | 6.658 | 6.75 | -1.354 | 6.777 | 6.821 | -0.6361 |
| 0.9 | 6.658 | 6.678 | -0.2934 | 6.777 | 6.824 | -0.6773 |
| 0.8 | 6.421 | 6.528 | -1.65 | 6.777 | 6.836 | -0.8578 |
| 0.7 | 6.302 | 6.37 | -1.067 | 6.777 | 6.859 | -1.197 |
| 0.6 | 6.064 | 6.2 | -2.201 | 6.777 | 6.897 | -1.738 |
| 0.5 | 5.945 | 6.018 | -1.226 | 6.851 | 6.954 | -1.473 |
| 0.4 | 5.737 | 5.82 | -1.431 | 6.978 | 7.036 | -0.8313 |
| 0.3 | 5.52 | 5.602 | -1.458 | 7.097 | 7.156 | -0.8129 |

### 3.3 Time-step convergence

| e_target | dt_scale | dt_s | force_steps | e_measured | e_clamped_analytic | err_vs_clamped_pct |
|---|---|---|---|---|---|---|
| 0.7 | 1 | 0.0001189 | 53 | 0.7221 | 0.7182 | 0.5433 |
| 0.7 | 0.5 | 5.945e-05 | 106 | 0.7199 | 0.7182 | 0.2411 |
| 0.7 | 0.25 | 2.972e-05 | 214 | 0.719 | 0.7182 | 0.111 |
| 0.5 | 1 | 0.0001008 | 59 | 0.5543 | 0.5503 | 0.7332 |
| 0.5 | 0.5 | 5.038e-05 | 119 | 0.5521 | 0.5503 | 0.334 |
| 0.5 | 0.25 | 2.519e-05 | 238 | 0.5511 | 0.5503 | 0.1459 |
| 0.3 | 1 | 6.066e-05 | 91 | 0.3994 | 0.3971 | 0.5642 |
| 0.3 | 0.5 | 3.033e-05 | 184 | 0.3981 | 0.3971 | 0.2458 |
| 0.3 | 0.25 | 1.517e-05 | 368 | 0.3975 | 0.3971 | 0.08862 |

### 3.4 Corrected clamped mapping

`restitution_to_damping_clamped` inverts the *clamped* model (bisection on zeta)
instead of the free-oscillator identity. Realised restitution against the target:

| e_target | e_measured | rel_err_corrected_pct | rel_err_old_mapping_pct |
|---|---|---|---|
| 0.9 | 0.9017 | 0.1839 | 0.409 |
| 0.7 | 0.7039 | 0.556 | 3.158 |
| 0.5 | 0.5033 | 0.6659 | 10.86 |
| 0.3 | 0.3013 | 0.4468 | 33.13 |

### 3.5 Rate independence (linear model)

| v0_m_s | e_measured | e_clamped_analytic | err_vs_clamped_pct |
|---|---|---|---|
| 0.25 | 0.5541 | 0.5503 | 0.6915 |
| 0.5 | 0.5543 | 0.5503 | 0.7332 |
| 1 | 0.5544 | 0.5503 | 0.7541 |

### 3.6 Derivation cross-check

| zeta | e_clamped_closed_form | e_clamped_rk4 | abs_diff | e_unclamped |
|---|---|---|---|---|
| 0.05 | 0.8588 | 0.8588 | 1.169e-11 | 0.8545 |
| 0.1 | 0.7441 | 0.7441 | 9.792e-12 | 0.7292 |
| 0.2 | 0.5717 | 0.5717 | 6.746e-12 | 0.5266 |
| 0.3 | 0.451 | 0.451 | 1.174e-11 | 0.3723 |
| 0.4 | 0.3635 | 0.3635 | 3.987e-12 | 0.2538 |
| 0.5 | 0.2984 | 0.2984 | 1.576e-12 | 0.163 |
| 0.6 | 0.2488 | 0.2488 | 3.337e-12 | 0.09478 |
| 0.69 | 0.2137 | 0.2137 | 2.818e-12 | 0.05004 |

## 4. Verdict

| quantity | max abs error | within 3 % band |
|---|---|---|
| solver restitution vs clamped analytic | 0.733 % | YES |
| solver restitution vs unclamped analytic (mapping basis) | 33.127 % | NO |
| force-contact duration vs analytic | 2.201 % | YES |
| overlap duration vs analytic | 1.738 % | YES |
| corrected mapping, realised vs target | 0.666 % | YES |
| existing (unclamped) mapping, realised vs target | 33.127 % | NO |
| momentum drift (total) | 0.000e+00 kg m/s | exact zero |
| force-law residual (total) | 0.000e+00 | machine precision |

Reading: the **solver is correct** and reproduces the clamped closed form to well
inside the 3 % band, converging under time-step refinement. The **mapping in
`contact_tangential` is the inaccurate part**: it inverts the free-oscillator
identity while the solver truncates the force at zero, so a requested
`e_target` is systematically overshot once dissipation is significant.

## 5. Notes, limitations and scope

1. The `max(0, ...)` truncation is a deliberate modelling choice (no tensile/adhesive
   contact), not a bug. The consequence is that the classic `e <-> gamma_n` identity
   is only valid in the weak-damping limit; the bias grows with `1-e_target`
   (see 3.1) and is already above the 3 % band at `e_target <= 0.7`.
2. The same truncation exists in the tool and platen branches, so the bias transfers
   to every normal contact in the solver; the pair case is used because it is the
   only configuration with an exact counterpart.
3. The closed-form `e_clamped` assumes `zeta < 1/sqrt(2)`; above that the force
   vanishes during the approach phase and the code falls back to RK4. Adhesive or
   Hertzian contacts are out of scope - this validates the linear law only.
4. The semi-implicit Euler integrator is first order; restitution errors are
   ~O(dt/t_contact). The default `recommended_dt` gives <1 % error for the cases
   tested here, but a soft-damping case with fewer than ~50 steps per contact must
   be re-checked.
5. Fracture, friction (Cundall-Strack) and tangential response are not exercised;
   the pair is bond-free and normal-only.
6. Recommended use: when a measured coefficient of restitution must be imposed,
   use `restitution_to_damping_clamped` (clamped solver) rather than
   `restitution_to_damping`; keep the latter only for the free-oscillator identity
   it documents.

Figure: `artifacts/dem3d_contact_impact/contact_impact_benchmark.png`.
