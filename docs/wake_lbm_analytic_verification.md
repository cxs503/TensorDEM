# D3Q19 propeller-race LBM: analytic verification of the lattice core

Status: **verification debt cleared for the core kernels** (equilibrium,
collision, viscosity, Smagorinsky closure, sound speed). The *driving* boundary
set (inlet / outlet / actuator disk) is explicitly **out of scope** here and
remains covered by the momentum- and mass-conservation audit only.

Debt being cleared: `tensordem.threefield.wake_lbm` (`WakeLBM`) is the
lattice-Boltzmann solver that produces the propeller race used by the
three-field ice–hull–propeller coupling. Its evidence was conservation-only —
the excess race momentum flux versus the imposed thrust, and the mean profile
versus the self-similar Gaussian jet model. Nothing bound the **effective
viscosity**, the **speed of sound** or the **equilibrium** to a closed-form
reference, although every lattice-to-SI conversion in `src/tensordem/threefield/coupling.py`
depends on them. `docs/threefield/FORMULAS.md` did not grade the LBM core at
all.

Producing script: `scripts/verify_wake_lbm_analytic.py`
(all numbers below are emitted by it, never hand-copied).
Tests: `tests/test_wake_lbm_analytic_verification.py` (7 tests, all green).
Artifacts: `docs/artifacts/wake_lbm_analytic/` (JSON, CSV, 3 PNG).

## 1. Case and parameters

| item | value | note |
|---|---|---|
| lattice | D3Q19, SRT/BGK, Guo-free (no body force in these cases) | shipped velocity/weight set |
| domain | 16×16×16 (12×16×16 for the closure check) = 4096 cells | small so the run is minutes not hours |
| x boundary | **periodic** (supplied by the harness) | shipped set is inlet/outlet — see §5 |
| y, z boundary | free-slip specular reflection (shipped tables) | walls at nodes `y=0`, `y=ny-1` |
| shear mode | `u_x(y,0) = A cos(K y)`, `K = m pi/(ny-1)`, `A = 0.01` | wall-normal derivative vanishes at both walls, i.e. compatible with free-slip |
| relaxation times | `tau0 = 0.6, 0.8, 1.0, 1.2, 1.5` | `nu = (tau0-1/2)/3 = 0.033 … 0.333` |
| Smagorinsky | `Cs = 0.16`, `tau0 = 0.5006` (the solver's operating point) | closure check only |
| precision | float64 | |
| run length | 2.5 viscous time constants (`1/(nu K^2)`), 206–2052 steps | |

Physical meaning of the case: a transverse (shear) wave in an incompressible
viscous fluid has the exact Navier–Stokes solution

    u_x(y,t) = A cos(K y) exp(-nu K^2 t).

Because the analytic solution is exponential with a rate that *is* the
viscosity, measuring the decay of the projected mode amplitude is an independent
measurement of the solver's viscosity — the one transport coefficient that
controls how fast a propeller race spreads and how much of the thrust the
far-field Gaussian fit attributes to spreading rather than to momentum loss.

## 2. Method

1. `PeriodicShearHarness` composes the **shipped** kernels —
   `_equilibrium`, `_macroscopic`, `_effective_tau`, `_stream`, and the
   free-slip reflection tables `reflect_y` / `reflect_z` — and supplies the one
   ingredient the shipped solver does not have: a periodic x link. Nothing in
   the physics path is re-implemented.
2. Mode amplitude is extracted with a trapezoid-weighted projection on
   `cos(K y)` (`a(t)`), then `ln|a|` is least-squares fitted against `t`;
   `nu_meas = -slope / K^2`.
3. Sound speed is measured from the zero-crossing half-period of an x-periodic
   acoustic mode, `omega = cs k`.
4. The Smagorinsky closure is checked on a *known* non-equilibrium field: a
   collision of an equilibrium field returns that equilibrium, so the field is
   produced by one streaming step of the analytic shear mode
   (`f_i(x) = feq_i(u(x - c_i))`), which has `Pi_ab ∝ strain rate ∝ amplitude`.
   The code is compared against a direction-by-direction Python evaluation of
   the published closure (Hou et al. 1996).

References: Krüger et al. (2017) *The Lattice Boltzmann Method*, Ch. 5
(shear-wave viscosity) and Ch. 9 (Smagorinsky); Hou, Sterling, Chen & Doolen
(1996) *Fields Inst. Commun.* 6, 151 (closure); Skordos (1993) *Phys. Rev. E*
48, 4823 (equilibrium moments).

## 3. Results

### 3.1 Equilibrium and lattice isotropy (machine precision)

| quantity | measured max error |
|---|---|
| `sum_i feq_i - rho` | 2.2e-16 |
| `sum_i feq_i c_i / rho - u` | 4.9e-17 |
| `sum_i feq_i c_ia c_ib - rho(cs^2 d_ab + u_a u_b)` | 1.1e-16 |
| `sum_i w_i - 1` | 0.0 |
| `sum_i w_i c_ia c_ib - cs^2 d_ab` | 1.1e-16 |
| `sum_i w_i c_ia c_ib c_ic c_id - cs^4(d_ab d_cd + d_ac d_bd + d_ad d_bc)` | 1.1e-16 |

### 3.2 Shear-wave viscosity versus `nu = (tau0 - 1/2)/3`

| `tau0` | `nu` exact | `nu` measured | error | steps | decay-fit `R^2` | mass drift | x-momentum drift |
|---|---|---|---|---|---|---|---|
| 0.60 | 0.033333 | 0.033451 | **+0.352 %** | 2052 | 1.000000 | 1.9e-13 | 6.7e-15 |
| 0.80 | 0.100000 | 0.100235 | **+0.235 %** | 684 | 1.000000 | 4.9e-14 | 2.8e-15 |
| 1.00 | 0.166667 | 0.166660 | **−0.004 %** | 411 | 1.000000 | 2.1e-14 | 5.9e-16 |
| 1.20 | 0.233333 | 0.232472 | **−0.369 %** | 294 | 0.999999 | 1.2e-14 | 8.7e-16 |
| 1.50 | 0.333333 | 0.329436 | **−1.169 %** | 206 | 0.999995 | 6.7e-15 | 6.3e-16 |

Mode scaling (diffusive check, `tau0 = 1.0`): `m = 2`, `K = 2pi/15 = 0.418879`,
`nu_meas = 0.166589` → error **−0.046 %**, `R^2 = 0.999993`. The decay rate
therefore scales as `K^2` and the viscosity is not a function of the mode.

### 3.3 Smagorinsky closure (`Cs = 0.16`, `tau0 = 0.5006`, `nu0 = 2.000e-4`)

| amplitude `A` | `mean(tau_eff - tau0)` | `nu_t` | `nu_t / nu0` |
|---|---|---|---|
| 0.01 | 1.341e-4 | 4.470e-5 | 0.224 |
| 0.05 | 6.702e-4 | 2.234e-4 | 1.117 |
| 0.10 | 1.340e-3 | 4.466e-4 | 2.233 |

* zero-stress limit: `tau_eff(Pi = 0) = tau0` exactly (error 0.0);
* fitted log-log exponent of `tau_eff - tau0` vs amplitude: **0.9996**
  (`|Pi|` is linear in the strain rate, so the eddy contribution is linear in
  the amplitude for a fixed wavenumber);
* code versus independent evaluation of the Hou et al. closure on the same
  field: max relative difference **0.0** (bit-identical).

### 3.4 Sound speed

`k = 2pi/16 = 0.392699`, `omega` measured from 29 zero crossings:
`omega = 0.226265` vs `cs k = 0.226725`; `cs_meas = 0.576179` vs `1/sqrt(3) =
0.577350` → error **−0.203 %**.

### 3.5 Figures

| file | content |
|---|---|
| `fig1_shear_decay.png` | semi-log `a(t)` for all five `tau0` against `A exp(-nu K^2 t)` |
| `fig2_viscosity_vs_tau.png` | measured `nu` vs `(tau0-1/2)/3` with per-point error labels |
| `fig3_sgs_and_acoustics.png` | closure scaling `tau_eff - tau0` vs amplitude, and the acoustic mode trace |

## 4. Error judgement (explicit, per key quantity)

| key quantity | reference | error | < 3 % ? |
|---|---|---|---|
| shear viscosity, `tau0 = 0.6` | `(tau0-1/2)/3` | +0.352 % | **yes** |
| shear viscosity, `tau0 = 0.8` | idem | +0.235 % | **yes** |
| shear viscosity, `tau0 = 1.0` | idem | −0.004 % | **yes** |
| shear viscosity, `tau0 = 1.2` | idem | −0.369 % | **yes** |
| shear viscosity, `tau0 = 1.5` | idem | −1.169 % | **yes** |
| viscosity at mode 2 (`k^2` scaling) | idem | −0.046 % | **yes** |
| lattice sound speed | `1/sqrt(3)` | −0.203 % | **yes** |
| equilibrium mass/momentum/stress | exact sums | ≤ 2.2e-16 (round-off) | **yes** |
| D3Q19 isotropy (2nd, 4th moments) | exact lattice sums | ≤ 1.1e-16 | **yes** |
| Smagorinsky closure value | Hou et al. (1996) closed form | 0.0 (same field) | **yes** |
| Smagorinsky amplitude exponent | 1 (linear in strain rate) | 0.0004 | **yes** |
| **absolute** eddy viscosity `nu_t` | *no independent reference* | – | **not claimed** |

The last row is the honest limit: the closure is verified as an *implementation*
of the published formula, not as a turbulence model. There is no measured
propeller-race turbulence data in the repository to calibrate `Cs` against.

## 5. Notes, pitfalls and scope

1. **The naive periodicity wrap is a mass source.** `out[0,:,:,i] =
   f[nx-1,:,:,i]` resurrects populations that `_stream` had already dropped
   through the y/z walls at the corner columns; that version gained ~2e-5 of the
   total mass per step and biased the fitted viscosity by −11 %. The wrap must
   repeat the same y/z slicing as `_stream`. This is the single most
   instructive trap of this verification: an 11 % "viscosity error" was a
   harness bug, not solver physics.
2. **A collision of an equilibrium field is that equilibrium.** The Smagorinsky
   closure is exactly inert on an equilibrium input (`f_neq = 0`), so a
   closure test must first generate gradients (one streaming step here).
3. **The Frobenius norm counts off-diagonals once.** `_outer` contains the six
   independent components; summing all nine entries of `Pi_ab Pi_ab` (which is
   what a naive manual check does) double-counts `xy`, `xz`, `yz` and shifts
   `tau_eff` by ~1.8e-3 — an error that would look like physics.
4. **Free-slip reflection conserves mass only if the wall-normal velocity
   vanishes.** The reflection copies the outgoing populations into the incoming
   slots; the reflected mass and the mass that leaves during the next streaming
   step match only when `u_n = 0` at the wall. In the shear mode `u_n` is zero to
   round-off, so the measured drift is 1e-14 … 2e-13 over 200–2000 steps. In a
   flow with a genuine wall-normal component this boundary is *not* exactly
   conservative, which is a property of the shipped scheme, not of this harness.
5. **Scope.** Only the core kernels are covered. The shipped inlet (uniform
   equilibrium plane), outlet (zero-gradient copy) and the actuator-disk force
   have no closed-form reference at all; their evidence remains the
   momentum/mass conservation audit. A Poiseuille- or Taylor–Green-level check
   of those boundaries would need no-slip or full-periodic support, which the
   class does not expose.
6. **Operating point.** At `tau0 = 0.5006` the subgrid term is *not* cosmetic:
   `nu_t/nu0 = 0.22` at `A = 0.01` and `2.23` at `A = 0.10`. The molecular part
   alone is a poor model of the race at operational amplitudes, so a change of
   `Cs` is a change of the physics, not a numerical knob.
7. **Largest residual** (`tau0 = 1.5`, −1.17 %): the run is also the shortest
   (206 steps, 2.5 time constants) and the mode the most strongly damped, so the
   fit sees the fewest e-folds. The error is inside the 3 % band but is the one
   number in this document I would re-measure with a longer run before quoting
   it in a paper.