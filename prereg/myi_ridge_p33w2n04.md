# Pre-registered design — P33-W2-N04 MYI ridge keel interaction

**Decision.** Does a plane-strain DEM crushing cell supply an extreme ridge load
`F_max` that scales with ice strength and keel depth as `F_max ~ sigma_ice^a h_keel^b`
with `a, b` in [0.8, 1.2], and is the resulting pressure acceptable for an EH36
ice-belt panel at `sigma_max < sigma_y / 1.15`?

**Alternatives considered.**
1. Plane-strain *confined* crushing cell (rigid floor + far-field wall + advancing
   full-depth plate, crushed ice extruded upward) — chosen: it is the only
   configuration that both resolves bond-level failure and yields a pressure
   (`p = f_plate / h_keel`).
2. Free-boundary ridge pushed by a plate without confinement — rejected before
   the run: with no confining stress the aggregate has no strength to mobilise
   and the plate measures bed inertia, not a crushing pressure.
3. Engineering limit analysis `F = p_design h_contact L_c` with a literature
   pressure — rejected as the primary route because it cannot produce a
   campaign-internal scaling law; it remains the fallback if (1) fails.

**Hypothesis.** `a` in [0.8, 1.2] (a cohesion-controlled crushing pressure is
proportional to the ice strength) and `b` in [0.8, 1.2] (the loaded contact
height grows with keel depth).

**Metrics and unit of analysis.** Per DEM case: plate force per unit width
`f` (peak over the run, and mean over the second half of the 0.8 m advance),
pressure `p = f / h_keel`, extreme load `F_max = p h_keel L_c`, energy-ledger
closure `|residual| / exchanged energy`, bond count and broken/exhausted bonds.
Unit = one DEM case (one parameter combination, one packing seed).

**Baseline / control.** Base point `sigma_ice = 7.5 MPa`, `h_keel = 20 m`,
`V = 2 m/s`, `n = 0.30`, seed 20261108, `L_c = 1.5 m`, advance 0.8 m.  Every
other case changes exactly one factor; all numerical settings, the packing
seed, the specimen length ratio (1.2 h), the disc radius (0.5 m) and the
freeze-bond fraction (0.10 sigma_ice) are held fixed, so a difference between
cases is attributable to the swept factor.

**Predeclared thresholds and stopping rule.**
* (a) sampled keel-depth mean within 5 % of target and section porosity within
  0.02 of the request;
* (b) ledger closure < 5 % per case;
* (c) `a, b` in [0.8, 1.2] (with the reported standard errors);
* (d) `sigma_max < sigma_y / 1.15` for the EH36 panel at the campaign design
  pressure (design pressure = the largest measured steady pressure);
* (e) >= 5 cases, `F_max` monotone increasing in `sigma_ice`, `h_keel`, `V`;
* (f) `pytest` >= 15 green; (g) identical results on a repeated run.
Each DEM case stops when the plate has advanced 0.8 m or after 400 000 steps.

**Variables and confounders.** Changed: `sigma_ice`, `h_keel`, `V`, `n`.
Controlled: packing seed, disc radius, bond fraction, friction coefficient,
stiffness set, damping, dt, advance, geometry ratio.  Known confounders: the
pressure is normalised by `h_keel` (so a constant `p` already implies
`F ~ h_keel`), porosity changes the solid area (reported separately), and the
packing is monodisperse (no block-size effect in the DEM).

**Repetition rationale.** One packing per case (fixed seed) is used because the
arguments are compared at a *paired* configuration: with the seed fixed the
lattice is identical across cases, so the comparison is a controlled contrast
rather than a sample from the packing distribution.  The packing-to-packing
scatter is not sampled and is therefore *not* included in the reported standard
errors (a stated limitation).

**Analysis.** Log-space two-regressor least squares with standard errors and
R^2; monotonicity tested on the one-factor sweeps with zero tolerance;
ledger-closure convergence checked by halving dt on the base case.

**Environment and resource limits.** torch 2.7.1 CPU, float64, `dt` from the
stiffest-contact stability limit (~4.0e-5 s at the base point), 10^4 steps per
case, <= 1000 bodies per case, nine cases run as separate processes.

**Safety / interpretation limits.** No threshold was changed after inspecting
results; the failure of criteria (b)/(c)/(d) is reported as measured.  The DEM
pressure is an *ordering* metric, not a design pressure, because the confined
two-dimensional cell is compaction dominated and the pressure is not
time-step converged (a dt halving changes it by 38 %).