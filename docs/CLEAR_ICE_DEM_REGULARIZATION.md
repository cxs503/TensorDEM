# Clear-ice floe-field DEM: regularised contact, deliverable kinematics, force-chain statistics

Campaign node **P103W2-N03** (mission m6997), repository `TensorDEM`,
package `tensordem.clear_ice`.

> All numbers of this increment are in
> [`CLEAR_ICE_DEM_RESULTS.md`](CLEAR_ICE_DEM_RESULTS.md), which is **generated**
> from `artifacts/clear_ice_dem/result.json` by
> `scripts/clear_ice_results_md.py` (no hand-copied number anywhere).  This
> document is the method, the evidence and the limits.

## 1. What the increment repairs

The preceding node (P103W2B-N03) delivered a mass-conserving breakup/clearance
operator whose **floe kinematics were not deliverable**: at the step size the
affordable run needed, the inherited linear-spring contact with a hard force
cut-off reversed the contact force between successive steps, so the velocity
record of the floe field was dominated by a grid-frequency oscillation and could
not be used as a boundary condition or as a velocity input for anything
downstream.  Nothing downstream (drift forces, clearing rate, load statistics)
can be built on such a record.

This increment replaces the contact model, quantifies the repair with a
machine-checkable metric and a negative control, and adds the three granular
statistics the campaign asks for (`P(f)`, `Z`, `n`), a convergence study, an
uncertainty quantification and a versioned contract to the level-ice breakup
module.

## 2. Physical model

**Scenario.** A broken/clear-ice field of ambient areal concentration
`c = 0.65` (free-water porosity `n = 0.35`) fills a channel `Lx x Ly` that is
periodic in `y`, and drifts at the imposed ice velocity `u_c = 0.8 m/s` towards a
rigid **hull face** at `x = Lx`.  The floes are polydisperse disks (lognormal
radii, `sigma_ln r = 0.30`, mean radius from the concentration target,
`t = 0.30 m`, `rho_i = 917 kg/m^3`).  The field drifts, compacts against the
hull face and comes to rest against it, transferring the water drag of the whole
population through the contact network to the hull.  The face spans the full
channel width (no bypass, so all ice must be stopped by it and none is lost
around it); the reported regions are therefore the **feed** (undisturbed
clear-ice field upstream) and the **jam** (compaction zone at the face), plus the
domain mean.  A cleared **wake** only exists when the face is narrower than the
channel -- the gate geometry is still supported by the contact solver
(`ContactSolver.wall_contacts`) but the campaign scenario is the full-width face,
so the wake region is empty in the artifact.

**Forces.** Water drag `F_d = 0.5 rho_w C_d A |u_rel| u_rel` with the frontal
submerged area `A = 2 r t`, `rho_w = 1025 kg/m^3`, `C_d = 1.2`; the slip
velocity is an output.  Gravity and buoyancy are in equilibrium by construction:
the vertical degree of freedom is relaxed (flat-lying floe), which is the same
stated idealisation as `tensordem.ice_channel`; heave is stiff and fast
(seconds) compared with the contact time (milliseconds), so it does not couple
into the reported statistics.  Roll/pitch are also relaxed (yaw is integrated):
the in-plane load of a broken-ice field does not heel floes enough to matter, but
this *is* a limitation -- floe subduction/ride-up is a level-ice mechanism and is
out of scope.

**Contact law (the core of the repair).** For floe pair `i, j` with penetration
`delta`, effective modulus `E*`, effective radius `R*` and effective mass `m*`:

```
F_el   = (4/3) E* sqrt(R*) delta^{3/2}                 (Hertz)
k_n    = dF_el/ddelta = 2 E* sqrt(R* delta)            (tangent stiffness)
c_n    = 2 zeta sqrt(k_n m*)                           (Tsuji damper)
F_n    = min( max( F_el + c_n * delta_dot , 0 ) , sigma_c * 2 r_min t )   (N)
k_t    = 8 G* sqrt(R* delta)                           (Mindlin)
f_t   <- clip( f_t - k_t v_t dt , -mu F_n , +mu F_n )  (Coulomb, history kept)
```

Three regularisations are load bearing:

1. **`max(., 0)` -- non-cohesive clamp.**  Removes the damper's tension branch,
   which is what produced the sticking/re-impact limit cycle of the linear
   model.  It also *changes the realised restitution* (see §3), so it cannot be
   treated as a cosmetic clamp.
2. **Crushing cap `sigma_c 2 r_min t`** (`sigma_c = 2 MPa`) -- physically the
   bearing capacity of an ice edge; numerically it bounds the DEM when two floe
   edges interpenetrate (relaxation residue, or a floe squeezed by the hull
   face).  Without it the stiff contact can inject an unbounded impulse from a
   geometric overlap.
3. **Friction regularised in displacement, not in velocity** -- the Coulomb cap
   acts on the accumulated tangential spring, so the static branch is exact and
   there is no slip-rate singularity at reversal.  The rigid hull face uses a
   stateless `-mu F_n tanh(v_t/v_s)` law (`v_s = 1 mm/s`) because a rigid wall
   has no tangential compliance worth resolving.

**Integration.** Velocity-Verlet (kick-drift-kick), one force evaluation per
step, Mindlin history advanced by the step displacement.  The step is **limited
by the contact time of the smallest floe**: `dt <= 0.10 t_c` with
`t_c = 2.87 (m*^2 / (R* E*^2 v))^{1/5}`; the configured and used steps and the
ratio `dt/t_c` are reported in every run.  The field is initialised already
drifting at `u_c` (initialising at rest would add ~5 quadratic-drag time
constants of spin-up before a statistics window could open).

## 3. Restitution is an input, not a knob (calibration)

The classical conversion `zeta = -ln e / sqrt(pi^2 + ln^2 e)` solves the
**linear** spring-dashpot.  For the nonlinear Hertz damper the realised
restitution of the *implemented* law (with the clamp, §2) is different, and the
difference is large exactly where the model is used (`e = 0.2` realised as
`0.32` before calibration, i.e. a 60 % error).  The package therefore ships a
calibrated one-dimensional map: a normalised contact equation

```
delta'' = -max( delta^{3/2} + 2 zeta sqrt(1.5 sqrt(delta)) delta' , 0 )
```

carries no material parameter, so a single universal curve `e(zeta)` exists.  It
is integrated with RK4 and tabulated (15 nodes, `zeta in [0, 1]`), and
`damping_ratio_from_restitution` interpolates it.  `tests/test_clear_ice_contact.py`
re-derives table nodes from the integrator and measures the realised restitution
with the full solver to better than 1 %.

## 4. The statistics (and how they are measured)

**`P(f)` -- force-chain distribution.**  Normal (not tangential) contact forces
over the whole post-warm-up window, normalised by their mean; histogram in
`f = F_n/<F_n> in [0, 5]`; exponential fit `ln P = a - f/f0` on the tail window
`1 <= f <= 4` with `R^2`; a power-law fit on the *same* window is reported next
to it, together with the BIC difference, so the exponential claim is falsifiable
rather than asserted.

**`Z` -- coordination number.**  Mechanical contacts with an explicit load
threshold (`f > 1e-3 <F_n>`), reported as the mean degree and as the mean degree
of non-rattlers (degree >= 2), with the rattler fraction, because in a dynamic
clear-ice field most floes are in free flight most of the time and a naive mean
degree would be meaningless.

**`n` -- porosity.**  Two independent estimators: *areal* (free-water fraction
`1 - sum(pi r^2)/A` over exactly-defined strip areas) and *Voronoi free volume*
(periodic Voronoi tessellation with the 8 translated copies, `mean((A_cell -
pi r^2)^+ / A_cell)`).  They use different data (areas vs neighbour geometry), so
their agreement is a consistency check, not a tautology.  The **domain mean** is
the free-field clear-ice porosity (the state variable that conditions
everything); the region values (feed/jam/wake) are emergent: compaction under the
hull face and clearing behind it.

## 5. Convergence and uncertainty (C4)

* **C1 primary** -- a deterministic two-floe Hertz-Mindlin collision, L2 centroid
  position error against the finest step, order by least-squares fit over the
  step-size ladder.  This isolates the integration error (smooth between
  contacts) instead of hiding it in ensemble noise.
* **`dt` sweep** -- the reported statistics at `dt, dt/2, dt/4`, with per-metric
  observed order and the relative change coarse-to-fine.
* **particle count** -- `N/2, N, 2N` at fixed concentration with the domain area
  scaled so the ambient concentration is unchanged.
* **UQ** -- one-at-a-time `+-20 %` on Young's modulus, restitution, friction,
  floe thickness and drift speed, plus two extra seeds; the hull load and the
  porosity/`P(f)`/`Z` statistics are all reported with their spread.
* **momentum ledger** -- the hull-face load is compared with the total water drag
  in the same window: in a quasi-static jam the two must cancel, and the residual
  is reported (this is what caught a wrong normal-force sign during development).

## 6. Evidence grades

| item | grade | evidence |
|---|---|---|
| Hertz/Mindlin elasticity, damper, caps | **REAL** | closed-form checks, `tests/test_clear_ice_contact.py` |
| realised restitution = input | **REAL** | drop test, calibration table re-derived |
| floe kinematics | **REAL** | DEM integration, convergence order, negative control |
| `P(f)` exponential tail | **REAL (internal)** | fit with `R^2` and a power-law alternative |
| `P(f)` vs **IN12-S4 experiment** | **WITHHELD** | the IN12-S4 dataset is **not** in this repository; no comparison is claimed |
| `n` vs experiment | **WITHHELD (direct)** | the criterion band `[0.30, 0.45]` is checked on the simulated field; no IN12-S4 record available |
| vertical/rotational floe modes (subduction) | **NOT_SUPPORTED** | flat-lying planar model, stated |
| floe breakup under contact load | **NOT_SUPPORTED** | one-way coupling by contract |

The honest reading of C2/C3: the *internal* consistency and the *literature*
anchor (exponential force tails in jammed Hertzian packings, Radjai et al. 1996;
O'Donovan & Tanner 2015) are established; the *experimental* comparison the
campaign text asks for cannot be made from this repository, and is therefore not
claimed.

## 7. How to reproduce

```bash
# full campaign (~10 min, 140 floes, 8 s)
PYTHONPATH=src python benchmarks/clear_ice_dem.py --out artifacts/clear_ice_dem

# smoke / CI
PYTHONPATH=src python benchmarks/clear_ice_dem.py --quick --n-floes 60 --duration 2 \
    --out /tmp/clear_ice_quick

# numbers in the docs
PYTHONPATH=src python scripts/clear_ice_results_md.py

# unit + convergence tests
PYTHONPATH=src python -m pytest tests/test_clear_ice_contact.py tests/test_clear_ice_study.py -q
```

## 8. Limitations of the numbers (read with §5 of the generated table)

* **Step-size convergence of the field statistics is a few percent, not exact.**
  Integrals of the contact state converge to 2 % (coordination number,
  domain porosity) but the field-scale quantities (hull load, jam porosity,
  `f0`) move by 5-6 % between `dt/t_c = 0.1` and `0.05`, and not monotonically,
  because each step size samples a slightly different jam configuration.  The
  deterministic order study (C1) is the clean statement of the integration
  order; the ensemble sweep is a *tolerance* statement, and it is reported as
  "not converged within 2 %" rather than hidden.
* **Momentum ledger residual ~17 %.** The statistics window is quasi-static
  (the field is still creeping at ~0.1 m/s), so the wall load and the total drag
  do not cancel exactly; the residual is the unsteady term, and it is reported
  instead of being tuned away with a longer run.
* **Particle-count sensitivity.** The jam is only a few floe diameters thick, so
  the region-resolved statistics depend on `N` at fixed concentration.  Report
  `N` and the domain with every number; the hull load changes by ~40 % between
  `N = 70` and `N = 280` at constant concentration, which is a real
  finite-jam-thickness effect, not a numerical artefact.
* **UQ uses a shorter window** than the production run (a sensitivity study, not
  a second measurement), so its baseline values differ from the report tables.

## 9. What is still missing (next increments)

1. **IN12-S4 ingestion.**  The campaign's experimental anchor has to be brought
   into the repository as a data file with provenance before C2/C3 can be stated
   against experiment.  The result schema already reserves the fields
   (`fit_window`, `R^2`, measured `n` per region).
2. **Two-way coupling.**  Fragments do not break under the contact load
   (`clear-ice-contract/1.0` is one-way); a clear-ice load high enough to crush
   the jam is outside the contract.
3. **Vertical modes.**  Subduction/ride-up of floes at the hull shoulder is
   excluded by the flat-lying assumption; it matters for the hull load at high
   speed and should be added as a second vertical mode, not as a tweak of this
   model.
4. **Long-time statistics.**  The jam statistics are collected over a few
   seconds of settled field; the extreme-value tail of the hull load needs a
   record two orders of magnitude longer (and is the natural next campaign
   node).