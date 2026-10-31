# Brash-ice channel resistance and escort spacing (W4-N07)

Operational layer for **channel (brash-ice) resistance** and **convoy escort
spacing optimisation**, built on a floating-piece DEM and calibrated to the
Riska (1989) channel-resistance form.

## 1. Physical model

### 1.1 DEM (`tensordem.brash_channel`)

A brash-ice channel is represented as a polydisperse cloud of spherical ice
pieces floating in a channel of rigid walls. A rigid hull — rounded wedge bow
plus parallel sides, extruded over the brash layer — translates along the
channel at a prescribed speed. Forces per piece (translational degrees of
freedom only):

| force | law |
| --- | --- |
| gravity | `-m g z` |
| buoyancy | Archimedes on the submerged sphere-cap fraction `f(u)`, `u = -z/r`, waterline `z = 0`; closed-form potential `U_b` |
| piece–piece contact | linear spring `k_n δ` + dashpot `c_n`, viscous-Coulomb friction `min(μ F_n, c_t |v_t|)` |
| hull contact | same law on the plan-view signed distance of a rounded wedge + parallel sides |
| channel walls | spring-dashpot + friction at `y = 0`, `y = W`, floor |
| hydrodynamic drag | linear `-c_d (r/r̄) v` |

Piece rotation is not resolved (no rolling resistance); the brash layer is
plan-view 2-D and pieces cannot dive under the hull. Those are the two dominant
model limitations (see §5).

Contact stiffness scales with the piece radius, `k_n ∝ r̄`, so the *relative*
penetration at a given load is resolution independent — this is what makes the
50k vs 100k resolution study meaningful. The inertial mass may be scaled
(`mass_scale`) to raise the explicit timestep; every force, and therefore the
force-balanced resistance, stays physical.

### 1.2 Energy ledger

Integration is symplectic (semi-implicit) Euler,

```
v_{n+1} = v_n + dt F_n / m ,      x_{n+1} = x_n + dt v_{n+1}
```

so the kinetic-energy increment equals `Σ F · v_mid dt` with
`v_mid = (v_n + v_{n+1})/2` **exactly**. Every force category is integrated with
that same trapezoidal velocity, and the ledger

```
W_bow = ΔKE + ΔU_gravity + ΔU_buoyancy + D_contact + D_hull_slide + D_wall + D_drag
```

therefore closes to round-off. `W_bow = R · v_bow · dt` is the work the drive
supplies, `D_hull_slide = W_bow − Σ F_hull · v_mid dt` is the sliding loss at the
hull surface. A closure residual is a genuine consistency check of the contact
laws, the hull reaction pair and the integrator — not an artefact of a small
`dt`.

## 2. Riska (1989) channel-resistance form

```
R = C_h (B h σ_f) (C/C_ref)^p (h/h_ref)^a (1 + k_v v)
```

`C_h` is the dimensionless channel coefficient; `p`, `a` are the concentration
and thickness exponents. Reference state: `C_ref = 0.6`, `h_ref = 1.0 m`,
`σ_f = 0.5 MPa`, `v_ref = 1 m/s`. `(C_h, p, a)` are identified by log-linear
least squares against the DEM sweep (relative error minimised uniformly), with a
leave-one-out check for stability.

## 3. Convoy energy and the escort spacing optimum

```
E(s) = R_lead/η_lead + n R_escort(s)/η_escort
R_escort(s) = R_ch [1 + α (1 − e^{−s/L_c}) + β e^{−(s−s_0)/λ}]
```

* `α, L_c` — **channel consolidation**: a follower arriving `s/V` later meets
  brash ice that has had time to close up, so its resistance grows with spacing.
* `β, λ` — **close-spacing interference**: ice crowded between the hulls (and
  wake spray) penalises small spacing.

Because the two length scales differ, `E(s)` has an interior minimum. The
optimiser reports the bracket, the curvature and an explicit `interior` flag, so
a boundary optimum can never be silently reported as an interior one.

## 4. Reproduce

```bash
export PYTHONPATH=src
# (concentration × thickness) sweep
python scripts/channel_campaign.py matrix --concentration 0.6 --thickness-m 1.0 \
    --length-m 7.0 --settle-steps 800 --spinup-travel-m 1.0 --measure-travel-m 1.5 \
    --case-dir runs/cases_long
# convoy at a given spacing (multiples of the ship length)
python scripts/channel_campaign.py convoy --spacing-over-length 1.0 --length-m 9.0 \
    --case-dir runs/channel_cases
# particle resolution pair
python scripts/channel_campaign.py convergence --particle-count 50000 --mass-scale 3000 \
    --case-dir runs/cases_conv
# consolidate -> artifacts/channel_resistance/consolidation.json + figures
python scripts/channel_aggregate.py
```

API contract: `docs/channel-resistance/API_CONTRACT.md`
(`POST /calculations/channel-resistance`, handler
`tensordem.channel_resistance.handle_channel_resistance`).

## 5. Limitations

1. **Plan-view 2-D ice**: pieces are confined between vertical walls; they can
   neither dive under the hull nor ride up the bow. The absolute resistance
   therefore reflects lateral displacement + friction, not the vertical
   ice-breaking work of a real bow. Only the *scaling* (with `C`, `h`) is used
   for the Riska calibration, and `C_h` absorbs the absolute level.
2. **No rotational dynamics**: the friction law is viscous-Coulomb on relative
   translation; rolling resistance is absent, so the friction budget is an upper
   bound for round pieces.
3. **No refreezing bonds**: consolidation enters the convoy model as a
   channel-age term fitted to the DEM spacing sweep, not as simulated sintering.
4. **Mass scaling**: `mass_scale` (100 for the sweep, 3000 for the resolution
   pair) accelerates the explicit integration; the force balance, and therefore
   the resistance, is unchanged, but the inertia-driven transients are damped
   differently. A sensitivity run is reported in the artefact.
5. **Velocity dependence is assumed, not fitted** (`k_v = 0`): the sweep is run
   at a single speed (0.5 m/s).
6. Time-averaged resistance is taken over a finite travel window; the reported
   `relative_std` (0.5–1.9) shows the stick-slip scatter of a jammed brash
   channel, which is why the mean, not a single sample, is calibrated.

## 6. Artefacts

| path | content |
| --- | --- |
| `artifacts/channel_resistance/consolidation.json` | matrix, calibration, convoy fit, convergence, cross-checks, criteria |
| `docs/channel-resistance/figures/fig1_resistance_vs_thickness.png` | DEM vs calibrated Riska form |
| `docs/channel-resistance/figures/fig2_parity.png` | parity, ±10 % band |
| `docs/channel-resistance/figures/fig3_convoy_spacing.png` | spacing sweep and U-curve |
| `docs/channel-resistance/figures/fig4_energy_ledger.png` | dissipation budget and closure residual |
| `docs/channel-resistance/figures/fig5_resolution.png` | 50k vs 100k resolution |
| `docs/channel-resistance/figures/fig6_field_snapshot.png` | plan-view field, speed-coloured |