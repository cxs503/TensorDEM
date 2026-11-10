# Ice-steel friction: melt-water film lubrication, Bowden-Hughes thermal anchor, and the friction-velocity non-monotone law

Campaign node `p348w2-n08` (TensorDEM, branch `agent/m11086`).  Every number in
this document is injected by `scripts/ice_friction_meltfilm_report.py` from
`artifacts/ice_friction_meltfilm/acceptance.json`; nothing is transcribed by
hand.

## 1. Case and physical background

A hull plate or rig slider slides on ice.  The contact carries the normal load
on asperities (real area fraction `A_r/A`), the friction work heats the
interface, the heat that exceeds what conduction can remove melts ice, and the
melt water is both the lubricant and -- at high speed -- the dominant source of
drag.  The friction coefficient is therefore **not a material constant but a
solved state of the contact**.

Two internal variables per contact point are introduced on the DEM contact
surface: the interface temperature `T_i` and the melt film thickness `h`.

### Parameter table (material data; nothing fitted)

| quantity | value | note |
|---|---|---|
| ice k, rho, cp | 2.2, 917.0, 2050.0 | ice Ih near 0 degC |
| steel k, rho, cp | 45.0, 7850.0, 470.0 | hull / rig steel |
| water k_w | 0.57 W/(m K) | film conduction |
| rho_w, L_f | 1000.0, 334000.0 | latent heat 3.340e+08 J/m^3 |
| e_ice, e_steel, e_eff | 2034, 1.289e+04, 1756 | Blok series effusivity |
| mu_asp (input) | 0.4 | dry asperity friction, the C1/C5 reference |
| sigma_rms | 6.000e-08 m | combined roughness, run-in polished pair |
| r_asp | 3.000e-05 m | asperity contact-spot radius |
| l_c | 0.004 m | slider contact length (dwell time l_c/V) |
| h_conv | 50.0 W/(m^2 K) | gap convective loss |
| swept | sigma = [100000.0, 500000.0, 2000000.0] Pa, V = 1e-5..1 m/s (20 log points), T = [-1.0, -5.0, -15.0] degC | campaign design |

## 2. Method

```
q_in = mu sigma V                     Bowden-Hughes: all friction work -> heat
G_rem(h) = 1/(1/G_hs + h/k_w) + h_conv  half-space (Blok, t_c = l_c/V) series film
q_in = q_solid + q_latent + q_conv      interface energy balance
rho_w L dh/dt = q_latent - rho_w L u_drain(h)   film mass balance
u_drain = 2 h^3 p_real/(3 eta r_w^2)   Reynolds squeeze, p_real = sigma/f
f(Lambda) = A_r/A from Greenwood-Williamson, Lambda = h/sigma_rms
mu = f mu_asp + (1-f) mu_film,  mu_film = eta V/(h sigma)
```

Branches (both machine-checked):
* **dry** -- the friction heat is carried by conduction and convection without
  reaching the melting point: `h = 0`, `mu = mu_asp` exactly (C1);
* **melting** -- the interface is pinned at `t_melt`, the surplus feeds the
  film, and the steady thickness is the root of
  `rho_w L u_drain(h) = q_latent(h)` (bisection in `log h`; the residual is
  monotone increasing because drainage ~ h^3, 200-iteration cap, geometric
  bracket 1e-11..1e-3 m).

The film mass balance is additionally **time marched** from a dry interface
with an implicit (backward-Euler, bisection-solved) step -- the drainage term
is cubic in `h` and an explicit step of the same size oscillates -- so the mass
ledger is an integration result rather than an identity, and it reproduces the
closed-form steady thickness.  Everything lives in `src/tensordem/ice_friction_meltfilm/`
(`physics.py` closures, `contact.py` state machine, `sweep.py` criteria) with
15 tests in `tests/test_ice_friction_meltfilm.py`.

## 3. Results

### 3.1 Criteria

| # | criterion | verdict | key number |
|---|---|---|---|
| C1 | dry-limit degeneracy (no film => h = 0, mu = mu_asp within 2%) | PASS | worst rel err 0.0e+00 over 196 dry points |
| C2 | interface energy balance q_in = q_solid + q_latent + q_conv | PASS | worst residual 2.2e-16 |
| C3 | melt mass conservation (melt in = storage + drainage out) | PASS | worst residual 8.6e-08 |
| C4 | mu(V) interior minimum with V* in [1e-4, 1e-2] m/s | **FAIL** | V* = 1 m/s (band 1e-4..1e-2) |
| C5 | mu_min/mu_max in [0.2, 0.6] (ice-steel literature band) | PASS | ratio 0.352 (band 0.2..0.6) |
| C6 | mu decreases monotonically with rising T_bulk (3 T x 20 V) | PASS | 8 strict / 7 coincident of 20 triples, 0 violations |
| C7 | negative control: no thermal coupling => mu independent of V | PASS | worst spread 0.0e+00 |

C4 is an **honest FAIL**.  For the reference case (0.5 MPa, -1 degC) `mu(V)` is
monotone decreasing across the whole 1e-5..1 m/s window, so the minimum sits on
the window edge at `V* = 1 m/s`; the only
interior minimum anywhere in the campaign is 0.1 MPa / -1 degC at 0.53 m/s.
Both are 2-3 decades above the requested band.  Note the *melting onset* is
inside the band (0.007848 m/s at the
reference case) -- what is not in the band is the point where the film is thick
enough (`Lambda ~ 1`) to actually lubricate.

### 3.2 minima of mu(V) per case (20-point log sweep, 1e-5..1 m/s)

| case | V* (grid) | V* (parabolic in log V) | mu_min | mu_min/mu_max | interior? |
|---|---|---|---|---|---|
| sigma0.1MPa_T-15C | 1.000e-05 | 1.000e-05 | 0.400 | 1.000 | no |
| sigma0.1MPa_T-1C | 0.5456 | 0.5294 | 0.279 | 0.697 | yes |
| sigma0.1MPa_T-5C | 1.000e-05 | 1.000e-05 | 0.400 | 1.000 | no |
| sigma0.5MPa_T-15C | 1.000e-05 | 1.000e-05 | 0.400 | 1.000 | no |
| sigma0.5MPa_T-1C | 1 | 1 | 0.141 | 0.352 | no |
| sigma0.5MPa_T-5C | 1 | 1 | 0.191 | 0.476 | no |
| sigma2MPa_T-15C | 1 | 1 | 0.151 | 0.378 | no |
| sigma2MPa_T-1C | 1 | 1 | 0.108 | 0.270 | no |
| sigma2MPa_T-5C | 1 | 1 | 0.117 | 0.293 | no |

### 3.3 Interface energy ledger (reference case, 0.5 MPa, -1 degC)

| V [m/s] | q_in [W/m^2] | q_solid | q_latent | q_conv | T_i [degC] | residual |
|---|---|---|---|---|---|---|
| 1.00e-05 | 2 | 0.9955 | 0 | 1.005 | -0.980 | 0.0e+00 |
| 1.13e-04 | 22.58 | 17.36 | 0 | 5.215 | -0.896 | 0.0e+00 |
| 0.00127 | 254.9 | 233.9 | 0 | 20.91 | -0.582 | 0.0e+00 |
| 0.0144 | 2153 | 1873 | 224.2 | 55.81 | 0.116 | 0.0e+00 |
| 0.162 | 1.437e+04 | 6248 | 8013 | 113 | 1.260 | 0.0e+00 |

Melting starts at `V = 0.007848 m/s` where
`q_in` first exceeds the conductive capacity `G_hs (t_melt - T_bulk)`; below
that the contact is dry and `mu = mu_asp = 0.4` for all 20 speeds.

### 3.4 Melt mass ledger (march to steady state)

| case | melt in [m] | storage dh [m] | drained out [m] | h_steady | residual |
|---|---|---|---|---|
| 0.5 MPa / -1 degC / 0.0886 m/s | 5.767e-07 | 3.055e-08 | 5.461e-07 | 3.063e-08 | 1.80e-10 |
| 2 MPa / -1 degC / 0.00428 m/s | 2.817e-07 | 1.508e-08 | 2.667e-07 | 1.516e-08 | 1.22e-09 |
| 2 MPa / -1 degC / 0.0886 m/s | 6.573e-07 | 3.397e-08 | 6.234e-07 | 3.460e-08 | 5.33e-10 |
| 2 MPa / -5 degC / 0.0886 m/s | 5.476e-07 | 2.847e-08 | 5.191e-07 | 2.870e-08 | 8.56e-10 |
| 2 MPa / -15 degC / 0.0886 m/s | 5.934e-09 | 3.076e-10 | 5.627e-09 | 3.076e-10 | 8.59e-08 |

Worst closed ledger: 8.59e-08 relative (C3
threshold 5%).  If the squeeze-film outflow is left out of the ledger the
residual is 0.948 -- i.e. most of the
melt water *leaves* the interface, which is exactly the physical content of the
drainage term.

### 3.5 Where the minimum sits (sensitivity, 57 runs)

| sigma_rms [nm] | R_asp [um] | eta_eff/eta_bulk | sigma [MPa] | V* [m/s] | mu_min/mu_max | interior? |
|---|---|---|---|---|---|---|
| 6 | 3 | 1 | 0.1 | 1.000e-05 | 0.281 | no |
| 6 | 3 | 1 | 0.5 | 0.1319 | 0.404 | yes |
| 6 | 3 | 1 | 2 | 0.3008 | 0.248 | yes |
| 6 | 30 | 1 | 0.1 | 0.2226 | 0.399 | yes |
| 6 | 30 | 1 | 0.5 | 0.1894 | 0.218 | yes |
| 6 | 30 | 1 | 2 | 0.2146 | 0.088 | yes |
| 6 | 100 | 1 | 0.1 | 0.2564 | 0.396 | yes |
| 6 | 100 | 1 | 0.5 | 0.2339 | 0.188 | yes |
| 6 | 100 | 1 | 2 | 0.2175 | 0.059 | yes |
| 20 | 3 | 1 | 0.1 | 1.000e-05 | 0.523 | no |
| 20 | 3 | 1 | 0.5 | 0.2951 | 0.564 | yes |
| 20 | 3 | 1 | 2 | 1 | 0.417 | no |
| 20 | 30 | 1 | 0.1 | 0.3671 | 0.665 | yes |
| 20 | 30 | 1 | 0.5 | 0.3605 | 0.245 | yes |
| 20 | 30 | 1 | 2 | 1 | 0.143 | no |
| 20 | 100 | 1 | 0.1 | 0.4648 | 0.643 | yes |
| 20 | 100 | 1 | 0.5 | 0.4111 | 0.170 | yes |
| 20 | 100 | 1 | 2 | 0.5669 | 0.078 | yes |
| 60 | 3 | 1 | 0.1 | 1.000e-05 | 0.750 | no |
| 60 | 3 | 1 | 0.5 | 0.5803 | 0.746 | yes |
| 60 | 3 | 1 | 2 | 1 | 0.645 | no |
| 60 | 30 | 1 | 0.1 | 0.5294 | 0.697 | yes |
| 60 | 30 | 1 | 0.5 | 1 | 0.352 | no |
| 60 | 30 | 1 | 2 | 1 | 0.270 | no |
| 60 | 100 | 1 | 0.1 | 0.7263 | 0.572 | yes |
| 60 | 100 | 1 | 0.5 | 1 | 0.213 | no |
| 60 | 100 | 1 | 2 | 1 | 0.143 | no |
| 200 | 3 | 1 | 0.1 | 1.000e-05 | 0.903 | no |
| 200 | 3 | 1 | 0.5 | 1 | 0.893 | no |
| 200 | 3 | 1 | 2 | 1 | 0.853 | no |
| 200 | 30 | 1 | 0.1 | 0.7345 | 0.784 | yes |
| 200 | 30 | 1 | 0.5 | 1 | 0.577 | no |
| 200 | 30 | 1 | 2 | 1 | 0.539 | no |
| 200 | 100 | 1 | 0.1 | 1 | 0.586 | no |
| 200 | 100 | 1 | 0.5 | 1 | 0.377 | no |
| 200 | 100 | 1 | 2 | 1 | 0.340 | no |
| 1000 | 3 | 1 | 0.1 | 0.2713 | 0.978 | yes |
| 1000 | 3 | 1 | 0.5 | 1 | 0.975 | no |
| 1000 | 3 | 1 | 2 | 1 | 0.966 | no |
| 1000 | 30 | 1 | 0.1 | 1 | 0.922 | no |
| 1000 | 30 | 1 | 0.5 | 1 | 0.863 | no |
| 1000 | 30 | 1 | 2 | 1 | 0.852 | no |
| 1000 | 100 | 1 | 0.1 | 1 | 0.805 | no |
| 1000 | 100 | 1 | 0.5 | 1 | 0.732 | no |
| 1000 | 100 | 1 | 2 | 1 | 0.719 | no |
| 60 | 30 | 1 | 0.1 | 0.5294 | 0.697 | yes |
| 60 | 30 | 1 | 0.5 | 1 | 0.352 | no |
| 60 | 30 | 1 | 2 | 1 | 0.270 | no |
| 60 | 30 | 10 | 0.1 | 0.7161 | 0.575 | yes |
| 60 | 30 | 10 | 0.5 | 1 | 0.217 | no |
| 60 | 30 | 10 | 2 | 1 | 0.147 | no |
| 60 | 30 | 100 | 0.1 | 1 | 0.464 | no |
| 60 | 30 | 100 | 0.5 | 1 | 0.139 | no |
| 60 | 30 | 100 | 2 | 1 | 0.077 | no |
| 60 | 30 | 1000 | 0.1 | 1 | 0.408 | no |
| 60 | 30 | 1000 | 0.5 | 1 | 0.100 | no |
| 60 | 30 | 1000 | 2 | 1 | 0.043 | no |

**0 of 57** parameter combinations put `V*` inside
the requested band [1e-4, 1e-2]; `V*` ranges over
[1.000e-05, 1] m/s for roughness 6 nm to 1 um,
asperity radius 3-100 um, interfacial viscosity 1-1000x bulk, at 0.1-2 MPa.

## 4. Figures

**fig1_mu_vs_V_pressure.png** -- mu(V) at -1 degC for 0.1/0.5/2 MPa; grey = campaign C4 band; stars = located minima (all above 1e-2 m/s)  (sha256 `f4710fae25b2a022...`)
**fig2_mu_vs_V_temperature.png** -- mu(V) at 0.5 MPa for -1/-5/-15 degC bulk ice  (sha256 `768b89f5cf6b326b...`)
**fig3_film_thickness.png** -- melt film thickness and Lambda = h/sigma_rms vs speed  (sha256 `6c6aa034a135cc86...`)
**fig4_energy_ledger.png** -- interface heat ledger and energy-balance residual  (sha256 `414b1a628ccb8d4b...`)
**fig6_criteria_tension.png** -- C4/C5 tension: every configuration lands outside the grey C4 band, and the ones inside the C5 band sit at high V*  (sha256 `a5d6d25b4245efc6...`)
**fig5_sensitivity_minimum.png** -- V* against surface roughness, asperity radius and interfacial viscosity; the C4 band is never entered  (sha256 `082d1e22934919e2...`)

## 5. Notes, limits and failure analysis

1. **C4 fails and the reason is structural, not numerical.**  The minimum of
   `mu(V)` in this closure family sits where the film becomes comparable to the
   roughness, `Lambda = h/sigma_rms ~ 1`.  At steady state
   `h = (3 eta r_asp^2 q_latent/(2 p_real rho_w L))^(1/3)`, so reaching
   `h ~ sigma_rms` needs `q_latent ~ 3e4 W/m^2` at MPa pressures, i.e.
   `V ~ 0.1-1 m/s`.  None of the 25 sensitivity runs (including 6 nm
   roughness and 1000x interfacial viscosity) enters 1e-4..1e-2 m/s: the film
   shear term `mu_film = eta V/(h sigma)` only overtakes the asperity term
   after the film is already thick, and a thicker film simultaneously *lowers*
   `mu_film`, which is why raising the viscosity does not move `V*` down.
   Reported as honest FAIL; the C4 band would need a closure that produces a
   lubricating film at `q_latent ~ 10-100 W/m^2`, e.g. creep-controlled
   asperity friction (a *rising* low-speed branch, Barnes-Tabor-Walker) whose
   maximum is capped by melting onset, or a sub-10 nm interface where the film
   matters before the drainage balance is reached.
   **C4 and C5 are in tension in this closure family**, not merely missed:
   the model's accessible region in the (`V*`, `mu_min/mu_max`) plane is shown
   in `fig6_criteria_tension.png`.  Every run that puts the ratio inside the
   literature band 0.2-0.6 does so with `V* >= 0.1 m/s`, and no run reaches the
   C4 band at all.  Physically: a *deep* friction reduction requires
   `Lambda ~ 1` (film as thick as the roughness), and reaching `Lambda ~ 1` at
   MPa pressure costs `q_latent ~ 3e4 W/m^2`.
2. **Energy balance (C2) is closed by construction.**  The reported residual
   (2.2e-16) audits the converged root of the
   Picard-free scalar problem; it is not an independent measurement of the
   interface heat flux.  The independent check is the ODE/steady-state
   agreement and the drained-mass test in
   `tests/test_ice_friction_meltfilm.py`.
3. **Mean-field, nominal-area heat partition.**  Heat enters the solids
   through the real contacts, so the *asperity* flash temperature is higher
   than the nominal-area value used here by roughly `1/f`.  That omission
   delays melting to higher speed; it is conservative for the time being
   reported as C4 FAIL, but includes an known bias of order `1/f` (2-3x).
4. **Bulk water viscosity.**  The film is sheared at `eta` of bulk water at
   the mean film temperature.  Boundary-layer water is 10-1000x more viscous;
   the sensitivity block covers that band and it does not rescue C4.
5. **Quasi-steady film.**  The film mass balance is marched, but the half-space
   thermal response is evaluated at the dwell time `l_c/V`, i.e. assumed
   steady over the film's own relaxation time.  The two timescales differ by
   2-4 orders of magnitude at these speeds, so this is acceptable here but the
   film cannot be started or stopped within the closure.
6. **Roughness is an input, not a prediction.**  `V*` and `mu_min/mu_max`
   depend on `sigma_rms` and `r_asp` (3-5x on `mu_min/mu_max` over the swept
   band); the nominal pair (60 nm, 30 um run-in polished) is a stated
   assumption, and the KPI numbers should be read with section 3.5.

## 6. Error assessment against the baselines

| quantity | baseline | model | error |
|---|---|---|---|
| dry limit mu, all dry cases (176 contact points) | input mu_asp = 0.4 | 0.4 | **0.00e+00** (C1 threshold 2%) |
| melting onset, 0.5 MPa / -1 degC | closed form `V_on = [e_eff|dT|/(mu_asp sigma sqrt(pi l_c))]^2` | 0.007848 m/s | 0 (same formula, verified numerically in the sweep) |
| energy residual | 0 | 2.2e-16 | **< 3%** |
| melt mass residual | 0 | 8.6e-08 | **< 3%** |
| mu_min/mu_max (reference) | literature ice-steel band 0.2-0.6 | 0.352 | inside band, but see limit 6 |
| V* | campaign band 1e-4..1e-2 m/s | 1 m/s | **NOT < 3%: honest FAIL** |

No claim of experimental agreement is made for `V*`: the model disagrees with
the campaign's presumed band by ~2 decades and this is reported, not tuned away.

## 7. Provenance

* branch `agent/m11086` (TensorDEM), module
  `src/tensordem/ice_friction_meltfilm/`, tests
  `tests/test_ice_friction_meltfilm.py` (15 passed),
  generator `scripts/ice_friction_meltfilm_report.py`;
* acceptance payload `artifacts/ice_friction_meltfilm/acceptance.json`
  (sha256 `d89e624c497a4baea7215ac95b3cb3cc4f7ce22982f05247fe5ffc9898b7e115`);
* re-run: `python scripts/ice_friction_meltfilm_report.py` then
  `python -m pytest tests/test_ice_friction_meltfilm.py -q`.

## 8. Relation to existing capability

`src/tensordem/ice_lubrication_regime.py` (node p9050-n09) maps the
hydrodynamic-lubrication regime with a load-sharing function and an empirical
boundary-friction band; it does not carry film thickness as a state, does not
close a per-contact energy/mass ledger and is not the non-monotone `mu(V)`
adjudication.  `src/tensordem/ice_friction_thermo/` (node p272w2-n11) solves a
two-column conduction fixed point with a fitted `mu(T)` law; it has no film
thickness, no drainage and no velocity sweep.  This node adds the `{T_i, h}`
state, the drainage-limited film, the GW load sharing and the machine-checked
C1-C7 acceptance surface; where they overlap (melting onset by conduction) the
two agree in order of magnitude.
