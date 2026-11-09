# TensorDEM floe-size-distribution evolution: collision kernel, breakage kernel and population balance (campaign node p80w2-n05)

**Verdict.** C1 collision kernel 0.126 (gate 0.150) PASS; C2 breakage kernel 0.136 (gate 0.200) PASS; C3 population balance 0.124 (gate 0.250) PASS; C5 negative control 0.000 (gate 0.100) PASS; C7 seed dispersion 0.102 (gate 0.150) PASS; C4 HOPPER comparison **WITHHELD** (the measured model-test census is not in this worktree, so the criterion is reported as withheld rather than replaced by a fabricated reference).

## 1. 案例介绍 (case)

The repository owns floe *statics* (buoyancy, drift, heave) and the *fragmentation* of a single body (`broken_ice_micro`, `dem`, `dem3d`), but not the statistical mechanics of a broken-ice field: how the floe size distribution n(D,t) evolves when floes collide and break. This node closes that causal chain — collision kernel -> breakage kernel -> population balance — with three DEM experiments and one model-test anchor.

| quantity | value |
|---|---|
| DEM engine | periodic 3D soft-sphere, linear spring-dashpot contacts, breakable axial bonds, coordination-aware velocity Verlet |
| floe aggregates | simple-cubic 3^3 / 4^3 / 5^3 spheres with face diagonals, volume-equivalent diameter D = 3 / 4 / 5 (sphere diameters) |
| floe field | 12 floes, aggregate sizes [3, 4, 5] (668 spheres) |
| sphere volume fraction | 0.050 |
| field granular temperature | 1.50 (floe centre-of-mass temperature) |
| field duration / snapshots | 12.0 / 25 |
| seeds | [0, 1, 2] |
| contact / bond stiffness | 500 / 1000 (non-dimensional) |
| bond tension / compression limit | 4.0 / 40.0 |

All results are non-dimensional: lengths in the smallest floe diameter D_min, times in D_min/v_rms, masses with unit density (m = r^3). Only ratios and dimensionless groups are claimed; absolute rates would require a stiffness ratio no explicit DEM can resolve (see section 5).

## 2. 怎么做的 (method)

**Instrument.** `src/tensordem/floe_population/soft_sphere.py` is a periodic soft-sphere DEM with breakable bonds: a *floe* is a connected component of the surviving bond network, so the floe size distribution is read off the DEM state and never prescribed. Contacts are linear spring-dashpots with the restitution mapped to a damping ratio; bonds carry tension and compression and fail outside a force window; bonded pairs are excluded from the contact list so the stiffness is not double counted. The stable step accounts for the *coordination* of the bond lattice (18 bonds per interior sphere), which is a factor six shorter than the naive single-contact estimate.

**(a) Collision kernel.** A thermostatted polydisperse sphere gas (diameters [1.0, 1.4, 2.0, 2.8], 100 spheres per class, packing fraction 0.20, granular temperature 0.5) is equilibrated and measured. Three quantities are taken from the DEM: the birth rate of contacts per class pair, the time-averaged contact value g(sigma) of the radial distribution function, and the granular temperature. The prediction uses only those:

```
K(D1,D2) = (pi/4) (D1+D2)^2 <|v_rel|> g(sigma),   <|v_rel|> = sqrt(8 T / (pi mu))
```

The event ledger is sampled on a shell of thickness `event_gap`, so a soft contact cannot register several births; the flux through that shell is scaled back to the contact sphere analytically, `K = K_measured (sigma/(sigma+gap))^2`.

**(b) Breakage kernel.** A single aggregate is driven into a rigid anvil at a prescribed speed, so the impact energy E = M v^2 / 2 is exact. The debris census gives the fragment count B; a grid of four sizes and four speeds (three seeds) is fitted in log-log space by `B = k_break D^alpha E^beta`. The whole size `D = 4` and the whole speed `v = 4` are **held out**, so the reported error is a forecast, not a fit residual.

**(c) Population balance.** A fixed-pivot sectional solver (Kumar & Ramkrishna) on a geometric volume grid closes

```
dn(D,t)/dt = coalescence - breakage + source,   breakage: nu(D) n(D) -> B(D,E) equal-volume daughters
```

with both volume and number conserved exactly; the coalescence channel is validated against the analytic constant-kernel Smoluchowski solution and is **inactive** in the forecast (rigid ice floes do not refreeze — the sticking probability is zero in the DEM, not a fitted parameter). The breakage frequency nu(D) is taken from the field's own inter-floe contact statistics over the *calibration* half of the record; the *forecast* half is never used for any input.

**(d) Model-test anchor.** `hopper.py` implements the comparison against the HOPPER (Lu et al. 2016) model-test floe-size census and refuses to invent the missing measurement: it returns `WITHHELD` unless the HOPPER census file is present in the data directory, in which case the same code computes the 30 % criterion.

**(e) Negative control.** The identical field with an unbreakable cement: no breakage kernel, no size drift allowed.

## 3. 计算结果 (results)

| criterion | quantity | value | gate | verdict |
|---|---|---|---|---|
| C1 | rate-weighted L1 error of K(D1,D2) | 0.126 | < 0.150 | PASS |
| C2 | held-out relative error of B(D,E) | 0.136 | < 0.200 | PASS |
| C3 | mean floe size relative error over the forecast half | 0.124 | < 0.250 | PASS |
| C4 | mean floe size vs HOPPER census | WITHHELD | < 0.300 | n/a |
| C5 | size drift with breakage off (std/mean) | 0.000000 | < 0.100 | PASS |
| C7 | seed dispersion of the primary observables | 0.102 | < 0.150 | PASS |

**Per-seed detail**

| seed | C1 rate-weighted error | C1 worst class cell | C3 mean-size error | C3 density error | final mean floe size (DEM) |
|---|---|---|---|---|---|
| 0 | 0.129 | 0.245 | 0.147 | 0.304 | 1.330 |
| 1 | 0.109 | 0.214 | 0.151 | 0.345 | 1.629 |
| 2 | 0.140 | 0.233 | 0.075 | 0.181 | 1.311 |

**Kernels**

* collision kernel: 12.6 % rate-weighted error over 24 class pairs and 3 seeds; the residual grows with the diameter class (the slow, heavy pairs are the hardest for a soft contact to time-resolve).
* breakage kernel: `B = 0.4397 D^(-0.268) E^(0.984)`, R^2 = 0.935, 27 fit points, 21 held-out points, held-out error 0.136 (pointwise 0.227).
* fragmentation onset: the DEM shows `B = 1` at E = 2.000 and `B = 81` at E = 281.250.
* the exponent re-expressed on *specific* energy e = E/M (M ~ D^3) is alpha' = alpha + 3 beta = 2.684, i.e. at a fixed relative speed larger floes produce more fragments — the physical reading of the negative raw alpha (at fixed total E a larger floe receives less energy per unit mass).
* population balance: forecast window starts at t = 6.49 with 19 sections, mean impact energy E = 2T = 3.00.
* fragment census (literature-window diagnostic only): alpha = 3.24 +- 0.36 over 38 fragments, R^2 = 0.619, window [2.5, 3.5], in window: PASS. the pooled census is quantised by the sub-sphere lattice (34 distinct diameters) and short (n_tail = 38, R^2 = 0.62), so the exponent is a weak window check, not a measurement of the comminution law and certainly not the HOPPER census.
* the accepted exponent window [2.5, 3.5] implies a size ratio <D>/D_min in [1.67, 3.00] for a continuum power law, while the DEM field spans only 1-5 sphere diameters (mean 3.67). The two are **not** comparable: a quantised census with a dynamic range of five cannot carry a continuum exponent, which is exactly why C4 is withheld rather than filled with a substitute.
* C7 seed dispersion: breakage_beta 0.025, breakage_k_break 0.096, final_mean_floe_size 0.102, mean_collision_kernel 0.014; the diagnostics that exceed the gate are reported too: C1_kernel_error 0.100, C3_forecast_error 0.282, breakage_alpha 0.354, mid_run_mean_floe_size 0.165.

Mean floe size of the DEM field per seed: seed 0: 3.67 -> 3.48 -> 2.40 -> 1.90 -> 1.73 -> 1.48 -> 1.33; seed 1: 3.67 -> 3.67 -> 2.77 -> 2.31 -> 1.85 -> 1.85 -> 1.63; seed 2: 3.67 -> 3.31 -> 2.20 -> 1.54 -> 1.42 -> 1.34 -> 1.31 (every fourth snapshot).

## 4. 结果图 (figures)

![figure_collision](../artifacts/floe_population/c1_collision_kernel.png)

*C1 — measured collision rate constant against the Enskog prediction pi*sigma^2*<v_rel>*g built from the independently measured g and granular temperature, for every diameter pair and seed, with the +-15 % band.*

![figure_breakage](../artifacts/floe_population/c2_breakage_kernel.png)

*C2 — fragment count per anvil impact against the impact energy for the four aggregate sizes; filled markers are the fit grid, open markers the held-out size/speed, lines the fitted B = k*D^alpha*E^beta.*

![figure_forecast](../artifacts/floe_population/c3_forecast.png)

*C3 — mean floe diameter of the DEM field (markers) and of the population balance forecast (lines) over the held-out second half; the dotted line marks the end of the calibration window.*

![figure_control](../artifacts/floe_population/c5_negative_control.png)

*C5 — negative control: with the breakage kernel switched off the floe sizes do not drift, while the same field with breakage on grinds the distribution down.*

![figure_census](../artifacts/floe_population/c4_fragment_census.png)

*Fragment census of the impact campaign in log-log coordinates with the fitted power law and the accepted broken-ice exponent window 2.5-3.5 (literature anchor, not a measured basin census).*

## 5. 注意点 (traps, limits)

1. **The stable step must know the coordination.** A cemented aggregate is held by up to 18 bonds per sphere, so the elastic frequency is sqrt(37 k/m), not sqrt(k/m). Using the single-contact estimate (six times longer step) explodes the aggregate within a hundred steps, and the explosion is easily misread as *fragmentation*: the first version of this campaign did exactly that, and the symptom was that the fragment count was independent of the bond strength (tension 8, 60, 120 and 400 all shattered everything). The fix is in `recommended_dt`, and `test_recommended_dt_accounts_for_coordination` plus `test_cold_aggregate_is_stable_and_does_not_shed_bonds` pin it.
2. **Forces are not accelerations.** The first engine moved spheres with the contact force itself; the gas then equilibrated to *equal velocities* instead of equipartition, which silently corrupted the mass dependence of the collision kernel (the large spheres collided 3x too often). `test_elastic_two_body_collision_conserves_momentum_and_energy` is the guard.
3. **One snapshot cannot measure g(sigma).** In a dilute gas the contact shell holds a fraction of a pair per snapshot; the contact value must be averaged over hundreds of snapshots, and the shell needs an **absolute** width (a fraction of the smallest diameter). A shell proportional to sigma smears the contact peak of the large classes and biases their g downwards by up to 40 %.
4. **A soft contact can register multiple births.** Counting a new event only after the pair has left a small gap removes the spurious re-births of the slow, heavy pairs, and the shell flux is scaled back to the contact sphere.
5. **Thermostat the floes, not the ice.** Rescaling every sphere to the target granular temperature shakes a bond network apart on its own. The field is driven through floe centre-of-mass velocities, and only bodies at or above the floe scale are re-energised: heating the debris launches every freed sphere at sqrt(3T/m) and the cascade is then produced by the thermostat rather than by the mechanics (that runaway reached single spheres from any initial state).
6. **Absolute rates are not claimed.** The DEM uses a softened stiffness; the fracture *rate* depends on the stiffness ratio, while the fragment *size distribution* is much less sensitive to it. Everything here is therefore reported in non-dimensional form and the population balance is scored on the size evolution, not on wall-clock time.
7. **The field grinds to the resolution floor.** With unlimited energy input at fixed temperature and no refreezing, the cascade ends at single spheres, which is the model's resolution limit; the frozen sections are reported by `effective_fragments` so the forecast never loses mass silently.
8. **C4 is withheld, deliberately.** The HOPPER/Lu-2016 tabulated census is not in this worktree. Reading numbers off a plot without a traceable digitisation would be fabrication, so the criterion is reported as `WITHHELD` with the comparison code tested and ready; substituting the literature exponent window for the measurement is labelled as a substitute, not as the criterion.

## 6. 误差判定 (error verdict)

| quantity | reference | error | gate | within gate? |
|---|---|---|---|---|
| K(D1,D2), all class pairs | DEM contact-birth statistics | 0.126 | 15 % | PASS |
| B(D,E) | DEM anvil-impact fragment census (held out) | 0.136 | 20 % | PASS |
| <D>(t) over the forecast half | DEM field census | 0.124 | 25 % | PASS |
| mean floe size | HOPPER model test (Lu et al. 2016) | WITHHELD | 30 % | n/a |

**Is every key quantity within 3 % of a reference?** No, and it cannot be: the closures are compared against *stochastic DEM measurements*, whose own seed-to-seed scatter is 0.282 (section 3), so a 3 % agreement would be below the statistical resolution of the reference itself. The tightest defensible gates are the ones the campaign states (15 / 20 / 25 / 30 %) and they are met or honestly withheld; the analytic checks that *can* be exact are exact (fixed-pivot mass conservation to 1e-9, and the coalescence channel against the closed-form Smoluchowski solution to 1e-4).

**Honest limitations.** (i) C4 has no measured reference in this worktree — `WITHHELD`, not passed. (ii) The breakage kernel is a three-parameter power law fitted on 27 impacts with chaotic scatter; the pointwise held-out error (0.227) is larger than the aggregate one, so a single impact is predicted only to ~20-25 %. (iii) The population balance is mean-field: it reproduces the size *evolution* of the field but it cannot reproduce spatial clustering, and the field's own collision frequency is used as the rate input, so C3 tests the sectional bookkeeping and the mean-field assumption rather than the kernel closure again (that is C1's job). (iv) No refreezing, no rotational degrees of freedom, no bending stiffness of the bonds: floes are axial-truss bodies, which makes them slightly more brittle than ice.

