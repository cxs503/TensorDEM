# Hull-ice friction: temperature dependence, two-scale calibration and integration into level-ice resistance

Status: implementation + calibration campaign in `TensorDEM` (branch `agent/m835`).
Numbers are **not** repeated in this document: every value lives in
`artifacts/hull_ice_friction/summary.json` and in the auto-generated companion
[`hull_ice_friction_temperature_results.md`](hull_ice_friction_temperature_results.md)
(written by `scripts/render_hull_ice_friction_report.py`, which reads the
artefacts programmatically).

## 1. Physical background

Ice sliding on steel is an adhesion-controlled process: the friction
coefficient is the ratio of the shear strength of the shear plane to the
indentation hardness of the ice surface, `mu = tau_shear / p_hardness`
(Bowden & Tabor 1954). Both quantities grow as the ice cools, and close to the
melting point a quasi-liquid layer carries the shear, so `mu` falls towards
0 degC and rises again with cold. For ice on steel the measured band at about
-10 degC and 0.1-1 m/s sliding speed is `mu = 0.15-0.20` (Bowden & Tabor 1954;
Oksanen & Keinonen 1982).

Because a hull-ice friction coefficient enters level-ice resistance formulas
only through the hull friction term, a temperature-dependent `mu(T)` changes
the predicted resistance between Arctic (-5 degC) and cold (-20/-25 degC)
operating conditions. Quantifying that change, with two independent
(frictional-contact DEM and molecular-dynamics) scale levels, is the scope of
this work.

## 2. What was implemented

### 2.1 Interfacial law and parametric family (`src/tensordem/hull_ice_friction.py`)

* `micro_contact_friction(T)` — `mu_c(T) = mu_ref * (|T - T_melt| /
  |T_ref - T_melt|)^exponent`, anchored at the literature value at -10 degC
  (default `mu_ref = 0.15`, `exponent = 0.45`).
* `friction_temperature_model(T)` — the calibration family
  `mu(T) = mu0 + alpha |T - T_melt|^beta`; `alpha, beta > 0` implies
  `dmu/dT < 0`.
* `fit_mu_temperature(T, mu)` — exact linear least squares in `(mu0, alpha)`
  on a `beta` grid plus bisection refinement (no SciPy dependency); returns
  coefficients, residuals, `R^2` and the monotonicity flag.

### 2.2 DEM sliding-friction experiment (same module)

* 2-D bonded-particle ice bed (square lattice, breakable axial + diagonal
  bonds, top row pinned) resting on a **rigid steel surface made of a flat
  wall plus a periodic row of roughness bumps**. The wall carries the normal
  reaction, so the normal direction is unambiguous; the bumps generate
  ploughing.
* The plate translates at a prescribed sliding velocity; the ice bed is loaded
  by its own weight plus an external normal pressure representative of the
  hull-ice contact pressure.
* Friction is measured as the time-averaged reaction of the ice on the steel
  surface, `mu = |<F_x>| / |<F_z>|`, and decomposed **exactly** into a
  tangential (adhesion) part and the x-projection of the bump normal forces
  (ploughing).
* `dem_temperature_sweep` runs the protocol: five temperatures from -5 to
  -25 degC, ten independent realisations (seeds) per temperature, sample mean
  and standard deviation reported.

### 2.3 MD ice/iron interface shear (`src/tensordem/md_interface_friction.py`)

* Frozen three-layer fcc(111)-like iron slab; two-layer honeycomb water film of
  single-site pseudo-molecules with **Lennard-Jones + Coulomb** interactions
  (alternating partial charges stand in for the directional hydrogen-bond
  network of the reduced model).
* The top film layer is driven by a harmonic spring at constant velocity; a
  Langevin thermostat holds the target temperature. `mu_MD = |<F_x>| /
  |<F_z>|` on the frozen substrate — the same operational definition as the DEM
  experiment, resolved at the molecular scale.
* The single free parameter (Fe/water dispersive well depth) is identified by a
  grid search **at the reference temperature only**; the value is then frozen
  and the remaining temperatures are genuine predictions.
* `film_mean_square_displacement` / `melting_scan` provide an order-parameter
  scan of the reduced water model for the limitations discussed in section 5.

### 2.4 Integration into level-ice resistance (`src/tensordem/ice_resistance.py`)

```
R(T) = R_crushing + R_breaking + R_submersion + R_friction(T)
R_friction(T) = mu(T) * cot(beta) * (R_crushing + R_breaking)
```

`beta` is the mean bow-plating inclination: the normal force on the hull is
`N = (R_crushing + R_breaking)/sin(beta)`, so the tangential (friction)
component contributes `mu cot(beta) (R_crushing + R_breaking)` to the
resistance. Published empirical formulas (Lindqvist, Riska, Jeong, Keinonen)
absorb hull friction inside fitted coefficients, so they cannot be reopened for
a temperature sweep without refitting; this decomposition keeps the friction
term explicit.

### 2.5 API endpoint (`src/tensordem/api_hull_ice_friction.py`)

`POST /calculations/hull-ice-friction`. `handle_hull_ice_friction(payload)` is a
framework-free handler with full request validation; `create_router()` exports a
FastAPI router for mounting in the ship platform:

```python
from tensordem.api_hull_ice_friction import create_router
app.include_router(create_router(prefix="/api/v1"), prefix="")
```

Request fields: `temperatures_C`, `mu_parameters` (calibrated coefficients,
optional), `interface_law` (`mu_ref`, `exponent`), `vessel` (geometry
overrides), `comparison` (warm/cold pair). The response returns the friction
coefficients, the resistance components per temperature, the warm/cold
comparison, an `evidence` block (REAL / WITHHELD statements) and a
machine-checkable `criteria` block.

## 3. Campaign artefacts

| artefact | content |
| --- | --- |
| `artifacts/hull_ice_friction/summary.json` | full campaign record (configuration, sweeps, fit, cross-scale, resistance, criteria, timings) |
| `artifacts/hull_ice_friction/dem_friction.csv` | DEM `mu(T)` with standard deviation, decomposition, normal load, repeats |
| `artifacts/hull_ice_friction/md_friction.csv` | MD `mu(T)`, standard deviation, thermostat temperature |
| `artifacts/hull_ice_friction/cross_scale.csv` | DEM/MD comparison per temperature |
| `artifacts/hull_ice_friction/resistance.csv` | resistance components versus temperature |
| `artifacts/hull_ice_friction/fig1_mu_vs_temperature.png` | `mu(T)`: DEM points + fitted curve, MD points, literature band |
| `artifacts/hull_ice_friction/fig2_cross_scale.png` | DEM versus MD 1:1 plot with the +-30 % band |
| `artifacts/hull_ice_friction/fig3_resistance_vs_temperature.png` | resistance decomposition versus temperature, annotated `delta R` |
| `artifacts/hull_ice_friction/fig4_dem_decomposition.png` | adhesion versus ploughing contribution |

Reproduce with:

```
python scripts/calibrate_hull_ice_friction.py --output artifacts/hull_ice_friction
python scripts/render_hull_ice_friction_report.py
python -m pytest tests/test_hull_ice_friction_temperature.py \
                 tests/test_md_interface_friction.py \
                 tests/test_ice_resistance_hull_friction.py
```

## 4. Acceptance criteria and how they are evaluated

| criterion | definition | implementation |
| --- | --- | --- |
| (a) monotonicity | `dmu/dT < 0` over -5 ... -25 degC | consecutive DEM means must increase towards colder temperatures; the fitted `alpha, beta > 0` |
| (b) literature value | `mu(-10 degC)` inside or within 20 % of 0.15-0.20 | fitted `mu` at the reference temperature versus the Bowden & Tabor band |
| (c) cross-scale | `|mu_DEM - mu_MD| / mu_MD < 0.30` at matched temperatures | per-temperature comparison, maximum reported |
| (d) resistance effect | `R(-20 degC) - R(-5 degC)` relative increase > 10 % | `resistance_temperature_comparison` |
| (e) tests | `pytest` fully green | 30 tests in the three new modules' test files |

The measured values of every criterion are in the generated companion document,
section 5.

## 5. Limitations and scope (honest notes)

1. **Reduced water model.** The MD film is a single-site Lennard-Jones +
   Coulomb surrogate, not a validated water model: it has no explicit hydrogens,
   no proton disorder and no experimentally calibrated melting point. The
   `melting_scan` order parameter shows no sharp melting transition of the free
   film between 150 K and 500 K, i.e. the surrogate is solid-like over the whole
   physical range. Consequently the MD temperature *sensitivity* is weaker than
   the literature law (which produces a factor ~2 change in `mu` over
   -5 ... -25 degC), and the cross-scale check is anchored by the single-point
   calibration at the reference temperature. This is a property of the reduced
   model, not of the calibration procedure, and it is the main reason any
   residual cross-scale disagreement exists.
2. **Both scale levels share the interfacial shear law.** In the DEM the local
   ice/steel contact friction is exactly `mu_c(T)`; the DEM contributes the
   mechanical conversion (normal-load distribution, ploughing, decomposition)
   and its statistical error bar, not an independent temperature law. The DEM
   measurement therefore *verifies the mechanical bookkeeping* (the ratio
   reproduces the imposed law within a few percent, see `fig4`) rather than
   re-deriving the law from first principles.
3. **Absolute level of the resistance formula is not a validation target.**
   The Lindqvist-type coefficient set is used to expose the friction share and
   the temperature sensitivity; the published MT Uikku comparison in
   `benchmarks/dem3d/ship_ice_validation/README.md` explicitly requires
   geometrically and dimensionally consistent ship/ice data plus explicit
   friction and buoyancy treatment before absolute resistances may be scored.
   The endpoint therefore reports `WITHHELD` for absolute resistance accuracy.
   The reported `delta R` is a model-internal sensitivity (friction share times
   the `mu` ratio), which is exactly what the acceptance criterion asks for.
4. **2-D DEM.** The sliding bed is two-dimensional with a per-unit-thickness
   interpretation; ploughing is captured through the bump geometry in the
   sliding plane, but out-of-plane debris escape and third-body layers are not
   represented.
5. **Soft-particle DEM.** Contact stiffness is reduced by roughly two orders of
   magnitude relative to the elastic modulus of ice (standard soft-particle
   practice). The stiffness enters the contact overlap and the numerics, not the
   force ratio, and the measured ratio reproduces the imposed contact law
   (`fig4`), but a stiffness sensitivity study with a production-scale DEM is
   still owed.
6. **Bonds do not fail** in the reported runs (`bond_failure = False` by
   default; with failure enabled the interface layer debonds and the drag
   measurement degenerates, because a free debris layer co-moves with the plate
   and carries no slip). Ice fracture at the sliding interface is therefore not
   resolved, and the ploughing contribution is a lower bound.
7. **MD thermostat bias.** The driven layer dissipates energy, so the measured
   kinetic temperature exceeds the thermostat set point by a few tens of kelvin
   in some runs; the values are reported per temperature in the generated
   results document, and no attempt is made to correct the friction values for
   that bias.

## 6. References

* Bowden, F.P. & Tabor, D. (1954) *The Friction and Lubrication of Solids*, Oxford.
* Oksanen, P. & Keinonen, J. (1982) The mechanism of friction of ice, *Wear* 78, 315-324.
* Lindqvist, G. (1989) A straightforward method for calculation of ice resistance of ships, *POAC*.
* Riska, K. (1997) On the mechanics of the ramming interaction between a ship and a massive ice floe, VTT.
* Hu, J. & Zhou, L. (2016) Further study on level ice resistance and channel resistance for an icebreaking vessel, *IJNAOE* 8(2), 169-176 (MT Uikku reference set used as context in `benchmarks/dem3d/ship_ice_validation/`).
* Allen, M.P. & Tildesley, D.J. (2017) *Computer Simulation of Liquids*, 2nd ed. (Langevin thermostat, LJ force conventions).