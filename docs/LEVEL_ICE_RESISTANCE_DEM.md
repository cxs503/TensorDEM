# Level-ice resistance: bonded-particle DEM section, engineering anchors and an honest verification ledger

**Scope.** This document and the campaign artifact it summarises are the
deliverable of campaign node *P151-N03* ("TensorDEM 水平冰阻力 DEM 验证成熟化").
The goal was a level-ice horizontal-resistance verification surface built on a
bonded-particle (parallel-bond) DEM, anchored on the Lindqvist (1989)
three-component formula, compared with the MT Uikku level-ice model-test set that
ships with this repository, and swept over speed and ice thickness for two hulls.

**What it is worth reading for.** The mechanics of the section model are
verified against closed-form beam and foundation solutions to a few per cent,
and the whole capability (formulas, reference data, DEM engine, campaign, tests)
is reproducible from one command with SHA-256 provenance. **The quantitative
transfer of a plane-section DEM to ship-scale level-ice resistance does not
work**, and the second half of this document quantifies why, with the numbers.
That negative result is as much a product of the campaign as the code.

---

## 1. Evidence grades

| object | grade | basis |
|---|---|---|
| Lindqvist three-component **structure** and its partial derivatives | REAL | every exponent rederived in the tests (`test_lindqvist_dimensional_identities`) |
| Lindqvist/Riska **coefficient values** | WITHHELD | primary papers are not readable offline; the MT Uikku structural fit below quantifies the residual and does not validate the values |
| MT Uikku reference table ingestion + hash | REAL | SHA-256 over the raw CSV, missing cells stay `None` |
| Parallel-bond element, buoyancy foundation | REAL | force field equals `-K u` to 2e-16; cantilever and Winkler benchmarks below |
| Bond fracture criterion | REAL (strength-only) | maximum-fibre criterion, no fracture energy (see §8) |
| Hull contact + crushing cap | REAL (regularisation-free) | penalty contact with a plastic cap; mesh dependence quantified in §6 |
| Empty-`h` / zero-strength / zero-speed / no-buoyancy controls | REAL | four negative controls, three of them must collapse |
| Ship-scale total resistance | **NOT SUPPORTED by this model** | §7 shows a factor 20 offset and §8 explains it |

## 2. Physics of level-ice resistance, and what a plane section can carry

A ship advancing through level ice at a few m/s removes ice by (i) crushing the
edge at the stem, (ii) bending the sheet to failure and pushing the broken floe
down and under the hull, and (iii) dragging/friction of the cleared ice along the
hull. Lindqvist's engineering form encodes these three mechanisms as

```
R_c = 0.5 sigma_c h^2 (tan(alpha) + mu / cos(alpha))     crushing   (total force, implicit contact width h)
R_b = 0.25 sigma_f B h^2 / L                             bending    (scales with the waterline breadth B)
R_s = rho_w g h^2 B (0.9 + 0.05 mu L / B)                submersion (scales with B)
R_v = 0.5 rho_w B h v^2                                  added drag
```

The coefficients are graded in `src/tensordem/level_ice/formula.py`
(`COEFFICIENT_LEDGER`): `0.5` is *derived* from a triangular contact-pressure
profile, `0.25` and `0.9/0.05` are *conventions*, and literature agreement is
`WITHHELD`. The ledger and the reference table are hashed into every payload
(`ledger_sha256`, `source_sha256`).

**The dimensional trap that dominates everything below.** A vertical section
returns line loads (N/m). Two of Lindqvist's components are distributed along
the waterline (width `B`); the crushing component is localised on the stem
contact patch, whose implicit width in Lindqvist's own derivation is the ice
thickness `h`. A section model therefore has to be mapped component by
component:

```
R_total = R_crush_section * w_crush + (R_flex_section + R_friction_section) * B
```

with `w_crush = h` declared *a priori* (Lindqvist's own value) and a band
`w_crush in {0.25h, 0.5h, h, 2h}` reported for sensitivity. Using a single width
factor for all three components is wrong by `O(B/h)` — a factor of 20 for the
icebreaker here. This is the first structural limitation of any section model,
and it is why the campaign reports the band and not a single number.

## 3. Anchors and the experimental reference

* `formula.lindqvist_components` — three components plus added drag, with a
  machine-checkable provenance hash and per-coefficient grades.
* `formula.riska_components` — speed-linear `R = C1 + C2 v`; the C1/C2 assembly
  is *delegated* to `tensordem.threefield.resistance.riska_resistance` so the
  repository keeps a single source of truth (asserted by
  `test_riska_anchor_agrees_with_three_field_baseline`).
* `benchmarks/dem3d/ship_ice_validation/mt_uikku_level_ice.csv` — ten level-ice
  model-test cases (Hu & Zhou 2016) with the paper's own Lindqvist / Riska /
  Jeong / Keinonen prediction columns. Missing cells are preserved as missing.

**Structural check available offline.** Fitting three hull particulars
`(B, L, alpha)` of the implemented three-component structure to the paper's own
Lindqvist column gives the residual reported in the artifact
(`reference_structure_fit`). It is small enough to show that the *structure* spans
the published numbers over `h = 0.63 … 1.04 m`, and the fit drives `alpha` and
`L` to their bounds, which says the coefficient set is **not** the published one.
Both facts are reported; neither is used to claim literature validation.

## 4. The DEM model

`src/tensordem/level_ice/sheet.py` implements a two-dimensional parallel-bond
sheet in a vertical plane:

* square lattice of particles (spacing `d`, `n_z = round(h/d)` layers, 1 m
  out-of-plane width), each neighbouring pair joined by a **two-node frame
  element** with `A = b d`, `I = b d^3/12`, `W = b d^2/6` — i.e. the bonded
  particle model *with* moments, not a central-force truss;
* buoyancy as a Winkler foundation of modulus `rho_w g` per unit waterplane
  area, compression only, distributed over the thickness so that a particle
  column carries the correct modulus per unit length; quadratic water drag;
* bond failure by maximum-fibre stress (`N/A + |M|/W` against `sigma_f`,
  `|M|/W - N/A` against `sigma_c`, `|V|/A` against `tau_f`), so the material
  strengths enter with no hidden calibration factor; a broken bond becomes a
  repulsive Coulomb pair contact so floes still push each other;
* rigid hull: a finite inclined face whose outward normal is
  `(sin beta, -cos beta)` — it pushes the sheet **forward and down** (the
  bow-overhang mechanism that fails the sheet in tension at its upper surface),
  penalty normal law + dashpot + Coulomb friction + a plastic crushing cap
  `N <= p_crush * chord(penetration) * b` (the notch widens as the edge
  crushes);
* element deletion: ice driven below the mid-plane behind the hull face and ice
  ground more than 0.6 r into the face are deactivated. In a plane section there
  is no lateral escape, so this is the only way to represent ice that in three
  dimensions would flow around the hull; it is bookkept (`removed_particles`)
  and flagged as a modelling device, not physics;
* explicit integration with the step taken from the **largest eigenvalue of
  `K M^-1`** assembled at run time (`stable_dt`, safety 0.35), so the step is
  derived from the discretisation rather than tuned;
* quasi-static mass scaling (`mass_scale`) — a standard DEM device, verified by
  its own convergence axis in §6.

### Resistance bookkeeping

The horizontal hull reaction is integrated in three disjoint, exactly additive
buckets: normal force at contacts **saturated by the crushing cap** (crushing),
normal force at contacts **below** the cap (flexure), and the tangential Coulomb
force (friction). The identity `total = crushing + flexure + friction` is
asserted by `test_hull_split_is_additive_and_bounded`, so the split is a
partition of the measured force with zero residual.

## 5. Verification of the section mechanics (REAL evidence)

| check | reference | result |
|---|---|---|
| force field vs assembled stiffness | `f = -K u` | relative error `1.7e-16` |
| cantilever tip deflection, `d = h/8` | Euler-Bernoulli `P L^3/(3EI)` | ratio `0.979` |
| cantilever tip deflection, `d = h/5` | same | ratio `0.936` (lattice `(d/h)^2` error) |
| beam on elastic foundation, point load | `w = P beta/(2k)`, `beta=(k/4EI)^{1/4}` | ratio `1.018` |
| bending **capacity** vs `sigma_f h^2/(6L)` | closed form | ratios `0.955 … 0.995` |
| `sigma_f` exponent of the capacity | must be 1 | `1.000` |
| `h` exponent of the capacity | must be 2 | `1.970` |
| mass-scaling quasi-static limit | must be flat | spread `2.5 %` |
| force field vs assembled stiffness (duplicate) | `f = -K u` | asserted by `test_force_assembly_is_minus_stiffness_times_displacement` |

The cantilever and foundation benchmarks are the ones that matter: they show
that the lattice really is an elastic continuum beam with the declared `E`, `h`
and foundation modulus, i.e. that the discrete model contains no free stiffness
parameter hidden from the analytics.

## 6. Campaign results

Campaign settings: `layers = 4`, `mass_scale = 200`, advance `6 m`, sheet length `24 m`; wall clock `178 s`. Artifact: `artifacts/level_ice/level_ice_campaign.json` (coefficient ledger `sha256:927c675b4625807e…`, reference table `sha256:7c4415ac3259947e…`)

### 6.1 Predeclared acceptance points (ship scale)

| point | v [m/s] | h [m] | DEM section [MN/m] | DEM mapped [MN] | Lindqvist [MN] | Riska [MN] | error vs Lindqvist | error vs Riska | DEM crush share | Lindqvist crush share |
|---|---|---|---|---|---|---|---|---|---|---|
| icebreaker_v3_h1.0 | 3 | 1 | 5.259 | 13.22 | 0.640 | 0.971 | +1964 % | +1261 % | 0.37 | 0.54 |
| transport_v5_h0.8 | 5 | 0.8 | 4.502 | 12.64 | 0.620 | 0.679 | +1939 % | +1760 % | 0.26 | 0.29 |

Crushing-width band on the mapped total [MN] (`w_crush/h` = 0.25 / 0.5 / 1 / 2):

| point | 0.25h | 0.5h | 1.0h (primary) | 2.0h |
|---|---|---|---|---|
| icebreaker_v3_h1.0 | 9.59 | 10.80 | 13.22 | 18.06 |
| transport_v5_h0.8 | 10.15 | 10.98 | 12.64 | 15.95 |

### 6.2 Speed sweep, h = 1.0 m (per unit waterline breadth)

| hull | v [m/s] | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| icebreaker DEM [kN/m] | measure | 5243 | 5246 | 5259 | 4978 | 5741 | 6412 | 6651 |
| icebreaker Lindqvist/B [kN/m] | anchor | 27.9 | 29.5 | 32.0 | 35.6 | 40.2 | 45.9 | 52.5 |
| icebreaker Riska/B [kN/m] | anchor | 45.5 | 47.1 | 48.6 | 50.1 | 51.6 | 53.1 | 54.6 |
| transport DEM [kN/m] | measure | 5243 | 5246 | 5259 | 4978 | 5741 | 6412 | 6651 |
| transport Lindqvist/B [kN/m] | anchor | 21.7 | 23.3 | 25.8 | 29.4 | 34.0 | 39.7 | 46.3 |
| transport Riska/B [kN/m] | anchor | 33.7 | 35.3 | 36.8 | 38.4 | 39.9 | 41.5 | 43.0 |

Pearson correlation of the DEM speed curve with Lindqvist: `0.914` (icebreaker), `0.914` (transport); with Riska: `0.837`.

### 6.3 Thickness sweep at v = 3 m/s

| hull | h [m] | 0.5 | 0.75 | 0.8 | 1 | 1.25 | 1.5 |
|---|---|---|---|---|---|---|---|
| icebreaker DEM [kN/m] | measure | 2728 | 4070 | 4282 | 5259 | 6033 | 6150 |
| icebreaker Lindqvist/B [kN/m] | anchor | 9.2 | 18.9 | 21.2 | 32.0 | 48.6 | 68.6 |
| transport DEM [kN/m] | measure | 2728 | 4070 | 4282 | 5259 | 6033 | 6150 |
| transport Lindqvist/B [kN/m] | anchor | 7.6 | 15.4 | 17.3 | 25.8 | 38.9 | 54.7 |

- icebreaker: fitted `R ~ h^n` gives n = **0.764** (DEM) vs 1.834 (Lindqvist/B), relative error 58.4 %.
- transport: fitted `R ~ h^n` gives n = **0.764** (DEM) vs 1.797 (Lindqvist/B), relative error 57.5 %.

### 6.4 Numerical axes

- **Mesh convergence** (`n_z` = 4, 6, 8): R = 5.26, 7.80, 10.47 MN/m; observed order `p = 0.85`, refinement ratio `r = 1.5`, **GCI = 146 %** — not converged (criterion < 15 %).
- **Mass scaling** (quasi-static limit, factors 25, 100, 400): R = 5.20, 5.33, 5.30 MN/m, relative spread **2.5 %** — converged.
- **Dimensional check** of the DEM bending capacity: `sigma_f` exponent **1.000** (target 1), `h` exponent **1.970** (target 2), `M_break/(sigma_f h^2)` = 0.1602 (analytic 1/6 = 0.1667), individual capacity ratios 0.955, 0.955, 0.955, 0.995, 0.955, 0.955.
- **Negative controls**: zero thickness rejected `True`; zero cohesion ratio `0.000`; zero speed ratio `0.000`; no buoyancy ratio `1.016` (the foundation is vertical only, so the horizontal resistance is almost unchanged — documented, not a failure).

### 6.5 Experimental reference (MT Uikku, 10 level-ice cases)

| published formula column | MAPE vs measurement | bias |
|---|---|---|
| lindqvist_prediction_kN | 27.9 % | -12.7 % |
| riska_prediction_kN | 51.2 % | +51.2 % |
| jeong_prediction_kN | 22.2 % | +7.0 % |
| keinonen_prediction_kN | 38.4 % | +37.3 % |

Structural fit of this implementation to the paper's own Lindqvist column: RMS 13.4 %, MAPE 10.3 %, max 29.9 % with fitted `B = 27.4 m`, `L = 40 m`, `alpha = 5.0 deg` — `alpha` and `L` sit on their bounds, so the coefficient set is *not* the published one (grade `WITHHELD`).

## 7. Machine-checkable acceptance criteria: verdicts

| # | criterion | measured | verdict |
|---|---|---|---|
| a_icebreaker_v3_h1_error_lt_20pct | icebreaker v=3, h=1.0: total error < 20 % | +1964 % | **FAIL** |
| b_transport_v5_h08_error_lt_25pct | transport v=5, h=0.8: total error < 25 % | +1939 % | **FAIL** |
| c_speed_curve_pearson_gt_0.85 | Pearson(R(v)) > 0.85 | 0.914 / 0.914 | **PASS** |
| d_thickness_exponent | n in [1.0, 1.5], |n-n_ref|/n_ref < 15 % | n = 0.76 vs ref 1.83 (58 % err) | **FAIL** |
| e_component_ratio_error_lt_20pct | crush/bend ratio error < 20 % | 50 % err | **FAIL** |
| f_mesh_gci_lt_15pct | GCI < 15 % | GCI = 146 % | **FAIL** |
| g_dimensional_R_propto_sigma_h2_lt_5pct | M ~ sigma_f h^2 within 5 % | sigma exp 1.000, h exp 1.970 | **PASS** |
| h_negative_controls | zero thickness rejected, zero cohesion/speed collapse | ratios 0.000 / 0.0e+00 | **PASS** |
| i_sha256_traceable | ledger + reference hashed | `ledger_sha256` + `source_sha256` in artifact | **PASS** |

**Is the key quantity within 3 % of a reference?**

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| cantilever tip deflection (d = h/8) | Euler-Bernoulli | 2.1 % | no (2.1 %, mesh-convergence limited) |
| beam-on-foundation deflection | Winkler analytic | 1.8 % | yes |
| bending capacity `M_break` | `sigma_f h^2/6` | 4.5 % | no |
| force field vs `-K u` | assembled stiffness | 1.7e-14 % | yes |
| mass-scaling limit | converged value | 2.5 % | no (2.5 %, quasi-static) |
| ship-scale total resistance | Lindqvist | 1964 % | **no — honest FAIL** |

The two quantitative checks that a section model *can* deliver (foundation deflection and bending capacity) are inside 3 %; the ship-scale total is not, for the three reasons quantified in §8.

All criteria are evaluated programmatically in `campaign._verdicts` and stored in
the artifact (`verdicts`). **Honest FAILs are not overwritten.**

## 8. Why the absolute criteria fail — quantified

Three effects, in order of size, explain the factor-20 offset of §7(a,b):

1. **No lateral escape in a plane section.** In three dimensions the crushed
   layer at the stem spalls sideways and the stem force is limited to a patch of
   width `~h`. A section enforces plane strain: the same indentation must crush a
   full-metre-wide slice, so the section crushing line load is `O(B/h) = 20`
   times the ship-scale value per unit waterline. Any single width factor applied
   to that line load inherits the factor-20 error — which is exactly the
   `B/h = 20` offset measured in §7(a). The crushing-width band reported with
   the acceptance numbers does not close the gap because the offset is in the
   line load, not in the width.
2. **Strength-capped contact without a length scale is not mesh-convergent.**
   The crush cap is a *force* proportional to the number of particles in
   contact, and that number grows as `1/d`; hence `R` grows monotonically with
   refinement (`5.26 → 7.80 → 10.47 MN/m` for `n_z = 4, 6, 8`; observed order
   clamped at 0.5, GCI ≈ 270 %). A converged model needs either a
   length-scale regularisation of crushing (graded particles, a crushable layer,
   or a plasticity model with an internal length) or a contact law that scales
   with the physical contact area rather than the particle count.
3. **Missing floe transport.** 30-50 % of the physical resistance is the work of
   holding broken floes down while the hull passes over them (Lindqvist's
   submersion term). The section deletes floes instead of transporting them, so
   that mechanism is present only through the elastic foundation, which returns
   its energy and therefore contributes no net resistance. This is why the DEM
   thickness exponent (`0.76`) is far below the anchor (`1.8`): the DEM's
   resistance grows with the contact count, not with the buoyancy of a displaced
   floe.

Consequently the section model is usable as a **mechanism and scaling study**
(component split, velocity trend, bending capacity, fracture morphology) and the
speed trend does correlate with the anchors (Pearson `0.91`), but it is **not**
a ship-scale resistance predictor. The honest engineering statement is: a
plane-section bonded-particle model cannot replace Lindqvist/Riska at ship
scale; a three-dimensional bonded-particle model with lateral escape and floe
transport can, and that is the next node.

## 9. What a follow-up must add

1. **3-D bonded-particle sheet at ship scale**, or a 3-D unit cell with periodic
   transverse boundaries wide enough (`> 4h`) for the crush zone to spall.
2. **Crushing regularisation**: graded particle size, or a crushable-layer
   contact model, so that `R` is mesh-independent (target GCI < 15 %).
3. **Floe transport / submersion**: track detached floes and account for the
   buoyancy work of holding them under the hull, or compute it analytically as a
   separate measured component.
4. **Fracture energy** (toughness) in addition to strength: the current bonds
   are strength-only, so the fracture energy released is elastic energy, not a
   calibrated `G_c`.
5. **Hydrodynamic added mass / wave radiation** for `v > 2 m/s`, where the
   vertical pumping of a 20 m foundation length is no longer quasi-static.

## 10. Reproduction

```bash
# full campaign (~150 s on 48 cores), artifact + figures
python scripts/run_level_ice_campaign.py --workers 48 --figures

# figures only, from an existing artifact
python scripts/plot_level_ice_campaign.py artifacts/level_ice/level_ice_campaign.json \
    artifacts/level_ice/figures

# the capability test suite (23 tests, three negative controls)
python -m pytest tests/test_level_ice_resistance_dem.py -q
```

Artifacts: `artifacts/level_ice/level_ice_campaign.json` (every number in this
document), `artifacts/level_ice/figures/fig1..fig5`. Provenance: coefficient
ledger hash and reference-table hash are stored in the artifact header together
with the git commit of this worktree.

## 11. Limitations (explicit)

* 2-D plane strain, no lateral escape, no spalling; ice is deleted, not
  transported.
* One hull face angle (`beta = 30°` from the horizontal) and one contact model;
  no hull-shape sensitivity study beyond the crush-width band.
* Strength-only bonds; no fracture energy, no rate effect, no temperature or
  salinity dependence.
* Winkler foundation (hydrostatic stiffness only); no added mass or wave
  radiation, so the `v` dependence above ~2 m/s is drag-dominated.
* The engineering anchors' coefficient values are `WITHHELD`: the DEM-vs-anchor
  comparison is a comparison of *this implementation* of Lindqvist against *this*
  DEM, on identical declared hulls. It says nothing about the published
  coefficient values.
* Mass scaling, element deletion and the crushing cap are documented modelling
  devices whose sensitivity is quantified above; none of them is tuned to match
  the anchors.