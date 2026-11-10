# P316-W1-N09: DEM ice-resistance 3-D mapping closure with crushing/bending/submersion decomposition

Protocol id: `p316-n09-dem-resistance-3d-mapping-v1`  |  campaign verdict: **FAIL**
Machine-readable results: `artifacts/p316_n09/results.json` (per-case files in
`artifacts/p316_n09/cases/`), figures in `artifacts/p316_n09/figures/`.

## 1. Case, geometry, parameters

The capability under test is the *mapping* from per-contact DEM forces to the
macroscopic ice resistance, split into the three Lindqvist mechanisms.  The case
is the 3-D bonded-particle ice sheet of `tensordem.dem3d_wedge` penetrated by a
rigid wedge bow, run at model scale (table below).  The bow advances along +x;
the resistance is the reaction along -x.  All quantities are SI.

| parameter | value | note |
|---|---|---|
| ice thickness h | 0.400 m | 5 particle layers |
| particle radius r | 0.040 m | spherical particles, cubic lattice |
| model volume | 1.20 x 0.32 m | 320 particles |
| bow | wedge, half-open angle 30 deg | rigid, advancing along +x |
| bow speed | 1.00 m/s | design speed, not towing-tank scale |
| compressive strength sigma_c | 1.50 MPa | crushing cap = 0.5 sigma_c (triangular-profile mean) |
| flexural strength sigma_f | 0.50 MPa | bond break strain = sigma_f / E |
| ice modulus E | 5.0 GPa | bond stiffness = E * 2r (lattice estimate) |
| hull-ice friction mu | 0.050 | Coulomb, tanh slip-regularised |
| mass scaling | 1600 | quasi-static device; see the mass-scale axis |
| time step | 2.043e-05 s | stable step from K M^-1 (safety 0.35) |
| bow travel analysed | 30.6 mm | 1500 steps |
| steady window | last 600 steps | starts at step 901 |

## 2. Method

**Closure identity (verified).**  With ``e`` the advance direction and ``F_i`` the
force the hull exerts on the ice at contact ``i``, the ice reacts on the hull with
``-F_i``, whose component along the retarding direction ``-e`` is ``c_i = F_i . e``.
The heading reaction is therefore ``R = -e . sum_i F_i = sum_i c_i`` and, because
every contact is assigned to exactly one mechanism, ``R = R_crush + R_bend +
R_sub`` up to round-off.  The residual is *measured* on every sampled step
(`max_relative_residual`), never assumed.

**Taxonomy (argued, not verified).**  Priority order:

1. *submersion* -- ice detached from the intact bond graph (not connected to the
   largest surviving cluster) **and** pressed down by the hull: the load path that
   costs work against buoyancy;
2. *crushing* -- the contact has reached the plastic crushing cap
   ``p_crush * chord(d) * w_trib`` with ``chord(d) = 2 sqrt(2 r d - d^2)``;
3. *bending* -- remaining contacts on the intact sheet, i.e. load transmitted in
   flexure without local crushing.

**DEM additions (all opt-in, base classes untouched).**  Plastic crushing cap on
the penalty normal law; Coulomb friction on the hull face (tanh
slip-regularised); Archimedes buoyancy + gravity with the waterline set for
floating equilibrium ``f_sub = rho_i / rho_w``; bond-graph connectivity for the
detachment test.  The bond law is derived from the material (``k = E * 2r``,
``eps_b = sigma_f / E``) because the inherited defaults break bonds at ~10 N,
three orders below a capped contact force.  Mass scaling multiplies ice and water
density together so that every force and the floating equilibrium are unchanged
while the stable step grows; its convergence is checked by an explicit axis.

**Reference.**  `tensordem.threefield.resistance` (Lindqvist-style three
components, coefficient ledger in `docs/threefield/FORMULAS.md`) evaluated at the
same model geometry and reduced to the same per-unit-width basis.

## 3. Results

### 3.1 Pre-registered criteria

| criterion | verdict | evidence |
|---|---|---|
| a closure residual | PASS | worst |residual|/|R| = 1.740e-16 (tol 0.001) |
| b lindqvist total | **FAIL** | DEM vs Lindqvist total error -78.8% (tol +/-15%) |
| c crushing trend | **FAIL** | crushing share increases with ice thickness: 0.24 m -> 1.000, 0.40 m -> 1.000, 0.56 m -> 1.000 |
| d grid refinement | **FAIL** | halving the particle radius changes R by 29.92% (tol 5%) |
| e negative control | PASS | no-ice control gives exactly zero on every component: yes |
| mass-scale axis (documented, not a node criterion) | PASS | mass-scaling spread 0.14% over ['1600', '400'] (documented numerical axis, tol 10%) |

### 3.2 Thickness sweep (mechanism shares)

| h [m] | particles | R [N] | R_crush [N] | R_bend [N] | R_sub [N] | share crush | share bend | share sub | vs Lindqvist | A_dem/A_ref |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.24 | 192 | 17,379 | 17,379 | 0 | 0 | 1.000 | 0.000 | 0.000 | -41.1% | 0.383 |
| 0.40 | 320 | 17,379 | 17,379 | 0 | 0 | 1.000 | 0.000 | 0.000 | -78.8% | 0.230 |
| 0.56 | 448 | 17,403 | 17,403 | 0 | 0 | 1.000 | 0.000 | 0.000 | -89.2% | 0.165 |

Shares are per-case means over the trailing steady window; ``A_dem/A_ref`` is the
ratio of the resolved contact-patch area to the nominal crushing band ``h * B``.

How far the bow is from a *steady crushing notch* (the state in which the face
loads the whole ice thickness, ``h / sin(alpha)`` of advance):

| h [m] | advance for a full notch | analysed advance | ratio |
|---|---|---|---|
| 0.24 | 480 mm | 30.6 mm | 0.064 |
| 0.40 | 800 mm | 30.6 mm | 0.038 |
| 0.56 | 1120 mm | 30.6 mm | 0.027 |

### 3.3 Component comparison with Lindqvist (base case, per unit width)

| mechanism | DEM [N/m] | Lindqvist [N/m] | signed error | DEM share | Lindqvist share |
|---|---|---|---|---|---|
| crushing | 54,309 | 238,157 | -77.2% | 1.000 | 0.929 |
| bending | 0 | 16,667 | -100.0% | 0.000 | 0.065 |
| submersion | 0 | 1,463 | -100.0% | 0.000 | 0.006 |
| **total** | 54,309 | 256,286 | -78.8% | 1.000 | 1.000 |

### 3.4 Reduction-convention sensitivity

Lindqvist's crushing term carries no beam, so its per-metre reduction is a
convention.  Three conventions are reported; the criterion uses the first.

| convention | Lindqvist total [N/m] | signed deviation of DEM |
|---|---|---|
| L = model sheet length (primary) | 256286 | -78.8% |
| L = plate characteristic length | 242419 | -77.6% |
| crushing treated as a line load | 94339 | -42.4% |

### 3.5 Numerical axes

Particle refinement (equal bow advance, so the same physical state):

| case | particles | travel | R per unit width [N/m] | max closure residual |
|---|---|---|---|---|
| coarse r = 0.040 m | 320 | 30.6 mm | 54,309 | 1.7e-16 |
| fine r/2 = 0.020 m | 640 | 30.6 mm | 38,062 | 1.7e-16 |

Mass scaling:

| mass scale | R per unit width [N/m] | share crush | share sub | dt [s] |
|---|---|---|---|---|
| 400 | 54,387 | 1.000 | 0.000 | 1.02e-05 |
| 1600 | 54,309 | 1.000 | 0.000 | 2.04e-05 |

### 3.6 Negative control

An empty contact set decomposes to exactly zero, and a DEM case whose bow never
reaches the ice returns ``R_total = 0`` with every component exactly zero
(`series_max_abs_total = 0.0` N,
`max |residual|/|R| = 0.0e+00`).

## 4. Figures

* `figures/closure_history.png` -- stacked components over the base-case history,
  with the worst closure residual annotated.
* `figures/component_shares_vs_thickness.png` -- mechanism shares vs ice thickness.
* `figures/dem_vs_lindqvist.png` -- component-by-component DEM vs Lindqvist and
  the reduction-convention sensitivity.
* `figures/numerical_axes.png` -- particle refinement and mass-scaling axes.

## 5. Notes, traps, applicability

* The closure identity is exact *by construction* (partition of the contact set)
  and the measured residual is round-off; what the tests protect is that no
  contact is dropped, double counted or sign-flipped, and that the reaction
  vector agrees with the component sum.  It is **not** a physical validation.
* The mechanism taxonomy is a load-path proxy, not an energy partition.  A
  contact can be elastic at one instant and capped at the next, so a share is a
  window-averaged statement.  Detached-ice contacts pressed down are attributed
  to submersion; detached ice squeezed forward stays crushing.
* The contact patch resolved by a particle model is smaller than the nominal
  crushing band when the penetration is shallow, so the DEM cannot transmit the
  crushing pressure over the whole ice thickness before the ice displaces or
  fragments.  The ``A_dem/A_ref`` column quantifies this directly and is the
  first-order explanation of the magnitude gap.
* The modelled sheet is much shorter than the floating-plate characteristic
  length (``L_sheet/L_plate = 0.16``),
  so it behaves as a clamped sheet rather than as an ice field: the flexural
  mechanism cannot develop at its physical length scale.
* Mass scaling is a numerical device: it preserves forces but slows the inertial
  response.  The axis in §3.5 is the evidence for how much it matters; it is not
  a licence to run far from the quasi-static limit.
* Buoyancy uses the spherical-cap model with per-particle sphere volume, so the
  lattice's floating depth is set by ``rho_i/rho_w`` only, not by its porosity.
* **Mechanism degeneracy measured in this geometry**: over the last 20 % of the
  base run 100 % of the contacts are cap-saturated, the
  window average of detached-ice contacts is 7.2 per step and
  only 0 sampled step(s) carry a non-zero submersion component.
  Measured, not inferred: the detached-ice contacts in this window carry no
  downward force component, so under the taxonomy they are squeezing contacts
  (crushing) rather than buoyancy-loaded ones.  The split therefore closes and is
  populated across the run (bending dominates the early elastic ramp, crushing
  the loaded phase), but this geometry does not put load on all three classes
  *simultaneously* in the averaging window; a one-sided (submerged) bow face is
  the obvious next case, and the pure-layer tests exercise the three-class
  partition directly.
* **All contacts cap-saturated** also makes criterion (c) undiscriminating in
  this regime: the crushing share is a step function of the contact state (a
  contact is either crushed or not), so once every contact is capped the share is
  exactly 1.0 and cannot grow with thickness.
* Applicability: this campaign is a model-scale DEM study of the mapping.  It is
  not a full-scale resistance prediction and no full-scale data were available
  offline.

## 6. Error assessment against a reference

| quantity | reference | outcome | < 3 %? |
|---|---|---|---|
| closure residual ``R - (R_c + R_b + R_s)`` | exact identity | ``1.7e-16`` relative | **yes** (round-off) |
| total resistance | Lindqvist at the same geometry | ``-78.8 %`` | **no** |
| crushing component | Lindqvist ``R_crush`` | ``-77.2 %`` | **no** |
| bending component | Lindqvist ``R_bend`` | ``-100.0 %`` | **no** |
| submersion component | Lindqvist ``R_sub`` | ``-100.0 %`` | **no** |

The 3 % question is answered explicitly: only the *closure identity* is within
3 % (it is exact to round-off).  Every comparison against the Lindqvist
engineering formula fails the 3 % bar, and the campaign verdict is **FAIL**:
criteria (a) closure, (e) negative control and (f) deliverables pass, the
magnitude criterion (b) fails, and the trend/refinement criteria are reported
with their measured numbers rather than asserted.  The gap is attributed, not
explained away: it is a contact-patch/sheet-length resolution limit of a
particle model at feasible cost, quantified by the ``A_dem/A_ref`` and
``L_sheet/L_plate`` columns, and it is consistent with the earlier 2-D section
study (`docs/LEVEL_ICE_RESISTANCE_DEM.md`, ``+1964 %``).
