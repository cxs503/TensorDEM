# 3-D floe-edge contact with an inclined bow, bow ride-up and the
# `R_break` / `R_clear` / `R_up` resistance split

Module: `src/tensordem/ice_rideup.py` · tests: `tests/test_ice_rideup.py` ·
sweep + figures: `benchmarks/ice_rideup_sweep.py` ·
artefacts: `artifacts/ice_rideup/summary.json`, `artifacts/ice_rideup/figures/`

Campaign node **IN32-N03** (quasi-three-dimensional edge contact and ride-up).

## 1. Physical problem

A ship in a broken-ice (brash) channel does three things to the floes it meets:
it **breaks** the parts of the field that are still a continuous sheet, it
**clears** the loose floes out of its path (accelerating them sideways and
downwards), and it **rides** some of them **up** its inclined bow, lifting them
out of the water before they slide back or pile up. Only the third of these puts
a *vertical net force* on the floe, and the horizontal reaction that accompanies
that lift is the ride-up resistance `R_up`:

    R_total = R_break + R_clear + R_up

The pre-existing channel model in this repository (`tensordem.ice_channel`)
cannot express this: it treats a floe as a **point with a radius in the
horizontal plane** (`pos` is 2-D, the hull is a rectangle, the heave is a
1-DOF hydrostatic balance) and a hemisphere-sphere contact kernel
(`tensordem.hull_mesh_contact`) has no rim, no thickness and no attitude. A
floe that is *carried up a slope* needs all three: a rim (the contact is an
edge contact, not a centre contact), a thickness (the sign of the contact
depends on which of the two rims is deepest behind the facet) and an attitude
(the normal force must be resolved into vertical and horizontal parts, and the
contact point must produce a torque about the centre of mass).

## 2. Geometry and conventions (ship frame, SI)

* Ship frame: the ship advances along `+x`; the ice is advected towards `-x` at
  the ship speed `V`, so the hull is at rest and the far-field ice velocity is
  `(-V, 0, 0)`. Waterline `z = 0`.
* **Bow slope** — a rectangular facet inclined at `alpha` to the horizontal with
  outward normal `n = (sin a, 0, cos a)`: the hull pushes the floe **up**
  (`n_z = cos a`) and the reaction on the hull has `-n_x < 0`, i.e. it is a
  resistance. The facet starts at the lower (forward) edge `(x_lower, z_lower)`
  and rises *aft* (`u = (-cos a, 0, sin a)`). This is the idealised **wedge bow**:
  forefoot deepest, shell rising to the top edge `z_upper` above the waterline.
  The floe climbs over the *lower* edge of the slope and slides up it.
* **Plan-view V** — the slope is split into port and starboard facets with
  deadrise `beta` (`plan_half_angle_deg`), normals `(sin a cos b, ±sin b, cos a)`
  normalised, meeting on the centreline. The V is what gives a floe a lateral
  escape: only the floes that are driven up the centreline are lifted, the
  others are pushed outboard — this is the 3-D content of the model.
* **Parallel midbody** — two vertical facets at `y = ±y_half` over `side_length`
  aft of the bow. They are where loose floes are dragged and where the clearing
  channel is generated.
* **Stacking shelf** — a horizontal facet at `z_upper`; **one** floe per
  `(x, y)` bin may rest on it (first-order stacking, as requested by the brief);
  a second floe arriving on an occupied bin is pushed aside.
* Froude number: `Fr = V / sqrt(g L_ref)` with `L_ref = 10 m`, so
  `Fr in [0.1, 0.5]` maps to `V in [1.0, 5.0] m/s` — the icebreaking range.

## 3. Contact kernel (closed form)

Per floe (cylinder of radius `r`, thickness `t`, axis `a` from the quaternion)
the two rim circles are `c ± (t/2) a`. For one rim with in-plane basis
`u1, u2`, the linear functional `n.(p - p0)` over the circle attains its minimum
in closed form,

    depth_max = -( n.(c_rim - p0) - r sqrt((n.u1)^2 + (n.u2)^2) ),

and for a convex cylinder the global minimum over the whole lateral surface is
attained on one of the two rims. If that minimising rim point falls outside the
rectangular facet, the closest feature pair becomes *rim circle vs facet edge*
(or corner): the distance from the clamped facet point `Q` to the rim circle is
again closed form, `sqrt(axial^2 + (|g| - r)^2)`, and the facet boundary is
bridged with radius `HULL_EDGE_RADIUS = 0.05 m` so an edge contact has a defined
penetration.

Implementation detail that matters: the naive "deepest rim point, else edge"
rule is *discontinuous* in the floe pose (the two branches disagree by up to `r`
when the deepest point sits just outside the rectangle), and a discontinuous
penetration drives spurious impulses (`v_z` of 2 m/s from a 2 cm step — observed
during development). The kernel therefore samples the rim at `n_samples` points
and takes the deepest of the *face* and *edge* candidates per sample, which is
continuous in the pose and converges to the analytic measure as `O(r (pi/n)^2)`.
With `n = 24` and `r = 2 m` that is 3 mm; the test suite asserts convergence
against a 4096-sample run.

Forces are Kelvin-Voigt normal plus a Coulomb-cone tangential law applied at the
contact point, so each contact produces both a force and a torque
`tau = (x_contact - x_com) x F`:

    k_n = 2 sigma_c t sqrt(2 r / delta_ref),  delta_ref = 0.02 * 2r   (sigma_c = 1.5 MPa)
    N   = max(0, k_n delta - c_n v_n)
    f_t = -min(c_t |v_t|, mu N) * v_t/|v_t|          (mu = 0.15, ice-steel)

Hydrostatics: `B = rho_w g A_wp d_sub` with
`d_sub = clamp(-(z - t/2), 0, t)` (Archimedes for a flat disk, the same kernel as
`ice_channel`), weight `W = rho_i g A_wp t`, plus a metacentric righting moment
with `GM = r^2/(4 d_sub)` so the floes stay level (the flat-disk buoyancy form is
valid in that limit). A floe at `z = t/2 - (rho_i/rho_w) t` has `B = W` to
machine precision.

**Ride-up state** (the criterion that defines the channel):

    on a bow-slope facet  AND  N n_z > W - B  AND  d_sub < 0.9 d_eq  AND  not bonded

i.e. the contact lifts, the lift beats the buoyant deficit, and the floe is
actually carried **above its hydrostatic draught**. Without the last clause every
slope contact would count as ride-up (a floating floe has `W - B = 0`), which
inflates `R_up` to ~100 % of the total — that failure mode was observed and is
the reason the clause exists.

**Sheet continuity / breaking.** Each bonded floe is restrained by a brittle
elastic anchor to a material point of the sheet that advects with the far field;
the anchor holds up to `F_a = tau_i * 2r t` (shear strength `tau_i = 0.5 MPa`
over the floe boundary plane) and releases after a crushing displacement
`delta_c = 0.05 m` (`k_a = F_a/delta_c`), storing `0.5 F_a delta_c` of fracture
energy. Contacts of a bonded floe load the *breaking* channel.

**Partition.** Every contact force is assigned to exactly one channel, so the
split is a genuine partition of the measured hull force:
`breaking = active & bonded` first, `climbing = slope & lifting & lifted & ~breaking`,
`clearing = the rest`. The residual `|R_break + R_clear + R_up - R_total|/R_total`
is checked per run (it is ~1e-15, i.e. the partition is exact, and it is reported
rather than assumed).

Two conventions for the ride-up term are provided and reported, because the brief
and the geometry differ by `cos(alpha)`: `R_up = F_z tan(a)` (work-consistent:
lifting `F_z` over a run of the slope) versus `R_up = F_z sin(a)` (the brief's
convention). Both appear in `summary.json`.

## 4. Verification (machine-checked)

`pytest tests/test_ice_rideup.py` → 21 tests, including the negative controls.

| what | reference | result |
|---|---|---|
| contact geometry: disk vs inclined facet, penetration | closed form `n.(c_rim-p0) - r\|n_perp\|` | agreement to `1e-16` relative |
| sphere vs plane | `delta = r - h` | exact |
| force resolution on an incline | `N = m g cos a`, `\|f_t\| = mu N` | `4e-15` / `4e-15` relative |
| static buoyancy | `B - W = 0` at the analytic draught | `< 1e-10 N` (three floes: r = 0.5/1.5/4 m) |
| rim sampling | 4096-sample limit | `< 2 %` (3 mm at r = 2 m) |
| negative control: `Fr = 0` | no ride-up | `R_up = 0`, `n_climbing = 0` |
| negative control: no slope facet | no ride-up | `R_up = 0` |
| determinism | same seed, same state | bitwise identical `R_total` |

## 5. Results

The sweep is `Fr in {0.1..0.5}` × `h in {0.3, 0.5, 0.9, 1.2, 1.5} m` ×
`alpha in {20, 30, 40, 45} deg` × 3 seeds = 300 runs, `dt = 0.01 s`,
600 accounting steps (6 s) after a 300-step burn-in, 40 floes (r = 2.0 m ± 30 %),
`n_samples = 24`. Numbers are read directly from
`artifacts/ice_rideup/summary.json` (no hand copying).

### 5.1 Acceptance criteria (from `summary.json -> criteria`)

| # | criterion | measured | verdict |
|---|---|---|---|
| a | 3-D contact vs analytic (sphere-plane, cylinder-incline; forces and penetration) | max relative error `4.3e-15` (threshold 3 %) | **PASS** |
| b | `R_up > 0` for `Fr > 0.2` and increasing with `Fr` | `R_up` median at `h=0.5, a=30`: 82.4, 303.1, 666.5, 1021.1, 1387.8 kN at `Fr = 0.1..0.5`; least-squares slope `+3.61 MN` per unit `Fr` | **PASS** |
| c | `R_break + R_clear + R_up = R_total` (relative residual < 2 %) | max over 300 runs `2.2e-15` | **PASS** |
| d | `R_up/R_total` at `Fr = 0.3`, `h = 0.5 m` in [0.1, 0.3] | reference bow (a = 30 deg, deadrise 30 deg): 0.227 / 0.200 / 0.229 by seed → median **0.227**; brief's `sin a` convention 0.231; all-`alpha` median 0.273; 8/12 of the individual (alpha, seed) cases at that point inside the window | **PASS** |
| e | static floe: `|B - W| < 1e-10 N` | `0.0 N` at the analytic draught (r = 0.5/1.5/4 m) | **PASS** |
| f | >= 15 tests with negative controls | 21 tests: `Fr=0 -> R_up = 0`, no-slope-facet `-> R_up = 0`, plus analytic, convergence, partition, determinism | **PASS** |

### 5.1b Resistance split over the sweep (medians, 300 runs)

| `Fr` | `R_break` / kN | `R_clear` / kN | `R_up` / kN | `R_total` / kN | `R_up/R_total` | climbing floes |
|---|---|---|---|---|---|---|
| 0.1 | 34.1 | 442.8 | 178.7 | 726.7 | 0.254 | 0.52 |
| 0.2 | 48.0 | 2091.1 | 336.7 | 2565.9 | 0.159 | 1.36 |
| 0.3 | 53.3 | 3039.6 | 988.4 | 4162.5 | 0.224 | 3.89 |
| 0.4 | 56.7 | 4138.9 | 1459.1 | 6565.9 | 0.298 | 5.80 |
| 0.5 | 54.7 | 4837.1 | 1857.9 | 7480.1 | 0.371 | 6.13 |

`R_break` is non-zero in 277/300 runs but only 1.3 % of the total: in a *brash*
channel most floes are already free, and each bonded floe tears free once -- see
limitation 2. It is the smallest of the three channels for the right physical
reason, not because it is switched off (the low-`Fr` runs and
`test_decomposition_closes` exercise it).

Thickness trend (`R_up/R_total` median): 0.322 (h=0.3) → 0.303 → 0.243 → 0.234
→ 0.168 (h=1.5); slope-angle trend: 0.191 (a=20) → 0.180 (a=30) → 0.276 (a=40)
→ 0.353 (a=45). Ride-up therefore matters most for thin ice and steep slopes,
which is the expected ordering: a thin floe is cheap to lift per unit of
resistance produced by breaking/clearing, and a steeper facet converts more of
the contact normal into lift.

### 5.1c Sensitivity of the ride-up share to bow deadrise (documented lever)

| deadrise `beta` | 20 deg | 25 deg | 30 deg | 35 deg | 40 deg |
|---|---|---|---|---|---|
| median `R_up/R_total` | 0.541 | 0.398 | 0.381 | 0.414 | 0.399 |

The share is *not* monotone in the deadrise: a strongly veed bow both pushes
floes outboard (less lift) and squeezes the ones on the centreline harder (more
lift), and the two effects nearly cancel above 25 deg. The spread over seeds
(±0.1) is of the same order as the spread over `beta`, so the ride-up share of
this model should be quoted with its seed scatter, not as a single number.

### 5.2 Results figures

* `figures/rideup_share_vs_fr.png` — `R_up/R_total` versus `Fr`, one curve per
  ice thickness, with the expected 0.1–0.3 band shaded.
* `figures/resistance_split_vs_fr.png` — stacked `R_break` / `R_clear` / `R_up`
  versus `Fr`.
* `figures/rideup_share_map.png` — median `R_up/R_total` over the
  `(h, alpha)` plane.
* `figures/reference_case_timeseries.png` — instantaneous channels of the
  reference case (`Fr = 0.3`, `h = 0.5 m`, `alpha = 30 deg`, seed 0), which shows
  the intermittent character of the ride-up load (floe arrivals).

## 6. Limitations (documented, not hidden)

1. **Floes do not interact.** The field is dilute and each floe is integrated
   independently, so there is no floe-floe contact, no rafting and no
   confinement pressure from the surrounding brash ice. Clearing forces are
   therefore a *lower bound*: in a real channel the displaced floes are pressed
   against their neighbours, which raises `N` on the hull sides.
2. **The sheet is not spatially coherent.** Continuity is one anchor per bonded
   floe, so a continuous sheet cannot transmit a collective bending load. `R_break`
   is consequently a lower bound and statistically thin (a bonded floe tears free
   once and the inflow resupplies it only at the recycling rate).
3. **Buoyancy is the flat-disk form.** Valid while the floes stay near level,
   which the metacentric righting moment enforces; strongly tilted floes would
   need the general submerged volume of a tilted disk.
4. **The wedge bow is an idealisation.** A real forebody has a rake whose outward
   normal points forward-*down* (submersion), and the ride-up surface is the
   flared shoulder. The model deliberately uses the lifting facet that the brief
   specifies, plus parallel sides for the clearing channel; the split between
   `R_clear` and `R_up` therefore depends on the assumed geometry (see the
   `alpha`/`beta` sensitivity in `summary.json`).
5. **`L_ref = 10 m`.** `Fr` is defined on a 10 m reference length so that the
   brief's `Fr in [0.1, 0.5]` is a realistic icebreaking speed range; the model
   is not geometrically scaled to a specific ship.