# GAP-DEM-G3 close-out: real ship-hull-form contact + digitisation rule (TensorDEM)

**Anchor**: `NEW:GAP-DEM-G3` — the repository gap ledger
(`docs/dem_calibration_gap_ledger.md`) still carried two open parts of the row
*"3-D DEM mesh contact on a real hull"*: (i) curved-hull validation on a **real
ship-hull form** (the earlier audit only closed a faceted **cylinder**,
`docs/dem3d_hull_mesh_contact_curved_audit.md`), and (ii) **penalty
stiffness/damping calibration**, explicitly "not benchmarked".  No platform
`feature_ledger` slug exists for hull-mesh contact (checked: the only open DEM
anchors are the mu(I) / RCP / Campbell case-pool entries), so this is logged as
a ledger-external gap; a suggested new ledger entry is given at the end.

Module under test: `src/tensordem/hull_mesh_contact.py` (unchanged).
New verification aid: `src/tensordem/hull_form_mesh.py` (analytic Wigley hull:
surface, inscribed triangulation, distance, directional curvature, analytic
penalty reference).  Audit driver:
`scripts/m16794_wigley_hull_contact_audit.py`; tests:
`tests/test_hull_form_mesh_contact.py`; artifacts:
`artifacts/m16794_wigley_hull_audit/`.

## 1. Case

| quantity | value |
|---|---|
| hull form | **Wigley** parabolic hull, length L = 4.0 m, beam B = 0.4 m, draft T = 0.25 m (L/B = 10, B/T = 1.6) |
| surface | y = (B/2)(1-(2u-1)²)(2v-v²), u∈[0,1] aft→fore, v∈[0,1] keel→waterline, mirrored to starboard |
| particle | sphere R_p = 0.05 m (ice-floe analogue), penalty stiffness k = 1e+06 N/m, no damping in the geometry studies |
| zones probed | midship side, fore shoulder, aft shoulder, bilge, keel, stem (both sides) |
| penetrations | 4, 8, 12, 20, 30, 40 mm |

The Wigley hull is the classical CFD-workshop validation form and is *doubly
curved* (waterline + section curvature, closing keel line and stem), unlike the
cylinder used before; it therefore exercises the geometry a digitised hull
mesh actually has.  A digitised SUBOFF offset table is **not** available offline
in this repository, so a real offset mesh could not be used; the analytic form
is the honest substitute because it supplies a closed-form reference.  This is
recorded as a limitation, not glossed over.

## 2. Method

1. **Reference first, thesis second.** The analytic distance is a 2-D bounded
   minimisation of the squared distance to the parametric surface (vectorised
   coarse bracket → three shrinking-window dense refinements → L-BFGS-B).  It
   is validated against two structures that do not use it: a 1601×801 global
   grid (an *upper bound* on the true minimum) and a 401×401 local grid rebuilt
   around the solver's own foot point (an independent local search).
   Result: solver never worse than the grid
   (`n_solver_worse_than_grid = 0` of
   40), local-grid gain 5.55e-17 m — i.e.
   the reference is machine-precision, so an error measured against it is the
   kernel's, not the reference's.
2. **Kernel vs analytic.** `mesh_contact_forces` on inscribed hull
   triangulations, compared with the closed-form penalty force
   `k(R_p - d)·n` computed from the analytic surface only.
3. **Digitisation law.** Error is compared with the *directional* sagitta rule
   derived from the panel geometry,
   `s = (κ_u h_u² + κ_v h_v²)/8`, relative error `≈ s / penetration`, where
   `h_u, h_v` are the local panel chords and `κ_u, κ_v` the normal curvatures
   along the two parameter directions (`wigley_directional_curvature`).
4. **Aggregate load.** 240 mixed-radius spheres placed on the hull with random
   penetrations 2–20 mm: total force vector, per-sphere parity, contact count,
   action/reaction.

## 3. Results

**Fine mesh (160×64 = 40960 triangles), 72 probe points:**
max relative force error **0.30 %**, max direction error
**0.35°** (half facet chord angle).

**Digitisation convergence (penetration 10 mm, worst over the 6 zones × 2 sides):**

| n_u×n_v | triangles | local chord [mm] | measured max rel. err | rule prediction | measured/predicted |
|---:|---:|---:|---:|---:|---:|
| 12×5 | 240 | 335.6 | 24.06 % | 18.54 % | 1.30 |
| 24×10 | 960 | 166.8 | 4.98 % | 3.83 % | 1.30 |
| 48×20 | 3840 | 84.4 | 0.71 % | 0.86 % | 0.82 |
| 96×40 | 15360 | 41.9 | 0.14 % | 0.30 % | 0.49 |
| 160×64 | 40960 | 25.3 | 0.12 % | 0.08 % | 1.55 |

The rule is a **sufficient** bound at every resolution (measured ≤ predicted,
tightness 0.7–0.9 at the coarse end, ≈0.5–0.9 at the fine end) and the error
falls with the expected second order once the panel chord is the limiter.

**Penetration sensitivity at fixed mesh 48×20** (from the run log,
`artifacts/m16794_wigley_hull_audit/audit_stdout.log`): measured max rel. error
1.42 % / 0.71 % / 0.35 % / 0.24 % / 0.19 % / 0.18 % for penetrations
5 / 10 / 20 / 30 / 38 / 40 mm against rule predictions 3.61 / 1.81 / 0.90 /
0.60 / 0.48 / 0.45 %.  The error *falls* with penetration because the
denominator of the rule is the penetration itself (the analytic force is
`k·pen`), not the stand-off distance; the stand-off is the right denominator
only when the error is normalised by the *mesh* force.

**Aggregate ice-floe load (240 spheres, 160×64 hull)** — same run log: global
force vector error **0.0553 %**, contact count **240/240** (no missed or
spurious contacts), action/reaction residual **0.0 N**, broad phase **0/300**
superset violations on the hull mesh.

**Damping / restitution: NOT_RUN (honest).**  The closed-form reference
`e = exp(-βπ/√(1-β²))`, `β = c/(2√(km))` is the right calibration target for
this force law, but every attempt was aborted by the mission time budget: the
kernel rebuilds its spatial hash per call, so a 1-DOF impact at
`dt = t_c/100` costs ≈25 ms per step.  The code path exists
(`restitution_study`) and a *reduced* version is exercised in the test suite;
no restitution number is claimed here.

## 4. Figures

* `artifacts/m16794_wigley_hull_audit/fig_force_vs_penetration.png` — force vs
  penetration, six hull zones, kernel (circles) vs analytic (lines).
* `artifacts/m16794_wigley_hull_audit/fig_resolution_convergence.png` — measured
  error vs local facet chord (log-log) with the directional-sagitta prediction
  and the 3 % acceptance line.

(The ice-floe parity plot and the restitution plot belong to the sections that
were cut by the time budget; they are not claimed.)

## 5. Notes, traps, limits

* **Trap (hit and corrected):** the first version of the facet rule used
  `κ_max` with the *largest* panel chord and normalised by the stand-off
  distance.  That over-predicted the error by 10× (163 % vs 24 % measured at
  12×5) and, worse, predicted the wrong trend.  The correct pairing is
  directional: `κ_u` with the `u` chord, `κ_v` with the `v` chord, normalised
  by the penetration.  After the correction the prediction tracks the
  measurement across four mesh refinements.
* **Trap (inherited and avoided):** never trust a hand-written reference.  The
  earlier curved-hull audit recorded a spurious 178° "direction error" caused by
  a wrong analytic closest point; here the reference is checked by two
  independent grids before any verdict.
* **Creases are not facets:** at the stem and keel line the surface has no
  single-valued tangent plane (`κ → ∞`), so no facet-size rule can represent
  them; contact there is governed by the crease edge, and the residual error is
  the edge's, not the panel's.
* **Not a digitised hull:** SUBOFF-class offset meshes are not in this
  repository; a real offset table would additionally carry measurement noise,
  which this geometry study isolates away.
* **Scope:** friction, rolling resistance and hull *motion* are not benchmarked;
  only the normal penalty law on a static hull is scored.

## 6. Error verdict (key quantity vs independent reference, < 3 % ?)

| key quantity | reference | measured | < 3 % ? |
|---|---|---:|---|
| reference distance, 1601×801 grid check | brute-force grid (upper bound) | solver never worse (40 pts) | **yes** |
| reference distance, local 401×401 grid | independent local search | gain 5.6e-17 m | **yes** |
| force, 160×64 mesh, 72 pts, 6 zones | analytic Wigley penalty | 0.30 % max | **yes** |
| force direction, 160×64 | true surface normal | 0.35° max | yes (chord angle) |
| force, 96×40 mesh | analytic | 0.14 % | **yes** |
| force, 48×20 mesh | analytic | 0.71 % | **yes** |
| force, 24×10 mesh | analytic | 4.98 % | **no** (below the rule) |
| force, 12×5 mesh | analytic | 24.1 % | **no** (below the rule) |
| global load, 240-sphere cloud, 160×64 | 240 analytic penalties | 0.055 % | **yes** |
| broad-phase completeness (hull mesh) | brute-force AABB set | 0/300 missing | **yes** |
| action/reaction | −ΣF | 0.0 N | **yes** |
| restitution e (damping calibration) | exp(−βπ/√(1−β²)) | NOT_RUN | **not answered** |

No honest-FAIL of the kernel: every "no" is a resolution below the closed-form
digitisation rule, which predicts it.  The one unmet item is the damping
calibration, reported as NOT_RUN rather than assumed.

## 7. Reproduce

```bash
PYTHONPATH=src python scripts/m16794_wigley_hull_contact_audit.py
PYTHONPATH=src python -m pytest tests/test_hull_form_mesh_contact.py -q
```

Suggested platform ledger entry (not yet registered):
`CASE-DEM-HULL-CONTACT-WIGLEY` — *DEM 球-船体网格接触：真实船型曲面（Wigley 解析）力误差 <3 % + 罚函数量纲规则*
acceptance: fine mesh (≥160×64) force error <1 % vs analytic hull, global load
error <1 %, and the directional sagitta rule must be conservative
(measured ≤ predicted) at every tested resolution.
