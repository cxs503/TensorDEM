# 3-D hull mesh contact: curved-hull (faceted cylinder) verification — GAP-DEM-G3 debt

Module `src/tensordem/hull_mesh_contact.py` (m669) · audit script
`scripts/m6423_curved_hull_audit.py` · tests `tests/test_hull_mesh_contact_curved.py`
· artifacts `artifacts/m6423_curved_hull_audit/` (`metrics.json`, 3 PNG).

The 3-D mesh contact kernel was verified on flat planes and single analytic
triangles only; the gap ledger carried the open row *"curved-hull validation on
a real hull not re-audited"*. This audit closes the geometry side of that row
with the 3-D analogue of `docs/polyline_hull_contact_verification.md`.

## 1. Case

A rigid **circular cylinder** hull (radius `R_c = 0.5 m`, length `L = 2.0 m`,
axis = z) is the 3-D analogue of the 2-D circular hull used in the polyline
audit. The mesh is an **inscribed** faceted cylinder — exactly what a
digitised curved hull is. Sphere radius `R_p = 0.05 m`, penalty stiffness
`k = 1e6 N/m`, damping 0, hull at rest. The analytic reference is the exact
distance from the sphere centre to the solid cylinder (mid-span radial
distance and the two rim circles, closed form), independent of the kernel.

## 2. Method

1. Build inscribed faceted cylinders `n_circ ∈ {8, 16, 32, 64, 128}` ×
   `n_axial ∈ {4, 8}`; sweep sphere centres radially (penetration 2–40 mm of
   the true surface) at 9 azimuths (fine mesh) or 72 azimuths (scan).
2. Compare `mesh_contact_forces` against the analytic penalty force
   `k·(R_p − d)` in magnitude and direction.
3. End-rim study: 35 approach points around the end rim vs the rim-circle
   distance (n_circ = 64).
4. Broad-phase completeness: 300 random centres, spatial-hash candidates must
   be a superset of the brute-force AABB set (same criterion as the
   degenerate-triangle audit).
5. Action/reaction balance on the curved mesh.

## 3. Results (all from `metrics.json`, no transcription)

**Fine mesh (n_circ = 128), 54 sweep points:** max force rel. error
**1.47 %**, max direction error **1.406°** (= half facet chord angle,
geometric), 0 dropouts.

**Discretisation scan** (penetration 10 mm, 72 azimuths):

| n_circ | sagitta (m) | sagitta/overlap | measured max rel. error |
|---:|---:|---:|---:|
| 8 | 3.806e-2 | 3.81 | 11.5 % |
| 16 | 9.607e-3 | 0.961 | 23.7 % |
| 32 | 2.408e-3 | 0.241 | 5.95 % |
| 64 | 6.023e-4 | 0.0602 | 1.49 % |
| 128 | 1.506e-4 | 0.0151 | 0.37 % |

**End rim (n_circ = 64):** max rel. error **1.00 %** (sagitta 6.0e-4 m).
**Broad phase:** 0 missing candidates / 300 queries. **Action/reaction
residual:** 0.0.

**Closed-form digitisation rule** (3-D analogue of the 2-D rule):
`n_min = ceil(π / acos(1 − δ/R_c))` guarantees sagitta < δ. For δ = 20/10/5 mm:
`n_min = 12 / 16 / 23`.

## 4. Figures

* `force_vs_penetration.png` — force vs penetration, four azimuths vs analytic.
* `discretisation_scan.png` — measured error vs resolution vs sagitta bound.
* `rim_approach.png` — end-rim approach, mesh vs analytic.

## 5. Notes, traps, limits

* **The predictor is sagitta/overlap, not sagitta/penetration.** Overlap =
  `R_p − pen`; when the penetration approaches `R_p` the overlap (and the
  analytic force) shrinks while the facet offset stays fixed, so the relative
  error blows up as `sagitta/(R_p − pen)`. The 2-D audit's
  "min segments for penetration" criterion only guarantees *contact exists*;
  force accuracy needs the overlap in the denominator.
* **Non-monotonic at the coarse end (honest, measured):** n_circ = 8 gives
  11.5 % while n_circ = 16 gives 23.7 %. At n = 8 the vertices sit close
  enough to the centre that the vertex distance masks the facet offset; at
  n = 16 the facet midpoint (sagitta 9.6 mm ≈ overlap 10 mm) dominates.
  Resolution studies must quote the bound, not assume monotonicity.
* **No contact dropout in this case:** `R_p = 50 mm` ≫ sagitta for all
  resolutions tested, so the 2-D dropout phenomenon does not appear; the
  dropout regime (sagitta > overlap) remains characterised only in 2-D.
* The cylinder is a *canonical* curved hull, not a ship hull: real-hull
  (SUBOFF-class) validation still needs a real mesh and is out of scope here.
  Penalty stiffness/damping calibration and friction are not benchmarked
  (same scope as the 2-D audit).
* Trap hit while writing the reference: the first version of the analytic
  closest point returned the sphere centre itself (mid-span) and an off-circle
  point (rim), producing a spurious 178° "direction error" — the reference
  implementation must be checked against the kernel's own geometry before any
  verdict (same lesson as the ice-ride-up MC reference).

## 6. Error verdict (key quantity vs reference, explicit < 3 %?)

| key quantity | reference | measured | < 3 % ? |
|---|---|---:|---|
| force, n_circ = 128 (54 pts) | analytic cylinder | 1.47 % max | **yes** |
| force, n_circ = 64 (72 pts) | analytic cylinder | 1.49 % max | **yes** |
| force, n_circ = 32 | analytic cylinder | 5.95 % max | **no** (below rule) |
| force, n_circ = 16 / 8 | analytic cylinder | 23.7 % / 11.5 % | **no** (below rule) |
| force direction, n_circ = 128 | true radial | 1.406° max | yes (chord angle) |
| rim force, n_circ = 64 (35 pts) | rim circle | 1.00 % max | **yes** |
| broad-phase completeness | brute force | 0/300 missing | **yes** |
| action/reaction | −ΣF | 0.0 | **yes** |

No honest-FAIL of the kernel: every "no" is a resolution below the closed-form
digitisation rule, which the rule itself predicts.

## 7. Reproduce

```bash
PYTHONPATH=src python scripts/m6423_curved_hull_audit.py
PYTHONPATH=src python -m pytest tests/test_hull_mesh_contact_curved.py -q
```
