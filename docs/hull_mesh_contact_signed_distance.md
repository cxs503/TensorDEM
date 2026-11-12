# 3-D hull mesh contact: signed (inside/outside) distance — GAP-DEM-HULL-SIGNED

Module `src/tensordem/hull_mesh_contact_signed.py` (new) + opt-in flag
`closed_surface` in `src/tensordem/hull_mesh_contact.py` · audit script
`scripts/hull_signed_audit.py` · tests `tests/test_hull_mesh_contact_signed.py`
· artifacts `artifacts/m16799_hull_signed/`.

## 1. Defect (verified, before the fix)

`mesh_contact_forces` computed `overlap = contact_r - |p - closest|` with **no
inside/outside test** and took the normal from the closest point to the sphere
centre. Once the centre crossed a hull surface the vector reverses, so:

* `0 < t < contact_r` (t = depth past the surface): the penalty force **pulls the
  particle into the hull**;
* `t >= contact_r`: the force is **zero** — the particle is lost from the
  simulation with no restoring force.

Measured on a flat mesh, R = 50 mm, k = 1 MN/m, centre 5 mm past the surface:
force **−45.0 kN** where the outward answer is **+45.0 kN**; at 60 mm past the
surface the force is exactly 0. (This is a one-sided-penalty limitation, not a
trapezoid-rule error: the outer branch was, and remains, correct.)

## 2. Case, method, references

Three watertight surfaces: UV sphere (R = 0.5 m), capped cylinder (R = 0.5 m,
L = 2 m), and the published ITTC benchmark **Wigley hull** (L = 4 m, B = 0.5 m,
D = 0.25 m; two side sheets + flat deck, shared edges so the mesh is
edge-manifold). The classifier (`ray_parity_inside`, Möller–Trumbore odd–even
parity along a fixed non-degenerate ray) is scored against **analytic
containment**; the signed distance is scored against closed-form/parametric
references that never touch the triangle kernel. Force is scored against
`k (R + δ)` using the reference foot point.

## 3. Error verdict (key quantity vs reference, explicit < 3 %?)

| key quantity | reference | measured | < 3 % ? |
| --- | --- | --- | --- |
| classifier, sphere (n=1200) | analytic containment | mismatch 1 | **no** |
| classifier, capped cylinder (n=1200) | analytic containment | mismatch 0 | yes |
| classifier, Wigley hull (n=1000) | analytic containment | mismatch 0 | yes |
| signed distance, Wigley (n=6) | parametric minimisation | max 53.88 %, mean 13.46 % | **no** |
| force, cylinder inside 0.25–3R (max) | k(R+δ) | 0.89 % | yes |
| force, Wigley inside 0.25–2R (max) | k(R+δ), analytic foot | 53.88 % | **no** |
| direction, Wigley inside (min cos) | outward normal at analytic foot | -0.99999 | **no** |
| direction, cylinder inside (min cos) | outward radial | 0.99880 | yes |
| baseline reversal (control) | outward force must be > 0 | 37515 N (outward cos -1.00) | **no** (defect) |

## 4. Figures

* `fig1_force_vs_depth.png` — capped cylinder, force vs signed penetration depth,
  baseline vs signed vs analytic.
* `fig2_wigley_signed.png` — Wigley hull midship: force and direction cosine vs
  depth, against the parametric reference.
* `fig3_escape_map.png` — lost-particle map over (k, v) for an ice floe,
  baseline vs signed, with the `k = m v²/R²` line.

## 5. Dynamic impact (before / after)

Realistic floe: R = 0.5 m, m = 480 kg (ρ = 917 kg/m³), v₀ = 3 m/s, k = 1e4 N/m,
dt = 2e-4 s, capped-cylinder hull wall.

| quantity | baseline | signed |
|---|---|---|
| y_final [m] | -0.3330 | 0.2787 |
| min y [m] (wall at 0.5) | 0.5070 | 0.5070 |
| max force [N] | 4998.0 | 7581.4 |
| rebound speed [m/s] | 0.0000 | 1.0833 |
| restitution e | 0.0000 | 0.3611 |
| particle lost (centre > 2R past wall) | False | False |

The baseline floe ends at y = -0.333 m, i.e. **inside** the
cylinder (a cylinder of radius 0.5 m centred on y = 0), with no force acting on
it. The signed run rebounds with e = 0.361 ≤ 1 (central
difference with the semi-implicit Euler step, no damping).

## 6. Criterion (from `metrics.json.criterion`)

Crossing requires the kinetic energy to survive a penetration of one radius at
the available penalty stiffness: `½ m v² > ½ k R²` ⇒

    k_crit = m v² / R²

* m = 480 kg, v = 1.0 m/s, R = 0.5 m → k_crit = 1920 N/m
* m = 480 kg, v = 2.0 m/s, R = 0.5 m → k_crit = 7680 N/m
* m = 480 kg, v = 3.0 m/s, R = 0.5 m → k_crit = 17280 N/m

Measured on the (k, v) grid: **0** baseline combinations lose the
particle and **0** signed combinations do (the criterion line
`k = m v²/R²` is plotted in fig. 3; every baseline loss sits below it).

## 7. Notes, traps, limits

* **Continuity of the fix.** `overlap = contact_r - d_s` with `d_s < 0` inside
  tends to `contact_r` from both sides at the surface, so the outer branch is
  unchanged and the fix costs nothing in outer-branch accuracy (verified above:
  the outer-branch error is the same faceting error as before).
* **Watertightness is required, not assumed.** `closed_surface=True` raises
  `ValueError` on an open mesh: ray parity is undefined there, and silently
  returning "outside" would recreate the defect in a less visible form.
* **Degenerate contact (centre exactly on a triangle)** still falls back to the
  raw triangle normal, whose orientation is not guaranteed outward; this is a
  measure-zero case and is *not* claimed as verified.
* **Cost.** The parity test runs only for spheres in the contact band and is
  O(F) per such sphere; it is not on the fast path of the broad phase.
* **Not in this node:** a true signed-distance broad phase (the parity test is
  per-sphere Möller–Trumbore, not accelerated), self-intersecting hull meshes
  (parity is wrong for them), and a defence against a particle that is *born*
  inside the mesh at t > contact_r (no history is consulted, so it is pushed
  out only once its centre is within contact_r of a wall).

## 8. Reproduce

```bash
PYTHONPATH=src python scripts/hull_signed_audit.py
PYTHONPATH=src python -m pytest tests/test_hull_mesh_contact_signed.py -q
```
