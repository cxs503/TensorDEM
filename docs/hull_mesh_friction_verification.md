# Hull-path Coulomb friction and rolling resistance — implementation + test evidence

**Anchor**: `NEW:GAP-DEM-G3` residual — the m605/m16794 gap ledger closed the
*normal* contact on a real ship-hull form (Wigley, doubly curved) but left
"friction / rolling resistance on the hull contact path" explicitly **open**
(`docs/dem_calibration_gap_ledger.md`, m16794 write-back). Suggested
`feature_ledger` entry:
`GAP-DEM-G3-HULL-FRICTION` — "hull-mesh contact path: Coulomb friction +
rolling resistance verified vs analytic incline anchors (<1 %)"; domain DEM /
ship-ice contact.

## ① Problem

`tensordem.hull_mesh_contact.mesh_contact_forces` is a **normal-only** penalty
law. Every ice-hull number produced with it is frictionless, which is not a
neutral simplification: the tangential traction on the hull is what turns
normal crushing pressure into longitudinal resistance (ice riding up the bow
and along the shoulder), and without friction a floe cannot even rest on the
hull. The gap was recorded, not disputed; this change closes it and states how
much of it is closed.

## ② What was done

1. **Geometry is now a single source of truth.** The narrow phase of
   `hull_mesh_contact` was extracted into
   `narrow_phase_contacts(positions, vertices, faces, ...)` returning flat
   arrays (`particle`, `triangle`, `closest`, `normal`, `overlap`, `inside`).
   `mesh_contact_forces` now sums over exactly those rows, so a tangential law
   cannot disagree with the normal law about *which* triangle, *where* the
   contact point is, or *how deep* the penetration is. The refactor is
   behaviour-preserving: the five pre-existing hull-kernel suites
   (`test_hull_mesh_contact`, `_curved`, `_signed`, `_deep_penetration`,
   `_degenerate`) still pass — 61 passed, 3 xpassed.
2. **New module** `src/tensordem/hull_mesh_friction.py`:
   * `HullFrictionConfig` — `mu`, `k_t`, `mu_r`, rolling model, and
     `carry_over_across_facets`.
   * `HullFrictionState` — per-`(particle, triangle)` Cundall–Strack tangential
     displacement + EPSD rolling state; pruned when a contact is lost.
   * `mesh_contact_forces_frictional(...)` — normal law *identical* to the
     frictionless kernel, plus
     `delta <- P_t (delta - v_slip dt)`, `|delta| <= mu f_n / k_t`,
     `F_t = k_t delta` on the sphere at the contact point, tangential moment
     `r x F_t`, rolling couple from the verified
     `rolling_resistance.rolling_resistance_torque` kernel, and a consistent
     hull reaction (force **and** moment about a hull origin, so a coupled
     ship-ice model conserves momentum).
   * The slip velocity uses the **material-point** velocities, i.e. the hull
     contributes `v_h + omega_h x (c - x_0)`, so a rotating (yawing) ship
     drives friction on ice correctly.
   * **Memorisé across facets**: when the nearest triangle changes, the stored
     tangential displacement is re-projected onto the new tangent plane instead
     of being dropped. Without it the friction force on a *curved* hull is a
     facet-periodic sawtooth between `0` and `mu f_n`.
   * Reuse, not duplication: the Coulomb cone comes from
     `contact_tangential.project_tangential_displacement` /
     `saturation_displacement`, the rolling couple from
     `rolling_resistance.rolling_resistance_torque` — the kernels already
     verified on the sphere–sphere path.
   * `integrate_sphere_on_mesh(...)` — semi-implicit Euler reference harness
     returning a *closed energy ledger* (`kinetic + potential + tangential
     spring + dissipated`).
3. **Tests** `tests/test_hull_mesh_friction.py` — 19 cases, all passing
   (40.9 s, `PYTHONPATH=src pytest tests/test_hull_mesh_friction.py -q`).

## ③ Numbers that were measured (this repo, this branch)

| quantity | measured | reference | verdict |
|---|---|---|---|
| normal ordinate with/without friction | diff `0.0` (bit-identical) | `mesh_contact_forces` | **gate: friction does not touch the normal law** |
| `mu = 0` limb vs frictionless kernel | max force diff `<1e-12` N | same kernel | pass |
| cone saturation `|F_t|/(mu f_n)` | `1.0` (rel. `<1e-12`) | Cundall–Strack cap | pass |
| friction direction vs `-v_slip` | `<1e-6` deg | anti-parallel | pass |
| tangential leakage into the normal | `<1e-15` | 0 | pass |
| sliding acceleration on an incline (20°) | slope vs `g(sin t - mu cos t)` | `incline_sliding_acceleration` | **`<1 %`** |
| static hold/slide boundary | held `8°` below `atan(mu)`, slides `8°` above, drift ratio `>100x` | `tan(theta_c) = mu` | pass (`<1 %` bisection pending in the campaign script) |
| rolling branch sign | accelerates downhill above `atan(mu_r)` | `critical_incline_angle_deg` | pass |
| energy ledger (`KE+PE+spring+diss`) | spread `3.6e-4` of `m g R` over 500 steps | closed ledger | pass (`<1 %`) |
| reaction force closure | `<1e-12` N | equal & opposite | pass |
| no relative motion ⇒ no friction | `<1e-12` N | kinematic requirement | pass |
| hull rotation enters slip velocity | equality to `1e-12` m/s | `omega_h x (c-x_0)` | pass |
| state reset on contact loss | 1 → 0 entries, zero force | hygiene contract | pass |

Reproduction:

```bash
PYTHONPATH=src python -m pytest tests/test_hull_mesh_friction.py -q          # 19 passed
PYTHONPATH=src python scripts/verify_hull_mesh_friction.py                   # full campaign
```

## ④ Figures

`scripts/verify_hull_mesh_friction.py` writes
`artifacts/hull_mesh_friction/{verification.json, fig_incline.png,
fig_thresholds.png, fig_facet_crossing.png}`: (a) sliding acceleration vs the
closed form, (b) the static-threshold bisection against `arctan(mu)`,
(c) rolling acceleration against `g(sin t - mu_r cos t)/(1+I/mR^2)`, (d)
`|F_t|/(mu F_n)` on the doubly curved Wigley hull **with** and **without** the
facet memory (the negative control for the carry-over claim).

## ⑤ Honest status / limits

* **NOT_RUN (budget)**: the campaign script completed its unit gates under
  pytest but the full artifact run (bisections + Wigley facet sweep + figures)
  did not finish inside this mission's wall-clock budget, so
  `artifacts/hull_mesh_friction/verification.json` is **not committed** and the
  three figures are **not** produced. The quantitative claims in §③ are the
  pytest-gated values and the diagnostic runs; the bisection-based
  `theta_c = atan(mu)` and `atan(mu_r)` statements are asserted in the tests as
  *hold vs slide* and *sign of the rolling acceleration*, not as a measured
  threshold. This is a NOT_RUN, not a pass.
* **Laboratory calibration of `mu` / `mu_r`** for ice against steel/coated hull
  is **WITHHELD**: no digitised ice-hull friction measurement set exists in this
  repository. What is verified is the *law and its inversion*, not the material
  constants. Any hull-resistance number that quotes ice friction must carry
  that provenance.
* The tangential dashpot is deliberately **absent** (`c_t = 0`): a sticking
  contact is undamped and oscillates elastically with `sqrt(k_t/m)`, which is
  visible in `test_static_hold_above_and_below_threshold` and is why the hold
  criterion is a *drift* criterion, not a velocity criterion. Adding `c_t` is
  the next node; it must not be done without re-scoring the anchors.
* One contact per sphere (nearest triangle) is inherited from the kernel. A
  floe straddling a keel/chine crease therefore gets one contact, not two; the
  normal law has the same limitation and the facet-carried tangential memory
  only partly compensates.
* Friction on the **normal-only** 2-D polyline hull path
  (`hull_mesh_contact`'s 2-D cousin) was not touched.

## ⑥ Gap-ledger write-back

Appended to `docs/dem_calibration_gap_ledger.md` (append-only):

* Row **GAP-DEM-G3 residual — "friction / rolling resistance on the hull
  contact path"**: `open` → **closed at the contact-law level, open at the
  calibration level**. Evidence: `src/tensordem/hull_mesh_friction.py`,
  `tests/test_hull_mesh_friction.py` (19 cases),
  `scripts/verify_hull_mesh_friction.py`, this document.
* Quantified: cone exact; sliding acceleration `<1 %` vs the closed form;
  static hold/slide separated by `>100x` in drift; rolling branch sign correct;
  energy ledger closed to `3.6e-4` of `m g R`; reaction closure `<1e-12` N.
* Still open: (a) material calibration of `mu`, `mu_r` for ice–hull
  (**WITHHELD**, no in-tree data); (b) tangential dashpot; (c) multi-contact
  (crease/chine) resolution; (d) the full campaign artifacts (NOT_RUN).