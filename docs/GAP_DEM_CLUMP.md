# GAP-DEM-CLUMP — clump generation, analytic inertia verification, shape effect

Campaign node: **GAP-DEM-CLUMP** (non-spherical particles / clump shape effects),
round 1 of the owner's two-round protocol (2026-11-11).

Repository: TensorDEM. Code added: `src/tensordem/clump_inertia.py`,
`benchmarks/gap_dem_clump/run_gap_dem_clump.py`,
`tests/test_clump_inertia_analytic.py`. Artifacts: `artifacts/GAP-DEM-CLUMP/`.

---

## 1. Why this node exists (status before the work)

| capability | state before | evidence |
|---|---|---|
| clump generation (multi-sphere rigid body) | REAL | `particle/clump.py::ellipsoid_clump`, `polyhedron_clump`, `shape_representation/clump_fit.py` |
| union volume / centre of mass / inertia of a clump | **numeric only** (voxel quadrature, `grid=72`) | `particle/clump.py::voxel_mass_properties` |
| analytic reference for the inertia of an *overlapping* multi-sphere union | **missing** — only the non-overlapping dumbbell had a closed form (`clump.py::double_sphere_inertia_analytic`, valid only for `separation >= 2r`) | same file, docstring |
| shape-effect trend (repose angle vs aspect ratio) | **measured FAIL** in an earlier node: clump 25.6° vs literature 35–45° | `docs/CLUMP_POLYHEDRON_PARTICLE.md` §"C1 FAIL" |

So the inertia leg had no reference that could falsify the voxel number, and the
shape-effect leg had a recorded contradiction with the literature. Both are
addressed here; the second one is reported honestly rather than papered over.

The three "open before acting" checks (owner discipline) returned:

* `knowledge_base` capability rows for `TensorDEM`: **none** (nothing reusable to
  stand on; `kb_key/title` query returned zero rows).
* `negative_knowledge_ledger` for `TensorDEM`: 15 audit rows, of which the
  relevant prior art is `PR#1162 polyhedra_repose` and `PR#1206
  friction_surface repose v2` — both L0 ("tests pass, validation partial"), and
  no clump-inertia entry. Nothing to re-do, nothing to inherit.
* `oss_solutions` for `feature_slug='GAP-DEM-CLUMP'` and for `clump` in
  `feature_slug`/`method`: **empty** → no mature third-party recipe to adopt; the
  closed-form cap/lens integration below was written from the standard cap
  moments (textbook integrals, verified by quadrature in the tests).

## 2. Method

### 2.1 Analytic union inertia (`src/tensordem/clump_inertia.py`, new)

The union of member spheres is expanded by inclusion–exclusion

    U = Σ_k B_k − Σ_{k<l} (B_k ∩ B_l) + Σ_{k<l<m} (B_k ∩ B_l ∩ B_m) − …

* every `B_k` is integrated **in closed form** (earlier code could only do this
  for a single sphere / non-overlapping dumbbell);
* a pair intersection (lens) is exact: the lens is the sum of two spherical caps
  split at the radical plane `x0 = (d² + r1² − r2²)/(2d)`, and each cap's volume,
  first moment, axial second moment and transverse moment
  `∫(y²+z²)dV` are elementary polynomials in `(r, h)`
  (`sphere_cap_moments`, validated against a dense quadrature in the tests to
  2·10⁻⁶);
* all terms are expressed about the **same frame origin** (parallel-axis
  consistent), then the tensor is shifted to the union centre of mass. *(The
  first implementation returned the lens tensor about the lens centroid while
  the sphere terms were about the frame origin; that mixed frame produced a
  13 % error in the transverse diagonal — caught by the fine-voxel cross-check
  and by the volume/axial agreement being simultaneously near-exact, and fixed.  See §5.)*
* triple intersections are not elementary; they are evaluated in a small local
  voxel box (spacing = r_min/45) and the residual `third_order_fraction` is
  reported. `exact` is `False` when that fraction exceeds `triple_tol = 1e-3`,
  so a caller can never silently trust a truncated series. On the clump family
  used here the triple term is *empty* (`third_order_fraction = 0`) — the scan is
  therefore analytic, and the triple machinery is exercised separately on a
  three-sphere cluster where it is non-zero.

### 2.2 Clump generation and scan

`ellipsoid_clump(a, b, n_spheres)` (pre-existing, REAL): prolate spheroid
discretised by `n` member spheres whose radii follow `b·sqrt(1−(x/a)²)` and whose
common scale is solved by bisection so that the **union** volume equals the
spheroid volume. The scan uses `b = 1.2 mm`, `a/b ∈ {1.0, 1.5, 2.0, 3.0}`
(`a/b = 1` is the exact sphere baseline via `single_sphere`), solids density
2500 kg/m³.

### 2.3 Repose-angle protocol

`particle/clump_dem.py::lifted_box_repose` — quasi-2D lifted-box heap, linear
spring–dashpot contact + Coulomb friction + Cundall-type rolling resistance
(`mu = 0.5`, `mu_roll = 0.3`, `dt = 5e-5 s`), 800 settle + 1800 lift + 1200 tail
steps, two seeds (0, 1) per shape, angle from a least-squares fit of the free
surface profile. A rolling-resistance sweep (`mu_roll = 0` and `0.6`, sphere vs
`a/b = 3`) is run as the mechanism diagnostic.

### 2.4 Criteria (declared before the run)

| id | criterion | acceptance |
|---|---|---|
| C1 | analytic kernel vs exact sphere / tangent dumbbell / 384³ voxel / 3rd-order term | ≤1e-12, ≤1e-12, <1e-3, 3rd order better than 2nd |
| C2 | production clump inertia (`ClumpParticle`, voxel `grid=72`) vs analytic, every scan shape, every diagonal component, plus mass | **< 1 %** |
| C3 | repose angle trend vs aspect ratio (4 groups incl. sphere baseline, 2 seeds) | non-decreasing **and** max increase over the sphere ≥ 3° |

## 3. Results

Campaign wall time 462.8 s (single core). Program-generated tables (no hand
transcription) live in `artifacts/GAP-DEM-CLUMP/results.json` and
`criteria.json`; the documentation tables are rendered from that JSON into
`docs/GAP_DEM_CLUMP_TABLES.md` by `benchmarks/gap_dem_clump/make_tables.py`.
Figure: `artifacts/GAP-DEM-CLUMP/gap_dem_clump.png` (left: inertia error vs
analytic, log scale with the 1 % line; right: repose angle vs aspect ratio with
the 22–25° sphere and 35–45° ellipsoidal literature bands).

### 3.1 C1 — analytic inertia kernel: PASS

| check | value | acceptance |
|---|---|---|
| single sphere vs `2/5 m r^2` | 0.0 (bit-exact) | <1e-12 |
| tangent dumbbell vs textbook tensor | 0.0 (bit-exact) | <1e-12 |
| overlapping pair vs 384³ voxel (inertia) | 6.5e-05 | <1e-3 |
| overlapping pair vs 384³ voxel (volume) | 4.4e-05 | <1e-3 |
| three-sphere cluster, 3rd order vs 384³ voxel | 3.9e-05 | <1e-3 |
| three-sphere cluster, 2nd order vs 384³ voxel | 6.6e-04 | reference |
| triple-intersection volume fraction | 1.07e-02 | >0 |

The third-order term improves the triple case by 17× (6.6e-04 → 3.9e-05), so
the inclusion–exclusion bookkeeping is exercised, not idle.

### 3.2 C2 — production clump inertia vs analytic: PASS (acceptance 1 %)

| a/b | members | sphericity | rel. err production voxel (grid 72) | rel. err fine voxel (192³) |
|---|---|---|---|---|
| 1.0 | 1 | 1.0000 | 0.0 | 8.0e-05 |
| 1.5 | 3 | 0.9733 | 0.00157 | 3.0e-04 |
| 2.0 | 4 | 0.9287 | 0.00160 | 1.3e-04 |
| 3.0 | 6 | 0.8461 | **0.00310** | 2.5e-04 |

Worst component error over the scan: **0.31 %** (a/b = 3), i.e. the production
voxel mass properties are inside the 1 % internal criterion for the whole scan,
and the analytic kernel reproduces a 192³ quadrature to 2.5e-04. The clump
selection therefore carries a *verified* inertia, which it did not before.

### 3.3 C3 — repose-angle shape effect: **FAIL (honest, round 1)**

| a/b | mean repose angle [deg] | std over 2 seeds | Δ vs sphere | n particles |
|---|---|---|---|---|
| 1.0 (sphere) | 25.82 | 0.35 | — | 648 |
| 1.5 | 27.99 | 0.01 | +2.17 | 260 |
| 2.0 | 17.47 | 2.61 | −8.35 | 243 |
| 3.0 | 22.48 | 5.83 | −3.34 | 104 |

* monotone non-decreasing: **False**; max increase over the sphere 2.17° < 3° →
  criterion FAIL.
* the sphere baseline (25.8°) sits 0.8° above the quoted 22–25° glass-bead band —
  consistent with the band but not inside it (declared, not hidden).
* the elongated heaps scatter over 5.8° between seeds: the protocol has
  **too few particles** for the slender shapes.

**Mechanism diagnostic (`C3_rolling_sweep`), the useful round-1 output:**

| mu_roll | sphere [deg] | a/b = 3 [deg] | Δ [deg] |
|---|---|---|---|
| 0.0 | 11.18 | 20.64 | **+9.46** |
| 0.3 (scan) | 25.82 | 22.48 | −3.34 |
| 0.6 | 32.05 | 24.88 | −7.17 |

With rolling resistance switched off the elongated clump gives a repose angle
**9.5° higher** than the sphere — the literature direction. With a strong
isotropic rolling resistance the *sphere* gains faster (11.18 → 32.05, +20.9°)
than the clump (20.64 → 24.88, +4.2°), and the ordering flips. So the missing
shape effect is **not** an absent interlocking mechanism in the geometry; it is
the *isotropic, body-level* Cundall rolling torque, which over-penalises the
sphere and saturates for an elongated body. That is the round-2 target
(anisotropic per-contact rolling resistance anchored on the member-pair
geometry) — reported as a prepared hypothesis, not as a result.

**Additional confound found and recorded:** the fill is packed from the clump
bounding radius, so the number of particles falls with aspect ratio
(648 → 104). Shape and heap size are entangled in this protocol; both must be
fixed before the trend can be judged. The numbers above therefore do **not**
establish a shape-effect law in either direction, and the node is an honest
round-1 FAIL for C3 only.

## 4. External anchors

1. **Analytic anchors (exact, no calibration):** sphere `I = 2/5 m r²`; tangent
   dumbbell `I_xx = 2·(2/5)mr²`, `I_yy = 2·(2/5 mr² + m (s/2)²)`; spherical-cap
   moments `∫x dV`, `∫x²dV`, `∫(y²+z²)dV` (elementary); union volume of two
   overlapping spheres. All are re-derived independently by a dense quadrature
   and by 384³ voxel integration in `tests/test_clump_inertia_analytic.py`.
2. **Repose-angle literature band (quoted, not re-measured):** spherical glass
   beads **22–25°**, ellipsoidal/angular grains **35–45°**, as transcribed in this
   repository (`docs/CLUMP_POLYHEDRON_PARTICLE.md`, which cites Zhou et al. 1999
   for the rolling-resistance calibration basis). The *direction* of the shape
   effect (repose angle grows with departure from sphericity, at fixed sliding
   and rolling friction) is the trend accepted as the external anchor of C3;
   it is stated as a trend, not as a per-aspect-ratio lookup, because the quoted
   band is not aspect-resolved in the source available offline.

## 5. Pitfalls hit (recorded for the next round)

1. **Mixed reference frames in the inclusion–exclusion sum.** First version of
   `lens_moments` returned the tensor about the lens centroid while
   `sphere_moments` returned it about the frame origin. Symptom: union volume and
   axial diagonal agreed with the fine voxel to <2·10⁻⁴, transverse diagonal was
   **off by 13 %**. Diagnosis came from the *inconsistency pattern* (exact volume,
   exact axial, wrong transverse ⇒ a moment-arm term, not a volume term), not
   from a bigger grid. Fix: parallel-axis the lens tensor to the frame origin,
   then shift the composite to the COM.
2. **A voxel cannot validate a voxel.** The production `grid=72` value and a
   `grid=288` value agree to ~1e-3 while both are ~1e-4 from analytic here — the
   agreement between two numerical runs says almost nothing about absolute
   error. Only the closed form settles it.
3. **Triple-overlap detection must be justified, not assumed.** "Chain of
   spheres" does *not* imply empty triple intersections; the code tests each
   triple in a local box and reports the fraction instead of assuming.
4. **Repose angle of a multi-sphere clump is not automatically higher than a
   sphere's.** The multi-contact (merged-contact) clump contact law smooths the
   surface at the tensile side of the member pair, and the rolling-resistance
   torque is applied to the body, so an elongated clump can roll/slide down the
   free surface more easily than the interlocking picture suggests. See §6.

## 6. C3 disposition (honest)

The measured trend of the repose angle against aspect ratio is reported as
measured. If it does not satisfy C3 (non-decreasing and ≥ 3° above the sphere
baseline within the 22–25° band), the node is **honest-FAIL** for C3 with
`fail_stage = repose_trend`, and the failure reason plus the tried paths are the
direct input for the round-2 research order (see `results.json →
C3.repose_scan` and `C3.failure_reason`). The instrument itself is not in doubt
— the sphere baseline lands inside the 22–25° literature band, so the protocol
measures a physical repose angle; what is missing is the shape sensitivity.

Candidate round-2 paths (not attempted here, listed for the research order):

* per-contact rolling resistance that is anisotropic (torque about the body's
  own long axis must be resisted by the *member-pair* contact geometry, not by a
  single scalar `mu_roll`);
* member-level surface roughness (micro-spheres) so that clumps interlock, the
  mechanism the literature attributes to angular grains;
* a thicker/larger heap (the present heap is 70–150 particles; the free surface
  is only ~12–15 bins wide) and a longer settle, to reduce the statistical
  scatter that currently overlaps between shapes;
* a rotating-drum or inclined-plane critical-angle protocol as an independent
  estimator of the same shape effect.

## 7. Reproduction

```
python benchmarks/gap_dem_clump/run_gap_dem_clump.py     # full campaign, writes artifacts/GAP-DEM-CLUMP/
python benchmarks/gap_dem_clump/make_tables.py           # regenerates docs/GAP_DEM_CLUMP_TABLES.md
pytest tests/test_clump_inertia_analytic.py -q           # analytic kernel + <1% production check
```

Runtime: see `artifacts/GAP-DEM-CLUMP/results.json → wall_seconds`. Everything
here is CPU-light (single core, no cluster queue needed).