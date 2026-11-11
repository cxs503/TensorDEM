# Floe attrition: Wadell roundness → Archard corner wear → μ_eff(R) → multi-pass ice resistance decay

**Node** p472w2-N05 · **Repository** TensorDEM · **Module** `tensordem.floe_attrition`
**Artifacts** `artifacts/floe_attrition/` (criteria.json, sphere_baseline.json, figures/)
**Pre-registration** `prereg/p472w2_n05_floe_attrition_prereg.json`
**Tests** `tests/test_floe_attrition.py` (26 tests)

## 1. Why this module exists

Broken-ice channels (brash ice, frozen rubble, ice ridges) are made of *angular*
floes.  Every passage of a hull blunts their corners, and rounding is one of the
mechanisms that makes a repeatedly used channel *easier* to transit — the
"multi-pass resistance decay" of ice-going operations.  TensorDEM could *measure*
shape descriptors of a vertex cloud (`polyhedra_shapes`: sphericity
`π^{1/3}(6V)^{2/3}/A`, angularity) but had **no shape family in which roundness is
a state variable that a wear law can drive**, no link from shape to an effective
friction, and no closure of the resulting resistance change.  This module adds
that chain:

```
Wadell roundness R  --(Archard wear, geometric inversion)-->  R(N transits)
        |                                                          |
        +--> interlocking index I(R) --(friction closure)--> mu_eff(R,C)
                                                                   |
                                          (multi-pass scenario) -->  R_ice(N)
```

Everything is exact-geometry based: `R` is not a fitted number, it is
`4πA/P²` of the *analytic* shape whose single parameter is the corner radius.

## 2. Geometry (criterion C1)

### 2.1 2D: `RoundedPolygon(n, a, r)`

Regular `n`-gon of edge `a` with all corners filleted by radius `r`
(`0 ≤ r ≤ r_max = a·cot(π/n)/2`).  At `r = r_max` the body is exactly a circle —
the limit that makes the chain consistent with a disk field.

Closed forms (α = (n−2)π/n, t = r·cot(α/2), Δ = π−α):

| quantity | expression |
|---|---|
| area | `A(r) = A₀ − n·r²·[cot(α/2) − Δ/2]` |
| perimeter | `P(r) = n·a − 2n·t + n·r·Δ` |
| Wadell roundness | `R = 4πA/P²` (1 for a circle, `π/4` for a square) |
| circumradius | `R_c(r) = R_c,sharp − r/cos(π/n) + r` |
| interlocking index | `I = R_c/R_ins − 1` (0 for a circle, `sec(π/n) − 1` for a sharp n-gon) |
| worn area | `A₀ − A(r) = n·[cot(α/2) − Δ/2]·r²` (exactly quadratic in `r`) |

### 2.2 3D: `RoundedCube(a, r)`

Minkowski sum `cube(a − 2r) ⊕ ball(r)`; with `L = a − 2r` (Steiner):

```
V(r) = L³ + 6L²r + 3πLr² + 4πr³/3
A(r) = 6L² + 6πLr + 4πr²          (= dV/dr at fixed L)
```

`r = a/2` is exactly the sphere (`Ψ = 1`, 3D Wadell roundness `r/R_ins = 1`).
Note that in the *constant-size* family (`a` fixed, material removed) `dV/dr ≠ A`;
that is precisely why the attrition rate acquires a geometric factor (§3).

### 2.3 Verification

* 2D closed forms vs an **exact** Green's-theorem boundary integral (facet
  shoelace term + closed arc term): worst relative error **6.6 × 10⁻¹⁵** over 42
  `(n, r/r_max)` cases (acceptance `1e-12`).
* circle limit `R = 1` to 1.1 × 10⁻¹⁵; sphere limit to machine precision.
* 3D: exact Platonic sphericity vs the existing `ConvexHull` descriptor
  (`polyhedra_shapes`) — tetra/cube/octa/icosa/dodeca all ≤ 5.3 × 10⁻¹⁶.
* Steiner identity `dV/dr = A` at fixed `L` by 64-node Gauss–Legendre: 1e-16.
* DEM-ready discretisation `rounded_polygon_vertices`: the sampled roundness
  converges to the analytic value as `O(m⁻²)` in the arc-point count (1.1e-3 at
  m = 4 → 4.2e-6 at m = 64).

## 3. Attrition law (criteria C2, C3, C7)

Archard: `dV = k·W·ds/H` with a **pressure-controlled** corner contact,
`W = p·A_patch(r)`, `A_patch = 2t + rΔ = χr` (2D) or `πr²` (3D).  Inverting the
quadratic `V(r) = c₂r²` exactly at every step gives

```
2D:  dr/ds = k p χ / (2 c₂ H) = const   → linear blunting, R(s) saturating
3D:  dr/ds ∝ r                          → exponential blunting
```

so the relaxation *sliding distance* `τ` scales as `p⁻¹` in both families.

Measured (C3, five pressures 0.5–8 × 10⁵ Pa): τ = 8433 / 4218 / 2110 / 1056 /
529 m, log–log slope **−0.99853** (deviation 0.15 % from the pre-registered −1),
fit `R² ≥ 0.999733`, fitted β ≈ 1.09.

C2 compares the simulated worn volume with the closed form of the same law:
relative error **0.42 %** at the production cadence (≤ 10 % required); the error
grows to 9.9 % only at a 25× coarser cadence (which is reported as the
discretisation limit, not hidden).

The *classical* Archard formula with a frozen initial patch departs from the
simulation as the sliding adds up — 2.2 % at 10 m, 6.5 % at 25 m, 13.6 % at
50 m, 28 % at 100 m, 287 % at 1000 m.  That is the patch-growth feedback, i.e. a
physical effect rather than a numerical error: while the blunting is small the
classical formula is within the 10 % band (≤ 25 m of sliding at these
parameters), which fixes its range of validity.

C7: the Archard ledger equals the geometric material loss to **4.4 × 10⁻¹⁵**
(2D) and 4.7 × 10⁻¹⁶ (3D); zero sliding is an exact fixed point (state
bit-identical, worn volume identically `0.0`).

## 4. Roundness → effective friction (criteria C4, C5)

```
μ_eff(R, C) = μ_sph(C) + κ_i · I(R) · Z(C)/Z_ref        (κ_i = 1, Z_ref = 4)
```

* `μ_sph(C)` = stress ratio `|τ|/p` of the **m9375 disk shear cell**
  (`ice_ice_collision`, deterministic seed), measured here at
  C = 0.3/0.5/0.7: **1.194 / 0.820 / 0.559** with coordination
  Z = 0.071 / 0.209 / 1.401 (`artifacts/floe_attrition/sphere_baseline.json`).
* `I(R)` is the exact geometric interlocking index of §2.
* The limit `R → 1` is **structurally exact**: `I = 0` so `μ_eff` returns
  `μ_sph(C)` **bit for bit** (C4: difference exactly `0.0` at all three
  concentrations; a near-circle polygon `R = 1 − 10⁻⁶` and the attrition chain
  run to full rounding stay within 7.6 × 10⁻¹⁰).

### 3×3 design (C5)

Levels `R = 0.80 / 0.90 / 0.99`, `C = 0.30 / 0.50 / 0.70`, 8 replicates per cell
(the replicate disorder is the regular-polygon side count `n ∈ {3,4,5,6}`, with
the fillet solved so the realised roundness hits the target to 5.6 × 10⁻¹⁵):

| source | SS | df | F | p | η² |
|---|---|---|---|---|---|
| roundness | 0.0429 | 2 | 83.7 | 1.8 × 10⁻¹⁸ | 1.14 % |
| concentration | 3.643 | 2 | 7103 | 6.5 × 10⁻⁷⁵ | 97.13 % |
| roundness × concentration | 0.0488 | 4 | 47.5 | 2.3 × 10⁻¹⁸ | 1.30 % |
| residual | 0.0162 | 63 | — | — | 0.43 % |

The interaction is **significant**: rounding and concentration are not separable
in this closure (the interlocking term is multiplicative in shape and contact
density), so the resistance decay below is quoted per concentration and the two
mechanisms are *not* narrated as one blended roundness story.  Main-effect means:
μ_eff decreases with roundness level (0.927 → 0.898 → 0.868, averaged over C) and
with concentration (1.199 → 0.834 → 0.659) — the latter is the *friction
coefficient*; the resistance rises with C through the buoyant normal load (§5).

## 5. Multi-pass resistance decay (C6)

Scenario (dense brash channel, busy headway; per metre of waterline):

| parameter | value |
|---|---|
| concentration | 0.70 |
| corner contact pressure | 8 × 10⁵ Pa |
| transits / slip per transit | 400 / 2.5 m (1000 m of sliding) |
| hull contact length | 5 m |
| rest time between passages | 300 s |
| freeze-bond clock `t_c` | 3.6 × 10⁵ s (100 h) |
| bond cohesion / bonded area | 20 kPa / 0.15 m² per m |
| load amplification by bonding β_c | 0.30 |
| breakup term `R_crush` | 4.0 kN/m |

Result: **R_ice = 14.79 → 14.32 kN/m** (3.2 % decay over 400 transits,
10.2 MJ/m of cumulative contact energy, 0.071 m³/m of ice worn off).
Shares of the *net* change: rounding **+474.8 %**, consolidation **−374.8 %**,
sum **100.0000 %** (closure residual 1.9 × 10⁻¹⁵ of the total).  The shares sum
to 100 % by construction (they are the two halves of the exact difference), and
the negative consolidation share is a result: freeze bonding gives back ~78 % of
what corner rounding saves, so the *net* decay is small.

Single-mechanism runs show the mechanisms are **coupled**: ΔR(rounding only) =
+2064 N/m, ΔR(consolidation only) = −1768 N/m, ΔR(both) = +472 N/m; the
superposition residual is 176 N/m = **37 % of the total**, because bonding
amplifies the same normal load that friction multiplies.  Reporting the two as
independent numbers would therefore be wrong — consistent with the significant
interaction of §4.

## 6. Traps found / fixed on the way

1. **`polyhedra_shapes.make_icosahedron` was not an icosahedron** — the second
   vertex loop reused the leaked `sy` of the first loop, producing 10 vertices
   with wrong coordinates (sphericity 0.8828 instead of the exact 0.93933).  Fixed
   and guarded by `test_icosahedron_is_regular` (12 vertices, 20 faces, 30 edges,
   edge-length spread < 1e-12).
2. **Steiner's identity is per fixed inner size.**  For the constant-size family
   `a = L + 2r` (which is what wear produces) `dV/dr ≠ A(r)`.  The verification
   now checks the identity at fixed `L`, and the difference is exactly what makes
   the wear rate pressure-independent in 2D.
3. A pressure-controlled patch on a *mathematically sharp* corner (`r = 0`) has
   zero width, hence zero load and no wear — a deadlock, not physics.  Real floes
   have a finite `r₀/a ≈ 0.02`; the API raises instead of silently stalling.

## 7. Evidence grading and limitations

| component | grade | note |
|---|---|---|
| roundness / sphericity / interlocking geometry | **REAL** | exact closed forms, 1e-15 vs independent integral |
| Archard attrition + conservation | **REAL** | law + exact inversion, 4e-15 ledger |
| `μ_sph(C)`, `Z(C)` | **REAL (DEM)** | m9375 shear cell, reduced shear window (1500 shear / 800 sampling steps, seed 20260101); a full-window rerun is the declared refinement |
| `κ_i = 1` | **WITHHELD (calibration missing)** | O(1) geometric coefficient, *not* fitted; a rounded-polygon shear cell is the missing calibration |
| consolidation model | **MOCK (declared)** | phenomenological `1 − e^{−t/t_c}` with load coupling, not a freeze-bond DEM |

Explicitly out of scope here: 3D friction closure calibrated by DEM, size
polydispersity, melting/refreezing of the worn debris, hull-surface wear
(the complementary Archard channel), and a DEM shear cell of rounded polygons
(the calibration that would replace `κ_i`).

## 8. Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_floe_attrition.py -q          # 26 tests
PYTHONPATH=src python scripts/floe_attrition_sphere_baseline.py          # DEM baseline
PYTHONPATH=src python scripts/floe_attrition_verification.py             # C1-C7 + figures
```