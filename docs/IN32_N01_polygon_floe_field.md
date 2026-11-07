# IN32-N01 — Unordered polygonal floe field: Laguerre tessellation + SAT contacts with lever arms

**Repository**: TensorDEM · **Node**: IN32-N01 (maturation of wave-1 IN22-N01) ·
**Modules**: `src/tensordem/laguerre_field.py`, `src/tensordem/polygon_contact.py`,
`src/tensordem/polygon_floe_field.py` ·
**Tests**: `tests/test_polygon_floe_field.py` (33 test functions, 52 collected cases, 20 of them negative controls) ·
**Harness**: `benchmarks/polygon_floe_field_acceptance.py` ·
**Artifacts**: `artifacts/in32_n01_polygon_floe_field.json`, `artifacts/in32_n01/*.png` ·
**Generated tables**: [`docs/IN32_N01_results.md`](IN32_N01_results.md) (produced from the
artifact JSON, never hand-transcribed).

---

## 1. Physical background and what was wrong in wave-1

A broken-ice cover in a channel is a *granular assembly of polygonal fragments*. Two
properties of that geometry drive the ship–ice physics:

1. **Topology.** Ice concentration is raised by *compaction*, not by growth. A field of
   equal-area disks on an ordered (size-graded) lattice — the wave-1 generator
   (`tensordem.fsd_packing`, node IN22-N01) — reproduces the coverage but not the
   topology: every floe has the same neighbour shell, there is no structural
   heterogeneity, and the packing jams at the random-close-packing limit, so the field
   could not be pushed to 8.5–9/10 ice without residual overlaps (the wave-1
   `phi >= 0.85` overlap FAIL).
2. **Load transfer.** Two disks touch on the line joining their centroids, so a disk
   contact force never has a lever arm: the torque on a floe is identically zero. In a
   real ice cover the contact between two fragments is an *edge-to-edge* or
   *vertex-to-edge* contact away from the centroid line, and the resulting torque is
   what makes floes rotate, interlock and form pressure ridges.

IN32-N01 therefore replaces the disk geometry by the real thing: every floe is a
convex polygon whose area is prescribed by the floe-size distribution (FSD), the
polygons are mutually disjoint with a plateau gap at any concentration up to
`phi = 0.90`, and contacts are convex-polygon SAT contacts that carry a contact point,
a lever arm and a torque. The polygon field is the front-end geometry that
`tensordem.ice_channel` (ship transit) and `tensordem.floe_6dof` (single-floe motion)
need in order to run on realistic fragment topology.

### Geometry and parameters

| item | value / source |
|---|---|
| domain | periodic rectangle `Lx x Ly`; `Lx Ly = sum(A_i)/phi`, aspect 1.6 |
| floe count | 200–4000 (sweep 200, completeness 2000, budget 4000) |
| FSD families | `PowerLawFSD(2.5)`, `LogNormalFSD(0.5)`, `BimodalFSD()`, `DeltaFSD(5.0)` (wave-1 classes, median floe radius 5 m) |
| floe area | `A_i = pi a_i^2`, `a_i` from stratified FSD quantiles `u_k=(k+1/2)/N`, shuffled over the sites |
| concentration | `phi in {0.60, 0.70, 0.80, 0.85, 0.90}` |
| plateau gap | `s = sqrt(phi)` shrink of every Laguerre cell about its own centroid |
| thickness | `0.1 x median r_eq` (only enters mass/inertia; geometry is 2D) |
| density | 917 kg/m³ (ice) |
| contact law | granular2d Hertz–Mindlin-type: `f_n = max(0, k_n delta + c_n n.v_rel)`, `c_n = 2 zeta sqrt(k_n m_eff)`, Coulomb cap `|f_t| <= mu f_n`, `k_n = 3e4`, `zeta = 0.2`, `mu = 0.5` |

## 2. Method

### 2.1 Prescribed-area Laguerre (power) tessellation

The Laguerre cell of site `x_i` with weight `w_i` is

```
C_i = { y : |y - x_i|^2 - w_i <= |y - x_j|^2 - w_j  for all j != i }
```

Each inequality is a half-plane: in site-relative coordinates `y' = y - x_i` and with
`d = x_j' - x_i` it reads `d . y' <= (|d|^2 + w_i - w_j)/2`. The cell is therefore a
**convex polygon** and the cells tile the torus exactly (interiors disjoint, areas
summing to `Lx Ly`).

Implementation (`laguerre_field.py`):

* **Batched Sutherland–Hodgman clipper** — all `N` cells advance simultaneously through
  the (up to) `k` neighbour half-planes; the emitted vertex set is compacted with a
  `cumsum` index (no Python loop over cells), the working capacity is the live vertex
  count plus slack, and cells whose remaining planes are provably farther than their
  circumradius leave the batch. Cost per tessellation is `O(N k)` vectorised work.
* **Neighbour relevance by plane distance**, not Euclidean distance: a large cell is
  bounded by *far-away large* floes whose bisectors sit close to its own site, so
  candidates from a KD-tree pool are sorted by `|0.5|d| + (w_i-w_j)/(2|d|)|`. This is
  what keeps `k` small (16–128 instead of `N`).
* **Self-validating coverage**: after clipping, `sum(A)/Lx Ly` must be 1 to `1e-6`
  (observed: 0 ≤ deviation ≤ 4.4e-16 in the sweep). If not, `k` is doubled and the
  tessellation rebuilt; a persistent mismatch is a hard `RuntimeError`. A silent overlap
  is therefore impossible by construction, unlike a purely geometric packing code.
* **Prescribed areas by Newton on the exact area Jacobian.** For an edge of length
  `l_ij` on the bisector of pair `(i,j)`, shifting the plane by `dw/(2|d_ij|)` sweeps an
  area `l_ij dw/(2|d_ij|)`, hence
  `dA_i/dw_i = sum_j l_ij/(2|d_ij|)` and `dA_i/dw_j = -l_ij/(2|d_ij|)` — a graph
  **Laplacian** (the semi-discrete optimal-transport Hessian). One sparse solve per
  iteration replaces hundreds of fixed-point sweeps. The iteration is
  **homotopy-continued** in the target vector (uniform → requested FSD) with a
  backtracking line search on `sum((A-T)^2/T)`, initialised at the plane-balance estimate
  `w_i ~ 1.074 sqrt(T_i Tbar) - Tbar`. Continuation is what prevents heavily
  under-weighted cells from vanishing (an empty cell has no shared edge, hence no
  Jacobian row). Measured: 9–18 iterations, final max relative area error `2e-6 … 1.9e-3`.

### 2.2 Plateau gap and the no-overlap guarantee

The FSD areas `A_i` are converted into *cell* targets `T_i = A_i/phi`, the box is sized so
`sum(T_i) = Lx Ly`, and every cell polygon is then scaled about its own (area) centroid
by `s = sqrt(phi)`. Two consequences:

* the realised floe area is exactly `A_i` (`s^2 * T_i = A_i`), so the FSD is honoured;
* because the cells are disjoint and shrink toward their own interiors, the gaps are
  strictly positive: the minimum SAT signed gap is `0.094 m` at `phi = 0.90` over the
  whole 240-field sweep (criterion (a)).

### 2.3 Convex-polygon contact with lever arms

`polygon_contact.py`:

* **Signed gap** is exact in both branches: separated → minimum vertex/edge distance over
  all edge pairs (exact for disjoint convex polygons); overlapping → minimum translation
  depth over all edge normals (SAT penetration).
* **Witness pair (CFP)** → contact point `p`; lever arms `r_i = p - x_i` (using the
  *minimum-image* position of the neighbour, so torques are correct across the periodic
  seam) and therefore a torque `tau_i = r_i x f_i`, `tau_j = -r_j x f_j`.
* **Pair search**: KD-tree over the 3x3 periodic images with a bounding-circle filter,
  minimum-image displacement per pair. Verified identical to the `O(N^2)` enumeration
  (criterion (c)).
* **Hull contact**: the same SAT routine applied to a two-vertex "polygon" (a hull
  segment), which gives floe–hull gaps and lever arms for the ship contact model.
* **Contact law**: the granular2d normal/tangential law is reused unchanged (Hertz-type
  normal spring, `c_n = 2 zeta sqrt(k_n m_eff)`, Coulomb cap on `f_t`, optional rolling
  resistance), with velocities taken at the **contact point**
  (`v + omega x r`), so both force and torque are consistent.

## 3. Results

All numbers live in [`docs/IN32_N01_results.md`](IN32_N01_results.md), generated from
`artifacts/in32_n01_polygon_floe_field.json`; the headline values are:

| criterion | requirement | measured | verdict |
|---|---|---|---|
| (a) no overlap, 240 fields (4 FSD x 12 seeds x 5 `phi`) | min SAT gap `>= -1e-9 m` | see table | PASS |
| (b) area moments vs analytic FSD | `< 1 %` | see table | PASS |
| (c) contact set vs `O(N^2)` at N=2000 | 0 FP / 0 FN | see table | PASS |
| (d) static field drift / force / torque bookkeeping | drift `< 1e-10 m`, sums `<= 1e-12` | see table | PASS |
| (e) monodisperse limit, `phi <= 0.70` | IoU vs circumscribed-disk field `>= 0.98` | see table | PASS |
| (f) generation budget, N=4000 | `< 30 s` | see table | PASS (cluster + local) |
| (g) tests | `>= 18` cases incl. negative controls | 52 collected cases (33 functions, 20 negative controls) | PASS |

## 4. Figures

Generated by the harness into `artifacts/in32_n01/`:

| file | content |
|---|---|
| `field_phi085.png` | polygon field (log-normal FSD, `phi=0.85`, N=420) coloured by `r_eq`, with the near-contact network; the plateau gap is visible between all neighbours |
| `gap_hist.png` | SAT signed-gap distribution of near pairs for a monodisperse field at `phi = 0.60/0.75/0.85/0.90`; no gap is negative, the median shrinks as `sqrt(phi)` |
| `moments_vs_n.png` | realised `m1`/`m2` relative error vs N (log-log) with the 1 % line, and the coordination number saturating at the planar value ~6 |
| `torque_lever_arm.png` | contact torque vs lever arm at `phi=0.90`: the polygonal contact point is off the centroid line, so the torque is non-zero (a disk model gives identically zero) |

## 5. Notes, pitfalls, limits

1. **A finite neighbour list can silently over-claim area.** The first implementation
   clipped each cell against its `k` Euclidean-nearest sites. That is exact only while no
   cell "swallows" a neighbour; as soon as the weights are disparate the true bounding
   planes come from *far* sites and the polygons overlap — the coverage identity detects
   it (`cover = 9.2` in the broken version). Hence the plane-distance ordering and the
   coverage-based `k` expansion. Any future optimisation of the neighbour search must
   keep the coverage check.
2. **`np.roll` is wrong for padded polygon batches.** The successor of the last live
   vertex is vertex 0, not the first padding slot; using `roll` corrupts every cell at
   the wrap and produced a 0.28 coverage in an early version.
3. **Capacity shrink must be reversible.** A batched clipper can exceed its capacity by
   two vertices (spikes); shrinking the scratch array without the ability to grow again
   truncated cells (coverage 0.186). The clipper now grows on demand.
4. **The fixed point on weights oscillates** for polydisperse FSDs (Jacobi iteration on a
   Laplacian is slow *and* unstable at high damping). Newton on the exact Laplacian plus
   continuation replaced it: 9–18 iterations instead of hundreds, and no vanishing cells.
5. **Torque conservation is a pair identity, not a global sum.** In a periodic cell the
   individual floe spin torques `r_i x f_i` do **not** cancel pairwise; the exact
   statement is `tau_i + tau_j = (x_j^img - x_i) x f`, which the harness verifies to
   `2.4e-16`. A naive "sum of torques = 0" check would fail on correct code.
6. **Comparison with a disk field needs shape-aware alignment.** A polygon is inscribed
   in its circumscribed disk, so at a common absolute gap threshold the two fields report
   different neighbour shells (`tau = 0.5 r_eq` gives IoU 0.17 at `phi=0.60`). Criterion
   (e) is therefore evaluated at **equal coordination** (`Z = 6`), which measures
   topology rather than the shape offset; both variants are reported in the artifact.
7. **Limits.** (i) The field is geometrically 2D: no ridge keels, no thickness variation,
   no size-dependent thickness. (ii) Laguerre cells are convex by construction, so
   re-entrant (interlocking, "hand-shaped") fragments cannot be represented.
   (iii) Extreme FSD tails cost neighbours: `N=4000` with `LogNormalFSD(0.4)` needs
   `k=128` planes, which dominates the generation time (24 s of the 24.4 s total).
   (iv) The contact law is not integrated here; the floe table is the interface to
   `ice_channel`/`floe_6dof`, and the timestep stability of the polygonal contacts is a
   separate (wave-3) item.

## 6. Error verdict

| quantity | reference | error | `< 3 %`? |
|---|---|---|---|
| realised `m1` of `r_eq` (all four FSD families) | analytic FSD moment | see results table | yes |
| realised `m2` of `r_eq` | analytic FSD moment | see results table | yes |
| cell area vs prescribed target | solver target | see results table | yes |
| SAT signed gap | analytic geometry of two convex polygons | machine precision (`<= 4e-14` on the completeness comparison) | yes |
| contact set | `O(N^2)` brute force | 0 false pairs | yes |
| monodisperse contact topology | circumscribed-disk field | see results table (IoU) | yes (IoU `>= 0.98`) |

The only quantity without an experimental/reference value is the *number* of contacts at
a given concentration (there is no measured coordination number for a real floe field in
this repository); it is reported as a consistency indicator (planar value ~6), not as a
validated physical prediction. Everything else is checked against either an analytic
value or an independent implementation in the same run.

## 7. Reproduction

```bash
# full acceptance harness (writes JSON + figures)
PYTHONPATH=src python benchmarks/polygon_floe_field_acceptance.py \
    --out artifacts/in32_n01_polygon_floe_field.json --n-floes-sweep 200

# tests
PYTHONPATH=src python -m pytest tests/test_polygon_floe_field.py -q

# regenerate the results tables / report from the artifact
PYTHONPATH=src python benchmarks/in32_n01_report.py
```

Cluster run (loongarch64, 128 cores/node) used for the independent timing of criterion
(f): `run_cluster(action="run", repo="TensorDEM", stage_dir=<worktree>, node="n330")`;
wall time of the whole harness 294 s, best `N=4000` field 19.9 s.