# Restoring the dropped `shape_representation` contract (P40-N10) + an import-contract gate

**Repo** TensorDEM · **branch** `agent/m13768` · **evidence** `artifacts/p40_n10_restoration/audit.json`
(every number below is written there by `benchmarks/p40_n10_shape_restoration_audit.py`; nothing is
hand-copied) · **scan artefact** `artifacts/import_contract/broken_imports.json`

---

## 1. What was broken (the defect this work fixes)

The suite could not even be collected:

```
ERROR tests/test_freeze_bond.py
ERROR tests/test_ice_structure_contact.py
ERROR tests/test_shape_representation.py
```

A repo-wide scan of every `from tensordem.X import a, b` in `tests/**`
(`tools/import_contract_scan.py`) found **46 unresolved symbols in 3 test modules**
(1074 lines of acceptance tests). The cause is visible in the commit log: deliveries whose
subject reads *"matured mission deliverables（N 个文件被门禁滤除）"* — the merge gate filtered the
**module** while the **acceptance test** was kept. A filtered test module is invisible debt: it
does not go red, it disappears from the verification surface.

| package | unresolved symbols | test module | state after this branch |
|---|---|---|---|
| `tensordem.shape_representation` | 10 | `tests/test_shape_representation.py` | **restored — 11 tests pass** |
| `tensordem.freeze_bond` | 21 | `tests/test_freeze_bond.py` | open (handed off) |
| `tensordem.ice_structure_contact` | 15 | `tests/test_ice_structure_contact.py` | open (handed off) |

## 2. Case, geometry and parameters

P40-N10 asks how far two *representations* of a non-spherical particle are from the true shape:
the elliptical-Hertz contact law and the N-sphere clump. Contact anchor: reduced modulus
`E* = 1e7 Pa`, normal load `F = 1 N`, curvature of the tight direction fixed at `A = 1/0.05 m⁻¹`,
curvature ratio `A/B` scanned 1 … 40. Clump shapes: prolate spheroid `a = b = 0.0194 m`,
`c = 0.0252 m` (aspect 1.3) and a superquadric block `0.022 × 0.018 × 0.016 m`, `n1 = n2 = 3`.
Pile: disks of radius `0.02 m`, density 2500 kg/m³, `E = 5e7 Pa`, `ν = 0.25`, `μ_slide = 0.55`,
`μ_roll = 0.06`, `dt = 1.2e-4 s`, restitution 0.1, per-step velocity factor 0.99.

## 3. Method

* `elliptic_hertz_quadrature` — the **reference**: for a trial contact ellipse `(a_tight, t·a_tight)`
  the half-space displacement under the Hertz pressure is obtained by direct quadrature of the
  Boussinesq kernel (`displacement_quadrature`, polar coordinates around the field point, `1/ρ`
  removed analytically). The compatibility condition `w = δ − (A x² + B y²)/2` fixes the axis ratio
  `t` from `A/B` alone (bisection on `|k_x|/|k_y| = A/B`), then `p0`, the semi-axes and `δ` follow
  from linearity in `p0` and in `p0 a/E*`. **No elliptic integral is used.**
* `elliptic_hertz` — the closed form, delegated to `hertz_ellipsoid.ellipsoid_plane_contact`
  (the library keeps exactly one elliptic-integral implementation of this geometry).
* `equivalent_radius_hertz` — the no-name isotropic shortcut `1/R_eq = (A+B)/2`: this is the error
  source the node quantifies.
* `fit_inscribed_clump` / `clump_surface_deviation` — radial-centre inscribed clump with the exact
  inscribed radius per ball (min boundary distance over 192 directions) and a two-sided surface
  error (body surface → clump and clump skin → body, the latter in the surface band `F > −0.25`).
* `_Pile` / `repose_angle_from_pile` / `run_pile` — explicit soft-particle 2-D DEM (Hertz normal
  spring with restitution-based critical damping, Coulomb tangential spring with per-pair history,
  rolling-resistance torque, walls, per-step viscous factor) and the free-surface repose estimator.

## 4. Results

### 4.1 Elliptical Hertz: closed form vs. independent quadrature

| `A/B` | `k_n` closed form (N/m) | `k_n` quadrature (N/m) | rel. error | equivalent-radius error |
|---|---|---|---|---|
| 1.0 | 31 072.3250595386 | 31 072.3250595386 | −2e-13 % | −1.2e-13 % |
| 1.5 | — | — | 6.8e-10 % | −0.90 % |
| 2.0 | 35 109.4811501 | 35 109.4811503 | −5.6e-10 % | −2.59 % |
| 3.0 | 37 936.4548054 | 37 936.4548055 | −2.8e-10 % | −6.24 % |
| 5.0 | 42 070.2009005 | 42 070.2009005 | −4.3e-11 % | −12.43 % |
| 8.0 | 46 515.2221059 | 46 515.2221064 | −1.1e-9 % | −19.08 % |
| 20.0 | — | — | ≤1e-9 % | — |
| 40.0 | — | — | ≤1e-9 % | −42.63 % |

Circular limit against the analytic Hertz law: `δ` error `−1.0e-13 %` (closed form) and `−2.0e-13 %`
(quadrature), `k_n` error `2.2e-14 %`. Complete elliptic integrals: `K(0.9) = 2.2805491384227703`,
`E(0.9) = 1.171697052781614` vs. the tabulated `1.1716970527816143` (A&S 17.1); the Legendre
relation `K E' + E K' − K K' = π/2` holds to `8.9e-16`.

### 4.2 Inscribed clump convergence

| shape | N = 8 | N = 32 | N = 128 | fill at N = 128 |
|---|---|---|---|---|
| spheroid 1.3 | 0.535 | 0.231 | 0.087 | 4.82 |
| superquadric `n = 3` | 0.547 | 0.257 | 0.087 | 4.36 |

`rel_max` = two-sided max surface error / mean semi-axis. Monotone in `N`, factor 6.2 from
`N = 8` to `N = 128`. The fill column is the **sum** of the ball volumes over the body volume, an
overlap-inflated upper bound (the balls are inscribed, so the *union* never exceeds the body;
`max_inward = 0` to numerical precision).

### 4.3 Pile test bed

| quantity | value | criterion |
|---|---|---|
| released energy (KE+PE at release) | 658.42 J | — |
| peak KE / released energy | 1.81e-3 | < 0.2 |
| final KE / released energy | 1.68e-3 | < 0.05 |
| lowest particle height | 0.0197 m | > 0 |
| static pseudo-dissipation | 1.22e-5 / step | < 1e-2 |
| repose estimator vs. synthetic 32° pile | 30.76° (−1.24°) | ≤ 3° |

## 5. Two defects found inside the restored contract

1. **`E(0.9)` was pinned to a wrong value.** The pre-gate test asserted
   `E(0.9) = 2.0953194538471797` "per Abramowitz & Stegun table 17.1". That number is
   `K(1 − Σ_{n≥1} 2^{n−1} c_n²)`, i.e. the AGM series with its leading `c_0²/2 = k²/2` term
   dropped; the tabulated value is `1.1716970527816143`. The test is corrected and now also pins
   `E ≤ π/2` and the Legendre relation, so the value cannot drift back.
2. **The one-sided tangency clump fit is degenerate.** Solving "ball at `(R(d) − r)d` of radius `r`
   touches the surface" for `r` has *no* isolated solution: the distance to the boundary along `d`
   equals `r` for every `r`, so the constraint set is the whole interval. Measured on the spheroid:
   the bisection drifted to the inscribed sphere, 30 % of the balls collapsed onto the centre and
   the error was **non-monotone in `N`** (0.191 / 0.196 / 0.186 for `N = 8/32/128`). The replaced
   construction (radial centres at `0.65 R(d)`, exact inscribed radii, 0.98 safety factor) is the
   only one in this branch and gives 0.535 → 0.231 → 0.087.

## 6. Limitations / what is *not* done

* `freeze_bond` (21 symbols) and `ice_structure_contact` (15 symbols) remain unresolved: their
  tests still cannot be collected. They are **not** marked `skip` — the debt stays red and visible,
  and two missions are dispatched with the scan artefact as the acceptance anchor.
* The 2-D pile engine is restored for the **`sphere` channel only**. `clump` (rigid disk clusters)
  and `superellipse` (support-function contact) raise `NotImplementedError` instead of silently
  degrading to disks, so `scripts/p40_n10_shape_verdict.py` pile stages covering those channels
  cannot run yet (honest FAIL of that sub-stage, not a silent pass).
* The DEM collapse run (`120` disks, `40 000` steps, per-step velocity factor 0.99) does **not**
  produce a measurable repose angle: under per-step drag the released block reaches a plateau
  instead of a wedge and the estimator correctly returns `NaN`. Reproducing the pre-gate
  repose-vs-shape table therefore needs a re-tuned release (drag factor and step budget) *and* the
  missing clump channel; that is handed off, not claimed.
* The repose estimator keeps the per-bin **maximum** envelope, but measures flank distance to the
  bin's inner edge (a bin-centre convention is biased low by ≈3.6° at 32°, measured).
  Single-seed scatter of the synthetic validation is ±1.5° for the acceptance seed and up to ±4°
  across seeds — the estimator is a 400-point estimator, not a precision instrument.

## 7. How to verify

```bash
PYTHONPATH=src python3 -m pytest tests/test_shape_representation.py -q   # 11 passed
PYTHONPATH=src python3 tools/import_contract_scan.py                     # 36 broken, exit 1
PYTHONPATH=src python3 benchmarks/p40_n10_shape_restoration_audit.py     # rewrites audit.json + png
```

`tools/import_contract_scan.py` is the reusable gate for this defect class: it exits non-zero while
any test import does not resolve, so a filtered delivery can no longer be merged green.

## 8. Error judgement (vs. reference)

| key quantity | reference | error | < 3 %? |
|---|---|---|---|
| `k_n`, `δ`, `a_tight` (elliptical Hertz) | Boussinesq quadrature | ≤ 1.1e-9 % | yes |
| `k_n`, `δ` at `A = B` | analytic Hertz `(9F²/16E*²R)^{1/3}` | ≤ 2.2e-14 % | yes |
| `K(0.9)`, `E(0.9)` | A&S table 17.1 | ≤ 1e-15 rel | yes |
| repose angle of a synthetic 32° pile | 32° | −1.24° (−3.9 %) | **no** (estimator noise, see §6) |
| equivalent-radius `k_n` at `A/B = 8` | quadrature | −19.1 % | **no — this is the node's finding** |