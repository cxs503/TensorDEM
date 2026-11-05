# Restoring the dropped `fracture_gen` verification contracts (m2910)

Protocol: `tensordem-fracture-gen-contract-restoration-v1`
Scope: `src/tensordem/fracture_gen.py` (+ 3 merged test/benchmark modules reconciled, +1 vectorisation)
Evidence: `scripts/fracture_gen_contract_report.py` →
`artifacts/fracture_gen_contract_restoration/{summary.json,tables.md,fig_*.png}`,
`artifacts/fracture_gen_audit/{audit.json,audit_tables.md,fracture_gen_audit.png}`

## 1. The debt: published audits that could not execute

`main` before this change shipped two merged verification notes —
`docs/dem3d_fracture_gen_audit.md` and `docs/fracture_gen_mass_footprint_audit.md` —
plus their executable contracts, but **none of the `src` hunks the notes describe**.
The delivery gate had dropped the `fracture_gen.py` side of both PRs (the same failure
mode documented for Hertz / restitution / friction-cone in
`docs/dem_analytic_verification_debt_restoration.md`, where the fifth contract —
`fracture_gen` mass footprint — was explicitly left open).

Observable state of a pristine checkout:

| symptom | before | after |
|---|---|---|
| `pytest tests/ --collect-only` | 1070 collected, **1 error** (`test_fracture_gen_mass_footprint.py`: `ImportError: cannot import name 'MASS_RADIUS_FACTOR'`) | 1087 collected, **0 errors** |
| `pytest tests/test_fracture_gen_audit.py` | **14 failed** / 18 passed (no `mass_mapping`, `face_extent`, `mass_accounting`, falsifiable `energy_accounting`) | 16 passed |
| `pytest tests/test_fracture_gen_mass_footprint.py` | not collectable | 16 passed |

In other words the two published audit *verdicts* (mass −47.64 %, face footprint
over-severed up to 16×) were `WITHHELD` evidence — the numbers existed only in prose.

## 2. What was restored (all opt-in, frozen default path untouched)

| API | semantics |
|---|---|
| `MASS_RADIUS_FACTOR = (6/π)^(1/3) = 1.240701` | width of the mass radius per geometric radius on `h = 2r`; `4/3 π r_m³ = h³` exactly |
| `FractureGenConfig.mass_mapping` | `"equal_volume_spheres"` / `"cell_volume"` (plus the `"equal_volume_sphere"` / `"lattice_cell"` spellings used by the other note); normalised to the internal `mass_mode` (`"sphere"` / `"cell"`). Unknown values raise `ValueError` |
| `FractureGenConfig.face_extent` | `"plane"` (frozen contract: unlimited plane) or `"disc"` (sever only bonds intersecting the face footprint). `footprint_aware=True` is an alias for `"disc"` |
| `FractureGenConfig.on_plane_tolerance` | a bond with exactly one endpoint within `tol` of the plane is severed; `0.0` = frozen strict sign test. Negative rejected |
| `FailureFace.extent_radius` | optional producer extent (m); `None` keeps the frozen JSON contract and derives `√(A/π)` |
| `face_footprint_radius(face)` | `extent_radius` if given, else `√(area/π)`; non-positive → `ValueError` |
| `identify_bonds_crossing_face(..., face_extent=, on_plane_tolerance=, footprint_aware=)` | degenerate (zero) face normal → `ValueError` |
| `FractureField.mass_radius`, `.volume_fraction`, `.mass_mapping`, `.face_extent` | inertial radius vs contact radius made explicit; `volume_fraction = m_p/(ρh³)` |
| `FractureLedger.mass_accounting(domain_volume=None)` | DEM vs `ρV`, signed + absolute residual, `within_15pct` |
| `FractureLedger.lattice_volume()` | `N h³`, the volume the realised lattice actually covers |
| `FractureLedger.energy_accounting()` | adds falsifiable keys `analytic_bond_count = ΣA_i/h²`, `analytic_unit_bond_energy`, `bond_count_relative_error`, `unit_bond_energy_relative_error`, `input_consistency_error = Σ|E_i − G_i A_i|`, `face_extent` |

Nothing was re-derived from the (absent) original PR branches: every helper is a closed
form taken from the merged notes and re-checked against independent lattice enumeration.

### 2.1 `build_bond_topology`: same pair list, 30–140× cheaper

The bond finder was a Python `i,j` double loop with one `torch.linalg.norm` per pair.
A 512-particle audit case therefore took ~4.6 s per field and the whole published audit
was impractical to run (which is *why* the debt went unnoticed). It is now a blocked
distance computation (`512 × N` blocks, upper triangle, same `i < j` lexicographic order).
The pair list is **bit-identical** to brute force (`test_vectorised_topology_matches_brute_force`
plus the table below), and the four fracture-gen suites now run in 9.6 s instead of minutes.

## 3. Results (all numbers machine-generated, `tables.md` / `summary.json`)

### 3.1 Mass mapping vs continuum `ρV` (lattice covers the domain exactly, n = 8, s = 0.25 m)

| mapping | DEM mass [kg] | continuum ρV [kg] | signed error | abs error | < 3 % ? |
|---|---|---|---|---|---|
| `equal_volume_spheres` (frozen) | 3841.120618 | 7336.000000 | −47.640122 % | 4.764e+01 % | **no** (FAIL) |
| `cell_volume` (trunk default) | 7336.000000 | 7336.000000 | +0.000000 % | 0.0 % | yes |

The frozen deficit equals `π/6 − 1` to 1e-15 and is resolution independent (pure geometry).

### 3.2 Bond severing vs lattice analytics

| case | measured | analytic | relative error | < 3 % ? |
|---|---|---|---|---|
| plane between layers | 64 | A/s² = 64 | +0.0 % | yes |
| plane on a particle layer (strict) | 0 | 2n² = 128 | −100.0 % | **no** (frozen) |
| plane on a layer, `on_plane_tolerance=1e-12` | 128 | 2n² = 128 | +0.0 % | yes (opt-in) |
| disc R = 0.25 m | 4 | enumeration 4 | 0 % | yes |
| disc R = 0.50 m | 12 | enumeration 12 | 0 % | yes |
| disc R = 0.75 m | 32 | enumeration 32 | 0 % | yes |
| disc R = 1.00 m | 52 | enumeration 52 | 0 % | yes |

Against the *continuum* `πR²/s²` the disc model is within −4.5 %…+13.2 % for R ≤ 1.0 m
(±½ spacing quantisation of a discrete lattice), while the frozen plane criterion severs
64 bonds for **every** radius — over-severing by up to **16×** at R = 0.25 m.

### 3.3 Error judgement against the platform bar

| quantity | baseline | error | < 3 % ? |
|---|---|---|---|
| DEM mass, `cell_volume` | ρV | 0.0 % | yes |
| DEM mass, `equal_volume_spheres` | ρV | 47.64 % | **no** — honest FAIL of the frozen mapping, documented, not used |
| severed bonds, full plane | A/s² | 0.0 % | yes |
| severed bonds, disc vs exact enumeration | enumeration | 0.0 % (all R) | yes |
| severed bonds, disc vs πR²/s² | πR²/s² | 3.5 %…27.3 % (R ≤ 1.0 m: ≤ 13.2 %) | no at the smallest radius — lattice quantisation, not a model error (see §4) |
| severed bonds, plane on a layer | 2n² | −100 % frozen / 0 % with tolerance | **no** with default — documented limitation |

## 4. Reconciliation of three contradictions inside the merged contracts

The two merged notes (and the m2536 mass-mapping restoration) disagree with each other;
this change had to pick one reading per conflict and says which:

1. **Default mapping.** `docs/dem3d_fracture_mass_mapping_audit.md` (m2536, merged code)
   made `cell_volume` the trunk default; `docs/fracture_gen_mass_footprint_audit.md`
   assumes the default is still the frozen sphere mapping. **Resolution: keep the trunk
   default `cell_volume`** — flipping it back would silently re-introduce a −47.64 %
   mass bias in every downstream DEM field and would change already-merged regression
   numbers. The two "frozen" cases in `tests/test_fracture_gen_mass_footprint.py` and in
   `benchmarks/fracture_gen_mass_footprint_audit.py` now request the legacy mapping
   explicitly, so the audit verdict (π/6 − 1, `within_15pct == False`) is unchanged and
   still enforced, and a new test pins the shipped default as volume consistent.
2. **Sign of the mass residual.** The audit note expects `|error|`, the footprint note
   expects a signed error (−47.64 %). Both are reported: `relative_error` is signed,
   `absolute_relative_error` is its magnitude (`test_lattice_quantisation_is_reported`
   now asserts the magnitude via the latter, plus the sign separately).
3. **Naming.** The two notes name the same mappings `equal_volume_spheres`/`cell_volume`
   and `equal_volume_sphere`/`lattice_cell`. All four spellings are accepted and
   normalised; the reporting name is canonical (`equal_volume_spheres` / `cell_volume`),
   so `mass_accounting()["mass_mapping"]` is stable.

## 5. Notes, caveats, limits

- The frozen `face_extent="plane"` and `on_plane_tolerance=0.0` remain the defaults:
  this change restores the ability to *audit* the frozen contract, it does not silently
  fix it. Downstream users must opt in to `"disc"` / a tolerance.
- Footprint shape is a disc. The FEM→DEM contract carries only area + centroid, so a
  narrow elongated cohesive face cannot be represented; the bias is O(perimeter) and is
  visible as the ±13 % band in §3.2. `FailureFace.extent_radius` is the extension point
  if the FEM leg ever emits a real outline.
- `πR²/s²` is only a valid baseline while the footprint stays inside the plane extent;
  the R = 1.25 m row (analytic 78.5 vs 64 available columns, −18.5 %) is a domain-clipping
  artefact, and the enumeration agrees with the disc model exactly there.
- A layer-coincident cut still severs only the bonds the tolerance rule selects; the DEM
  field keeps transmitting contact force through the on-plane particles, so a "cut" is a
  bond-severed field, not a true geometric separation.
- `test_in_plane_translation_keeps_bond_count` passes only because `torch.arange` renders
  the middle lattice layer as `0.30000000000000004` ≠ plane `0.3`, i.e. the very
  floating-point-coincidence degeneracy documented in §3.2. It is left untouched (it
  passes and its intent — x/y translation invariance — holds), but it is not a robust
  alignment test and should not be cited as one.

## 6. Reproduction

```
python -m pytest tests/test_fracture_gen_audit.py tests/test_fracture_gen_mass_footprint.py \
                 tests/test_fracture_gen_mass_mapping.py tests/test_fracture_gen.py -q   # 65 passed
python -m pytest tests/ -q --collect-only      # 1087 collected, 0 errors
python benchmarks/fracture_gen_mass_footprint_audit.py     # audit.json + audit_tables.md + png
python scripts/fracture_gen_contract_report.py             # summary.json + tables.md + 3 png
```