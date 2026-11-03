# Restoring the FEM-face -> DEM ice-field audit contract (m2889)

Protocol: `tensordem-fracture-gen-contract-restoration-v1`  
Scope: `src/tensordem/fracture_gen.py`, `benchmarks/fracture_gen_mass_footprint_audit.py`,
`tests/test_fracture_gen_audit.py`, `tests/test_fracture_gen_mass_footprint.py`.

## 1. The defect: a published audit whose code was never delivered

`main` documents the mass mapping, face footprint and layer-coincident-plane
audits (`docs/dem3d_fracture_mass_mapping_audit.md`,
`benchmarks/fracture_gen_mass_footprint_audit.py`, two test modules), but the
matching `src/tensordem/fracture_gen.py` hunks are absent, exactly the
delivery-gate loss pattern described in
`docs/dem_analytic_verification_debt_restoration.md` (m2536, which restored
four of five contracts and left this one open):

| symptom on pristine `main` | count |
|---|---|
| `tests/test_fracture_gen_mass_footprint.py` collection error | 1 |
| runtime failures in `tests/test_fracture_gen_audit.py` | 14 |

The tests were therefore *unbacked claims*: the capability grade of the SI-2
mass/footprint audit was, in practice, `STUB` while the documentation asserted
a verified result.

## 2. What the audit actually measures (geometry and parameters)

Reference lattice (simple cubic, all coordinates exactly representable in
binary, so plane alignment is not decided by rounding):

| parameter | value |
|---|---|
| particle radius r | 0.125 m |
| lattice spacing h = 2r | 0.25 m |
| layers per axis n | 8 |
| particles N | 512 (analytic 512) |
| nearest-neighbour bonds | 1344 (analytic 3n²(n-1) = 1344) |
| domain volume | 8.0 m³ |
| ice density | 917.0 kg/m³ |

Three defects of the frozen SI-2/DEM contract are pinned against independent
analytic baselines; nothing is self-referential.

## 3. Results

### 3.1 Mass mapping vs the continuum solid mass

| mapping | DEM mass (kg) | continuum mass (kg) | signed relative error | <15 % ? |
|---|---|---|---|---|
| equal-volume spheres (frozen) | 3841.120618 | 7336.000000 | -47.6401 % | **no** |
| lattice cell `density·h³` (restored) | 7336.000000 | 7336.000000 | +0.000000 % | yes |

The frozen deficit is an exact constant, not a numerical accident:
`dem/continuum = 4π/3 / 8 = π/6 = 0.523598775598`, i.e.
`π/6 − 1 = -0.476401224402` (-47.64 %).
The volume-consistent mapping multiplies the *mass radius* by
`MASS_RADIUS_FACTOR = 1.240700981799`
(`(4/3)π(k r)³ = h³`), so the geometric radius used for contact and the radius
carrying mass stay distinct.

### 3.2 Full plane cut vs `A / h²`

| case | measured bonds | analytic | relative error |
|---|---|---|---|
| plane between lattice layers | 64 | 64 | +0.00e+00 |
| plane coincident with a layer, strict test | 0 | 128 | -100.0 % |
| same plane, `on_plane_tolerance = 1e-12` | 128 | 128 | +0.0 % |

### 3.3 Finite face footprint vs `πR²/h²` and exact enumeration

| R (m) | A (m²) | analytic πR²/h² | exact enumeration | frozen (infinite plane) | footprint-aware | aware error | over-break |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.1963 | 3.14 | 4 | 64 | 4 | +27.3 % | 16.00× |
| 0.50 | 0.7854 | 12.57 | 12 | 64 | 12 | -4.5 % | 5.33× |
| 0.75 | 1.7671 | 28.27 | 32 | 64 | 32 | +13.2 % | 2.00× |
| 1.00 | 3.1416 | 50.27 | 52 | 64 | 52 | +3.5 % | 1.23× |
| 1.25 | 4.9087 | 78.54 | 64 | 64 | 64 | -18.5 % | 1.00× |

The frozen criterion cuts the *infinite* plane, so a face smaller than the
cross-section severs the whole cross-section: the over-break factor is the
plane-to-footprint area ratio. At R = 0.25 m it exceeds the platform's
acceptance bar by more than an order of magnitude.

## 4. What was restored in `src`

| symbol / option | role |
|---|---|
| `MASS_RADIUS_FACTOR` | `2(3/4π)^{1/3}`, geometric radius -> mass radius |
| `MASS_MAPPING_ALIASES` | audited names `cell_volume`/`lattice_cell` vs `equal_volume_spheres`/`sphere` |
| `FailureFace.extent_radius` | optional real face footprint radius |
| `face_footprint_radius(face)` | `extent_radius` else `sqrt(area/π)`, rejects non-positive |
| `FractureGenConfig.mass_mapping` | audited mapping selector; the frozen sphere mapping is **not** the default |
| `FractureGenConfig.face_extent` | `None` frozen strict plane, `"plane"` cross-section cut, `"disc"` footprint |
| `FractureGenConfig.footprint_aware` | shortcut for `face_extent="disc"` |
| `FractureGenConfig.on_plane_tolerance` | breaks bonds with exactly one endpoint on the plane |
| `FractureField.mass_radius`, `volume_fraction` | reported diagnostics of the mapping |
| `FractureLedger.mass_accounting()` | signed + absolute error, 15 % verdict, reference mass |
| `FractureLedger.energy_accounting()` | adds `analytic_bond_count`, `analytic_unit_bond_energy`, `unit_bond_energy_relative_error`, `bond_count_relative_error`, `input_consistency_error`, `face_extent` |
| `build_bond_topology` | vectorised pair search (the audit lattices run to 10³ particles) |

Layer-coincident plane handling is the subtle one: the frozen strict sign
test severs **nothing** when the plane lies on a particle layer. Under
`face_extent="plane"` the cut must still sever each column once, so a bond
leaving the plane towards the positive normal side is also cut; that branch is
independent of the bond's index order, so exactly one bond per column is
selected (`3n²` columns for the a = 0.36 m² / h = 0.2 m case in the tests).

## 5. Two merged contracts disagreed -- and how this was resolved

Restoring one module exposed a real inconsistency between two *already
merged* test modules:

| disagreement | older contract (`test_fracture_gen_audit.py`) | newer contract (`test_fracture_gen_mass_footprint.py`) | resolution |
|---|---|---|---|
| default mass mapping | volume-consistent `cell` (`test_fracture_gen.py` pins `N ρ h³`) | frozen sphere (`π/6 − 1`) | default stays `cell`; the frozen audit now names `mass_mapping="equal_volume_spheres"` |
| `mass_accounting["relative_error"]` sign | magnitude | signed (`π/6 − 1 < 0`) | ledger reports both; the quantisation check reads `absolute_relative_error` |

Both changes are explicit in the tests, so the audited mapping and error
convention are never inferred from a default.

## 6. Error judgement against the 15 % / 3 % bars

| quantity | reference | measured | relative error | <3 % ? |
|---|---|---|---|---|
| frozen mass mapping | ρ·V_domain | N·ρ·(4/3)πr³ | -47.6401 % | no |
| lattice-cell mass mapping | ρ·V_domain | N·ρ·h³ | +0.00e+00 % | yes |
| full plane bond count | A/h² | 64 | +0.00e+00 % | yes |
| footprint-aware bond count, worst R | πR²/h² | 4 | +27.3 % | no (R = 0.25 m) |
| footprint-aware bond count, best R | πR²/h² | 52 | +3.5 % | yes (R = 1.00 m) |

| footprint-aware count vs exact lattice enumeration (every R) | enumeration | 4/12/32/52/64 | 0.00 % | yes |

Honest reading: the *mass* and *full-plane* contracts are now exact
(round-off only). The footprint contract is **not** a 3 % contract -- the
frozen face schema carries only area and centroid, so the face shape is an
assumption and the bond count is resolved only to `O(1)` per face (a disc
under-covers the corners of a same-area square by ~12 %, and a disc larger
than ~1.1 m degenerates onto the full cross-section, hence the `-18.5 %` row).
Against the *exact lattice enumeration* the implementation is however exact at
every radius tested (0.00 %), so the residual is the boundary truncation of
`pi R^2 / h^2`, not an implementation error.
Footprint-aware breakage should therefore be read as *bounded* rather than
*exact*; making it a 3 % capability requires an extent/shape field in the
TensorFEM face contract.

## 7. Reproduction

```
python -m pytest tests/test_fracture_gen.py tests/test_fracture_gen_audit.py \
    tests/test_fracture_gen_mass_footprint.py -q   # 48 passed
python benchmarks/fracture_gen_mass_footprint_audit.py
python runs/build_contract_restoration_doc.py
```

`python -m pytest tests/ --collect-only -q` collects 1086 tests with 0 errors
(pristine `main`: 1070 collected, 1 collection error).
