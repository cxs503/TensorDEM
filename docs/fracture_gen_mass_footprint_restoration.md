# SI-2/DEM fracture-field generator: restoring the unbacked verification debt

**Scope of this document.** `tensorDEM`'s FEM-failure-face → DEM ice-field generator
(`src/tensordem/fracture_gen.py`, frozen input contract of PR #70) carries three merged
verification artefacts — an audit document (`docs/dem3d_fracture_gen_audit.md`, m2287),
a footprint audit document + benchmark (`docs/fracture_gen_mass_footprint_audit.md`,
`benchmarks/fracture_gen_mass_footprint_audit.py`, m2245) and four test modules — while the
*source implementation those artefacts test* was repeatedly dropped by the delivery gate
(`m2185`, `m2245`, `m2287` commits all end with "N 个文件被门禁滤除"). The result was the
classic "verified but unbacked" state: `tests/test_fracture_gen_mass_footprint.py` could not
even be **collected** (`ImportError: MASS_RADIUS_FACTOR`, `face_footprint_radius`), and
`tests/test_fracture_gen_audit.py` failed with `TypeError: unexpected keyword argument
'mass_mapping'`. Published audit tables therefore had no reproducible code behind them.

This increment restores the missing implementation without touching the fused physics, and
records — with numbers — which of the published claims become reproducible and which remain
open.

## 1. What was restored

`src/tensordem/fracture_gen.py` (426 → 691 lines, single file, default numerical path of the
bond-cutting unchanged):

| restored API | semantics | fixes |
|---|---|---|
| `MASS_MAPPINGS = {equal_volume_spheres, cell_volume, lattice_cell}`, `MASS_MAPPING_ALIASES = {sphere→equal_volume_spheres, cell→cell_volume}`, `MASS_RADIUS_FACTOR = 2(3/4π)^{1/3} = 1.2407009818` | three named mass mappings, legacy `mass_mode` spellings still accepted | `ImportError` in the footprint test |
| `FailureFace.extent_radius` | in-plane footprint radius; `None` → area-equivalent `sqrt(A/π)` | `ImportError` |
| `FractureGenConfig.{mass_mapping, face_extent, footprint_aware, on_plane_tolerance}` | validated opt-ins; unknown values raise `ValueError` | `TypeError` in the audit test |
| `FractureField.{mass_mapping, mass_mode (legacy spelling), cell_volume, mass_radius, volume_fraction}` | `mass_radius` is the *mass-carrying* radius (`MASS_RADIUS_FACTOR·r` for `lattice_cell`, `r` otherwise) | API gap |
| `face_footprint_radius(face)` | explicit `extent_radius` wins, else `sqrt(area/π)`; rejects `≤ 0` | `ImportError` |
| `identify_bonds_crossing_face(..., face_extent="plane"|"disc", footprint_aware, on_plane_tolerance)` | `disc` additionally requires the bond/plane intersection to fall inside the footprint disc; `on_plane_tolerance > 0` severs bonds with exactly one endpoint on the plane and keeps coplanar bonds; degenerate normals rejected | footprint over-break, layer-coincident-plane blind spot |
| `FractureLedger.{mass_accounting, lattice_volume}` | signed `relative_error = (M_DEM − ρV)/ρV`, `absolute_relative_error`, `within_15pct`, `lattice_volume_covered`, `fem_continuum_mass`, `dem_mass`, `mass_mapping` | missing `KeyError` keys |
| `FractureLedger.energy_accounting` additions | `analytic_bond_count = ΣA/(2r)²`, `analytic_unit_bond_energy = Ḡ(2r)²`, `unit_bond_energy_relative_error`, `input_consistency_error = Σ|released_energy − G·A|` | missing falsifiable keys |

Defaults are deliberately *not* silently improved:
`mass_mapping` defaults to the frozen `equal_volume_spheres`, `face_extent` defaults to
`"plane"`. Those are exactly the two settings the audit proves to be wrong (−47.64 % mass,
up to 16× over-break); they stay default so the audited FAIL remains reproducible and no
published SI-2 number changes meaning. `cell_volume` / `disc` are the documented opt-ins.

## 2. Contract reconciliation (the only test edits)

Two merged test modules were written against incompatible drafts of the same API, and one
merged test predates the audit contract. Three minimal, semantics-preserving edits reconcile
them, each cited to the *newer* merged document:

1. `tests/test_fracture_gen.py` — four construction sites get `mass_mapping="cell_volume"`.
   The file (m2185, 09:36) asserts "total mass = ρ·N·h³" from a default config; the audit doc
   merged later (`docs/dem3d_fracture_gen_audit.md`, m2287, 11:40) freezes
   `"equal_volume_spheres"` as the default and `cell_volume` as the opt-in. The assertions
   themselves are unchanged; only the mapping they ask for is now explicit.
2. `tests/test_fracture_gen_audit.py` — one assertion compares the *signed*
   `relative_error` with a magnitude `abs(0.4³ − 0.5³)/0.5³`; it now compares magnitudes.
   The sign convention (DEM relative to continuum, negative = mass lost) is required by the
   footprint test (`relative_error == π/6 − 1`) and by the published tables, so the
   unsigned expectation was the stale side.

No other test was modified; no threshold was relaxed. Everything else passes unmodified.

## 3. Verification

```
$ python -m pytest tests/test_fracture_gen.py tests/test_fracture_gen_audit.py \
      tests/test_fracture_gen_mass_footprint.py tests/test_fracture_gen_mass_mapping.py -q
64 passed
$ python benchmarks/fracture_gen_mass_footprint_audit.py    # exit 0, writes artifacts/fracture_gen_audit/
```

The audit is regenerated programmatically (`artifacts/fracture_gen_audit/{audit.json,
audit_tables.md,fracture_gen_audit.png}`), so no number below is transcribed by hand.
Reference lattice: r = 0.125 m, h = 0.25 m, 8³ = 512 particles, 1344 = 3n²(n−1) bonds.

### A2 mass mapping vs the continuum reference ρ·V = 7336.0 kg

| mapping | M_DEM [kg] | ρV [kg] | relative error | < 15 % ? |
|---|---|---|---|---|
| `equal_volume_spheres` (frozen default) | 3841.1206 | 7336.0000 | **−47.6401 %** | no |
| `lattice_cell` / `cell_volume` (opt-in) | 7336.0000 | 7336.0000 | +0.0e+00 | yes |

The deficit is the simple-cubic packing fraction, `π/6 − 1 = −0.4764012244`, matched to
1.1e-16: structural, not refinable. The corrected mapping reproduces `ρV` to round-off
because the domain edge is an integer multiple of `h`; lattice quantisation is a *separate*
loss and is reported by `mass_accounting()["lattice_volume_covered"]` (e.g. an L = 0.5 m box
with h = 0.2 m covers only 0.4 m per axis → −48.8 %).

### A3 plane-cut and footprint bond counts vs analytic §A/(2r)²

| case | measured | analytic | error |
|---|---|---|---|
| full plane between layers | 64 | 64 | 0 % |
| plane *on* a particle layer, strict sign test | 0 | 128 | −100 % |
| same, `on_plane_tolerance = 1e-12` | 128 | 128 | 0 % |
| footprint R = 0.50 m, frozen infinite plane | 64 | 12.57 | +409 % (5.33× over-break) |
| footprint R = 0.50 m, `footprint_aware` | 12 | 12.57 / 12 exact | −4.5 % vs πR²/h², 0 % vs enumeration |
| footprint R = 0.25 m, `footprint_aware` | 4 | 3.14 / 4 exact | +27.3 % vs πR²/h², 0 % vs enumeration |

The πR²/h² column is a continuum estimate; the lattice-exact enumeration is the meaningful
reference and `footprint_aware` matches it exactly at every radius tested (0.25–1.25 m). The
residual "error" against πR²/h² at small R is the discretisation of a 4-bond disc, not a code
error — hence the 15 % band in the test is applied to the analytic area estimate only.

## 4. Error verdict against the platform criteria

| key quantity | reference | relative error | < 3 % ? |
|---|---|---|---|
| lattice particle / bond count (512 / 1344) | analytic n³ / 3n²(n−1) | 0 | yes |
| `cell_volume` total mass vs ρV | 7336.0 kg | 4.3e-16 | yes |
| `equal_volume_spheres` total mass vs ρV | 7336.0 kg | **47.64 %** | **no — honest FAIL, by design** |
| full-plane broken bonds | A/h² = 64 | 0 | yes |
| layer-coincident bonds, tolerant test | 2n² = 128 | 0 | yes |
| footprint disc count (R = 0.25…1.25 m) | exact lattice enumeration | 0 | yes |
| interpolated branch `mass_radius` vs analytic `2(3/4π)^{1/3}` | 1.2407009817988 | 0 | yes |

## 5. Standing limitations (not fixed here)

* `equal_volume_spheres` remains the **default**: any caller that does not pass
  `mass_mapping="cell_volume"` produces an ice field with 52.4 % of the intended mass —
  inertia, added-mass ratio, Archimedes draught and kinetic-energy share are all affected.
  This is an audited FAIL, not a resolved one.
* `identify_bonds_crossing_face` still cuts the *infinite* plane by default; the frozen
  over-break persists unless `face_extent="disc"`/`footprint_aware=True` is set.
* The energy accounting keys are bookkeeping identities: `input_consistency_error` tests the
  *input* (`released_energy` vs `G·A`), it does not verify that the solver dissipates that
  energy at bond deletion.
* Nothing here is physical validation: the anchors are lattice geometry and closed forms for
  the generator, not measured ice loads.
* Pre-existing, unrelated red tests on this commit: `tests/test_dem3d_contact_impact.py`
  (5 failures, reproduced with the pre-change `fracture_gen.py` restored, i.e. not caused by
  this increment).