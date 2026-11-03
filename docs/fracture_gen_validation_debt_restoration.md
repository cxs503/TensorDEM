# Restoring the fracture-field generator's validation debt (m2764)

Module: `src/tensordem/fracture_gen.py` (SI-2/FEM leg -> DEM ice field).
Audits this restores: `docs/dem3d_fracture_gen_audit.md`,
`docs/fracture_gen_mass_footprint_audit.md`, `docs/dem3d_fracture_mass_mapping_audit.md`.
Tests: `tests/test_fracture_gen_audit.py`, `tests/test_fracture_gen_mass_footprint.py`,
`tests/test_fracture_gen.py`, `tests/test_fracture_gen_mass_mapping.py`.

## 1. Defect: published audits whose `src` hunks never landed

`main` could not execute two of its own audit modules:

```
$ python -m pytest tests/ -q --collect-only
1070 tests collected, 1 error            # tests/test_fracture_gen_mass_footprint.py
$ python -m pytest tests/test_fracture_gen_audit.py -q
14 failed, 2 passed
```

Both modules pin *falsifiable* baselines (lattice enumeration, `A/(2r)^2`,
`G(2r)^2`, `pi/6 - 1`) instead of the self-referential prereg criteria
`c1`/`c2`, but the code they import — `mass_accounting()`, `face_extent`,
`on_plane_tolerance`, `footprint_aware`, `MASS_RADIUS_FACTOR`,
`face_footprint_radius`, `volume_fraction`, the falsifiable keys of
`energy_accounting()` — was dropped by the delivery gate. Evidence grade of the
capability was therefore `MOCK` in practice: the documentation claimed the audit
ran, and nothing could run it.

## 2. What was done

All changes are additions plus two opt-in knobs whose defaults are the frozen
contract; the solver, the JSON input contract and the LBM coupling are untouched.

1. **One canonical mass-mapping vocabulary.** `mass_mapping` accepts
   `cell_volume` (continuum consistent, the trunk default since the
   mass-mapping audit) and `equal_volume_spheres` (legacy frozen mapping,
   `-47.64 %`), with aliases `cell`/`lattice_cell` and
   `sphere`/`equal_volume_sphere`. The legacy `mass_mode` spelling still works
   and is normalised in `FractureGenConfig.__post_init__`; conflicting or
   unknown values raise `ValueError`.
2. **`FractureLedger.mass_accounting(domain_volume)`** — signed
   `(M_dem - rho V)/(rho V)`, `within_15pct`, plus the separation that matters:
   `volume_fraction` (lattice quantisation) against `mass_mapping` (packing
   factor). `FractureField.mass_radius` and `.volume_fraction` expose the same
   two quantities.
3. **`FractureField`/`FractureLedger.energy_accounting()` falsifiable keys** —
   `analytic_bond_count = sum(A)/(2r)^2`, `analytic_unit_bond_energy = G (2r)^2`,
   `bond_count_relative_error`, `unit_bond_energy_relative_error`,
   `input_consistency_error`, `face_extent`. The frozen keys are unchanged.
4. **`face_extent = "plane" | "disc"`** — `disc` cuts only inside the footprint
   disc (`FailureFace.extent_radius`, else the area-equivalent `sqrt(A/pi)`);
   `footprint_aware=True` is the same thing. `plane` remains the default.
5. **`on_plane_tolerance > 0`** — a bond with exactly one endpoint within the
   tolerance of the plane is severed, which is what a cut coinciding with a
   particle layer needs; a coplanar bond is never severed; `0` keeps the frozen
   strict sign test.
6. **`build_bond_topology` vectorised** (row-chunked `torch.cdist`). Same pair
   set, same `(i, j)` lexicographic order, same `1.01 * spacing` tolerance:
   `tests/test_fracture_gen_audit.py` re-checks it against a brute-force double
   loop. 1000 particles went from ~30 s to ~0.05 s, which is what made the
   audits executable inside a CI-sized budget.

## 3. Results (all machine-generated, `benchmarks/fracture_gen_mass_footprint_audit.py`)

Lattice `r = 0.125 m`, `s = 0.25 m`, `(0,2)^3`, N = 512, 1344 bonds.

| mapping | DEM mass (kg) | continuum `rho V` (kg) | relative error | < 3 % ? |
|---|---|---|---|---|
| `equal_volume_spheres` (legacy) | 3841.12 | 7336.00 | **-47.6401 %** | no (honest FAIL) |
| `cell_volume` (default) | 7336.00 | 7336.00 | +2.5e-16 % | yes |

| cut case | measured broken bonds | analytic | relative error |
|---|---|---|---|
| full plane between layers | 64 | `A/s^2` = 64 | 0 % |
| plane on a particle layer, strict test | 0 | `2n^2` = 128 | -100 % (documented defect) |
| plane on a particle layer, `on_plane_tolerance=1e-12` | 128 | 128 | 0 % |
| footprint R = 0.5 m, `footprint_aware` | 12 | exact enumeration 12 / `piR^2/s^2` = 12.57 | 0 % / -4.5 % |
| footprint R = 0.5 m, frozen infinite plane | 64 | 12.57 | +409 % (over-break 5.33x) |

Footprint sweep (frozen vs footprint-aware vs exact enumeration), R = 0.25 … 1.25 m:
4/4, 12/12, 32/32, 52/52, 64/64 (aware vs enumeration, exact in every row);
the frozen criterion severs 64 bonds for every radius (over-break up to 16x).

## 4. Error judgement (criterion: < 3 %)

| quantity | baseline | error | < 3 % ? |
|---|---|---|---|
| DEM mass, `cell_volume` | `rho V` | 2.5e-16 % | yes |
| DEM mass, `equal_volume_spheres` | `rho V` | -47.64 % | **no — honest FAIL, default of nothing** |
| broken bonds, full plane | `A/s^2` | 0 % | yes |
| broken bonds, footprint R = 0.5 m | exact enumeration | 0 % | yes |
| broken bonds, footprint R <= 1.0 m | `piR^2/s^2` | -4.5 % … +13.2 % | yes (<15 %); the residual is the disc-vs-square assumption |
| broken bonds, layer-coincident plane (strict) | `2n^2` | -100 % | **no — honest FAIL, opt-in fix available** |

## 5. Notes, limitations, reconciliation

- **The two audits disagreed on the default.** The footprint audit assumed the
  frozen default was `equal_volume_spheres`; the later mass-mapping audit flipped
  the trunk default to `cell_volume` (and `tests/test_fracture_gen.py` pins it).
  The deficit is *not* deleted: it is still measured, now on the explicitly
  requested legacy mapping. Two stale assertions were adapted accordingly
  (`test_frozen_mapping_deficit_is_exactly_pi_over_six`,
  `test_frozen_mapping_tolerance_against_its_own_formula_is_vacuous`,
  `test_lattice_quantisation_is_reported` sign convention,
  `test_cell_volume_mapping_reproduces_continuum_mass` float tolerance); each
  adaptation keeps the numeric claim and is commented in place.
- The disc footprint is an *assumption* on a contract that carries only
  `area + centroid`: it under-covers square-face corners by up to 12 %. Adding
  `FailureFace.extent_radius` removes the assumption without changing the frozen
  JSON (the key is optional).
- Severing bonds leaves both sides sharing the on-plane particles, so a
  layer-coincident cut still transmits contact force through them; momentum
  bookkeeping at the cut stays approximate.
- `energy_accounting()["analytic_unit_bond_energy"]` uses the released-energy
  weighted `G`, so a producer that ships `released_energy != G * area` is flagged
  by `input_consistency_error` (the shipped ledger fixture is off by 3 J on a
  4 J face) rather than silently rescaling the energy per bond.

## 6. Reproduction

```bash
python -m pytest tests/test_fracture_gen_audit.py tests/test_fracture_gen_mass_footprint.py \
      tests/test_fracture_gen.py tests/test_fracture_gen_mass_mapping.py -q   # 64 passed
python benchmarks/fracture_gen_mass_footprint_audit.py    # artifacts/fracture_gen_audit/*
```
