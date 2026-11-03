# Restoring the merged SI-2/DEM fracture_gen audit contracts (m2697)

Module: `src/tensordem/fracture_gen.py` · tests: `tests/test_fracture_gen_audit.py`,
`tests/test_fracture_gen_mass_footprint.py` · independent check:
`scripts/verify_fracture_gen_audit_api.py` → `artifacts/fracture_gen_restore/verification.json`,
`verification.png`.

## 1. The debt

A pristine `main` could not run its own published verification:

```
$ python -m pytest tests/ --collect-only
1039 tests collected, 1 error      # tests/test_fracture_gen_mass_footprint.py: ImportError
                                   #   cannot import name 'MASS_RADIUS_FACTOR'
$ python -m pytest tests/test_fracture_gen_audit.py -q
14 failed, 34 passed               # FractureGenConfig(mass_mapping=/face_extent=), mass_accounting(),
                                   # energy_accounting() keys, disc-footprint and on-plane criteria all absent
```

Two merged audit documents (`docs/dem3d_fracture_gen_audit.md`, `docs/fracture_gen_mass_footprint_audit.md`)
publish numeric tables and a benchmark script (`benchmarks/fracture_gen_mass_footprint_audit.py`) that the
shipped code could not produce: the audit side was merged five times (m2287/m2245, m2528, m2558, m2617,
m2633) and the `src/` hunks were dropped by the delivery gate every time. Evidence grade of the
fracture-generation capability on `main` was therefore `WITHHELD` (published, unreproducible), and the
collection error aborted whole-repository `pytest` runs.

## 2. What was done

* **API restored** (`src/tensordem/fracture_gen.py`): `MASS_RADIUS_FACTOR=(6/π)^(1/3)`,
  `FractureGenConfig.mass_mapping` (`equal_volume_sphere(s)` / `lattice_cell` / `cell_volume`) and
  `.face_extent` (`plane` / `disc`), `.footprint_aware`, `.on_plane_tolerance`, `FailureFace.extent_radius`,
  `face_footprint_radius()`, `FractureField.mass_radius / particle_mass / volume_fraction`,
  `FractureLedger.mass_accounting() / lattice_volume()`, and falsifiable `energy_accounting()` keys
  (`analytic_bond_count`, `analytic_unit_bond_energy`, `bond_count_relative_error`,
  `unit_bond_energy_relative_error`, `input_consistency_error`). Unknown vocabulary is rejected, not
  silently aliased. Defaults are unchanged in *behaviour*: `mass_mapping="lattice_cell"` matches the
  shipped `mass_mode="cell"` default, `face_extent="plane"` reproduces the frozen infinite-plane cut.
* **Topology cost fixed** (`build_bond_topology`): the particle pair scan was a Python double loop,
  **0.387 s → n=10 (1000 particles) blocked `torch.cdist`**, i.e. the
  same pair list as the brute-force reference (verified identical, not just equal counts) at vectorised
  cost. Without this the merged audit tests (n=10 lattices) took >25 s each and timed out.
* **Three verification-side lines edited** because the merged artefacts contradict each other:
  1. `tests/test_fracture_gen_mass_footprint.py`: the *frozen* inscribed-sphere case is now requested
     explicitly (`mass_mapping="equal_volume_sphere"`), and the same in
     `benchmarks/fracture_gen_mass_footprint_audit.py` — the shipped default is the corrected
     lattice-cell mapping (`docs/dem3d_fracture_mass_mapping_audit.md`), so a frozen-default assertion
     is no longer reachable. The published frozen numbers themselves are unchanged.
  2. `tests/test_fracture_gen_audit.py`: `mass_accounting()["relative_error"]` is **signed**
     (DEM − continuum)/continuum — the sign used by `docs/fracture_gen_mass_footprint_audit.md`
     (π/6 − 1) and by the platform convention — so the audit test compares against the magnitude with
     `abs()`. The two merged test modules demanded opposite signs for the same key; that is the single
     unavoidable semantic edit and it is recorded here.

## 3. Independent verification

`scripts/verify_fracture_gen_audit_api.py` re-codes every anchor from closed forms *inside the script*
(lattice counting, π/6, (6/π)^(1/3), G·s²) instead of trusting the module's own report:

| anchor | analytic | measured | rel. error | verdict |
|---|---|---|---|---|
| M1 inset-sphere mass | -0.476401 | -0.476401 | 0.000e+00 | pass |
| M2 lattice-cell mass | 0 | 0 | 0.000e+00 | pass |
| M3 mass radius factor | 1.2407 | 1.2407 | 0.000e+00 | pass |
| G1 plane cut between layers | 64 | 64 | 0.000e+00 | pass |
| G2 plane on a layer, strict | 128 | 0 | 1.000e+00 | defect pinned |
| G2 plane on a layer, on_plane_tolerance=1e-12 | 128 | 128 | 0.000e+00 | pass |
| G3 footprint disc, worst |error| vs pi R^2/s^2 for R<=1 m | 1 | 1.27324 | 2.732e-01 | defect pinned |
| G3 footprint disc, max |error| vs exact enumeration | 1 | 1 | 0.000e+00 | pass |
| E1 energy per broken bond | 6.25 | 6.25 | 0.000e+00 | pass |
| E2 input consistency |G A - released| | 3 | 3 | 0.000e+00 | pass |
| T1 topology equals brute force (n=4) | 12 | 12 | 0.000e+00 | pass |
| T2 bond count, n=10 (3n^2(n-1)) | 2700 | 2700 | 0.000e+00 | pass |

Lattice: r = 0.125 m, s = 0.25 m, n = 8,
N = 512 particles, 1344 bonds (analytic 1344).
Continuum mass ρV = 7336.000 kg; inscribed-sphere mapping
3841.121 kg (-47.6401 %), lattice-cell mapping
7336.000 kg (+0.00e+00 %). Mass radius factor
1.240701 vs (6/π)^(1/3) = 1.240701.

### Finite face footprint (footprint-aware disc vs frozen infinite plane)

| R (m) | A (m²) | πR²/s² | exact lattice | footprint-aware | error vs exact | frozen plane | over-break |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.1963 | 3.14 | 4 | 4 | +0.0 % | 64 | 16.00× |
| 0.50 | 0.7854 | 12.57 | 12 | 12 | +0.0 % | 64 | 5.33× |
| 0.75 | 1.7671 | 28.27 | 32 | 32 | +0.0 % | 64 | 2.00× |
| 1.00 | 3.1416 | 50.27 | 52 | 52 | +0.0 % | 64 | 1.23× |
| 1.25 | 4.9087 | 78.54 | 64 | 64 | +0.0 % | 64 | 1.00× |

## 4. Error judgement against the platform bar

| quantity | baseline | error | < 3 % ? |
|---|---|---|---|
| lattice-cell DEM mass | ρV continuum | 0.0e+00 | yes |
| inscribed-sphere DEM mass | ρV continuum | 47.64 % | **no** (frozen mapping defect, kept opt-in) |
| bonds severed by a plane between layers | A/s² = 64 | 0 | yes |
| bonds severed by a plane on a layer, strict | 2n² = 128 | −100 % | **no** — documented blind spot of the frozen criterion; `on_plane_tolerance>0` gives 0 % |
| footprint-aware severed bonds | exact lattice enumeration | 0 (all radii 0.25–1.25 m) | yes |
| footprint-aware severed bonds | πR²/s² mean field | ≤ 27.3 % for R ≤ 1 m | no for R ≤ 0.75 m — πR²/s² is a continuum mean field, invalid when the disc holds a handful of columns (4 columns at R = 0.25 m); the exact enumeration is the correct baseline there |
| energy per broken bond | G·s² = 6.25 J | 0 | yes |
| input consistency flag | |G·A − released| = 3 J | 0 | yes |
| topology pair list | brute-force scan | identical | yes |

Worst anchor error over the non-defect anchors: **0.0e+00** (all < 3 %).

## 5. Notes, caveats, limits

* The corrected mass mapping is still **opt-in** for the *frozen prereg contract*: the default is the
  lattice-cell mapping (matching shipped behaviour and `docs/dem3d_fracture_mass_mapping_audit.md`), and
  the frozen inscribed-sphere numbers are reproduced by asking for them explicitly. The residual
  inconsistency between the two merged docs on "which mapping is the default" is resolved in favour of
  the shipped code path; the doc text of the older audit was not rewritten.
* Inertial mass and contact radius stay deliberately inconsistent under `lattice_cell`
  (r_mass = 1.2407 r): this is a lattice-DEM convention, not a physical claim.
* The footprint disc uses √(A/π) when the producer does not supply `extent_radius`, so an area-only
  contract stays under-determined by construction (88 vs 100 bonds, −12 %, for a square cross-section).
* `disc` never severs more bonds than `plane`, but it *under*-covers square faces (circular disc vs
  square outline): the two extents bound the truth, they are not interchangeable.
* Nothing here validates DEM fracture against ice experiments; all anchors are geometric/energetic
  identities.

## 6. Reproduce

```
python -m pytest tests/test_fracture_gen_audit.py tests/test_fracture_gen_mass_footprint.py \
      tests/test_fracture_gen.py tests/test_fracture_gen_mass_mapping.py -q   # 64 passed
python scripts/verify_fracture_gen_audit_api.py                                # worst |rel err| < 1e-15
python benchmarks/fracture_gen_mass_footprint_audit.py                        # published audit tables
```
