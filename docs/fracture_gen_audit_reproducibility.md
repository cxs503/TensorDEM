# Restoring reproducibility of the merged SI-2 fracture-generation audits

Status: done · module `src/tensordem/fracture_gen.py` · branch `agent/m2965`

## 1. The debt being paid

Two audit modules were merged onto `main` while the API they specify was not. Measured on
pristine `HEAD` (`git stash`, before any change of this branch):

| artefact | state on `HEAD` |
|---|---|
| `tests/test_fracture_gen_mass_footprint.py` | **collection error** — `ImportError: cannot import name 'MASS_RADIUS_FACTOR'`; it could never execute a single assertion |
| `tests/test_fracture_gen_audit.py` | **14 failed / 34 passed** |
| `benchmarks/fracture_gen_mass_footprint_audit.py` | **cannot import** — the three-defect audit and every number in its tables were unreproducible |
| `benchmarks/fracture_gen_audit.py` | crashed at `KeyError: 'volume_fraction'` |

So the repository published audit tables (mass mapping, face footprint, plane-cut bond count) whose
generating scripts did not run. That is a verification debt of the worst kind: the *claims* were
merged, the *evidence producer* was not.

Root cause is not one missing function but **three mass-mapping naming conventions that grew up
independently**, plus a face-extent model that only existed in prose:

* `fracture_mass_mapping.py` (merged, authoritative): `mass_mode ∈ {cell, sphere}`;
* SI-2 footprint audit: `mass_mapping ∈ {lattice_cell, equal_volume_sphere}`;
* ledger audit: `mass_mapping ∈ {cell_volume, equal_volume_spheres}`.

## 2. What was implemented (additive, default numerical path untouched)

1. **Alias table** `MASS_MAPPING_ALIASES` normalising all three conventions onto the two internal
   modes; the caller's label is retained (`FractureField.mass_mapping`) so audit rows stay traceable
   to the convention that produced them. Unknown labels (`lumped`, `unit_solid_volume`) still raise.
2. **`FailureFace.extent_radius`** (optional) and **`face_footprint_radius()`**: explicit in-plane
   radius, else the area-equivalent disc `sqrt(area/pi)`; non-positive/non-finite rejected.
3. **`identify_bonds_crossing_face(..., face_extent, footprint_aware, on_plane_tolerance)`**:
   - `"plane"` — frozen contract, unchanged: every bond straddling the infinite plane breaks;
   - `"disc"` — additionally requires the bond/plane intersection point to lie within the footprint
     disc, so the broken-bond count tracks the declared face area;
   - `on_plane_tolerance > 0` — also severs bonds with exactly one endpoint on the plane, which
     repairs the layer-coincident degeneracy;
   - the face normal is now normalised and a degenerate normal raises instead of silently producing
     a zero-distance "cut".
4. **`FractureGenConfig`** gained `mass_mapping`, `face_extent`, `footprint_aware`,
   `on_plane_tolerance` (all validated); **`FractureField`** propagates them, and gained
   `mass_radius` (`r_m = (6/pi)^(1/3) r` under the volume-consistent mapping), `domain_volume`,
   `volume_fraction` (exact coverage snapped to the integer so audits do not report `1 + O(eps)`).
5. **`FractureLedger.mass_accounting()`** — the external baseline the audits need: reference mass is
   `density * V`, *not* the particle-mass formula, so a mapping defect is visible instead of being
   certified. Accepts `mass_accounting(V)` and `mass_accounting(domain_volume=V)`; zero-argument form
   uses the generated domain; non-positive volume raises (`mass_conservation_error` remains the old
   self-referential check).
6. **`FractureLedger.energy_accounting()`** enriched with falsifiable keys
   (`analytic_bond_count`, `analytic_unit_bond_energy`, `unit_bond_energy_relative_error`,
   `bond_count_relative_error`, `input_consistency_error`, `face_extent`) — criterion c2 was an
   algebraic identity before.
7. **`build_bond_topology` vectorised** (blocked `torch.cdist`, 1024-row blocks) — the pure-Python
   O(N²) double loop made the 512-particle audits take ~6.4 s per field. Verified **bit-identical**:
   pair list and rest lengths equal (`max|diff| = 0.0`) on the 512-particle and a 120-particle
   non-cubic lattice, 14× faster (6.36 s → 0.455 s).

### Contract reconciliation (explicit, not silent)

`main` repaired the mass default: `mass_mode="cell"` is the default and the −47.64 % deficit is
documented as *fixed* (`docs/dem3d_fracture_mass_mapping_audit.md`, `tests/test_fracture_gen_mass_mapping.py`).
The stale SI-2 audit/test assumed the *frozen* mapping was the default. **The repaired default was
kept**; the frozen baseline is now selected by name (`mass_mapping="sphere"`) in the audit and in two
test cases. The honest FAIL is preserved, not papered over: the frozen row still reports −47.64 %
and `within_15pct = False`.

One assertion in `tests/test_fracture_gen_audit.py` was changed from expecting an unsigned
`relative_error` to asserting the signed value **and** `absolute_relative_error`; the module was red
on `HEAD` so no consumer depended on the old sign. All tolerances were left untouched.

## 3. Results (programmatically injected from `artifacts/fracture_gen_audit/audit.json`)

Lattice: r = 0.125 m, s = 2r = 0.25 m, domain (0,2)³ → 8³ = 512 particles, 1344 bonds.

| quantity | measured | baseline | rel. error | < 3 % ? |
|---|---|---|---|---|
| frozen mass mapping (sphere) | 3841.12 kg | 7336.00 kg (rho·V) | **−47.640 %** | no (honest FAIL, unchanged) |
| volume-consistent mapping (cell) | 7336.00 kg | 7336.00 kg | **0.000 %** | yes |
| full-plane cut, bond count | 64 | 64 (`A/s²`) | 0.000 % | yes |
| plane on a particle layer, strict test | 0 | 128 (`2n²`) | **−100 %** | no (degeneracy) |
| same, `on_plane_tolerance = 1e-12` | 128 | 128 | 0.000 % | yes |

Finite-face footprint (frozen infinite plane cuts all 64 bonds regardless of face area):

| R (m) | analytic πR²/s² | exact enumeration | frozen | footprint-aware | vs analytic | vs enumeration | over-break |
|---|---|---|---|---|---|---|---|
| 0.25 | 3.14 | 4 | 64 | 4 | +27.3 % | 0 % | 16.00× |
| 0.50 | 12.57 | 12 | 64 | 12 | −4.5 % | 0 % | 5.33× |
| 0.75 | 28.27 | 32 | 64 | 32 | +13.2 % | 0 % | 2.00× |
| 1.00 | 50.27 | 52 | 64 | 52 | +3.5 % | 0 % | 1.23× |
| 1.25 | 78.54 | 64 | 64 | 64 | −18.5 % | 0 % | 1.00× |

Interpretation of the footprint column: against the **exact lattice enumeration** the disc criterion
is exact (0 %); against the continuum `πR²/s²` it carries the irreducible ±O(1 bond) lattice
quantisation plus, for R = 1.25 m, domain saturation (the disc leaves the 2 m box, so only 64 columns
exist and the continuum baseline is meaningless there). The frozen criterion over-breaks by up to
16× for small faces — that is the audited defect, and it is now *measured by a script that runs*.

## 4. Verification

```
$ python -m pytest tests/test_fracture_gen_audit.py tests/test_fracture_gen_mass_footprint.py \
      tests/test_fracture_gen_mass_mapping.py tests/test_fracture_gen.py -q
64 passed                                       # was 34 passed / 14 failed + 1 collection error
$ python benchmarks/fracture_gen_mass_footprint_audit.py     # exit 0, writes artifacts/fracture_gen_audit/
$ python benchmarks/fracture_gen_audit.py                    # exit 0, writes benchmarks/fracture_gen_audit_report.json
$ python -m pytest tests/test_fracture_gen_mass_footprint.py -q
16 passed
```

Bit-identity check for the topology refactor:

```
big: identical pairs True | (1344,2) | max|d rest_lengths| 0.0 | 0.455 s   (was 6.36 s)
odd: identical pairs True | (286,2)  | max|d rest_lengths| 0.0
```

Isolation check: with the full suite running single-failure-stopped, the first failure was
`tests/test_dem3d_contact_impact.py::SolverImpactTests::test_contact_durations`. Verified **pre-existing
and unrelated**: it fails identically on the stashed baseline (`git stash`), and no file it touches was
modified here.

## 5. Limitations / deliberately not done

* **Frozen defaults were not changed.** `mass_mode="cell"`, `face_extent="plane"` remain the defaults,
  so existing regression artefacts are byte-identical; the new behaviours are opt-in.
* **The footprint contract is area-only.** A circular disc cannot resolve a square face better than
  O(1) bond per face (88 vs 100 bonds for A = 1 m², 12 %); fixing that needs a shape field in the frozen
  FEM→DEM contract, which is an upstream (TensorFEM) change.
* **No TensorFEM cross-library run.** The reference mass is analytic `rho·V`; the FEM-produced mass
  remains a debt in `docs/dem3d_fracture_mass_mapping_audit.md` §6.1.
* The 1.25 m footprint row is reported but not used as an acceptance case (domain saturation).

## 6. Reproduction

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_fracture_gen_audit.py tests/test_fracture_gen_mass_footprint.py -q
python benchmarks/fracture_gen_mass_footprint_audit.py
python benchmarks/fracture_gen_audit.py
```