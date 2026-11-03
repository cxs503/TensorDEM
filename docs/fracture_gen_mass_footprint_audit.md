# Validation debt audit — FEM failure face → DEM ice field (SI-2/DEM)

Module audited: `src/tensordem/fracture_gen.py` (merged 2026-10-21, PR #70).
Audit script: `benchmarks/fracture_gen_mass_footprint_audit.py`
Tests: `tests/test_fracture_gen_mass_footprint.py`
Machine-readable result: `artifacts/fracture_gen_audit/audit.json`
Figure: `artifacts/fracture_gen_audit/fracture_gen_audit.png`

## 1. Why this audit

The module advertises three acceptance criteria (roadmap §"Acceptance criteria"),
the first of which is *mass conservation* with tolerance 1e-12 kg. Read against
the code, criterion 1 compares the generated DEM mass with
`N_particles × density × (4/3)πr³` — **the same formula used to build it**. The
test therefore passes at machine precision while saying nothing about whether
the DEM field carries the mass of the FEM continuum it is supposed to replace.
That is a vacuous criterion, i.e. verification debt, not verification.

Likewise the bond-breakage criterion treats each FEM failure face as an
*infinite plane*: any bond whose endpoints straddle the plane's zero-level set
is severed, independent of the face's area, shape or location. For a genuinely
local cut this over-severs the field, which changes the initial damage state
handed to `IceDEM3D`.

This audit installs independent analytic baselines for both, plus a third
defect found while doing so (plane/layer coincidence), and provides opt-in
behaviour that restores the expected physics while leaving the frozen default
contract untouched.

## 2. Method

Lattice / geometry (exactly representable in binary, so plane alignment is not
decided by floating-point noise): particle radius r = 0.125 m, simple-cubic
spacing s = 2r = 0.25 m, domain (0, 2)³ → n = 8 layers per axis, N = 512
particles, 1344 bonds (3n²(n−1)). Ice density 917 kg/m³. Reference values come
from closed-form lattice counting:

| quantity | analytic baseline |
|---|---|
| particle count | n³ = 512 |
| bond count | 3n²(n−1) = 1344 |
| bonds severed by a plane between layers | A_plane/s² = n² = 64 |
| bonds severed by a face footprint (disc R) | πR²/s², exact enumeration for the discrete lattice |
| bonds severed by a plane through a layer | 2n² = 128 |
| continuum solid mass (reference) | ρ·V = 917 × 8 = 7336 kg |
| DEM mass, frozen mapping | ρ·N·(π/6)s³ = (π/6)·ρV |

Three opt-in knobs were added (all default-off, so the frozen contract and all
pre-existing tests are unchanged):

- `FractureGenConfig.mass_mapping = "equal_volume_sphere" | "lattice_cell"` —
  the corrected mapping sets the particle mass radius to
  `(3/4π)^(1/3)·s = 1.24070 r`, i.e. particle mass = ρs³ (sphere volume = cell
  volume). It is implemented as a separate `mass_radius` property so the
  geometric/contact radius is untouched.
- `FractureGenConfig.footprint_aware` — the segment/plane intersection point is
  projected onto the face plane and required to fall inside a footprint disc of
  radius `FailureFace.extent_radius` (or area-equivalent `√(A/π)` when the
  producer does not supply one; the frozen JSON contract is unchanged because
  the new field is optional).
- `FractureGenConfig.on_plane_tolerance` — counts a bond as severed when exactly
  one endpoint lies within the tolerance of the plane, which matters when the
  cut surface coincides with a particle layer.

New API surface: `FractureLedger.mass_accounting(domain_volume)` (DEM vs
continuum mass with signed relative error and a `within_15pct` flag),
`FractureLedger.lattice_volume()`, `face_footprint_radius(face)`.

## 3. Results

### Mass mapping (continuum reference density × volume)

| mapping | DEM mass (kg) | continuum mass (kg) | relative error | <15 % ? |
|---|---|---|---|---|
| equal_volume_sphere (frozen) | 3841.120618 | 7336.000000 | -47.6401 % | **no** |
| lattice_cell (corrected) | 7336.000000 | 7336.000000 | +0.000000 % | yes |

### Plane-cut bond counts vs analytic A / s²

| case | measured | analytic | relative error |
|---|---|---|---|
| full plane between layers | 64 | 64 | +0.00e+00 |
| plane on a particle layer, strict test | 0 | 128 | -100.0 % |
| plane on a particle layer, on_plane_tolerance=1e-12 | 128 | 128 | +0.0 % |

### Finite face footprint (n = 8, s = 0.25 m)

| R (m) | A (m²) | analytic πR²/s² | exact enumeration | frozen (infinite plane) | footprint_aware | aware error | over-break |
|---|---|---|---|---|---|---|---|
| 0.25 | 0.1963 | 3.14 | 4 | 64 | 4 | +27.3 % | 16.00× |
| 0.50 | 0.7854 | 12.57 | 12 | 64 | 12 | -4.5 % | 5.33× |
| 0.75 | 1.7671 | 28.27 | 32 | 64 | 32 | +13.2 % | 2.00× |
| 1.00 | 3.1416 | 50.27 | 52 | 64 | 52 | +3.5 % | 1.23× |
| 1.25 | 4.9087 | 78.54 | 64 | 64 | 64 | -18.5 % | 1.00× |

Key numbers, all read from `audit.json`:

- **Mass:** frozen mapping gives 3841.12 kg against a continuum 7336.00 kg →
  **−47.64 %** (identically π/6 − 1), i.e. 3.2× the 15 % acceptability bar.
  The corrected mapping gives 7336.00 kg → **+2.5e−16 %**.
- **Full plane between layers:** 64 severed bonds vs A/s² = 64 → **0 %** error,
  confirming the plane-intersection geometry itself is right.
- **Finite face footprint:** frozen criterion severs 64 bonds for *every* face
  radius (0.25 … 1.25 m); footprint-aware severing tracks πR²/s² to −4.5 %…+13.2 % for R ≤ 1.0 m
  and matches the exact lattice enumeration exactly. Over-break factor of the
  frozen criterion reaches **16×** at R = 0.25 m.
- **Layer-coincident plane:** with the strict sign test, a cut plane exactly on
  a lattice layer severs **0** bonds (vs 128 expected) — the FEM cut surface is
  invisible to the DEM field. `on_plane_tolerance = 1e-12` restores exactly 128.

## 4. Error judgement against the platform bar (>15 % unacceptable)

| quantity | baseline | measured error | <3 % ? |
|---|---|---|---|
| DEM mass, frozen mapping | ρV continuum | −47.64 % | **no** (FAIL) |
| DEM mass, lattice_cell mapping | ρV continuum | 2.5e−16 % | yes |
| severed bonds, full plane | A/s² = 64 | 0 % | yes |
| severed bonds, footprint R = 0.5 m | exact enumeration = 12 | 0 % (enumeration), −4.5 % vs πR²/s² | yes |
| severed bonds, footprint R = 0.5 m, frozen criterion | 12 | +433 % | **no** (FAIL) |
| severed bonds, plane on a layer | 2n² = 128 | −100 % (strict), 0 % (tolerant) | **no** with default; yes opt-in |

## 5. Notes, caveats, limits

- The corrected mass mapping is **not** a change of the frozen prereg decision:
  the default remains `equal_volume_sphere`. Downstream work that needs
  continuum-consistent mass (e.g. momentum/energy budgets, buoyancy, added-mass
  comparisons with FEM or experiment) must opt in explicitly, and must accept
  that the *contact* radius stays r while the inertial mass corresponds to a
  larger sphere — a known, deliberate inconsistency of lattice DEM.
- `πR²/s²` is only a valid baseline while the footprint stays inside the plane
  extent; at R = 1.25 m on a 2 m cross-section the footprint clips the domain
  and the analytic count (78.5) exceeds the available 64 columns, giving
  −18.5 %. The enumeration and footprint_aware values agree exactly there.
- Footprint shape is a disc. Non-circular (polygonal) faces are not represented;
  the area-equivalent disc is an assumption that must be revisited if the FEM
  leg emits narrow, elongated cohesive faces (bias is O(perimeter)).
- The bond-cut criterion remains a geometric zero-level-set test: curved
  surfaces, sub-particle cracks, and duplicated-node "true" separation are not
  modelled. Severing bonds leaves both sides sharing the on-plane particles, so
  a layer-coincident cut still transmits contact force through those particles;
  momentum bookkeeping at the cut is therefore approximate.
- Nothing in this audit touched the solver, the LBM coupling, or the frozen JSON
  contract; only `fracture_gen.py`, a new benchmark script, a new test module
  and this document.

## 6. Reproduce

```bash
cd TensorDEM
python3 benchmarks/fracture_gen_mass_footprint_audit.py      # writes artifacts/fracture_gen_audit/*
python3 -m pytest tests/test_fracture_gen_mass_footprint.py tests/test_fracture_gen.py -q
```

All numbers in §3–§4 are emitted by the script into `audit.json` /
`audit_tables.md`; nothing here is transcribed by hand.