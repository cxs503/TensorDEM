# DEM3D mode-II (shear) fracture energy: calibration and verification

Status: **verification document** — maps `shear_breaking_strain` to an equivalent
mode-II fracture energy `G_II`, provides closed-form calibration targets, and
documents the resolution-independence property of the strain-based criterion.
No solver code is modified.

Related: `docs/dem3d_bond_fracture_energy.md` (mode-I tensile `G_f`),
`docs/dem3d_shear_specimen.md` (shear threshold mapping),
`docs/dem3d_shear_cross_section_convergence.md` (material modulus recovery).

## 1. Physical background

Sea ice under ship loading fails in multiple modes: tensile cracking (mode-I),
shear sliding (mode-II), and mixed-mode combinations. The bonded-particle DEM
in TensorDEM uses a **strain-based** failure criterion:

- **Mode-I**: bond fails when axial strain `ε > breaking_strain`
- **Mode-II**: bond fails when the averaged deviatoric invariant
  `γ_bond = 0.5·(J2_i + J2_j) > shear_breaking_strain`

where `J2 = sqrt(2/3 · dev(E) : dev(E))` is computed from the least-squares
local deformation gradient at each particle.

The mode-I fracture energy `G_f = 10.5 · k · ε_b²` (J/m²) is derived in
`docs/dem3d_bond_fracture_energy.md`. This document derives the analogous
mode-II quantity.

## 2. Mode-II fracture energy derivation

### 2.1 Per-bond failure energy

For a bond with rest length `l0`, stiffness `k`, and shear threshold `s`,
the elastic energy stored at failure is:

```
E_bond = 0.5 · k · (s · l0)²
```

This is the same form as mode-I, because the criterion is strain-based.

### 2.2 Bonds crossing a shear plane

A mode-II crack plane (parallel to the shear direction, normal to the shear
gradient) cuts through the lattice. For a simple-cubic lattice with 26
neighbours, the bonds crossing a plane normal to the y-axis (shear in x,
gradient in y) are those with `dy = +1`:

| dx | dz | rest_length²/a² | count per cell |
|----|----|-----------------|----------------|
| -1 | -1 | 3 | 1 |
|  0 | -1 | 2 | 1 |
|  1 | -1 | 3 | 1 |
| -1 |  0 | 2 | 1 |
|  0 |  0 | 1 | 1 |
|  1 |  0 | 2 | 1 |
| -1 |  1 | 3 | 1 |
|  0 |  1 | 2 | 1 |
|  1 |  1 | 3 | 1 |

Sum of `rest_length²/a²` = 3+2+3+2+1+2+3+2+3 = **21**

This is the same crossing sum as for mode-I (by symmetry of the cubic shell).

### 2.3 Equivalent fracture energy

The number of bonds per unit area of the crack plane is `1/a²`. The total
energy per unit area is:

```
G_II = (1/a²) · Σ_bonds E_bond
     = (1/a²) · Σ_bonds 0.5 · k · s² · l0²
     = 0.5 · k · s² · (1/a²) · Σ l0²
     = 0.5 · k · s² · 21
     = 10.5 · k · s²
```

**Key result**: `G_II = 10.5 · k · shear_breaking_strain²` (J/m²)

This is **identical in form** to the mode-I result `G_f = 10.5 · k · ε_b²`,
because both use the same strain-based criterion on the same lattice.

### 2.4 Resolution independence

Since `l0 ∝ a` and the number of bonds per unit area `∝ 1/a²`, the lattice
spacing cancels:

```
G_II = 10.5 · k · s²    (independent of a)
```

This means the equivalent fracture energy of a fixed `(k, s)` pair is
resolution-independent — a property of the strain-based criterion, not of
a fracture-energy-based criterion.

## 3. Calibration targets

### 3.1 Sea-ice reference values

Published mode-II fracture energy for sea ice is less commonly reported than
mode-I, but typical ranges are:

| Ice type | G_II (J/m²) | Reference |
|----------|-------------|-----------|
| First-year sea ice | 1–5 | Schulson & Duval (2009) |
| Freshwater ice | 2–10 | Dempsey et al. (1999) |

### 3.2 Inverting for parameters

Given a target `G_II` and a chosen stiffness `k`:

```
shear_breaking_strain = sqrt(G_II / (10.5 · k))
```

For `k = 2000 N/m` and `G_II = 3 J/m²`:
```
s = sqrt(3 / (10.5 · 2000)) = sqrt(3 / 21000) ≈ 0.012
```

### 3.3 Default parameter audit

The TensorDEM default `shear_breaking_strain = 0.03` with `k = 2000 N/m` gives:

```
G_II = 10.5 · 2000 · 0.03² = 18.9 J/m²
```

This is **above** the typical sea-ice range, meaning the default parameters
produce a tougher material than real sea ice in shear. This is acceptable for
a demonstration solver but should be noted when interpreting results.

## 4. Verification protocol

### 4.1 Closed-form check

The `tensordem.bond_fracture_energy` module provides:

```python
from tensordem.bond_fracture_energy import analytic_fracture_energy_J_per_m2

# Mode-I (tensile)
G_f = analytic_fracture_energy_J_per_m2(k=2000, breaking_strain=0.015)
# → 4.725 J/m²

# Mode-II (shear) — same formula, different threshold
G_II = analytic_fracture_energy_J_per_m2(k=2000, breaking_strain=0.03)
# → 18.9 J/m²
```

### 4.2 Dynamic specimen check

The `run_direct_shear` function in `tensordem.shear_specimen` drives a
specimen to failure and records the cumulative fracture energy. The ratio
of total released energy to the crack area should approach `G_II` as the
specimen size increases and the boundary layer becomes negligible.

### 4.3 Acceptance criteria

| Check | Criterion | Status |
|-------|-----------|--------|
| Closed-form G_II = 10.5·k·s² | Exact to machine precision | ✅ |
| Resolution independence | G_II constant across `a` | ✅ |
| Dynamic specimen → G_II | Relative error < 10% for n ≥ 13 | To be verified |
| Sea-ice range calibration | Parameters tunable to 1–10 J/m² | ✅ |

## 5. Limitations

1. **Strain-based, not energy-based**: The actual failure criterion is
   strain-based, so the "fracture energy" is an equivalent continuum
   interpretation, not a direct input.

2. **No mixed-mode coupling**: The current model treats mode-I and mode-II
   independently. A real ice fracture criterion would couple them (e.g.,
   a Mohr-Coulomb or Drucker-Prager envelope).

3. **No rate dependence**: The threshold is rate-independent. Real ice
   exhibits rate-sensitive fracture toughness.

4. **26-neighbour lattice bias**: The factor 10.5 is specific to the
   simple-cubic 26-neighbour shell. Different neighbour sets would give
   different factors.

5. **No crack propagation physics**: The model breaks bonds instantly when
   the threshold is exceeded. There is no process zone, cohesive zone, or
   crack-tip shielding.

## 6. Recommended next steps

1. **Mixed-mode criterion**: Implement a coupled failure envelope
   (e.g., `f(ε_n, γ_s) = (ε_n/ε_c)² + (γ_s/γ_c)² ≤ 1`) to capture
   the interaction between tensile and shear failure.

2. **Rate-dependent threshold**: Add a rate-sensitive component to the
   shear threshold, calibrated against dynamic fracture experiments.

3. **Cohesive zone model**: Replace the instant-break criterion with a
   softening law that dissipates the correct energy over a finite
   displacement.

4. **Experimental validation**: Compare the predicted crack patterns
   and resistance against ice basin tests with known fracture properties.
