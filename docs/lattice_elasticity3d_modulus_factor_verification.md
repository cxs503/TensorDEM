# Lattice Elasticity 3D — Modulus Factor Verification

## 1. Case Introduction

**Physical background:** The `IceDEM3D` solver uses a simple-cubic bonded-sphere lattice
with 26 neighbours per interior particle (6 face + 12 edge-diagonal + 8 body-diagonal
bonds, stored as 13 half-space bonds per node).  The bond law is a central axial spring
with stiffness `k` (N/m) and breaking strain `eps_b`.

The **engineering Young's modulus** `E` of this lattice is not `k/a` (where `a = 2r` is
the lattice spacing).  It is derived from the affine (Born) homogenisation of the bond
network:

```
C_ijkl = (1/a³) Σ_bonds k·l²·n_i·n_j·n_k·n_l
```

For the 26-neighbour shell this gives:

| Constant | Value (in k/a) |
|----------|----------------|
| C11      | 13/3 ≈ 4.333   |
| C12      | 7/3  ≈ 2.333   |
| C44      | 7/3  ≈ 2.333   |

The uniaxial-stress modulus is:

```
E = C11 - 2·C12²/(C11+C12) = 2.7 · k/a
```

The **modulus factor** `f = E/(k/a) = 2.7` is a topological invariant of the
26-neighbour simple-cubic lattice.

**Why this matters:** The function `bond_stiffness_for_target_modulus(E, a, f)`
inverts `E = f·k/a` to recover `k`.  If `f` is wrong (e.g. the lattice topology
changes), the recovered stiffness will be wrong and the solver will produce
incorrect elastic responses.  The existing test used `f=2.7` directly without
deriving it from the topology, so a topology change would not be detected.

## 2. Method

### 2.1 Independent derivation of the modulus factor

1. Compute `C11, C12, C44` from `periodic_elastic_constants(k, a)` using the
   exact Born homogenisation formula over the 13 half-space neighbour offsets.
2. Compute `E` from `derived_moduli(C11, C12, C44)` using the cubic-crystal
   formula `E = C11 - 2·C12²/(C11+C12)`.
3. Derive `f = E/(k/a)`.

### 2.2 Round-trip verification

For each `(k, a)` pair:
1. Derive `f` from the Born pipeline.
2. Compute `E = f·k/a`.
3. Recover `k' = bond_stiffness_for_target_modulus(E, a, f)`.
4. Assert `|k' - k| / k < 1e-8`.

### 2.3 Cross-checks

- `nu = C12/(C11+C12)` must equal `isotropic_poisson_from_cauchy`.
- `cauchy_residual = C12 - C44` must be zero (central-force network).
- `K = (C11 + 2·C12)/3` must hold.
- `Zener = 2·C44/(C11-C12)` must hold.
- `G_reuss < G_voigt` must hold.

## 3. Results

| Test | Result |
|------|--------|
| `factor_derived_from_constants_matches_hand_value` | PASS (f = 2.700000000000, hand = 2.700000000000) |
| `round_trip_exact_for_derived_factor` | PASS (k = 2000.0, k' = 2000.0) |
| `round_trip_exact_for_hardcoded_factor` | PASS (k = 2000.0, k' = 2000.0) |
| `round_trip_across_stiffness_range` | PASS (k = 100…1e6, max rel error < 1e-10) |
| `round_trip_across_spacing_range` | PASS (a = 0.01…0.5, max rel error < 1e-10) |
| `factor_is_resolution_independent` | PASS (spread = 0.0) |
| `poisson_from_cauchy_matches_formula` | PASS (diff < 1e-15) |
| `cauchy_residual_is_zero` | PASS (diff < 1e-12) |
| `bulk_modulus_formula` | PASS (diff < 1e-12) |
| `zener_anisotropy_formula` | PASS (diff < 1e-12) |
| `reuss_shear_below_voigt` | PASS (G_reuss < G_voigt) |
| `invalid_arguments_rejected` | PASS (all 9 invalid inputs rejected) |

**All 13 tests pass.**

## 4. Error Assessment

| Quantity | Analytic | Measured | Relative Error | < 3%? |
|----------|----------|----------|----------------|-------|
| Modulus factor f | 2.700000000000 | 2.700000000000 | < 1e-12 | ✅ Yes |
| Round-trip k (k=2000) | 2000.0 | 2000.0 | < 1e-12 | ✅ Yes |
| Round-trip k (k=1e6) | 1000000.0 | 999999.9999999999 | < 1e-10 | ✅ Yes |
| Cauchy residual | 0.0 | < 1e-12 | N/A | ✅ Yes |

## 5. Notes

- The modulus factor `f = 2.7` is a **topological invariant** of the 26-neighbour
  simple-cubic lattice.  It is independent of `k` and `a`.
- If the lattice topology changes (e.g. adding body diagonals, switching to a
  different coordination), the factor must be re-derived from the Born pipeline.
- The `bond_stiffness_for_target_modulus` function itself is a simple algebraic
  inversion: `k = E·a/f`.  The test verifies that the factor `f` used in the
  function matches the actual topology.
- The existing test `test_required_stiffness_round_trip` in
  `test_dem3d_lattice_elasticity3d.py` also passes but uses a hardcoded `f=2.7`
  without deriving it.  This new test closes that gap.
