# Threefield Fragmentation Analytic Verification

**Date**: 2026-11-05  
**Mission**: m4772 (selfstart generalist)  
**Module**: `tensordem.threefield.fragmentation`  
**Test file**: `tests/test_threefield_fragmentation_analytic.py`

## Summary

This document records the analytical verification of the turbulent fragmentation
physics core (`threefield/fragmentation.py`).  The module implements a breakup
criterion for ice floes drifting in a propeller race: floes exceeding a critical
length `L_crit` are fragmented by turbulent bending stresses until all survivors
are below the threshold.  The verification confirms the *model's own internal
consistency* (not physical accuracy against experiments).

## Physics background

### Bending stress formula

A floe of characteristic length `L` and thickness `h` drifting in water with
turbulent fluctuation velocity `u'` feels a distributed drag load:

```
q = 0.5 ρ_w C_d u'² h    [N/m per unit width]
```

Treating the floe as a free-free Euler-Bernoulli beam of unit width supported
at the centre by buoyancy, the maximum bending moment occurs at the centre:

```
M_max = q L² / 8
```

The section modulus for a rectangular cross-section (unit width) is:

```
S = h² / 6
```

Peak bending stress:

```
σ_b = M_max / S = (q L² / 8) / (h² / 6) = 3 q L² / (4 h²)
```

Substituting `q`:

```
σ_b = 3 ρ_w C_d u'² L² / (8 h)
```

### Critical floe length

Setting `σ_b = σ_f` (flexural strength) and solving for `L`:

```
L_crit = √(8 σ_f h / (3 ρ_w C_d u'²))
```

### Floe volume

Floe is modelled as a square plate: `V = L² h` (unit width implied).

## Verification results

### 1. Bending stress formula

| Test | Result |
|------|--------|
| `test_analytic_formula_directly` | PASSED — code matches `3 q L² / (4 h²)` to 10 decimal places |
| `test_bending_stress_at_various_parameters` | PASSED — 4 parameter combinations verified |

### 2. Critical floe length is exact inverse

| Test | Result |
|------|--------|
| `test_round_trip_at_various_parameters` | PASSED — 80 combinations (4 σ_f × 4 h × 5 u × 4 C_d), stress at L_crit equals σ_f to 12 decimal places |
| `test_zero_turbulence_gives_infinity` | PASSED |
| `test_scaling_with_turbulence` | PASSED — L_crit ∝ 1/u' confirmed |
| `test_scaling_with_strength` | PASSED — L_crit ∝ √σ_f confirmed |

### 3. Mass conservation

| Test | Result |
|------|--------|
| `test_mass_conserved_for_various_densities` | PASSED — 5 densities × 4 turbulence levels, mass conserved to 12 decimal places |
| `test_mass_conserved_when_no_breakup` | PASSED — mass conserved when L < L_crit |
| `test_mass_conserved_under_extreme_breakup` | PASSED — mass conserved even when all floes fragment to min size |

### 4. Weibull fit recovery

| Test | Result |
|------|--------|
| `test_recovers_known_parameters_large_sample` | PASSED — shape recovered to <2%, scale to <1% at n=50,000 |
| `test_recovers_exponential_distribution` | PASSED — Weibull shape=1 (exponential) recovered |
| `test_cdf_agreement_r2_high_for_weibull_data` | PASSED — R² > 0.99 |

### 5. KS statistic correctness

| Test | Result |
|------|--------|
| `test_matches_independent_computation` | PASSED — matches independent two-sided KS to 12 decimal places |
| `test_ks_small_for_exact_match` | PASSED — KS < 0.05 for quantile-matched samples |

### 6. Dimensional consistency

| Test | Result |
|------|--------|
| `test_bending_stress_scales_with_L_squared` | PASSED — σ ∝ L² |
| `test_bending_stress_scales_with_u_squared` | PASSED — σ ∝ u'² |
| `test_bending_stress_scales_inversely_with_h` | PASSED — σ ∝ 1/h |
| `test_floe_volume_scales_with_L_squared` | PASSED — V ∝ L² (square floe) |

## Error assessment

All 18 tests pass.  Key quantities verified:

| Quantity | vs基准 | 误差 | <3%? |
|----------|--------|------|------|
| Bending stress formula | Euler-Bernoulli beam theory | <1e-10 (machine precision) | ✅ YES |
| Critical length inverse | bending_stress(L_crit) = σ_f | <1e-12 (machine precision) | ✅ YES |
| Mass conservation | initial_mass = final_mass | <1e-12 (machine precision) | ✅ YES |
| Weibull shape recovery | true shape | <2% at n=50k | ✅ YES |
| Weibull scale recovery | true scale | <1% at n=50k | ✅ YES |
| KS statistic | independent computation | <1e-12 (machine precision) | ✅ YES |

**No baseline experiment exists** for the turbulent breakup criterion itself
(the model is a simplified engineering criterion, not a physical experiment).
The verification therefore confirms *internal consistency* of the model's own
formulas, not physical accuracy.  This is explicitly stated in the test module
docstring and must be quoted with any number taken from this verification.

## Caveats and limitations

1. **Beam model simplification**: real floes experience non-uniform pressure,
   wave-induced dynamic loading, and ice-ice contact.  The free-free beam under
   UDL is a simplified model.

2. **No added mass**: the model does not include added mass effects, which would
   modify the effective inertia of the floe.

3. **No free-surface deformation**: the model assumes the water surface is flat
   and the buoyancy force is static.

4. **No rotation or horizontal drift**: the model treats the floe as a 1-DOF
   oscillator in the vertical direction only.

5. **Weibull fit is empirical**: the two-parameter Weibull is used as a
   convenient functional form for the fragment-size distribution, not as a
   physical law.

## Conclusion

The `threefield/fragmentation.py` module is internally consistent: the bending
stress formula, critical floe length, mass conservation, Weibull fitting, and
KS statistic all pass analytical verification against independent references.
The verification confirms the *model's own* formulas, not physical accuracy
against experiments (which do not exist for this simplified criterion).

## Files

- Test: `tests/test_threefield_fragmentation_analytic.py` (18 tests, all green)
- Source: `src/tensordem/threefield/fragmentation.py`
- Doc: `docs/threefield_fragmentation_analytic_verification.md` (this file)
