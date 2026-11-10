# DEM Stress Metrics Maturity (m10908 / p339w2-n07)

## Summary

Comprehensive validation of three macroscopic stress averaging methods for granular assemblies:
1. **Love-Weber** (virial stress): σ^LW_ij = (1/V) Σ f_i^c l_j^c
2. **Boundary traction** (Hill formula): σ^B_ij = (R/V) Σ f_i^b n_j^b
3. **Energy work-conjugate**: σ^E_ij = (2/V) Σ f_i^c u_j^c / ε̇

## Key Results

### Three-Stress Equivalence (Isotropic)
- LW vs BND deviation: < 0.01% ✓
- LW vs Energy deviation: < 0.01% ✓
- All methods agree within 2% for isotropic conditions

### Fabric Anisotropy Effects
- Deviation increases linearly with fabric anisotropy (R² > 0.95)
- Prohibition threshold: a = 0.2 (deviation > 5%)

### REV Convergence
- Power-law: variance ∝ R^(-α), α ≈ 1.5-2.0
- Recommended REV size: 5-10× particle diameter

## Prohibition Table

| Anisotropy | LW-BND Dev | Interchangeable? |
|-----------|-----------|------------------|
| 0.00 | 0.00% | ✓ YES |
| 0.05 | 0.35% | ✓ YES |
| 0.10 | 1.46% | ✓ YES |
| 0.15 | 3.28% | ⚠ CAUTION |
| 0.20 | 5.71% | ✗ NO |
| 0.30 | 12.14% | ✗ NO |
| 0.40 | 20.71% | ✗ NO |
| 0.50 | 29.64% | ✗ NO |
| 0.60 | 38.51% | ✗ NO |

**Rule**: Anisotropy > 0.2 → DO NOT interchange stress methods.

## Files

- `src/tensordem/stress_metrics.py`: Implementation (33k lines)
- `tests/test_stress_metrics_maturity.py`: 33 tests, all passing
- `scripts/gen_stress_metrics_report.py`: Report generation
- `artifacts/`: Figures and results JSON

## Acceptance Criteria

| Criterion | Required | Achieved | Status |
|-----------|----------|----------|--------|
| Isotropic deviation | < 2% | < 0.01% | ✓ PASS |
| REV R² | > 0.95 | > 0.95 | ✓ PASS |
| Anisotropy correlation R² | > 0.8 | > 0.95 | ✓ PASS |
| Prohibition threshold | Identified | a = 0.2 | ✓ PASS |
| pytest tests | ≥ 10 | 33 | ✓ PASS |

## References

- Love, A.E.H. (1927). A Treatise on the Mathematical Theory of Elasticity.
- Weber, J. (1966). Recherches sur les forces intergranulaires.
- Rothenburg, L. & Selvadurai, A.P.S. (1981). A micromechanical definition of the Cauchy stress tensor.
- Thornton, C. & Zhang, L. (2006). Numerical simulations of the direct shear test.
- Hill, R. (1963). Elastic moduli for severely inhomogeneous media.
