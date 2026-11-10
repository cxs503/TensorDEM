# DEM Ice-Structure Contact Friction Uncertainty Quantification

**Campaign Node:** P316-W1-N06 | **Mission:** m9982 | **Date:** 2026-11-10

## 1. Case Introduction

Ice-structure interaction in broken ice fields is governed by discrete floe-hull contacts. The ice resistance depends on normal contact forces (crushing-limited, ∝ σ_c) and tangential friction forces (Coulomb, ∝ μ·f_n).

The friction coefficient μ for ice-steel contact exhibits variability (CV≈20%) due to surface roughness, temperature, salinity, and velocity effects. Literature: cold ice μ≈0.05-0.15 (Løset et al., 2005).

**Standard case:** MT Uikku equivalent (L=105m, B=26m, V=3m/s, h=1m, C=0.6, σ_c=2.477MPa).

## 2. Methodology

### Surrogate Model
R(μ, σ_c) = σ_c · (a + b · μ), calibrated from DEM contact law physics (R²=0.9986).

### Uncertainty Propagation
- μ ~ N(0.10, 0.02), σ_c ~ N(2.477MPa, 0.495MPa)
- Monte Carlo: 10,000 samples
- Sobol indices: analytical for linear model
- Joint uncertainty: μ + σ_c coupled

## 3. Results

### Surrogate: a=1.189e-7 kN/Pa, b=2.547e-6 kN/Pa, R²=0.9986

### μ-only propagation:
- Mean: 0.93 kN, Std: 0.13 kN, 95% CI: [0.72, 1.13] kN

### Sensitivity (Sobol):
- S_μ = 0.313 (31.3%), S_σ = 0.674 (67.4%), S_interaction = 0.013 (1.3%)
- Ranking: σ_c > μ

### Joint uncertainty (μ + σ_c):
- Mean: 0.92 kN, Std: 0.23 kN, 95% CI: [0.57, 1.32] kN
- Variance decomposition error: 0.87% (<5%)

## 4. Figures
- fig1_distributions.png: μ and R probability densities
- fig2_monotonicity.png: R(μ) monotonic increase (ρ=1.0)
- fig3_sobol_indices.png: Variance decomposition bar chart
- fig4_joint_uncertainty.png: Joint sampling and comparison
- fig5_ci_linearity.png: CI width ∝ μ_std (R²>0.95)

## 5. Acceptance Criteria
| Criterion | Result | Status |
|-----------|--------|--------|
| (a) μ mean within 0.1±0.005 | 0.1000 | ✅ |
| (b) Monotonicity ρ=1 | ρ=1.0 | ✅ |
| (c) CI linearity R²>0.95 | R²=0.9998 | ✅ |
| (d) Variance decomp <5% | 0.87% | ✅ |
| (e) μ=0 negative control | R(0)<R(>0) | ✅ |
| (f) ≥12 pytest cases | 19 cases | ✅ |

## 6. Conclusions
1. μ uncertainty produces ±14% resistance variability
2. σ_c dominates (67.4%) over μ (31.3%) in variance contribution
3. Linear surrogate adequate (R²>0.998)
4. Joint uncertainty increases 95% CI width by 83%
5. All acceptance criteria satisfied

## 7. Deliverables
- src/tensordem/uncertainty/ (5 modules)
- tests/test_mu_uncertainty.py (19 tests)
- artifacts/m9982_results.json
- artifacts/fig*.png (5 figures)
