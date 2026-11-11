# Ice Ridge Resistance with Pore Fluid Coupling

## Campaign P226W3-N01 | Mission m14461

### Overview

This module implements the ice ridge resistance model with pore water pressure-consolidation
coupling for the ice ridge keel DEM skeleton. It addresses the gap identified in the original
`ridge_dem.py` which explicitly stated "there is no pore fluid inside the DEM."

### Components

#### 1. Sail Resistance (Kujala 1994 / Riska 1997)

**File:** `src/tensordem/ridge_sail_resistance.py`

Implements closed-form formulas for sail resistance based on full-scale measurements:

- **Breaking resistance** (Kujala 1994):
  ```
  F_sb = C_b · σ_f · h_s² · B / sin(α)
  ```
  where C_b ≈ 0.7 (calibrated from icebreaker Sisu/Urho measurements), σ_f is ice flexural
  strength, h_s is sail height, B is ship beam, α is hull waterline angle.

- **Pushing resistance** (Riska 1997):
  ```
  F_sp = C_p · |γ'| · h_s² · B · K_p / 2
  ```
  where γ' = (1-n)·(ρ_ice - ρ_water)·g is the effective unit weight of porous ice rubble,
  K_p = tan²(45° + φ/2) is the passive earth pressure coefficient.

- **Total sail resistance**: F_sail = F_sb + F_sp

**Key insight:** For typical parameters, breaking resistance dominates (≈99.8% of total)
because the effective unit weight of ice rubble in seawater is very small (buoyant).

#### 2. Pore Fluid Module (Terzaghi 1D Consolidation)

**File:** `src/tensordem/ridge_pore_fluid.py`

Implements the coupling between the DEM particle skeleton and pore fluid:

- **Terzaghi's effective stress principle**: σ' = σ - u
- **Mohr-Coulomb shear strength**: τ = c' + σ'·tan(φ')
- **Hydraulic conductivity** (Kozeny-Carman): k = (g/μ)·(n³/(1-n)²)·(d²/C)
- **Degree of consolidation**: U(T_v) from Terzaghi's series solution
- **Analytical excess pore pressure**: Δu(z,t) for suddenly applied surface load
- **Finite difference consolidation solver**: FTCS scheme with stability check

**Coupling mechanism:**
1. DEM contact forces → total stress σ
2. Excess pore pressure Δu accumulates under loading
3. Effective stress σ' = σ - (u_0 + Δu) decreases
4. Shear strength τ = c' + σ'·tan(φ') degrades
5. If σ' → 0, the skeleton loses load-bearing capacity (liquefaction)

#### 3. Ridge Resistance Quantiles

**File:** `src/tensordem/ridge_resistance_quantiles.py`

Monte Carlo analysis of ridge resistance variability:

- Input parameter distributions (sail height, keel depth, porosity, ice strength, friction angle)
- P10/P50/P90 quantiles of total resistance and grounding depth
- One-at-a-time sensitivity analysis

### Verification

#### Test Coverage

- **75 tests** across 3 test files:
  - `test_ridge_sail_resistance.py`: 25 tests
  - `test_ridge_pore_fluid.py`: 30 tests
  - `test_ridge_resistance_quantiles.py`: 20 tests

#### Machine Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| Breaking resistance > 0 | ✅ PASS |
| Pushing resistance > 0 | ✅ PASS |
| Total = F_sb + F_sp | ✅ PASS |
| F_sb ∝ σ_f, h_s² | ✅ PASS |
| F_sp ∝ (1-n), h_s² | ✅ PASS |
| K_p > 1 for φ > 0 | ✅ PASS |
| γ' < 0 for ice in seawater | ✅ PASS |
| σ' = σ - u (Terzaghi) | ✅ PASS |
| U → 1 as T_v → ∞ | ✅ PASS |
| Shear strength decreases with Δu | ✅ PASS |
| k > 0 from Kozeny-Carman | ✅ PASS |
| Strength reduction ≤ 1 | ✅ PASS |
| P10 < P50 < P90 | ✅ PASS |
| F_total = F_sail + F_keel | ✅ PASS |
| Grounding depth < keel depth | ✅ PASS |
| SHA256 reproducibility | ✅ PASS |

### Results Summary

#### Sail Resistance (h_s=2m, B=20m, σ_f=0.5MPa, n=0.35, φ=40°)

| Component | Value |
|-----------|-------|
| Breaking (Kujala) | 56.00 MN |
| Pushing (Riska) | 0.13 MN |
| **Total** | **56.13 MN** |

#### Pore Fluid (keel depth=10m, c_v=1e-5 m²/s, applied load=100 kPa)

| Quantity | Value |
|----------|-------|
| Max excess pore pressure | 90.9 kPa |
| Mean effective stress | 0.0 kPa (liquefied) |
| Mean shear strength | 5.0 kPa (cohesion only) |

#### Total Resistance Quantiles (Monte Carlo, n=500)

| Quantile | F_total (MN) | Grounding depth (m) |
|----------|-------------|---------------------|
| P10 | 429.13 | 6.34 |
| P50 | 1025.20 | 9.66 |
| P90 | 1906.81 | 12.85 |

### Artifacts

- `artifacts/m14461/fig01_sail_resistance_components.png` — Sail resistance vs sail height
- `artifacts/m14461/fig02_pore_pressure_dissipation.png` — Consolidation curves
- `artifacts/m14461/fig03_resistance_quantiles.png` — Resistance distributions
- `artifacts/m14461/fig04_sensitivity.png` — Sensitivity tornado chart
- `artifacts/m14461/results_table.json` — Numerical results

### References

1. Kujala, K. (1994). "Ice ridge loads on ship hulls — full-scale measurements on icebreakers
   Sisu and Urho." *Proceedings of IAHR Ice Symposium*, Trondheim, Norway.
2. Riska, K. (1997). "A model for ice ridge keel resistance." *Proceedings of POAC 97*,
   Vol. 2, pp. 454–463.
3. Terzaghi, K. (1925). "Erdbaumechanik auf bodenphysikalischer Grundlage." Franz Deuticke, Wien.
4. Terzaghi, K., Peck, R.B., & Mesri, G. (1996). "Soil Mechanics in Engineering Practice."
   3rd ed., Wiley.
5. ISO 19906:2019. "Petroleum and natural gas industries — Arctic offshore structures."
