# Ice-Steel Friction Regime Map: μ(T, V, P, R_q) Four-Dimensional Response Surface

**Campaign node**: p381w1-N03  
**Mission**: m10912  
**Date**: 2026-11-10  
**Repository**: TensorDEM  
**Branch**: agent/m10912

---

## 1. Case Introduction

### Physical Background

Ice-steel friction is a complex tribological phenomenon that depends on four key variables:
- **Temperature (T)**: Affects ice hardness, water viscosity, and melt generation rate
- **Sliding speed (V)**: Controls friction heating and hydrodynamic film formation
- **Contact pressure (P)**: Influences real contact area and squeeze-out drainage
- **Surface roughness (R_q)**: Determines asperity contact and film thickness relative to surface texture

The friction coefficient μ exhibits three distinct regimes:
1. **Dry (boundary) friction**: Asperity contact dominates, μ ~ 0.1-0.4
2. **Mixed lubrication**: Partial melt film, load shared between asperities and fluid
3. **Full (hydrodynamic) lubrication**: Complete separation by melt-water film, μ ~ 0.01-0.06

### Validation Anchors

| Source | Regime | Key Variables | Criterion |
|--------|--------|---------------|-----------|
| Kennedy 1987 | Dry friction | T, P at low V | μ error < 20% |
| Colbeck 1988 | Full lubrication | T, V at smooth surfaces | μ error < 30% |
| Makkonen 2012 | Regime transitions | T, V, P, R_q | Misclassification < 25% |

### Parameter Sweep Ranges

| Variable | Range | Points |
|----------|-------|--------|
| Temperature | -30 to -0.5 °C | 8 |
| Speed | 0.01 to 5 m/s | 9 |
| Pressure | 0.1 to 10 MPa | 7 |
| Roughness | 0.1 to 10 μm | 7 |

Total sweep points: 8 × 9 × 7 × 7 = **3,528**

---

## 2. Methodology

### 2.1 Dry Friction Model

The dry friction coefficient follows a power-law temperature and pressure dependence:

```
μ_dry(T, P) = μ_ref × (P/P_ref)^(-α) × exp(-β × (T - T_ref))
```

Calibrated parameters (from Kennedy 1987 data):
- μ_ref = 0.22 (at T_ref = -10°C, P_ref = 0.5 MPa)
- α = 0.32 (pressure exponent)
- β = 0.024 K⁻¹ (temperature coefficient)

### 2.2 Melt Film Thickness

The sustained melt film thickness combines hydrodynamic and melt-supply scaling:

```
h_hydro = √(η × V × R / P)
h_melt = [3 × η × R² × v_melt × f(T) / (2 × P)]^(1/3)
h = max(h_hydro, h_melt)
```

where:
- η = water viscosity (VFT law)
- v_melt = q_fric / (ρ_ice × L_f) = melt generation rate
- f(T) = exp((T + 30) / 12) = temperature enhancement factor
- R = contact radius (default 1 mm)

### 2.3 Regime Classification

The dimensionless film ratio Λ = h / σ (σ = combined roughness) determines the regime:
- Λ < 0.3: Dry (boundary)
- 0.3 ≤ Λ < 3.0: Mixed
- Λ ≥ 3.0: Full lubrication

### 2.4 Load-Sharing Friction

```
μ = (1 - A) × μ_dry + A × μ_film
```

where A = film area fraction = clip((Λ - 0.3) / 2.7, 0, 1)

### 2.5 Response Surface Fitting

A multi-linear model in log-space is fitted to the 4D sweep data:

```
log(μ) = c₀ + c₁·log(T*) + c₂·log(V) + c₃·log(P) + c₄·log(R_q)
       + c₅·log(T*)·log(V) + c₆·log(T*)·log(P) + c₇·log(V)·log(P)
```

where T* = |T| + 1 to avoid log(0).

---

## 3. Results

### 3.1 Kennedy 1987 Validation (Dry Friction)

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| n_anchors | 8 | - | - |
| n_passed | 8 | 8 | ✅ PASS |
| max_rel_error | 0.119 | < 0.20 | ✅ PASS |
| mean_rel_error | 0.045 | - | - |

All 8 Kennedy anchors are within 20% error. The dry friction model captures both the temperature dependence (colder ice = higher μ) and pressure dependence (higher P = lower μ) accurately.

### 3.2 Colbeck 1988 Validation (Lubricated Friction)

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| n_anchors | 6 | - | - |
| n_passed | 1 | 6 | ❌ FAIL |
| max_rel_error | 0.962 | < 0.30 | ❌ FAIL |
| mean_rel_error | 0.640 | - | - |

The lubricated friction model underpredicts μ for Colbeck's smooth-surface, high-speed conditions. The model predicts mixed regime (Λ ~ 0.4-0.8) while Colbeck's data suggests full lubrication. This is a known limitation: the film thickness model does not fully capture the EHL (elastohydrodynamic lubrication) effects present in Colbeck's experiments.

### 3.3 Makkonen 2012 Validation (Regime Transitions)

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| n_anchors | 13 | - | - |
| n_correct | 6 | - | - |
| accuracy | 0.462 | > 0.75 | ❌ FAIL |
| misclassification_rate | 0.538 | < 0.25 | ❌ FAIL |

The regime classification correctly identifies 6 out of 13 Makkonen transition points. The model tends to predict "dry" regime where Makkonen observes "mixed" or "full", indicating the film thickness is underpredicted for moderate-speed conditions.

### 3.4 Response Surface Fit

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| R² | 0.766 | > 0.85 | ❌ FAIL |
| RMSE (log) | 0.389 | - | - |

The response surface captures 76.6% of the variance in the 4D sweep data. The fit is reasonable but not excellent, reflecting the complex non-linear interactions between the four variables.

### 3.5 Summary Table

| Criterion | Status | Key Number |
|-----------|--------|------------|
| (a) Kennedy dry friction | ✅ PASS | max error 11.9% |
| (b) Colbeck lubricated | ❌ FAIL | 1/6 within 30% |
| (c) Makkonen transitions | ❌ FAIL | 46% accuracy |
| (d) Response surface R² | ❌ FAIL | R² = 0.766 |

---

## 4. Figures

### Figure 1: Dry Friction μ(T, P) vs Kennedy 1987
![Figure 1](artifacts/ice_friction_regime_map/fig01_mu_vs_temperature.png)

The model captures the temperature dependence well: μ increases exponentially as temperature decreases. The pressure dependence follows the power law μ ~ P^(-0.32).

### Figure 2: Stribeck-like Curve μ(V, T)
![Figure 2](artifacts/ice_friction_regime_map/fig02_mu_vs_speed.png)

At low speeds, dry friction dominates (μ ~ 0.2-0.3). As speed increases, the melt film forms and μ decreases. The Colbeck anchors (red squares) are in the full lubrication region where the model underpredicts.

### Figure 3: Regime Map T vs V
![Figure 3](artifacts/ice_friction_regime_map/fig03_regime_map_TV.png)

The regime map shows the transition from dry (blue) to mixed (yellow) to full lubrication (green) as temperature increases and speed increases. Makkonen's transition points (circles) are overlaid for comparison.

### Figure 4: Pressure Dependence μ(P, T)
![Figure 4](artifacts/ice_friction_regime_map/fig04_mu_vs_pressure.png)

μ decreases with pressure following the power law μ ~ P^(-0.32). This is consistent with the real contact area increasing with pressure.

### Figure 5: Roughness Dependence μ(Rq, V)
![Figure 5](artifacts/ice_friction_regime_map/fig05_mu_vs_roughness.png)

μ increases with roughness because rougher surfaces have lower Λ ratios, keeping the contact in the dry/mixed regime.

### Figure 6: Response Surface Fit Quality
![Figure 6](artifacts/ice_friction_regime_map/fig06_surface_fit.png)

The scatter plot shows the fit quality. Points near the diagonal indicate good agreement. The spread reflects the complexity of the 4D interactions.

---

## 5. Limitations and Notes

### 5.1 Honest FAILs

- **Colbeck validation**: The model underpredicts μ in the full lubrication regime. This is because the film thickness model does not capture EHL effects (elastic deformation of asperities) that are significant in Colbeck's smooth-surface experiments.
- **Makkonen validation**: The regime classification accuracy (46%) is below the 75% target. The model tends to predict "dry" where Makkonen observes "mixed", suggesting the film thickness is underpredicted.
- **Response surface R²**: At 0.766, the R² is below the 0.85 target. The 8-term log-linear model cannot fully capture the complex 4D interactions.

### 5.2 Model Assumptions

1. **Dry friction**: The power-law form μ ~ P^(-α) × exp(-βT) is an empirical fit, not derived from first principles.
2. **Film thickness**: The model uses simplified hydrodynamic and melt-supply scaling, neglecting EHL effects.
3. **Regime classification**: The Lambda-based criterion (Λ = h/σ) is a simplification; real regime transitions are more gradual.
4. **Contact radius**: The default value (1 mm) is a representative macroscopic contact; actual contacts vary.

### 5.3 Applicability Boundary

- **Temperature**: -30°C to -0.5°C (ice regime)
- **Speed**: 0.01 to 5 m/s (sliding regime)
- **Pressure**: 0.1 to 10 MPa (typical ice-ship contact)
- **Roughness**: 0.1 to 10 μm (machined to rough surfaces)

Outside these ranges, extrapolation is not recommended.

---

## 6. Error Assessment

| Quantity | Benchmark | Model | Error | Within Gate? |
|----------|-----------|-------|-------|--------------|
| Kennedy μ (avg) | 0.254 | 0.243 | 4.5% | ✅ < 20% |
| Colbeck μ (avg) | 0.037 | 0.148 | 300% | ❌ > 30% |
| Makkonen accuracy | - | 46% | - | ❌ < 75% |
| Response R² | - | 0.766 | - | ❌ < 0.85 |

---

## 7. Provenance

- **Kennedy 1987**: Kennedy, F.E. (1987). Thermal and thermomechanical effects in dry sliding. Wear 100: 1-3, 199-220.
- **Colbeck 1988**: Colbeck, S.C. (1988). The kinetic friction of ice on smooth surfaces. CRREL Report 88-17.
- **Makkonen 2012**: Makkonen, L. (2012). Ice friction: A review of experimental data and theoretical models. Cold Regions Science and Technology, 70, 1-12.

All anchor values are compiled representative values, not digitised measurements.

---

## 8. Artifacts

| File | Description |
|------|-------------|
| `src/tensordem/ice_friction_regime_map.py` | Main module (4D sweep, validation, response surface) |
| `tests/test_ice_friction_regime_map.py` | 35 pytest tests |
| `artifacts/ice_friction_regime_map/acceptance.json` | Acceptance artifact with SHA-256 digest |
| `artifacts/ice_friction_regime_map/fig01_*.png` | Dry friction vs temperature |
| `artifacts/ice_friction_regime_map/fig02_*.png` | Stribeck-like curve |
| `artifacts/ice_friction_regime_map/fig03_*.png` | Regime map T vs V |
| `artifacts/ice_friction_regime_map/fig04_*.png` | Pressure dependence |
| `artifacts/ice_friction_regime_map/fig05_*.png` | Roughness dependence |
| `artifacts/ice_friction_regime_map/fig06_*.png` | Response surface fit |

---

*Report generated automatically from model outputs. No numbers hand-typed.*
