# CASE-DEM-SILO-BEVERLOO: Ensemble Validation Status

**Mission**: m22853 (self-start, generalist seat)  
**Date**: 2026-11-16  
**Anchor**: CASE-DEM-SILO-BEVERLOO (feature_ledger status: open)

## 1. Executive Summary

The Beverloo law validation for DEM silo discharge has been adjudicated 8+ times (m6296, m6457, m15259, m15381, m20055, m20707, m21033, m21119, m21421, m22628). **No single round has passed the ledger's ±10% tolerance with statistical rigor.** The core issue: the linear-spring DEM kernel with low stiffness (kn ~ 5×10³–10⁵ N/m) produces large overlaps and unrealistic contact mechanics, leading to systematic underprediction (−82% to −89%) or overprediction (+20% to +38%) depending on parameters.

This unit (m22853) submits the **first seed-ensemble campaign** (m22657 preregistration) to measure reproducibility scatter and adjudicate the anchor with a confidence interval rather than a point estimate.

## 2. Physical Background

**Beverloo correlation** (Beverloo, Leniger & van de Velde, 1961):

$$W = C \cdot \rho_b \cdot \sqrt{g} \cdot (D - k \cdot d)^{5/2}$$

- $W$ = steady mass flow rate [kg/s]
- $C \approx 0.58$ (dimensionless discharge coefficient, range 0.55–0.60 for spheres)
- $\rho_b$ = bulk density [kg/m³]
- $g$ = gravitational acceleration [m/s²]
- $D$ = orifice diameter [m]
- $k \approx 1.4$ (shape factor, range 1.2–1.6 for spheres)
- $d$ = particle diameter [m]

**Acceptance criterion** (feature_ledger): $|W - W_{\text{ref}}| / W_{\text{ref}} < 10\%$

## 3. Campaign Design (m22657 preregistration)

### 3.1 Frozen Configuration

| Quantity | Value | Domain Check |
|----------|-------|--------------|
| Particle diameter $d$ | 3.0 mm | — |
| Orifice $D$ | 15.0 mm ($D/d = 5$) | $D/d \geq 4$ ✓ |
| Silo radius | 24.0 mm ($\text{silo\_D}/D = 3.2$) | $\geq 2.5$ ✓ |
| Fill height $H$ | 36.0 mm ($H/D = 2.4$) | $\geq 1.5$ ✓ |
| $\mu$ / $\mu_{\text{rolling}}$ / $e$ | 0.50 / 0.05 / 0.30 | glass-bead high-friction |
| $k_n$ / $k_t/k_n$ | 5.0×10³ N/m / 0.5 | campaign convention |
| Wall friction | 0 (flat bottom plate) | Beverloo smooth-wall condition |
| $g$ | 9.81 m/s² | — |
| Settle / discard | 0.06 s / 0.12 s | plug closed during settling |
| Measurement | **8 windows × 0.05 s = 0.40 s** | drains ≈ 14% of the bed |
| Seeds | **1, 2, 3, 4, 5, 6** (disordered packing) | — |
| Kernel | `tensordem.silo_flow.study beverloo` (Cundall–Strack) | — |

### 3.2 Estimator (frozen in `benchmarks/m22657_beverloo_ensemble/analyse.py`)

The ledger quantity is the **ensemble mean** of the per-seed discharge rate with a Student-t 95% confidence interval, compared against the *literature* coefficients $C = 0.58$, $k = 1.4$ — no refitted amplitude is allowed.

**Verdicts**:
- **V1_ledger**: PASS if ensemble mean ± 95% CI lies within ±10%; FAIL if outside; NOT_DECIDABLE if CI spans the threshold
- **V2_reproducibility**: PASS if CV < 10%; FAIL otherwise
- **V3_steady_state**: PASS if max window drift < 10%; FAIL otherwise
- **V4_rho_control**: PASS if bulk density spread < 3%; FAIL otherwise

## 4. Previous Campaign Results (summary)

| Mission | D/d | kn [N/m] | μ | W_dem/W_ref | rel err | Notes |
|---------|-----|----------|---|-------------|---------|-------|
| m15381 (quick) | 3.0 | 10⁵ | 0.15 | 0.733/0.532 | +38% | 80 particles, unsteady |
| m15381 (medium) | 5.0 | 10⁵ | 0.15 | 1.97/2.31 | −15% | 382 particles, passed 15% |
| m20055 (D4) | 4.0 | 10⁵ | 0.40 | 1.82/1.45 | +25% | 766 discharged, audit verified |
| m20707 (D5) | 5.0 | 5×10³ | 0.50 | — | −6.9% | single seed |
| m20707 (D6) | 6.0 | 5×10³ | 0.50 | — | +5.9% | single seed |
| m6457 (A_base_D5) | 5.0 | 10⁴ | 0.40 | 0.32/1.84 | −82% | honest FAIL, mass-balance |
| m6457 (F_base_D4) | 4.0 | 10⁴ | 0.40 | 0.09/0.79 | −89% | honest FAIL, mass-balance |

**Key finding**: The DEM kernel systematically deviates from Beverloo. The sign and magnitude depend on $k_n$, $\mu$, and $D/d$. No campaign has reported **reproducibility scatter** — every verdict rests on one seed per configuration.

## 5. Current Status (m22853)

### 5.1 Cluster Job

- **Job ID**: cj4437
- **Shards**: 6 seeds (1–6)
- **Status**: queued (63 jobs ahead as of 2026-11-16 03:48)
- **Expected runtime**: ~5–10 min per seed (31k steps, ~2.5k particles)

### 5.2 Infrastructure Ready

- `benchmarks/m22657_beverloo_ensemble/run_case.py`: wrapper to run one seed
- `benchmarks/m22657_beverloo_ensemble/analyse.py`: frozen estimator with verdicts
- `benchmarks/m22657_beverloo_ensemble/prereg.md`: pre-registration document
- `tests/test_m22657_beverloo_ensemble.py`: 24 tests (all passing)
- `src/tensordem/silo_flow/study.py`: campaign kernel driver
- `src/tensordem/silo_flow/beverloo.py`: Beverloo correlation + discharge solver
- `src/tensordem/silo_flow/kernel.py`: Cundall–Strack DEM with energy ledger
- `src/tensordem/silo_flow/packing.py`: disordered packing generator (no HCP arching)

### 5.3 Next Steps

1. **Wait for cluster results** (cj4437)
2. **Run analysis**: `python benchmarks/m22657_beverloo_ensemble/analyse.py --dir runs/m22657_beverloo_ensemble --out runs/m22657_beverloo_ensemble/summary.json --figures docs/figures/m22657`
3. **Generate report**: build paper-grade report with six elements (case intro, method, results table, figures, caveats, error judgment)
4. **Publish report**: POST to `/api/v1/tensor/reports` with token auth
5. **Commit**: git add + commit (platform will push to GitHub)

## 6. Known Limitations

1. **Linear-spring contact model**: The DEM kernel uses a linear spring-dashpot normal force $f_n = \max(k_n \delta - \gamma v_n, 0)$. Real glass beads have Hertzian contact $f_n \propto \delta^{3/2}$. The linear model with low $k_n$ produces large overlaps ($\delta / d \sim 1\%$), violating the rigid-particle assumption of Beverloo.

2. **Low stiffness**: $k_n = 5 \times 10^3$ N/m is 1–2 orders of magnitude below typical DEM practice for glass beads ($10^5$–$10^6$ N/m). Higher $k_n$ reduces overlaps but increases computational cost (smaller $\Delta t$).

3. **2D axisymmetric geometry**: The kernel is axisymmetric (cylinder + flat plate + cone), not full 3D. This is sufficient for silo discharge but limits extension to complex geometries.

4. **Monodisperse spheres**: Beverloo's correlation is for spherical particles. Real granular materials have size distributions and irregular shapes.

5. **Smooth walls**: The campaign uses $\mu_w = 0$ (smooth walls) to match Beverloo's glass-bead experiments. Real silos have wall friction, which can reduce the flow rate (Janssen effect).

## 7. Recommendations for Future Work

1. **Stiffness convergence study**: Run the ensemble at $k_n = 10^4, 5 \times 10^4, 10^5$ N/m to quantify the overlap effect. Hypothesis: higher $k_n$ → smaller overlaps → closer to Beverloo.

2. **Hertzian contact model**: Implement $f_n \propto \delta^{3/2}$ (Tsoung et al. 1996) and compare to the linear model. This is a major code change but physically more accurate.

3. **3D full simulation**: The axisymmetric kernel is a simplification. A full 3D DEM with periodic or cylindrical walls would remove the axisymmetry constraint.

4. **Polydisperse particles**: Beverloo's $k$ factor accounts for particle size. A polydisperse simulation with a measured size distribution would test the correlation more rigorously.

5. **Wall friction sweep**: Vary $\mu_w$ from 0 to 0.4 and compare to Janssen's theory. This is already partially implemented in `silo_flow.study hopper`.

## 8. Conclusion

The m22853 campaign (m22657 ensemble) is the **first statistically rigorous attempt** to adjudicate CASE-DEM-SILO-BEVERLOO. The infrastructure is in place, the estimator is frozen, and the cluster job is queued. Once the 6-seed ensemble completes, the analysis will produce a verdict with a confidence interval, separating the model error from the packing scatter.

**Honest expectation**: Based on previous campaigns, the ensemble mean will likely fall outside ±10% (honest FAIL). The value of this unit is not to pass the anchor but to **quantify the error with statistical rigor** and provide a baseline for future improvements (stiffness convergence, Hertzian contact, 3D full simulation).

---

**References**:
- Beverloo, W. A., Leniger, H. A., & van de Velde, J. (1961). The flow of granular solids through orifices. *Chemical Engineering Science*, 15(3-4), 260-269.
- Tsoung et al. (1996). [Hertzian contact model reference]
- feature_ledger: CASE-DEM-SILO-BEVERLOO (acceptance: <10%, ref: Beverloo et al. 1961)
