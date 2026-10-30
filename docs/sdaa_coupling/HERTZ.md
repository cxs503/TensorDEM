# Hertz Contact Benchmark — DEM-FEM Coupling Interface

## 1. Case Introduction

### Physical Background
Hertz contact theory describes the elastic contact between two curved bodies. For a rigid sphere of radius $R$ pressing into an elastic half-space with effective modulus $E^*$, the contact force follows:

$$F = \frac{4}{3} E^* \sqrt{R} \, \delta^{3/2}$$

where $\delta$ is the indentation depth. The tangent contact stiffness is:

$$k = \frac{dF}{d\delta} = 2 E^* \sqrt{R \delta}$$

### Geometry & Parameters

| Parameter | Value | Unit |
|-----------|-------|------|
| DEM particle radius | 0.02 | m |
| Bond stiffness | 1×10⁵ | N/m |
| Contact stiffness | 1×10⁵ | N/m |
| Tool (indenter) radius | 0.1 | m |
| Tool speed | 5.0 | m/s |
| Max indentation | 0.05 | m |
| Estimated E_dem | 2.5×10⁶ | Pa |
| Estimated E_eff | 1.33×10⁶ | Pa |
| Grid (main) | 4×3×4 | — |
| Device | CPU | — |

### Honest-FAIL Declaration
**TensorFEM is not available in this workspace.** The benchmark uses TensorDEM's DEM3D bonded-sphere assembly as an elastic half-space analogue. The discrete lattice does not reproduce continuum Hertz contact behavior. A beam deflection benchmark (FD Euler-Bernoulli) is provided as fallback with analytical comparison.

## 2. Methodology

### DEM3D Setup
- Bonded-sphere DEM with simple-cubic lattice
- Spherical prescribed indenter (tool) moving at constant velocity
- No bond breaking (high breaking strain = 0.5)
- No damping (contact_damping = 0, drag = 0) for energy conservation check
- Bottom boundary fixed, side boundaries free

### Effective Modulus Estimation
For a simple-cubic bonded-sphere lattice:
$$E_{dem} \approx \frac{k_{bond}}{2r}$$
$$E^* = \frac{E_{dem}}{2(1-\nu^2)} \quad \text{with} \quad \nu = 0.25$$

### Beam Fallback
Simply-supported beam under uniform load solved via finite differences:
$$EI \frac{d^4w}{dx^4} = q$$
BCs: $w(0)=w(L)=0$, $w''(0)=w''(L)=0$

Analytical: $\delta_{max} = \frac{5qL^4}{384EI}$

## 3. Results

### Hertz Contact (DEM3D)

| Metric | Value | Target | Pass |
|--------|-------|--------|------|
| Fitting exponent | 1.4108 | 1.5 ± 0.05 | ❌ FAIL |
| Stiffness error | 44.1% | ≤ 3% | ❌ FAIL |
| Energy drift | 58.6% | ≤ 1% | ❌ FAIL |
| Mesh convergence | 8.4% | < 5% | ❌ FAIL |

**Data points:** 13 contact samples over 651 DEM steps

### Beam Deflection (FD Fallback)

| Resolution | Stiffness Error |
|------------|-----------------|
| n=20 | 0.2000% |
| n=40 | 0.0500% |
| n=80 | 0.0125% |

**Convergence:** Error reduces by ~75% when doubling resolution (second-order convergence expected for FD).

## 4. Figures

### Force-Displacement Curve
The fitting curve data is in `artifacts/hertz_curve_data.json` containing:
- Measured DEM force vs displacement
- Power-law fit F = a·δ^b
- Hertz analytical prediction

### Beam Convergence
Deflection error decreases as O(1/n²) with mesh refinement, confirming second-order accuracy of the FD scheme.

## 5. Notes & Limitations

### Why Hertz Fails with DEM
1. **Discrete lattice ≠ continuum**: The bonded-sphere assembly has directional stiffness bias and cannot reproduce isotropic elastic half-space behavior
2. **Effective modulus uncertainty**: The E_dem ≈ k_bond/(2r) estimate is approximate; actual lattice modulus depends on coordination number and topology
3. **Contact mechanics mismatch**: DEM contact is sphere-sphere overlap, not smooth surface contact
4. **Small data range**: Only 13 data points in the contact regime limits fitting accuracy

### Beam Benchmark Strengths
- Analytical solution is exact for Euler-Bernoulli assumptions
- FD scheme shows clear convergence
- Error < 0.2% at coarsest resolution

### Applicability
- The coupling interface schema (`schemas/dem_fem_coupling_v1.json`) is valid and jsonschema-compliant
- True DEM-FEM coupling requires TensorFEM with contact element support
- The schema defines positions/velocities/contact_forces/dt/step with dtype and device declarations

## 6. Error Assessment

| Quantity | Benchmark | vs Analytical | Error | < 3%? |
|----------|-----------|---------------|-------|-------|
| Exponent | Hertz (DEM) | 1.5 | 5.9% | ❌ No |
| Stiffness | Hertz (DEM) | Hertz tangent | 44.1% | ❌ No |
| Energy | Hertz (DEM) | Hertz stored | 58.6% | ❌ No |
| Stiffness | Beam (FD, n=80) | 384EI/(5L³) | 0.0125% | ✅ Yes |

**Conclusion:** The Hertz benchmark fails all criteria due to the discrete nature of the DEM lattice. The beam fallback benchmark passes with < 0.02% error. The coupling interface schema is valid and ready for use when TensorFEM becomes available.
