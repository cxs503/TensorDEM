# CASE-DEM-SILO-BEVERLOO: Beverloo Law Validation

## Overview

This case validates the Beverloo orifice-discharge law (Beverloo, Leniger & van de Velde, 1961)
against DEM simulations using the `tensordem.silo_flow` kernel.

**Beverloo correlation:**

$$W = C \cdot \rho_b \cdot \sqrt{g} \cdot (D - k \cdot d)^{5/2}$$

where:
- $W$ = steady mass flow rate [kg/s]
- $C$ ≈ 0.58 (dimensionless discharge coefficient, range 0.55–0.60 for spheres)
- $\rho_b$ = bulk density [kg/m³]
- $g$ = gravitational acceleration [m/s²]
- $D$ = orifice diameter [m]
- $k$ ≈ 1.4 (shape factor, range 1.2–1.6 for spheres)
- $d$ = particle diameter [m]

**Acceptance criterion:** |W − W_ref| / W_ref < 10% for the primary case (D/d = 5).

## Physical Background

The Beverloo law describes the steady-state gravity-driven discharge of cohesionless
granular material through a circular orifice in a flat-bottom silo. The key physics:

1. The flow rate is set by the free-fall arch above the orifice — particles accelerate
   under gravity through the aperture, so $W \propto \sqrt{g}$.
2. The effective aperture is reduced by the particle diameter via the "empty annulus"
   correction $(D - kd)$, reflecting that particle centers cannot approach closer than
   one radius to the orifice edge.
3. The 5/2 exponent arises from dimensional analysis: the flow rate scales with the
   cross-sectional area of the effective orifice times the gravitational velocity.
4. The flow rate is independent of the filling height (for sufficiently deep beds),
   distinguishing it from fluid orifice flow.

## Case Matrix

| Case ID | D/d | D [mm] | μ_p | μ_r | g [m/s²] | Purpose |
|---------|-----|--------|-----|-----|-----------|---------|
| D4 | 4 | 12 | 0.50 | 0.05 | 9.81 | Law check point |
| D5 | 5 | 15 | 0.50 | 0.05 | 9.81 | **Primary case (verdict)** |
| D6 | 6 | 18 | 0.50 | 0.05 | 9.81 | Law check point |
| D5_g4 | 5 | 15 | 0.50 | 0.05 | 2.45 | Gravity scaling (g/4) |

**Shared parameters:**
- Particle diameter: d = 3 mm
- Particle density: ρ_p = 2500 kg/m³
- Silo radius: R = 25 mm
- Fill height: H = 60 mm
- Wall friction: μ_w = 0 (smooth walls, matching Beverloo's glass-bead experiments)
- Normal stiffness: k_n = 5×10³ N/m
- Tangential/normal stiffness ratio: k_t/k_n = 0.5
- Restitution: e = 0.30
- Seed: 20707

## DEM Kernel

The `tensordem.silo_flow.kernel` module implements:

- **Normal contact:** Linear spring–dashpot with restitution-derived damping
- **Tangential contact:** Cundall–Strack incremental spring, saturated at μ·F_n
- **Rolling resistance:** Coulomb torque |M| ≤ μ_r · R_eff · F_n
- **Wall friction:** Cundall–Strack per-surface tangential history (cylinder, plate, rim)
- **Packing:** Disordered deposition (not HCP lattice) to avoid arching artifacts
- **Energy ledger:** Closed per-timestep balance (kinetic + gravitational + elastic + dissipated + exit)

## Verification Chain

1. **Primary criterion:** D5 case flow rate within 10% of Beverloo prediction
2. **Recovered C:** Must lie in extended literature range [0.45, 0.75]
3. **Power-law exponent:** Fitted from D4/D5/D6 must be near 2.5 (|p − 2.5| ≤ 0.3)
4. **Gravity scaling:** W(g/4)/W(g) ≈ 0.5 (within ±10%)
5. **Steadiness:** Window-to-window coefficient of variation < 0.15

## Running the Campaign

```bash
cd /srv/tensor/agent/work/m22262
export PYTHONPATH=src

# Run individual cases
python benchmarks/m20707_beverloo/campaign.py --case D5
python benchmarks/m20707_beverloo/campaign.py --case D6
python benchmarks/m20707_beverloo/campaign.py --case D4
python benchmarks/m20707_beverloo/campaign.py --case D5_g4

# Or run all at once
python benchmarks/m20707_beverloo/campaign.py --all

# Analyze results
python benchmarks/m20707_beverloo/analyse.py

# Generate figures
python benchmarks/m20707_beverloo/render_figures.py
```

## Output Files

- `runs/m20707_beverloo/{case_id}.json` — Per-case results with DEM measurements and Beverloo comparison
- `runs/m20707_beverloo/summary.json` — Aggregated analysis with power-law fit and verdict
- `docs/case_beverloo/figures/` — Comparison plots

## Reference

Beverloo, W. A., Leniger, H. A., & van de Velde, J. (1961). The flow of granular solids
through orifices. *Chemical Engineering Science*, 15(3–4), 260–269.
doi:10.1016/0009-2509(61)80059-4
