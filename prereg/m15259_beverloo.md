# Preregistration: Beverloo Hopper-Discharge Benchmark (m15259)

## Decision
Compare DEM-simulated granular mass flow rate through a hopper orifice against the Beverloo (1961) correlation to verify TensorDEM's granular flow capability.

## Hypothesis
The DEM simulation will reproduce the Beverloo correlation within 15% relative error when using:
- Disordered packing (gap=0.15, jitter=0.08r)
- Low inter-particle friction (μ=0.15)
- Smooth walls (μ_w=0.0)
- D/d = 5.0

## Alternatives
- A: DEM simulation (this benchmark)
- B: Beverloo analytic correlation (reference)
- C: Previous m6457 results (baseline for comparison)

## Metrics
- Primary: Mass flow rate W (kg/s)
- Secondary: Relative error vs Beverloo (%), recovered C coefficient, flow rate std (kg/s)
- Conservation: Mass ledger residual, energy ledger residual

## Baseline/Control
Beverloo (1961) correlation with C=0.58, k=1.4, using measured bulk density.

## Predeclared Thresholds
- Pass: |relative error| < 15%
- Flow steadiness: CV of windows 2-6 < 20%
- Mass conservation: |mass_ledger_residual| < 1e-10

## Variables
- Controlled: particle properties (ρ=2500, μ=0.15, e=0.5), geometry (D/d=5, α=30°), gravity (9.81)
- Measured: bulk density, flow rate, packing fraction
- Confounders: packing disorder (controlled via gap/jitter), wall friction (set to 0)

## Repetition Rationale
Single seed (42) for this benchmark pool entry; m6457 already established multi-seed sensitivity.

## Environment
- Solver: TensorDEM vectorised hopper kernel
- Platform: 702 cluster (loong64)
- Python 3.11, NumPy, SciPy

## Analysis
- Compute W_meas = mean(flow_rate_windows)
- Compute W_bev = C·ρ_b·√g·(D-k·d)^(5/2)
- Compute rel_err = (W_meas - W_bev) / W_bev
- Compare against 15% threshold

## Safety/Resource Limits
- Timeout: 3600 s
- Cores: 8
- Memory: standard

## Interpretation Limits
- Beverloo correlation is empirical; ±10% scatter is typical even in experiments
- DEM contact model (linear spring-dashpot) is a simplification of real particle contacts
- Results apply to monodisperse spheres only; polydispersity would change flow behavior
