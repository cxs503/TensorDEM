# Ice–Structure Contact Mechanics: Rolling-Friction Proxy Bias Quantification
## Mission m5754 / IN34-W3-N08

### Physical Background

Broken ice in a ship channel is represented in DEM either as:
- **Angular polyhedra** (expensive, geometrically faithful), or
- **Spheres + rolling friction μ_r** (cheap, but angularity is *proxied*).

The central question: **how large is the systematic bias δ(μ_r) when we replace
true angular ice pieces with spheres + rolling friction?**

### Verification Pillars

1. **Contact pressure spatial statistics** — p(x,y) spatial correlation R(Δx,Δy)
   and peak factor γ = max(p)/mean(p).
2. **Polyhedron → sphere degeneracy** — contact kernel recovers Hertz to
   machine precision (≤ 1e-12) when polyhedron is a sphere.
3. **Rolling-friction bias quantification** — δ(μ_r) = (R_sphere - R_poly) / R_poly.

### Acceptance Criteria

| Criterion | Description | Threshold | Status |
|-----------|-------------|-----------|--------|
| (a) | Hertz degeneracy | rel. error ≤ 1e-12 | ✅ PASS |
| (b) | Sphere μ_r=0 bias | < 2% | ✅ PASS |
| (c) | Held-out shape bias | < 15% | ✅ PASS |
| (d) | Identifiability | cond# reduction ≥ 10× (if needed) | ✅ PASS |
| (e) | KS test (3 seeds) | p > 0.05 | ✅ PASS |
| (f) | Monotonic scan | R_total non-decreasing | ✅ PASS |

### Physical Model

The resistance model has two components:

1. **Form resistance**: R_form = Σ F_n max(0, n·d) — pressure drag from contacts
   opposing motion.
2. **Rolling resistance**: R_rolling = μ_r Σ F_n — direct resistance force from
   rolling friction / interlocking.

For angular particles (angularity ψ):
- Form resistance increases by factor (1 + 0.2ψ) — stress concentration at edges.
- Rolling resistance coefficient increases by μ_eff = 0.5ψ — interlocking.

The sphere+μ_r model captures the rolling resistance exactly by setting
μ_r = μ_eff(ψ), with residual bias from the form resistance increase:
δ ≈ 0.2ψ × R_form / R_total.

### Key Results

- **Hertz degeneracy**: max relative error = 0.0 (machine precision)
- **Optimal μ_r for ψ=0.4**: μ_r ≈ 0.20, bias ≈ 8%
- **Peak factor**: γ ≈ 2-5 (typical of granular force chains)
- **Correlation length**: ~10-20 mm (set by particle size)
- **Parameters identifiable**: |corr(μ_roll, ψ)| ≈ 0 (grid design)

### Figures

- `fig1_hertz_degeneracy.png` — Force-overlap curve and relative error
- `fig2_rolling_bias.png` — Bias vs μ_r for three angularity values
- `fig3_monotonic_scan.png` — Negative control: R_total vs μ_roll
- `fig4_contact_pressure.png` — Pressure field, correlation, peak factor
- `fig5_identifiability.png` — Sensitivity surface and contours

### Limitations

1. **Synthetic contact network**: The bias quantification uses a synthetic
   contact network (random normals in forward hemisphere, log-normal forces).
   Real ice-structure contacts have spatial correlations and force chains that
   are not captured.
2. **Simplified angularity model**: The angularity factor (1 + 0.2ψ) is
   empirical. Real angular particles have complex contact geometries that
   depend on the specific shape distribution.
3. **No DEM simulation**: This verification uses analytic/semi-analytic models.
   Full DEM validation against experiments is needed for production use.
4. **2D pressure field**: The contact pressure is binned onto a 2D grid, which
   loses information about the 3D contact network topology.

### Files

- `src/tensordem/ice_structure_contact.py` — Main verification module
- `tests/test_ice_structure_contact.py` — 29 pytest tests (all pass)
- `scripts/m5754_figures.py` — Figure generation script
- `runs/m5754_verification_results.json` — Machine-readable results
- `docs/m5754_figures/` — Generated figures
