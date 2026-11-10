# Ice Ridge Keel–Hull Interaction Mechanics

## Overview

This module implements the ice ridge keel–hull interaction mechanics for the campaign node
`p256w1-n05-ice-ridge-keel`. It provides three layers of prediction:

1. **Rankine passive earth-pressure upper bound** — classical soil mechanics solution
2. **ISO 19906 ridge load model** — regulatory form with size effect
3. **DEM verification** — 2-D granular ridge indented by rigid wedge

## Module: `tensordem.ice_ridge_keel_hull`

### Key Functions

| Function | Description |
|---|---|
| `rankine_passive_coefficient(phi_deg)` | K_p = (1+sinφ)/(1−sinφ) |
| `rankine_passive_force(h_k, W_k, phi_deg, ...)` | F = 0.5·|γ'|·h_k²·K_p·W_k |
| `rankine_shear_correction(h_k, W_k, phi_deg, cohesion, ...)` | F_cohesion = c·h_k·W_k·K_c |
| `rankine_total_force(...)` | F_total = F_passive + F_cohesion |
| `iso19906_ridge_load(A_contact, params)` | F = p_0·A·(A/A_0)^(-0.4) |
| `iso19906_ridge_load_full(h_k, W_k, alpha_deg, penetration, ...)` | Full model from geometry |
| `level_ice_crushing_load(h_ice, sigma_c, width)` | F = σ_c·h_ice·width |
| `compute_calibration(F_analytical, F_DEM_peak)` | γ_R = F_analytical / F_DEM |
| `ridge_load_envelope(h_k_range, alpha_range, ...)` | h_k × α scan |
| `dem_ridge_wedge_test(...)` | DEM verification harness |
| `run_full_campaign(...)` | End-to-end campaign runner |

### Negative Controls

- `negative_control_phi_zero()`: φ → 0 ⇒ K_p → 1 (hydrostatic limit)
- `negative_control_hk_zero()`: h_k → 0 ⇒ F_ridge → 0
- `negative_control_no_bond()`: σ_b = 0 ⇒ F_bonded > F_unbonded

### Provenance

All constants are sha256-anchored:
```python
PROVENANCE_SHA256 = hashlib.sha256(json.dumps(PROVENANCE, sort_keys=True).encode()).hexdigest()
```

## Verification Results

| Criterion | Target | Status |
|---|---|---|
| (a) Rankine vs textbook | ≤2% | ✓ PASS |
| (b) ISO 19906 vs Annex C | ≤10% | ✓ PASS |
| (c) DEM γ_R ∈ [0.6, 1.0] | 0.6–1.0 | ⚠ (small assembly) |
| (d) Negative controls | All pass | ✓ PASS |
| (e) Level-ice degradation | ≤15% | ⚠ (simplified) |
| (f) Provenance sha256 | Anchored | ✓ PASS |

## Test Suite

Run with: `pytest tests/test_ice_ridge_keel_hull.py -v`

45 tests covering:
- Rankine coefficient (6 tests)
- Submerged unit weight (4 tests)
- Rankine passive force (4 tests)
- Shear correction (3 tests)
- Total force (2 tests)
- ISO 19906 contact area (3 tests)
- ISO 19906 ridge load (3 tests)
- Level-ice crushing (2 tests)
- Calibration (3 tests)
- Envelope (3 tests)
- Ridge/level ratio (1 test)
- Negative controls (3 tests)
- DEM verification (2 tests)
- Full campaign (3 tests)
- Provenance (2 tests)

## References

- Terzaghi, K. & Peck, R.B. (1948). *Soil Mechanics in Engineering Practice*.
- ISO 19906:2010. *Petroleum and natural gas industries — Arctic offshore structures*.
- Hopky, G. & Lawrence, W.F. (1983). Ridge sail/keel geometry.
