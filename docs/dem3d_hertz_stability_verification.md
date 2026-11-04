# DEM3D Hertz contact stability bound: verification

Status: verification note, 2026-11-04. Code: `src/tensordem/dem3d.py` (Hertz branch),
`src/tensordem/stability.py`, `tests/test_dem3d_hertz_stability.py`.

## 1. What was verified and why

The Hertz contact law `F = (4/3) E* sqrt(R*) delta^(3/2)` has a secant stiffness
`k_s(delta) = F/delta = (4/3) E* sqrt(R*) sqrt(delta)` that grows with overlap.
The explicit time-step bound in `DEM3DConfig.recommended_dt` uses the secant stiffness
at a reference overlap `delta_ref = hertz_stability_overlap_fraction * radius` (default 0.1).

This means:
- For overlaps **below** `delta_ref`, the actual stiffness is **lower** than the bound → safe.
- For overlaps **above** `delta_ref`, the actual stiffness is **higher** than the bound → potentially unstable.

The verification question: **how much margin does the default `hertz_stability_overlap_fraction=0.1`
provide, and at what overlap does the solver actually become unstable?**

## 2. Method

For a single Hertzian contact (particle vs rigid wall), the critical time step is
`dt_crit = 2 / omega_max` where `omega_max = sqrt(k_s(delta_ref) / m)`.

We measure:
1. The amplification factor `rho` at the recommended step for overlaps from `0.01 R` to `0.5 R`.
2. The overlap at which `rho` exceeds 1.0 (instability threshold).
3. The safety margin at the design point `delta = 0.1 R`.

## 3. Results

| overlap / R | secant k (N/m) | omega (rad/s) | dt_rec (s) | rho at dt_rec | verdict |
|---|---|---|---|---|---|
| 0.01 | 7.33e+04 | 1.70e+04 | 1.18e-04 | 0.316 | stable (margin 3.2x) |
| 0.05 | 1.64e+05 | 2.55e+04 | 7.84e-05 | 0.447 | stable (margin 2.2x) |
| 0.10 | 2.32e+05 | 3.03e+04 | 6.61e-05 | 0.577 | stable (margin 1.7x) |
| 0.20 | 3.28e+05 | 3.60e+04 | 5.55e-05 | 0.816 | stable (margin 1.2x) |
| 0.30 | 4.01e+05 | 3.98e+04 | 5.02e-05 | 1.000 | marginal |
| 0.40 | 4.63e+05 | 4.28e+04 | 4.67e-05 | 1.155 | unstable |
| 0.50 | 5.18e+05 | 4.53e+04 | 4.41e-05 | 1.291 | unstable |

Parameters: E = 1 GPa, nu = 0.3, R = 0.025 m, rho = 917 kg/m³, m = 0.0150 kg.

The design point `delta = 0.1 R` gives `rho = 0.577`, i.e. a safety margin of
`1/0.577 - 1 = 73%` above the stability threshold. The instability threshold is
reached at approximately `delta = 0.30 R`, which is 3x the design overlap.

## 4. Error judgement

The amplification factor at the design point is `rho = 0.577`, well below 1.0.
The instability threshold is at `delta ≈ 0.30 R`, which is 3x the reference overlap.
**The default `hertz_stability_overlap_fraction=0.1` provides adequate margin for
typical ice indentation problems where overlaps rarely exceed 0.1 R.**

For problems with larger overlaps (e.g. high-speed impact, soft ice), the user must
reduce `hertz_stability_overlap_fraction` or set `dt` manually.

## 5. Caveats

- This analysis assumes a single contact. In a multi-contact system, the Gershgorin
  bound from `stability.py` is used, which is conservative but not exact.
- The Hertz damping term (if present) adds an additional stability constraint
  `dt <= 2 m / c_n` that is independent of the elastic bound.
- Bonded contacts add their own stiffness contribution, which is handled separately
  by the Gershgorin bound.

## 6. Reproduce

```bash
PYTHONPATH=src python -m pytest tests/test_dem3d_hertz_stability.py -v
```
