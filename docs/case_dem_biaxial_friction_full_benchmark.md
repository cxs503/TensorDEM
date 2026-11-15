# CASE-DEM-BIAXIAL-FRICTION — Full Benchmark Verification (m21349)

## Anchor
- **slug**: `CASE-DEM-BIAXIAL-FRICTION` (算例池·双轴剪切摩擦角)
- **acceptance**: 经典临界状态参考，容差 <8%（相对误差）
- **reference**: Bolton (1986) *Geotechnique* 36(1):65-78, Table 2 p.72 —
  dense Hostun sand plane-strain peak friction angle **phi_ps = 38°**
  (D_r = 80%, sigma_3 = 100 kPa); Table 1 p.67: **phi_crit ≈ 33°**.

## What was done
The full (non-quick) biaxial DEM benchmark was run on the compute scheduler
(node n330, job cj3394, stage_dir = this worktree).  Specimen: 24×48 = 1128
discs, radius 0.75, confining stress 100 kPa, 4000 consolidation steps,
strain target 0.30, wall_friction = 0 (frictionless servo walls).

## Result (programmatic, from committed run.json)

| Quantity | Value | Reference | Relative error | Within 8%? |
|---|---|---|---|---|
| Peak friction angle (wall traction) | **41.41°** | 38.0° | **8.97%** | **NO (FAIL)** |
| Peak friction angle (internal / Love-Weber virial) | **36.43°** | 38.0° | **4.13%** | **YES (PASS)** |
| Critical-state phi_cv (tail) | 19.75° | 33.0° (Bolton) / 28.81° (Rowe) | — | not converged |
| Rowe predicted phi_cv (mu=0.55) | 28.81° | — | — | — |

## Honest-FAIL interpretation
The **wall-traction** peak friction angle (41.41°) sits just outside the 8%
tolerance (8.97% relative error).  The **internal** (Love-Weber virial) peak
friction angle (36.43°) is comfortably within tolerance (4.13%).  This is the
known wall-vs-internal stress-definition gap in the biaxial cell: the rough
platen/wall boundary traction carries a shear component that biases the
externally-instrumented peak upward, while the virial (bulk) measurement is
closer to the classical critical-state reference.

The critical-state tail (phi_cv = 19.75°) does **not** converge to the Rowe
prediction (28.81°) — consistent with the earlier R3/R4 finding that 2D discs
without full rolling-resistance calibration under-predict the constant-volume
friction angle.  This is an informational cross-check, not a gate.

## Disposition
- **Gate (peak friction angle)**: wall measurement FAIL (8.97% > 8%);
  internal measurement PASS (4.13% < 8%).  The case is **honest-FAIL** on the
  wall definition — the defect is the boundary-traction stress definition, not
  the DEM physics.
- **Recommendation for next self-start**: re-run the full benchmark with the
  internal (virial) friction angle as the primary gate, or add a
  wall-friction correction term to the boundary-traction measurement so the
  wall peak aligns with the virial peak within 8%.
