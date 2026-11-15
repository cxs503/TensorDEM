# CASE-DEM-PLANE-SHEAR-MUIDI — Round 18: Higher-N Resolution Study

> **Anchor**: `CASE-DEM-PLANE-SHEAR-MUIDI` (feature_ledger, status=open)
> **Repository**: TensorDEM
> **Mission**: m21111
> **Kind**: verification debt — round-14 recommendation R4 (N > 1000)
> **Artifacts**: `high_n_sweep.py`, `runs/high_n/point_*.json`, `runs/high_n/result.json`

## ① Case Introduction

Dense plane shear of frictional spheres in a constant-stress servo shear cell,
confronted with the local rheology `mu(I)`, `phi(I)` of da Cruz et al. (2005,
PRE 72:021309) and GDR MiDi (2004, Eur. Phys. J. E 14:341) over the
inertial-number range `I = 1e-3 ... 0.3`.

The ledger acceptance criterion is **<10% error on both mu(I) and phi(I)
curves**.

## ② Why this round exists (the verification debt)

Rounds 1-17 established an honest **FAIL** verdict on the shipped device
(N = 294, box 0.12 × 0.05 × 0.06 m):

- **mu(I)**: 13-33 % below the reference at high I (worst -19.5 % sweep,
  -27.8 % full-cell); only the mid-I points (I ≈ 0.01-0.03) sit within 10 %.
- **noise floor**: the std of the final 25 % window exceeds the 10 % band at
  *every* point (5-32 %), i.e. the curve is *unresolvable* at this particle
  count regardless of protocol (round-16 verdict).
- **phi(I)**: the bed solid fraction reads ~10-15 % low and the dilatancy sign
  is contaminated by the servo-compaction / settled-height artefacts.

Round 14's synthesis listed the root causes and gave an explicit path forward,
recommendation **R4: "increase particle count (N > 1000) for proper bulk
statistics"**.  This round is that verification debt: it re-runs the same
device (same contact law, same P/kn = 1, same seed 5279) on a **deeper + wider
box** (0.16 × 0.08 × 0.09 m → N = 1276, a 4.3× increase) to test whether the
mu(I) deficit and the noise floor are a *particle-count / boundary-layer*
artefact or a genuine physics mismatch.

## ③ Method

- **Device**: `tensordem.mu_i_rheology.simulate_servo_shear`, constant-stress
  servo shear cell, periodic x, walled y/z.
- **Contact**: `mu = 0.5`, `mu_rolling = 0.05`, `e = 0.3`, `kn = 6.25e4`,
  `P/kn = 1` (identical to the baseline arm of all prior rounds).
- **Box**: `0.16 × 0.08 × 0.09` m (32d × 16d × 18d), N = 1276.
- **Sweep**: I targets `[1e-2, 3e-2, 1e-1, 3e-1]`, sheared to strain 0.5
  (past the low-I transient that rounds 8/14 flagged), seed 5279.
- **Reference**: `mu_s = 0.36, mu_2 = 0.84, I_0 = 0.30` (da Cruz / GDR MiDi);
  `phi_c = 0.58, phi_j = 0.64` (hand-copied, PROVENANCE-UNVERIFIED).
- **Cluster**: 4 shards (one I per shard), job `cj3204`.

## ④ Pre-registered verdicts

- **P1 (noise floor)**: mu std (final 25 % window) < 10 % at I ∈ [1e-2, 3e-2]
  — if the noise is a particle-count artefact.
- **P2 (mu(I) convergence)**: |mu_meas/mu_ref − 1| < 10 % at I ∈ [1e-2, 3e-2]
  — if the low-I deficit is a boundary-layer / resolution artefact.
- **P3 (honest verdict)**: full-curve PASS/FAIL reported with the worst error,
  no silent pass.

## ⑤ Results (injected from `runs/high_n/result.json`)

*Table populated by `high_n_sweep.py --assemble` after the cluster shards
return.*

| I_target | I_meas | N | mu_meas | mu_ref | err mu [%] | mu std [%] | phi_meas | phi_ref | err phi [%] |
|---|---|---|---|---|---|---|---|---|---|
| (pending) | | | | | | | | | | |

## ⑥ Error judgement

The key question this round answers: **does the mu(I) deficit shrink toward
the 10 % band when N is raised from 294 to 1276?**  The verdict is reported
honestly in `runs/high_n/result.json` — PASS if the mid-I points are within
10 % and the noise floor is below the band, FAIL otherwise (with the worst
error stated).  No silent pass.

## ⑦ Note / limitation

- The high-I branch (I ≥ 0.1) is the most expensive and the most
  boundary-sensitive; if it remains outside the band even at N = 1276, that
  is strong evidence the deficit is a *genuine* high-I physics mismatch
  (wall-slip / servo-compaction), not a resolution artefact — and the honest
  verdict stays FAIL with the root cause narrowed.
- `phi` reference magnitudes remain PROVENANCE-UNVERIFIED (no in-repo
  literature anchor); the phi(I) comparison is reported with that caveat.
