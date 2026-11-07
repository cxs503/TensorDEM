# DEM calibration gap ledger (TensorDEM side)

Living write-back of the DEM rows of the m605 capability inventory / gap matrix
(report id 542, `docs/fem_dem_capability_inventory.md` in that worktree) to this
repository, with the closing evidence for each row. Rows are only closed when a
verification surface and a test file exist here; everything else stays open with
the reason.

Ledger keys follow the m605 gap ids so the two documents can be diffed.

| id | m605 state | now | evidence (this repo) | open part |
|---|---|---|---|---|
| **GAP-DEM-G4** tangential friction / rolling resistance | 🟠 high, "no tangential friction / rolling resistance" | **closed for sphere/plane, clump/plane, clump/sphere and clump/clump** | sliding spring: `src/tensordem/contact_tangential.py`, `docs/dem3d_tangential_friction_verification.md`; rolling law (sphere): `src/tensordem/rolling_resistance.py`, `docs/dem3d_rolling_resistance.md`; **rolling + drilling on the clump path: `src/tensordem/clump_rolling.py` + `docs/clump_rolling_resistance_calibration.md` (mission m5052)** | **clump–clump: `src/tensordem/clump_clump.py` + `docs/clump_clump_contact.md` (mission m5186)** — Coulomb tangential spring on merged Hertz-Mindlin junctions; drilling resistance still has no elastic branch and rolling resistance is not defined for a clump pair |
| **GAP-DEM-G6** cell-list / parallel performance | 🟡 medium, "GPU cell list synchronous blocking" | **open** | scale benchmark exists (`docs/dem3d_scale_benchmark.md`, `results-dem3d-scale/`) | no GPU/domain decomposition; not addressed by m5052 |
| **GAP-DEM-G7** large-scale stress test (> 10k particles) | 🟡 medium, "no large-scale test" | **open (partially superseded)** | 10k/50k/100k wall-clock study in `docs/dem3d_scale_benchmark.md` | the 100k case is a wall-clock study, not a physics validation; throughput on the 702 cluster not yet recorded |
| **GAP-DEM-G1** 3-D DEM ↔ fluid two-way coupling | 🔴 blocking | open | 2-D only: `src/tensordem/calibrated_wet_coupling.py`, `docs/calibrated-wet/` | 3-D coupling not implemented |
| **GAP-DEM-G2** buoyancy / wet coupling (3-D) | 🔴 blocking | partially closed | `src/tensordem/buoyancy_wet3d.py`, `docs/dem3d_buoyancy_float.md` | not re-audited by m5052 |
| **GAP-DEM-G3** SUBOFF curved-hull contact | 🔴 blocking | partially closed | `src/tensordem/hull_mesh_contact.py` | curved-hull validation on a real hull not re-audited by m5052 |

## m5052 write-back (clump rolling resistance)

* **Row changed**: `docs/clump_multisphere_particles.md`, table row
  `rolling resistance / torsion friction for clumps`: `NOT_SUPPORTED` →
  **REAL** on the clump–plane and clump–sphere paths
  (`ContactConfig.rolling_friction`, `ContactConfig.drilling_friction`,
  `ContactConfig.rolling_model`, `resistance_moment`).
* **Verification surface**: 16 tests in `tests/test_clump_rolling_resistance.py`
  plus the calibration harness `scripts/m5052_clump_rolling_calibration.py`
  (numbers in `artifacts/m5052/clump_rolling_calibration.json`).
* **Quantified calibration**: tilt threshold `tan(theta_c) = min(mu_s, (a + mu_r r)/h)`
  reproduced to 0.19 … 1.46 % across five `mu_r` values and four clump shapes;
  `mu_r` recovered from the measured angle to ≤ 1.47 %; drilling spin-down exact
  to round-off; no energy creation, stored spring energy inside its analytic
  bound.
* **Applicability stated, not implied**: for equal-radius clumps in a flat pose
  `h = r` and `a/h ≥ 1`, so a tilt test on lumpy clumps measures **sliding**
  friction (41.99° for `mu_s = 0.9`, independent of `mu_r`), not the rolling
  coefficient. `mu_r` is identifiable only for near-spherical supports or from
  dynamic (repose / drum) tests, which remain open.
* **Withheld**: no digitised laboratory tilt-table or repose measurement is in
  this repository, so the *laboratory* calibration of `mu_r` is flagged
  **WITHHELD**; only the protocol and its inversion are verified.
* **Honest FAIL carried forward**: the viscous branch at its default
  `omega_ref = 1e-3` is numerically unstable at `dt = 2e-4` (its measured
  "threshold" of 5.74° is an artefact and is not claimed). Stability requires
  `dt < I omega_ref / M_lim`; the calibration uses `omega_ref = 0.1` for that
  branch.
* **Not addressed by m5052**: GAP-DEM-G6, GAP-DEM-G7 (scale/performance).
  Clump–clump contacts were closed separately in m5186
  (`docs/clump_clump_contact.md`).

## Reproduction

```bash
PYTHONPATH=src python -m pytest tests/test_clump_rolling_resistance.py -q
PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --workers 8
PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --extra-only
PYTHONPATH=src python scripts/m5052_clump_rolling_calibration.py --figures-only
```