"""Pre-declared thresholds, grid and reference anchors for the XVAL.

Every number here was fixed in ``prereg/xval_dem_round2_static_coulomb_com_lever.md``
*before* the cross-validation ran (2026-11-13).  Reference anchors are quoted from
the OSS artifact ``TensorDEM/self`` with their file/§ provenance (zero hand copy:
each value can be re-read at the cited location).
"""

from __future__ import annotations

# --- pre-declared tolerances (prereg §3) -----------------------------------
TOL = {
    "P1_torque_rel_err": 1e-12,      # impl vs independent oracle, closed form
    "P2_saturation_abs_err": 1e-10,  # delivered fraction vs min(1, w/w_eps)
    "P3_residual_rel_err": 1e-12,    # one-step residual vs max(0, w - w_eps)
    "P4_sphere_lever_abs": 1e-15,    # com lever == member lever on a sphere
    "P5_lever_ratio_rel": 1e-12,     # tip/flank arm vs the clump's exact geometry
    "P6_engine_step_slack": 1.0,     # steps to arrest vs ceil(w0/w_eps)
    "P6_energy_rel": 1e-12,          # kinetic energy must not increase
    "P7_repose_pct": 15.0,           # owner precision standard
    "P7_bitwise_deg": 1e-9,          # published bit-level value reproduction
}

# --- leg A parameter grid (dimensionless groups in the report) -------------
MU_ROLL_GRID = (0.1, 0.3, 0.5)
LEVER_GRID = (1.2e-3, 3.0e-3)          # R [m]: member radius / typical clump arm
FN_GRID = (1.0e-4, 1.0e-3)             # normal force [N]
INERTIA_GRID = (1.1e-11, 1.1e-10)      # I_eff [kg m^2]
DT_GRID = (2.0e-5, 5.0e-5, 2.0e-4)     # time step [s] (sensitivity point i)
RATIO_GRID = (1e-3, 1e-1, 0.5, 1.0, 10.0, 100.0, 1e4)   # w / w_eps

# --- leg B: published round-2 repose anchors ------------------------------
# docs/GAP_DEM_CLUMP_ROUND2.md §3, table "有效测量" (ar=1.0, seed=0, mu_roll=0.3)
REFERENCE_REPOSE_DEG = {
    "viscous/member": 17.47,
    "coulomb/member": 14.86,
    "coulomb/com": 15.09,
}
# docs/AUDIT_m15187_GAP-DEM-CLUMP.md line 20: bit-level reproducible value
BITWISE_REPOSE = {"config": "coulomb/member", "angle_profile_deg": 14.858310520973625}

# --- physical / protocol constants shared with the round-2 benchmark ------
MU_ROLL = 0.3            # benchmarks/gap_dem_clump/run_round2_scan.py: MU_ROLL
BASE_RADIUS = 1.2e-3     # m, single-sphere baseline of the clump scan
DENSITY = 2650.0         # kg/m^3
G = 9.81                 # m/s^2
AR_LIST = (1.0, 1.5, 2.0, 3.0)


def verdict(value: float, tol: float) -> bool:
    return bool(abs(value) <= tol)