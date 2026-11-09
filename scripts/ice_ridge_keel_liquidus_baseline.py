"""Before/after baseline for the liquidus transposition defect.

Reconstructs the *pre-fix* mirrored liquidus (the temperature anchor list sorted
ascending against a descending brine-salinity list) next to the upstream
``tensorphysics.ice.adfreeze`` table and writes the measured damage to
``artifacts/ice_ridge_keel/liquidus_defect_baseline.json``.

The reconstruction is a local function, so the baseline stays reproducible after
the fix: the file records what the defect cost, not a claim about it.

Usage::

    PYTHONPATH=src python scripts/ice_ridge_keel_liquidus_baseline.py
"""

from __future__ import annotations

import json
import os

import numpy as np

from tensordem import ice_ridge_keel_freezebond as keel
from tensordem import adfreeze_key_adapter as adapter

OUT = os.path.join("artifacts", "ice_ridge_keel", "liquidus_defect_baseline.json")

#: the pre-fix anchor tables (temperature ascending, salinity descending)
T_OLD = np.array([0.0, -1.0, -2.0, -5.0, -10.0, -15.0, -20.0, -25.0])
SB_OLD = np.array([255.0, 240.0, 197.0, 143.0, 78.0, 35.0, 17.5, 2.0])

GRID_T = (-2.0, -5.0, -10.0, -20.0)
GRID_TIME = (1.0e3, 3.6e3, 1.0e4)
GRID_S = (0.0, 5.0, 32.0, 60.0)
GRID_RA = (2.5, 20.0)


def tf_old(salinity_gkg: float) -> float:
    if salinity_gkg == 0.0:
        return 0.0
    order = np.argsort(SB_OLD)
    return float(np.interp(salinity_gkg, SB_OLD[order], T_OLD[order]))


def salinity_factor_old(temperature_c: float, salinity_gkg: float) -> float:
    if salinity_gkg == 0.0:
        return 1.0
    if temperature_c >= tf_old(salinity_gkg):
        return 0.0
    t = min(max(temperature_c, -23.0), 0.0)
    order = np.argsort(T_OLD)
    sb = float(np.interp(t, T_OLD[order], SB_OLD[order])) / 1000.0
    rho, vb = 917.0, 0.0
    for _ in range(8):
        vb = rho * salinity_gkg / 1000.0 / (1024.0 * sb)
        rho = 917.0 * (1.0 - vb) + 1024.0 * vb
    return float((1.0 - min(max(vb, 0.0), 0.6)) ** 3)


def tau_old(temperature_c: float, time_s: float, salinity_gkg: float,
            roughness_um: float) -> float:
    dt = tf_old(salinity_gkg) - temperature_c
    if dt < 0.0:
        return 0.0
    tau_ref = min(5.8301 * dt / (dt + 76.1562), 2.0)
    h = np.sqrt(2.0 * 2.2 * dt * time_s / (917.0 * 3.34e5))
    eta = 1.0 - np.exp(-h / 5.0e-3)
    return float(tau_ref * eta * (1.0 + 0.8 * (1.0 - np.exp(-roughness_um / 20.0)))
                 * salinity_factor_old(temperature_c, salinity_gkg))


def main() -> dict:
    rows = []
    worst_old = 0.0
    nonzero_old = 0
    nonzero_ref = 0
    zeroed_saline = 0
    saline_points = 0
    for T in GRID_T:
        for t in GRID_TIME:
            for S in GRID_S:
                for Ra in GRID_RA:
                    old = tau_old(T, t, S, Ra)
                    ref = 1.0e6 * _upstream_strength(T, t, S, Ra)
                    new = adapter.tau_adfreeze_pa(adapter.AdfreezeKeyParams(
                        temperature_c=T, contact_time_s=t, salinity_gkg=S,
                        roughness_um=Ra))
                    denom = abs(ref) if ref != 0.0 else 1.0
                    worst_old = max(worst_old, abs(old - ref) / denom)
                    nonzero_old += int(old > 0.0)
                    nonzero_ref += int(ref > 0.0)
                    if S == 32.0:
                        saline_points += 1
                        zeroed_saline += int(old == 0.0)
                    rows.append({"T_C": T, "t_s": t, "S_gkg": S, "Ra_um": Ra,
                                 "pre_fix_Pa": old, "post_fix_Pa": new,
                                 "upstream_Pa": ref})
    out = {
        "defect": "seawater liquidus anchor table transposed in the TensorDEM mirror",
        "pre_fix_max_rel_error_vs_upstream": float(worst_old),
        "post_fix_max_rel_error_vs_upstream": float(adapter.platform_parity()["max_rel_error"]),
        "n_points": len(rows),
        "nonzero_pre_fix": nonzero_old,
        "nonzero_upstream": nonzero_ref,
        "Tf_32gkg_pre_fix_C": tf_old(32.0),
        "Tf_32gkg_upstream_C": float(_upstream_Tf(32.0)),
        "Tf_32gkg_post_fix_C": keel.seawater_freezing_point(32.0),
        "tau_minus10C_S32_pre_fix_Pa": tau_old(-10.0, 3.6e3, 32.0, 2.5),
        "tau_minus10C_S32_upstream_Pa": 1.0e6 * _upstream_strength(-10.0, 3.6e3, 32.0, 2.5),
        "saline_points_zeroed_pre_fix": zeroed_saline,
        "saline_points_total": saline_points,
        "grid": rows[:5],
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))
    print("wrote", OUT)
    return out


def _upstream():
    import importlib.util

    if importlib.util.find_spec("tensorphysics") is None:
        return None
    from tensorphysics.ice import adfreeze as ref

    return ref


def _upstream_strength(T, t, S, Ra):
    ref = _upstream()
    if ref is None:                     # fall back to the (now audited) mirror
        return keel.AdfreezeClosure(contact_time_s=t, salinity_gkg=S,
                                    roughness_um=Ra).tau_mpa(T)
    return ref.adfreeze_strength(T, t, S, Ra)


def _upstream_Tf(S):
    ref = _upstream()
    if ref is None:
        return keel.seawater_freezing_point(S)
    return ref.seawater_freezing_point(S)


if __name__ == "__main__":
    main()