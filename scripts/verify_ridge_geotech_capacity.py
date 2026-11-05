"""Verify the ridge-grounding undrained capacity chain against Skempton/Vesic.

Run with ``PYTHONPATH=src python scripts/verify_ridge_geotech_capacity.py``.
Writes ``artifacts/ridge_capacity_verification/{capacity_verification.json,png}``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensordem.ridge_geotech import (
    SKEMPTON_NC_STRIP_MAX,
    MohrCoulombSoil,
    bearing_capacity_factors,
    skempton_undrained_nc,
)

OUT = Path("artifacts/ridge_capacity_verification")
VESIC_TABLE = {
    0.0: (5.14, 1.00, 0.00), 10.0: (8.35, 2.47, 1.22), 20.0: (14.83, 6.40, 5.39),
    25.0: (20.72, 10.66, 10.88), 30.0: (30.14, 18.40, 22.40),
    35.0: (46.12, 33.30, 48.03), 40.0: (75.31, 64.20, 109.41),
}
SKEMPTON_STRIP_TABLE = {0.0: 5.14, 0.5: 5.65, 1.0: 6.17, 2.0: 7.20, 2.5: 7.50, 4.0: 7.50}

DIAMETER = 0.914          # Beaufort reference pipe (36 in)
COVER_DEPTH = 0.5         # main coupled case in scripts/run_ice_ridge_grounding.py
SU = 30.0e3               # undrained shear strength of the reference seabed


def prerefactor_nc(z_over_b: float) -> float:
    """The pre-m2983 implementation: Skempton depth factor *times* Meyerhof."""
    return min(5.14 * (1.0 + 0.2 * z_over_b), 9.0) * (1.0 + 0.2 * min(z_over_b, 1.0))


def rel(a: float, b: float) -> float:
    return (a - b) / b


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    vesic = []
    for phi, (rc, rq, rg) in sorted(VESIC_TABLE.items()):
        c, q, g = bearing_capacity_factors(math.radians(phi))
        vesic.append({"phi_deg": phi, "Nc_code": c, "Nc_ref": rc,
                      "Nq_code": q, "Nq_ref": rq,
                      "Ngamma_code": g, "Ngamma_ref": rg,
                      "max_abs_err": max(abs(c - rc), abs(q - rq), abs(g - rg))})

    skempton = []
    for ratio, ref in sorted(SKEMPTON_STRIP_TABLE.items()):
        code = skempton_undrained_nc(ratio * DIAMETER, DIAMETER)
        skempton.append({"z_over_B": ratio, "Nc_code": code, "Nc_ref": ref,
                         "rel_err": rel(code, ref)})

    grid = np.linspace(0.0, 4.0, 401)
    curve = []
    for r in grid:
        ref = skempton_undrained_nc(float(r) * DIAMETER, DIAMETER)
        old = prerefactor_nc(float(r))
        curve.append({"z_over_B": float(r), "Nc_reference": ref, "Nc_prefix": old,
                      "Nc_current": ref, "rel_err_prefix": rel(old, ref)})

    z_over_b_case = (COVER_DEPTH + DIAMETER) / DIAMETER
    soil = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=SU,
                           saturated_unit_weight=18.0e3, water_unit_weight=10.06e3,
                           undrained=True)
    q_current = soil.vertical_pressure_capacity(DIAMETER, COVER_DEPTH + DIAMETER)
    q_prefix = SU * prerefactor_nc(z_over_b_case)
    p_current = soil.vertical_capacity_per_length(DIAMETER, COVER_DEPTH)
    p_prefix = q_prefix * DIAMETER

    case = {
        "diameter_m": DIAMETER, "cover_depth_m": COVER_DEPTH,
        "su_Pa": SU, "z_over_B": z_over_b_case,
        "Nc_current": q_current / SU, "Nc_prefix": q_prefix / SU,
        "Nc_skempton": skempton_undrained_nc(COVER_DEPTH + DIAMETER, DIAMETER),
        "q_current_Pa": q_current, "q_prefix_Pa": q_prefix,
        "capacity_per_length_current_N_per_m": p_current,
        "capacity_per_length_prefix_N_per_m": p_prefix,
        "rel_err_prefix": rel(q_prefix, q_current),
    }

    summary = {
        "protocol": "tensordem-ridge-capacity-skempton-audit-v1",
        "reference": "Prandtl (1920) slip line Nc=2+pi; Vesic (1973) factors; "
                     "Skempton (1951) strip embedment, cap 7.5 (strip) / 9.0 (square)",
        "vesic_table": vesic, "skempton_table": skempton,
        "curve": curve, "reference_case": case,
        "verdict": {
            "vesic_max_abs_err": max(r["max_abs_err"] for r in vesic),
            "skempton_max_abs_ratio_err": max(abs(r["rel_err"]) for r in skempton),
            "prefix_max_ratio_err": max(abs(c["rel_err_prefix"]) for c in curve),
            "acceptance": "<15 % relative error on the capacity factor",
        },
    }
    (OUT / "capacity_verification.json").write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8))
    ratios = np.array([c["z_over_B"] for c in curve])
    old = np.array([c["Nc_prefix"] for c in curve])
    ref = np.array([c["Nc_reference"] for c in curve])

    axes[0].plot(ratios, ref, "k-", lw=2.0, label="Skempton (1951) reference")
    axes[0].plot(ratios, old, "r--", lw=1.6, label="pre-fix implementation")
    axes[0].plot([r["z_over_B"] for r in skempton], [r["Nc_ref"] for r in skempton],
                 "ko", ms=5, label="table points")
    axes[0].axhline(SKEMPTON_NC_STRIP_MAX, color="grey", ls=":", lw=1.0)
    axes[0].set_xlabel("relative depth  z/B  (-)")
    axes[0].set_ylabel("undrained factor  $N_c$  (-)")
    axes[0].set_title("(a) undrained bearing factor")
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(alpha=0.3)

    err_pct = 100.0 * np.array([c["rel_err_prefix"] for c in curve])
    axes[1].plot(ratios, err_pct, "r-", lw=1.6, label="pre-fix vs Skempton")
    axes[1].axhline(15.0, color="k", ls="--", lw=1.0, label="15 % acceptance limit")
    axes[1].plot([z_over_b_case], [100.0 * case["rel_err_prefix"]], "b*", ms=12,
                 label=f"main case z/B={z_over_b_case:.3f}")
    axes[1].set_xlabel("relative depth  z/B  (-)")
    axes[1].set_ylabel("relative error  (%)")
    axes[1].set_title("(b) capacity-factor error of the pre-fix code")
    axes[1].legend(fontsize=7.5, loc="upper left")
    axes[1].grid(alpha=0.3)

    covers = np.linspace(0.0, 2.0, 201)
    p_new = [soil.vertical_capacity_per_length(DIAMETER, float(c)) / 1e3 for c in covers]
    p_old = [SU * prerefactor_nc((float(c) + DIAMETER) / DIAMETER) * DIAMETER / 1e3
             for c in covers]
    axes[2].plot(covers, p_new, "b-", lw=1.8, label="fixed (Skempton, once)")
    axes[2].plot(covers, p_old, "r--", lw=1.6, label="pre-fix (double depth)")
    axes[2].plot([COVER_DEPTH], [p_current / 1e3], "b*", ms=12)
    axes[2].set_xlabel("pipe cover depth above mudline (m)")
    axes[2].set_ylabel("ultimate soil capacity (kN/m)")
    axes[2].set_title(f"(c) reference pipe D={DIAMETER} m, $s_u$={SU/1e3:.0f} kPa")
    axes[2].legend(fontsize=7.5, loc="upper left")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "capacity_verification.png", dpi=160)
    print(json.dumps(summary["verdict"], indent=2))
    print(json.dumps(case, indent=2))


if __name__ == "__main__":
    main()