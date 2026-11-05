"""Verify the ridge-grounding subgrade capacity chain against published references.

Two independent verification stages live here; the module is the union of the
two branches that fixed the same defect (the Skempton embedment factor was
applied twice in the undrained branch of :class:`MohrCoulombSoil`).

Stage 1 (``--audit``, default): closed-form reference audit.  Vesic (1973)
factors, the Skempton (1951) undrained strip table, the Prandtl (1920) surface
anchor ``2 + pi`` and a before/after curve of the undrained factor.  Writes
``artifacts/ridge_capacity_verification/{capacity_verification.json,png}``.

Stage 2 (``--ledger``, default): the full reference-check ledger produced by
:func:`tensordem.ridge_geotech_verification.verify_subgrade_capacity_model`
(43 checks rolled up by evidence grade), the engineering impact table and the
figures used by ``docs/ridge_geotech_capacity_verification.md``.  Writes
``artifacts/ridge_geotech_capacity/{verification.json,figures/}``.

    PYTHONPATH=src python scripts/verify_ridge_geotech_capacity.py
    PYTHONPATH=src python scripts/verify_ridge_geotech_capacity.py --audit
    PYTHONPATH=src python scripts/verify_ridge_geotech_capacity.py --ledger
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ridge_geotech import (  # noqa: E402
    SKEMPTON_NC_STRIP_MAX,
    MohrCoulombSoil,
    SofteningVerticalSpring,
    bearing_capacity_factors,
    skempton_nc,
    skempton_undrained_nc,
)
from tensordem.ridge_geotech_verification import (  # noqa: E402
    SKEMPTON_STRIP_TABLE as SKEMPTON_STRIP_TABLE_REF,
    undrained_capacity_impact,
    verify_subgrade_capacity_model,
)

# --------------------------------------------------------------- stage 1 setup
OUT_AUDIT = Path("artifacts/ridge_capacity_verification")
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

def run_reference_audit() -> None:
    OUT_AUDIT.mkdir(parents=True, exist_ok=True)

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
    (OUT_AUDIT / "capacity_verification.json").write_text(json.dumps(summary, indent=2))

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
    fig.savefig(OUT_AUDIT / "capacity_verification.png", dpi=160)
    print(json.dumps(summary["verdict"], indent=2))
    print(json.dumps(case, indent=2))


# --------------------------------------------------------------- stage 2 setup
OUT_LEDGER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "artifacts", "ridge_geotech_capacity")
FIG_LEDGER = os.path.join(OUT_LEDGER, "figures")
EMBEDMENT = COVER_DEPTH       # 0.5 m cover, same reference case as stage 1


def _legacy_nc(z_over_b: float) -> float:
    """The pre-fix formula (kept for the figure only)."""
    k = min(z_over_b, 1.0)
    return min(5.14 * (1.0 + 0.2 * z_over_b), 9.0) * (1.0 + 0.2 * k)


def _figure(result: dict) -> list[str]:
    os.makedirs(FIG_LEDGER, exist_ok=True)
    paths = []

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))

    # (a) Skempton N_c: corrected model vs published table vs legacy defect
    ax = axes[0]
    z = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    legacy = [_legacy_nc(v) for v in z]
    corrected = [skempton_nc(v) for v in z]
    ax.plot(z, legacy, "s--", color="tab:red", ms=4,
            label="pre-fix: embedment factor twice")
    ax.plot(z, corrected, "-", color="tab:blue", lw=2,
            label="corrected: Skempton applied once")
    ax.plot([row[0] for row in SKEMPTON_STRIP_TABLE_REF],
            [row[1] for row in SKEMPTON_STRIP_TABLE_REF], "o", mfc="none",
            color="black", ms=8, label="Skempton (1951) table 3.3")
    ax.axhline(7.5, color="grey", ls=":", lw=1.2, label="strip cap N_c = 7.5")
    ax.axhline(2.0 + 3.141592653589793, color="green", ls=":", lw=1.2,
               label="Prandtl 2 + pi at the surface")
    ax.set_xlabel("relative embedment z/B (-)")
    ax.set_ylabel("undrained N_c (-)")
    ax.set_title("(a) undrained bearing factor")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    # (b) engineering impact: pipeline capacity vs s_u, pre-fix vs corrected
    ax = axes[1]
    strengths = [10.0e3, 20.0e3, 30.0e3, 45.0e3, 60.0e3]
    legacy_cap, corrected_cap = [], []
    for strength in strengths:
        imp = undrained_capacity_impact(diameter=DIAMETER, embedment=EMBEDMENT,
                                        cohesion=strength)
        legacy_cap.append(imp["legacy_capacity_per_length"] / 1e3)
        corrected_cap.append(imp["corrected_capacity_per_length"] / 1e3)
    ax.plot([s / 1e3 for s in strengths], legacy_cap, "s--", color="tab:red",
            label="pre-fix (over-predicted)")
    ax.plot([s / 1e3 for s in strengths], corrected_cap, "-o", color="tab:blue",
            label="corrected (Skempton, once)")
    ax.set_xlabel("undrained shear strength s_u (kPa)")
    ax.set_ylabel("ultimate capacity p_ult (kN/m)")
    ax.set_title(f"(b) 36 in pipe, D = {DIAMETER} m, cover {EMBEDMENT} m")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) relative error of every reference check
    ax = axes[2]
    rows = [row for row in result["checks"] if not row["compared_absolutely"]]
    rows.sort(key=lambda row: row["relative_error"], reverse=True)
    ax.bar(range(len(rows)), [max(row["relative_error"], 1e-16) for row in rows],
           color="tab:blue")
    ax.set_yscale("log")
    ax.axhline(0.03, color="tab:red", ls="--", lw=1.2, label="3 % acceptance line")
    ax.set_xlabel("reference check (sorted, worst first)")
    ax.set_ylabel("relative error vs reference (-)")
    ax.set_title(f"(c) {len(rows)} reference checks, max "
                 f"{result['summary']['max_relative_error'] * 100:.3f} %")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(FIG_LEDGER, "capacity_verification.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)

    # (d) reaction curve of the softening spring on the corrected capacity
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    soil = MohrCoulombSoil(friction_angle_deg=0.0, cohesion=30.0e3,
                           saturated_unit_weight=18.0e3, young_modulus=10.0e6,
                           water_unit_weight=10.06e3, undrained=True)
    spring = SofteningVerticalSpring(soil, DIAMETER, EMBEDMENT)
    displacements = [0.01 * i for i in range(1, 400)]
    ax.plot(displacements, [spring.reaction(w) / 1e3 for w in displacements],
            color="tab:blue", label="peak-to-residual subgrade spring")
    ax.axhline(spring.peak / 1e3, color="tab:orange", ls="--", lw=1.2,
               label=f"peak = 0.9 p_ult = {spring.peak / 1e3:.1f} kN/m")
    ax.axhline(spring.residual / 1e3, color="tab:green", ls=":", lw=1.2,
               label=f"residual = 0.8 peak = {spring.residual / 1e3:.1f} kN/m")
    ax.set_xlabel("vertical pipe displacement (m)")
    ax.set_ylabel("reaction per unit length (kN/m)")
    ax.set_title("corrected subgrade reaction, s_u = 30 kPa")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(FIG_LEDGER, "subgrade_reaction_corrected.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)
    return paths


def run_reference_ledger() -> int:
    os.makedirs(OUT_LEDGER, exist_ok=True)
    result = verify_subgrade_capacity_model(diameter=DIAMETER, embedment=EMBEDMENT)
    figures = _figure(result)
    payload = dict(result)
    payload["figures"] = [os.path.relpath(p, os.path.join(OUT_LEDGER, "..", ".."))
                          for p in figures]
    path = os.path.join(OUT_LEDGER, "verification.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["impact"], indent=2))
    print(f"wrote {path}")
    for row in payload["checks"]:
        flag = "ok  " if row["passed"] else "FAIL"
        print(f"{flag} {row['id']:<28} model={row['model']:<12.6g} "
              f"ref={row['reference']:<12.6g} rel={row['relative_error']:.3e} "
              f"[{row['evidence']}]")
    return 0 if payload["summary"]["all_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Run both verification stages (or the one named on the command line)."""
    args = list(sys.argv[1:] if argv is None else argv)
    unknown = [a for a in args if a not in ("--audit", "--ledger")]
    if unknown:
        raise SystemExit(f"unknown arguments: {unknown}")
    if not args:
        args = ["--audit", "--ledger"]
    status = 0
    if "--audit" in args:
        run_reference_audit()
    if "--ledger" in args:
        status = run_reference_ledger()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
