"""Verify the buried-pipeline subgrade capacity model against published references.

Runs :func:`tensordem.ridge_geotech_verification.verify_subgrade_capacity_model`,
writes ``artifacts/ridge_geotech_capacity/verification.json`` and the figures used
by the campaign report, and prints the ledger.  Nothing is transcribed by hand.

    python scripts/verify_ridge_geotech_capacity.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensordem.ridge_geotech import (MohrCoulombSoil,  # noqa: E402
                                     SofteningVerticalSpring, skempton_nc)
from tensordem.ridge_geotech_verification import (  # noqa: E402
    SKEMPTON_STRIP_TABLE, undrained_capacity_impact, verify_subgrade_capacity_model)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "artifacts", "ridge_geotech_capacity")
FIG = os.path.join(OUT, "figures")

DIAMETER = 0.914
EMBEDMENT = 0.5


def _legacy_nc(z_over_b: float) -> float:
    """The pre-fix formula (kept for the figure only)."""
    k = min(z_over_b, 1.0)
    return min(5.14 * (1.0 + 0.2 * z_over_b), 9.0) * (1.0 + 0.2 * k)


def _figure(result: dict) -> list[str]:
    os.makedirs(FIG, exist_ok=True)
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
    ax.plot([row[0] for row in SKEMPTON_STRIP_TABLE],
            [row[1] for row in SKEMPTON_STRIP_TABLE], "o", mfc="none",
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
    path = os.path.join(FIG, "capacity_verification.png")
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
    path = os.path.join(FIG, "subgrade_reaction_corrected.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    result = verify_subgrade_capacity_model(diameter=DIAMETER, embedment=EMBEDMENT)
    figures = _figure(result)
    payload = dict(result)
    payload["figures"] = [os.path.relpath(p, os.path.join(OUT, "..", ".."))
                          for p in figures]
    path = os.path.join(OUT, "verification.json")
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


if __name__ == "__main__":
    raise SystemExit(main())