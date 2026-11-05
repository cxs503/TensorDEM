"""Closed-loop restitution verification evidence generator (m3129).

Re-runs the four previously red closed-loop checks of
``tests/test_dem3d_restitution_verification.py`` and dumps every number into
``artifacts/restitution_closed_loop_m3129/evidence.json`` plus a two-panel
figure.  All report tables are injected from this JSON (zero hand-copying).

Run:  python3 scripts/verify_restitution_closed_loop.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensordem.contact_restitution import (
    classic_restitution_ratio,
    clamped_restitution_ratio,
    damping_for_target_restitution,
    measure_dem3d_impact,
)
from tensordem.contact_tangential import restitution_to_damping

RADIUS = 0.025
DENSITY = 917.0
K_N = 5_000.0
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3


def main() -> None:
    out_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts",
                           "restitution_closed_loop_m3129")
    os.makedirs(out_dir, exist_ok=True)
    evidence: dict = {
        "k_n": K_N, "mass": MASS, "radius": RADIUS, "density": DENSITY,
        "solver": "IceDEM3D 2x2x2 carved pair, contact_damping_model='absolute'",
        "rows": [],
    }

    # 1) solver vs clamped reference vs classic, gamma sweep
    for gamma in (1.0, 3.0, 5.0, 10.0):
        ref = clamped_restitution_ratio(gamma, K_N, MASS)
        run = measure_dem3d_impact(gamma)
        zeta = run["zeta"]
        classic = classic_restitution_ratio(zeta)
        evidence["rows"].append({
            "gamma_n": gamma,
            "zeta": zeta,
            "e_solver": run["e_measured"],
            "e_reference": ref["e_reference"],
            "e_classic": classic,
            "solver_vs_reference_rel": abs(run["e_measured"] - ref["e_reference"]) / ref["e_reference"],
            "classic_vs_reference_rel": abs(classic - ref["e_reference"]) / ref["e_reference"],
            "contact_time_s": run["contact_time_s"],
            "ref_contact_time_s": ref["contact_time_s"],
            "dt": run["dt"],
        })

    # 2) target-restitution closed loop (e_target = 0.6)
    target = 0.6
    cal = damping_for_target_restitution(target, K_N, MASS)
    run_cal = measure_dem3d_impact(cal["gamma_n"])
    classic_gamma = restitution_to_damping(target, K_N, MASS)
    run_classic = measure_dem3d_impact(classic_gamma)
    evidence["closed_loop"] = {
        "e_target": target,
        "gamma_calibrated": cal["gamma_n"],
        "zeta_calibrated": cal["zeta"],
        "e_solver_calibrated": run_cal["e_measured"],
        "err_calibrated_rel": abs(run_cal["e_measured"] - target) / target,
        "gamma_classic": classic_gamma,
        "e_solver_classic": run_classic["e_measured"],
        "err_classic_rel": abs(run_classic["e_measured"] - target) / target,
    }

    # 3) equal-mass pair (reduced mass m/2)
    run_pair = measure_dem3d_impact(2.0, both_free=True)
    ref_pair = clamped_restitution_ratio(2.0, K_N, 0.5 * MASS)
    evidence["equal_mass_pair"] = {
        "m_eff": run_pair["m_eff"],
        "m_expected": 0.5 * MASS,
        "e_solver": run_pair["e_measured"],
        "e_reference": ref_pair["e_reference"],
        "rel": abs(run_pair["e_measured"] - ref_pair["e_reference"]) / ref_pair["e_reference"],
    }

    # 4) time-step refinement at gamma = 10 (model error survives refinement)
    base = measure_dem3d_impact(10.0)
    refined = measure_dem3d_impact(10.0, dt_scale=0.5)
    evidence["refinement"] = {
        "e_base": base["e_measured"],
        "e_refined": refined["e_measured"],
        "rel": abs(refined["e_measured"] - base["e_measured"]) / base["e_measured"],
    }

    with open(os.path.join(out_dir, "evidence.json"), "w") as fh:
        json.dump(evidence, fh, indent=2)

    # figure: left = e vs zeta (solver / reference / classic), right = closed loop
    rows = evidence["rows"]
    zetas = [r["zeta"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(zetas, [r["e_reference"] for r in rows], "o-", label="clamped reference (RK4)")
    ax1.plot(zetas, [r["e_solver"] for r in rows], "s--", label="DEM3D solver (this fix)")
    ax1.plot(zetas, [r["e_classic"] for r in rows], "^:", label="classic unclamped")
    ax1.set_xlabel(r"$\zeta = \gamma_n / (2\sqrt{k_n m_{eff}})$")
    ax1.set_ylabel("coefficient of restitution $e$")
    ax1.set_title("Two-sphere impact, $k_n=5{,}000$ N/m, $m=0.0600$ kg")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    cl = evidence["closed_loop"]
    bars = ax2.bar(["calibrated\n(clamped inverse)", "classic mapping"],
                   [cl["err_calibrated_rel"], cl["err_classic_rel"]],
                   color=["#2a788e", "#c0392b"])
    ax2.axhline(0.015, color="k", ls="--", lw=1)
    ax2.text(1.35, 0.015, "1.5% gate", fontsize=8, va="bottom")
    for b, r in zip(bars, (cl["err_calibrated_rel"], cl["err_classic_rel"])):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                 f"{r:.3%}", ha="center", fontsize=9)
    ax2.set_ylabel(f"|e - {cl['e_target']}| / {cl['e_target']}")
    ax2.set_title("Closed loop at $e_{target}=0.6$")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "restitution_closed_loop.png"), dpi=150)
    print(json.dumps({"rows": rows, "closed_loop": cl,
                      "equal_mass_pair": evidence["equal_mass_pair"],
                      "refinement": evidence["refinement"]}, indent=2))


if __name__ == "__main__":
    main()
