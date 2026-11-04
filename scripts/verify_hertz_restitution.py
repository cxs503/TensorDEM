"""Verification of the Hertzian restitution target in ``tensordem.dem3d``.

Two levels are measured and cross-checked:

1. **Reference level** - ``tensordem.hertz_restitution`` integrates the contact
   law the solver implements (rigid Hertz spring + secant-scaled linear damper,
   compressive-only) with RK4.  This gives the COR that a given damping ratio
   really produces.
2. **Solver level** - a free-flying pair of ``IceDEM3D`` spheres with
   ``contact_model="hertz"`` is impacted and the rebound speed ratio is measured,
   so the reference is validated against the shipped code rather than itself.

Outputs (JSON + CSV + figure) are written under ``--output``; every number in
``docs/dem3d_hertz_restitution_verification.md`` comes from these files.

Usage::

    python scripts/verify_hertz_restitution.py \
        --output docs/m3608/outputs/hertz_restitution --figure
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

from tensordem.hertz_restitution import (
    HertzImpactSpec,
    hertz_impact_restitution,
    hertz_restitution_bias_table,
    hertz_zeta_for_target,
    measure_solver_impact,
)

E_YOUNG = 1.0e9
NU = 0.3
RADIUS = 0.025
DENSITY = 917.0
E_TARGETS = (0.9, 0.8, 0.7, 0.5, 0.3)
SPEEDS = (0.1, 0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def reference_table(e_targets=E_TARGETS, speeds=SPEEDS, steps: int = 2_000) -> list[dict]:
    return hertz_restitution_bias_table(e_targets, speeds=speeds, steps=steps)


def solver_table(e_targets=E_TARGETS, *, dt_scale: float = 0.35) -> list[dict]:
    rows = []
    for e_target in e_targets:
        for verified in (False, True):
            res = measure_solver_impact(e_target=e_target, verified=verified,
                                        dt_scale=dt_scale)
            rows.append({
                "e_target": e_target,
                "mapping": "verified" if verified else "classic",
                "e_measured": res["e_measured"],
                "bias_pct": 100.0 * (res["e_measured"] / e_target - 1.0),
                "dt_scale": dt_scale,
                "dt_s": res["dt_s"],
                "contact_steps": res["contact_steps"],
                "steps_to_separation": res["steps_to_separation"],
                "peak_overlap_m": res["peak_overlap_m"],
            })
    return rows


def dt_convergence(e_target: float = 0.5, scales=(0.5, 0.35, 0.2)) -> list[dict]:
    rows = []
    for verified in (False, True):
        for scale in scales:
            res = measure_solver_impact(e_target=e_target, verified=verified,
                                        dt_scale=scale)
            rows.append({
                "e_target": e_target,
                "mapping": "verified" if verified else "classic",
                "dt_scale": scale,
                "dt_s": res["dt_s"],
                "contact_steps": res["contact_steps"],
                "e_measured": res["e_measured"],
                "bias_pct": 100.0 * (res["e_measured"] / e_target - 1.0),
            })
    return rows


def speed_independence(e_target: float = 0.5, speeds=(0.25, 0.5, 1.0)) -> list[dict]:
    rows = []
    for verified in (False, True):
        for v0 in speeds:
            res = measure_solver_impact(e_target=e_target, verified=verified, v0=v0)
            rows.append({
                "e_target": e_target,
                "mapping": "verified" if verified else "classic",
                "v0_each_m_s": v0,
                "closing_speed_m_s": 2.0 * v0,
                "e_measured": res["e_measured"],
                "bias_pct": 100.0 * (res["e_measured"] / e_target - 1.0),
            })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in keys})


def make_figure(path: Path, reference: list[dict], solver: list[dict],
                trace: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    ax = axes[0]
    targets = sorted({row["e_target"] for row in reference})
    ship = [max(abs(r["bias_shipped_pct"]) for r in reference if r["e_target"] == t)
            for t in targets]
    corr = [max(abs(r["bias_corrected_pct"]) for r in reference if r["e_target"] == t)
            for t in targets]
    ax.plot(targets, ship, "o-", label="shipped mapping (classic zeta)")
    ax.plot(targets, corr, "s-", label="verified mapping")
    ax.axhline(3.0, color="crimson", ls="--", lw=1.0, label="3 % accuracy band")
    ax.set_yscale("log")
    ax.set_xlabel("requested restitution $e_{target}$ [-]")
    ax.set_ylabel("|relative COR error| [%]")
    ax.set_title("Reference model: bias vs requested COR")
    ax.invert_xaxis()
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    ax = axes[1]
    for mapping, style in (("classic", "o-"), ("verified", "s-")):
        pts = sorted((r["e_target"], r["e_measured"]) for r in solver
                     if r["mapping"] == mapping)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], style, label=f"solver, {mapping}")
    ax.plot([0.25, 1.0], [0.25, 1.0], "k:", lw=1.0, label="1:1")
    ax.set_xlabel("requested restitution $e_{target}$ [-]")
    ax.set_ylabel("measured COR [-]")
    ax.set_title("IceDEM3D pair impact (hertz law)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    t_ref, d_ref, f_ref = trace["reference"]
    t_sol, d_sol, f_sol = trace["solver"]
    ax.plot(t_sol, [d * 1e3 for d in d_sol], "o", ms=3, label="solver overlap")
    ax.plot(t_ref, [d * 1e3 for d in d_ref], "-", lw=1.2, label="reference overlap")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("overlap [mm]")
    ax2 = ax.twinx()
    ax2.plot(t_sol, f_sol, "o", ms=3, color="tab:orange", label="solver force")
    ax2.plot(t_ref, f_ref, "-", lw=1.2, color="tab:red", label="reference force")
    ax2.set_ylabel("normal force [N]")
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    ax.legend(handles=handles, fontsize=7)
    ax.set_title("Contact trace, $e_{target}$=0.5, verified mapping")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/m3608/outputs/hertz_restitution")
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    reference = reference_table()
    solver = solver_table()
    convergence = dt_convergence()
    speeds = speed_independence()

    summary = {
        "reference_rows": reference,
        "solver_rows": solver,
        "dt_convergence_rows": convergence,
        "speed_independence_rows": speeds,
        "worst_shipped_bias_pct": max(abs(r["bias_shipped_pct"]) for r in reference),
        "worst_verified_bias_pct": max(abs(r["bias_corrected_pct"]) for r in reference),
        "worst_solver_shipped_bias_pct": max(
            abs(r["bias_pct"]) for r in solver if r["mapping"] == "classic"),
        "worst_solver_verified_bias_pct": max(
            abs(r["bias_pct"]) for r in solver if r["mapping"] == "verified"),
        "reference_model": {
            "e_star_pa": 1.0 / (2.0 * (1.0 - NU**2) / E_YOUNG),
            "r_eff_m": 0.5 * RADIUS,
            "mass_kg": DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3,
        },
    }
    (out / "hertz_restitution_verification.json").write_text(
        json.dumps(summary, indent=2), encoding="utf8")
    write_csv(out / "reference_bias.csv", reference)
    write_csv(out / "solver_bias.csv", solver)
    write_csv(out / "solver_dt_convergence.csv", convergence)
    write_csv(out / "solver_speed_independence.csv", speeds)

    if args.figure:
        spec = HertzImpactSpec(v_in=1.0)
        zeta = hertz_zeta_for_target(0.5)["zeta_eff"]
        ref = hertz_impact_restitution(zeta, spec, record=True)
        sol = measure_solver_impact(e_target=0.5, verified=True, record=True)
        trace = {
            "reference": ([h[0] for h in ref["history"]],
                          [h[1] for h in ref["history"]],
                          [h[2] for h in ref["history"]]),
            "solver": ([h[0] for h in sol["history"]],
                       [h[1] for h in sol["history"]],
                       [h[2] for h in sol["history"]]),
        }
        make_figure(out / "hertz_restitution_bias.png", reference, solver, trace)

    print(json.dumps({k: v for k, v in summary.items()
                      if k.endswith("bias_pct") or k == "reference_model"}, indent=2))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())