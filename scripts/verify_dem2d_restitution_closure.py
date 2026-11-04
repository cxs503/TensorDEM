"""Generate the 2-D restitution closure evidence (JSON + CSV + figure).

Run from the repository root::

    python scripts/verify_dem2d_restitution_closure.py

Writes ``outputs/dem2d_restitution_closure/``:

* ``closure_table.json``  -- per-damping measured/predicted restitution;
* ``closure_table.csv``   -- same table, flat;
* ``inverse_map.json``    -- recovered damping for two target restitutions;
* ``timestep_convergence.json`` -- refined-dt error series;
* ``closure.png``         -- measured vs both closed forms and dt convergence.

Every number in ``docs/dem2d_contact_damping_restitution_closure.md`` is copied
from these files, so the document cannot drift away from the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem.dem2d_restitution import (  # noqa: E402
    discretization_bias,
    measure_restitution,
    recommended_classical_damping,
    recommended_damping,
)

STIFFNESS = 2_000.0
DAMPINGS = (0.0, 1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0)
TARGETS = (0.90, 0.70, 0.40)
FIELDS = [
    "damping_N_s_per_m",
    "damping_ratio",
    "measured_restitution",
    "unilateral_prediction",
    "classical_prediction",
    "relative_error_vs_unilateral",
    "relative_error_vs_classical",
    "dt_s",
    "steps",
]


def closure_table() -> list[dict[str, float | int]]:
    rows = []
    for damping in DAMPINGS:
        result = measure_restitution(stiffness=STIFFNESS, damping=damping)
        payload = result.as_dict()
        rows.append({field: payload[field] for field in FIELDS})
    return rows


def inverse_map() -> list[dict[str, float | int]]:
    rows = []
    for target in TARGETS:
        for name, damping in (
            ("unilateral", recommended_damping(target, stiffness=STIFFNESS)),
            ("classical", recommended_classical_damping(target, stiffness=STIFFNESS)),
        ):
            result = measure_restitution(stiffness=STIFFNESS, damping=damping)
            rows.append(
                {
                    "target_restitution": target,
                    "inverse": name,
                    "damping_N_s_per_m": damping,
                    "measured_restitution": result.measured_restitution,
                    "signed_error": (result.measured_restitution - target) / target,
                    "within_3pct": abs(result.measured_restitution - target) / target < 0.03,
                }
            )
    return rows


def timestep_convergence() -> list[dict[str, float | int]]:
    rows = []
    for damping in (5.0, 20.0):
        for level in discretization_bias(
            stiffness=STIFFNESS, damping=damping, refinement=4, levels=3
        ):
            payload = level.as_dict()
            rows.append(
                {
                    "damping_N_s_per_m": damping,
                    "dt_s": payload["dt_s"],
                    "measured_restitution": payload["measured_restitution"],
                    "unilateral_prediction": payload["unilateral_prediction"],
                    "relative_error_vs_unilateral": payload["relative_error_vs_unilateral"],
                }
            )
    return rows


def write_figure(
    table: list[dict[str, float | int]],
    convergence: list[dict[str, float | int]],
    output: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional plotting dependency
        print("matplotlib unavailable; skipping figure", file=sys.stderr)
        return

    ratios = [row["damping_ratio"] for row in table]
    # Dense continuous branches for the two closed forms.
    import math

    from tensordem.dem2d_restitution import particle_mass
    from tensordem.restitution import classical_restitution, unilateral_restitution

    mass = particle_mass()
    dense = [i * 0.005 for i in range(0, 241)]
    damping_of = [2.0 * zeta * math.sqrt(STIFFNESS * mass) for zeta in dense]
    unilateral_curve = [unilateral_restitution(c, STIFFNESS, mass, 1.0) for c in damping_of]
    classical_curve = [classical_restitution(c, STIFFNESS, mass) for c in damping_of]

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    left.plot(dense, unilateral_curve, "-", label="unilateral (clipped law) prediction")
    left.plot(dense, classical_curve, "--", label="classical bilateral formula")
    left.plot(
        ratios,
        [row["measured_restitution"] for row in table],
        "ko",
        markersize=5,
        label="IceDEM 2-D measured",
    )
    left.set_xlabel("damping ratio  $\\zeta = c/(2\\sqrt{km})$")
    left.set_ylabel("coefficient of restitution  $e$")
    left.set_title("2-D engine restitution: measured vs closed forms")
    left.grid(alpha=0.3)
    left.legend(fontsize=8)

    for damping, style in ((5.0, "o-"), (20.0, "s-")):
        series = [r for r in convergence if r["damping_N_s_per_m"] == damping]
        dts = [r["dt_s"] for r in series]
        errors = [r["relative_error_vs_unilateral"] * 100.0 for r in series]
        right.loglog(dts, errors, style, label=f"c = {damping:g} N*s/m")
    right.set_xlabel("time step  $\\Delta t$  [s]")
    right.set_ylabel("closure error  [%]")
    right.set_title("Semi-implicit Euler bias: first order in $\\Delta t$")
    right.grid(alpha=0.3, which="both")
    right.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "dem2d_restitution_closure"
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    table = closure_table()
    inverse = inverse_map()
    convergence = timestep_convergence()

    (output / "closure_table.json").write_text(
        json.dumps(
            {
                "protocol": "tensordem-dem2d-restitution-closure-v1",
                "solver": "tensordem.dem.IceDEM (2-D bonded disks, semi-implicit Euler)",
                "contact_law": "F_n = max(0, k_n*delta - c_n*v_n), unilateral clip",
                "probe": "single free particle vs kinematic circular tool (single DOF)",
                "stiffness_N_per_m": STIFFNESS,
                "rows": table,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output / "inverse_map.json").write_text(
        json.dumps({"stiffness_N_per_m": STIFFNESS, "rows": inverse}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "timestep_convergence.json").write_text(
        json.dumps({"stiffness_N_per_m": STIFFNESS, "rows": convergence}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "closure_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(table)
    if not args.no_figure:
        write_figure(table, convergence, output / "closure.png")

    worst_uni = max(row["relative_error_vs_unilateral"] for row in table)
    worst_cls = max(row["relative_error_vs_classical"] for row in table)
    print(json.dumps(
        {
            "rows": len(table),
            "worst_error_vs_unilateral_pct": 100.0 * worst_uni,
            "worst_error_vs_classical_pct": 100.0 * worst_cls,
            "inverse_hits": sum(row["within_3pct"] for row in inverse),
            "inverse_cases": len(inverse),
            "output_dir": str(output),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()