"""Audit: polyline vs exact circular-arc bow contact geometry (2-D DEM hull).

Quantifies the secant-geometry bias of ``polyline_contact_forces`` against the
independent closed-form circular-bow reference (``delta = r_p + R - r``), and
checks the exact arc kernel in ``tensordem.hull_geometry``.

The audit is a pure geometry/force-law study: no ice sheet, no fracture, no
integration.  Every disk is placed at a prescribed position and the three
contact descriptions are evaluated once:

    * polyline hull        (shipped kernel, ``hull_contact``)
    * sagitta-corrected polyline (mitigation, ``hull_geometry``)
    * exact circular arc   (reference kernel, ``hull_geometry``)

Run:  python scripts/audit_hull_polyline_geometry.py --output outputs/dem2d_hull_geometry
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from tensordem.hull_contact import polyline_contact_forces
from tensordem.hull_geometry import (
    CircularArc,
    arc_hull_contact_forces,
    arc_polyline,
    polyline_error_bounds,
    sagitta,
    sagitta_corrected_polyline,
    segments_for_tolerance,
)

DTYPE = torch.float64


def _angle_of(vector: torch.Tensor) -> float:
    return math.atan2(float(vector[1]), float(vector[0]))


def _angle_between_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    cos = max(-1.0, min(1.0, float((a * b).sum()) / (na * nb)))
    return math.degrees(math.acos(cos))


class BowProbe:
    """Evaluate the three contact descriptions for one disk position."""

    def __init__(
        self,
        radius: float,
        span: float,
        n_segments: int,
        particle_radius: float,
        stiffness: float,
        start_angle: float = -0.5 * math.pi,
    ) -> None:
        self.radius = radius
        self.span = span
        self.n_segments = n_segments
        self.particle_radius = particle_radius
        self.stiffness = stiffness
        self.start_angle = start_angle
        self.theta_segment = span / n_segments
        self.polyline = arc_polyline(radius, span, n_segments, start_angle=start_angle)
        self.corrected = sagitta_corrected_polyline(
            radius, span, n_segments, start_angle=start_angle
        )
        self.arc = CircularArc((0.0, 0.0), radius, start_angle, span)
        self.zero_velocity = torch.zeros((1, 2), dtype=DTYPE)
        self.zero_hull_velocity = torch.zeros(2, dtype=DTYPE)

    # -- exact closed-form reference (independent of both kernels) ----------
    def reference_force(self, theta: float, penetration: float) -> torch.Tensor:
        direction = torch.tensor([math.cos(theta), math.sin(theta)], dtype=DTYPE)
        return self.stiffness * penetration * direction

    def exact_penetration(self, radial_position: float) -> float:
        return max(0.0, self.particle_radius + self.radius - radial_position)

    # -- kernels -----------------------------------------------------------
    def _polyline_force(self, vertices: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        force, _ = polyline_contact_forces(
            position.reshape(1, 2),
            self.zero_velocity,
            vertices,
            self.zero_hull_velocity,
            particle_radius=self.particle_radius,
            hull_radius=0.0,
            stiffness=self.stiffness,
        )
        return force[0]

    def polyline_force(self, position: torch.Tensor) -> torch.Tensor:
        return self._polyline_force(self.polyline, position)

    def corrected_force(self, position: torch.Tensor) -> torch.Tensor:
        return self._polyline_force(self.corrected, position)

    def arc_force(self, position: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        force, reaction = arc_hull_contact_forces(
            position.reshape(1, 2),
            self.zero_velocity,
            [self.arc],
            self.zero_hull_velocity,
            particle_radius=self.particle_radius,
            stiffness=self.stiffness,
        )
        return force[0], reaction

    def place(self, segment: int, xi: float, penetration: float) -> tuple[torch.Tensor, float]:
        """Disk centre on the bow, ``xi`` in [-1, 1] across the segment."""
        theta = self.start_angle + (segment + 0.5 + 0.5 * xi) * self.theta_segment
        radial = self.radius + self.particle_radius - penetration
        position = torch.tensor(
            [radial * math.cos(theta), radial * math.sin(theta)], dtype=DTYPE
        )
        return position, theta


def relative_error(measured: float, reference: float) -> float:
    if reference <= 0.0:
        return 0.0 if measured <= 0.0 else math.inf
    return (measured - reference) / reference


def sweep_mid_chord(
    radius_ratio: float,
    particle_radius: float,
    span: float,
    n_values: Sequence[int],
    penetration_ratios: Sequence[float],
) -> list[dict[str, Any]]:
    radius = radius_ratio * particle_radius
    rows: list[dict[str, Any]] = []
    for n_segments in n_values:
        probe = BowProbe(radius, span, n_segments, particle_radius, stiffness=1.0)
        segment = n_segments // 2
        for penetration_ratio in penetration_ratios:
            penetration = penetration_ratio * particle_radius
            position, theta = probe.place(segment, 0.0, penetration)
            reference = probe.reference_force(theta, penetration)
            chord = probe.polyline_force(position)
            corrected = probe.corrected_force(position)
            arc_force, reaction = probe.arc_force(position)
            bounds = polyline_error_bounds(radius, span, n_segments, penetration)
            action_reaction = float(
                torch.linalg.vector_norm(arc_force.reshape(1, 2).sum(dim=0) + reaction)
            )
            rows.append(
                {
                    "radius_ratio": radius_ratio,
                    "particle_radius": particle_radius,
                    "radius": radius,
                    "n_segments": n_segments,
                    "segment_angle": bounds["segment_angle"],
                    "sagitta": bounds["sagitta"],
                    "penetration_ratio": penetration_ratio,
                    "penetration": penetration,
                    "reference_force": float(torch.linalg.vector_norm(reference)),
                    "polyline_force": float(torch.linalg.vector_norm(chord)),
                    "corrected_force": float(torch.linalg.vector_norm(corrected)),
                    "arc_force": float(torch.linalg.vector_norm(arc_force)),
                    "polyline_relative_error": relative_error(
                        float(torch.linalg.vector_norm(chord)),
                        float(torch.linalg.vector_norm(reference)),
                    ),
                    "corrected_relative_error": relative_error(
                        float(torch.linalg.vector_norm(corrected)),
                        float(torch.linalg.vector_norm(reference)),
                    ),
                    "arc_relative_error": relative_error(
                        float(torch.linalg.vector_norm(arc_force)),
                        float(torch.linalg.vector_norm(reference)),
                    ),
                    "predicted_relative_error": bounds["relative_force_error"],
                    "predicted_force_ratio": bounds["force_ratio"],
                    "polyline_direction_error_deg": _angle_between_deg(chord, reference),
                    "arc_direction_error_deg": _angle_between_deg(arc_force, reference),
                    "action_reaction_residual_N": action_reaction,
                    "contact_lost": bool(float(torch.linalg.vector_norm(chord)) == 0.0),
                }
            )
    return rows


def sweep_washboard(
    radius_ratio: float,
    particle_radius: float,
    span: float,
    n_segments: int,
    penetration_ratio: float,
    samples: int,
) -> list[dict[str, Any]]:
    radius = radius_ratio * particle_radius
    probe = BowProbe(radius, span, n_segments, particle_radius, stiffness=1.0)
    segment = n_segments // 2
    rows: list[dict[str, Any]] = []
    for index in range(samples):
        xi = -1.0 + 2.0 * index / (samples - 1)
        penetration = penetration_ratio * particle_radius
        position, theta = probe.place(segment, xi, penetration)
        reference = probe.reference_force(theta, penetration)
        reference_norm = float(torch.linalg.vector_norm(reference))
        chord = probe.polyline_force(position)
        corrected = probe.corrected_force(position)
        arc_force, _ = probe.arc_force(position)
        offset = xi * 0.5 * probe.theta_segment
        predicted_penetration = max(
            0.0,
            probe.particle_radius + radius * math.cos(0.5 * probe.theta_segment)
            - (radius + particle_radius - penetration) * math.cos(offset),
        )
        rows.append(
            {
                "xi": xi,
                "angle_offset_deg": math.degrees(offset),
                "predicted_polyline_penetration": predicted_penetration,
                "polyline_relative_error": relative_error(
                    float(torch.linalg.vector_norm(chord)), reference_norm
                ),
                "corrected_relative_error": relative_error(
                    float(torch.linalg.vector_norm(corrected)), reference_norm
                ),
                "arc_relative_error": relative_error(
                    float(torch.linalg.vector_norm(arc_force)), reference_norm
                ),
                "polyline_direction_error_deg": _angle_between_deg(chord, reference),
                "corrected_direction_error_deg": _angle_between_deg(corrected, reference),
                "arc_direction_error_deg": _angle_between_deg(arc_force, reference),
            }
        )
    return rows


def dropout_threshold(
    radius_ratio: float,
    particle_radius: float,
    span: float,
    n_segments: int,
    tolerance: float = 1e-12,
) -> float:
    """Bisect the smallest penetration for which the chord still contacts."""
    radius = radius_ratio * particle_radius
    probe = BowProbe(radius, span, n_segments, particle_radius, stiffness=1.0)
    segment = n_segments // 2
    low = 1e-12 * particle_radius
    high = max(4.0 * sagitta(radius, span, n_segments), particle_radius)
    for _ in range(200):
        mid = 0.5 * (low + high)
        position, _theta = probe.place(segment, 0.0, mid)
        if float(torch.linalg.vector_norm(probe.polyline_force(position))) > tolerance:
            high = mid
        else:
            low = mid
    return high


def make_figures(
    mid_chord_rows: list[dict[str, Any]],
    washboard_rows: list[dict[str, Any]],
    penetration_ratios: Sequence[float],
    tolerance_table: list[dict[str, Any]],
    output: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names: list[str] = []

    # Figure 1: mid-chord force error vs segment count, measured vs closed form.
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ratios = sorted({row["penetration_ratio"] for row in mid_chord_rows})
    colors = plt.cm.viridis([i / max(1, len(ratios) - 1) for i in range(len(ratios))])
    for color, ratio in zip(colors, ratios):
        rows = [
            row for row in mid_chord_rows
            if row["penetration_ratio"] == ratio and row["radius_ratio"] == 10.0
        ]
        rows.sort(key=lambda row: row["n_segments"])
        n = [row["n_segments"] for row in rows]
        measured = [abs(row["polyline_relative_error"]) for row in rows]
        predicted = [abs(row["predicted_relative_error"]) for row in rows]
        ax.loglog(n, measured, "o", color=color, label=f"measured, $\\delta/r_p$={ratio:g}")
        ax.loglog(n, predicted, "--", lw=1.0, color=color)
    ax.axhline(0.03, color="crimson", lw=1.2, ls=":")
    ax.text(4.2, 0.034, "3% tolerance", color="crimson", fontsize=9)
    ax.set_xlabel("chords per 180$^\\circ$ bow  $N$")
    ax.set_ylabel("mid-chord relative force error  $|F_N/F_{arc}-1|$")
    ax.set_title("Secant bow bias: measured polyline kernel vs closed form $-s/\\delta$\n"
                 "(markers = engine, dashed = prediction, $R/r_p$=10)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output / "fig1_force_error_vs_segments.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path.name)

    # Figure 2: washboard sweep across one segment.
    fig, (ax_force, ax_dir) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    xi = [row["xi"] for row in washboard_rows]
    ax_force.plot(xi, [100 * row["polyline_relative_error"] for row in washboard_rows],
                  "o-", ms=3, label="polyline (shipped)")
    ax_force.plot(xi, [100 * row["corrected_relative_error"] for row in washboard_rows],
                  "s-", ms=3, label="sagitta-corrected")
    ax_force.plot(xi, [100 * row["arc_relative_error"] for row in washboard_rows],
                  "^-", ms=3, label="exact arc kernel")
    ax_force.axhline(3.0, color="crimson", ls=":", lw=1.1)
    ax_force.axhline(-3.0, color="crimson", ls=":", lw=1.1)
    ax_force.set_xlabel("position within one segment  $\\xi$")
    ax_force.set_ylabel("relative force error  [%]")
    ax_force.set_title("Force magnitude through one segment")
    ax_force.grid(alpha=0.3)
    ax_force.legend(fontsize=8)
    ax_dir.plot(xi, [row["polyline_direction_error_deg"] for row in washboard_rows],
                "o-", ms=3, label="polyline (shipped)")
    ax_dir.plot(xi, [row["corrected_direction_error_deg"] for row in washboard_rows],
                "s-", ms=3, label="sagitta-corrected")
    ax_dir.plot(xi, [row["arc_direction_error_deg"] for row in washboard_rows],
                "^-", ms=3, label="exact arc kernel")
    ax_dir.set_xlabel("position within one segment  $\\xi$")
    ax_dir.set_ylabel("normal-direction error  [deg]")
    ax_dir.set_title("Contact normal through one segment")
    ax_dir.grid(alpha=0.3)
    ax_dir.legend(fontsize=8)
    fig.tight_layout()
    path = output / "fig2_washboard.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path.name)

    # Figure 3: segments required for a tolerance.
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    tolerances = sorted({row["tolerance"] for row in tolerance_table})
    ratios_r = sorted({row["radius_ratio"] for row in tolerance_table})
    for tolerance in tolerances:
        for index, ratio in enumerate(ratios_r):
            rows = [
                row for row in tolerance_table
                if row["tolerance"] == tolerance and row["radius_ratio"] == ratio
            ]
            rows.sort(key=lambda row: row["penetration_ratio"])
            style = ["-", "--", ":", "-."][index % 4]
            ax.loglog(
                [row["penetration_ratio"] for row in rows],
                [row["n_segments_required"] for row in rows],
                style, lw=1.4,
                label=f"tol={100 * tolerance:g}%, $R/r_p$={ratio:g}",
            )
    ax.set_xlabel("working penetration  $\\delta/r_p$")
    ax.set_ylabel("chords required per 180$^\\circ$ bow")
    ax.set_title("Bow discretisation requirement from $s = \\delta\\cdot$tol")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = output / "fig3_required_segments.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    names.append(path.name)
    return names


def write_tables_markdown(payload: dict[str, Any], output: Path) -> str:
    """Render the result tables as markdown so the docs never hand-copy numbers."""
    mid_chord = payload["mid_chord"]
    rows = [row for row in mid_chord if row["radius_ratio"] == 10.0]
    penetration_ratios = sorted({row["penetration_ratio"] for row in rows})
    n_values = sorted({row["n_segments"] for row in rows})
    lines = [
        "### Table 1 - mid-chord relative force error of the shipped polyline kernel",
        "",
        "Bow span 180 deg, $R/r_p$ = 10. `drop` marks a total loss of contact "
        "(penetration below the sagitta).",
        "",
        "| $\\delta/r_p$ | " + " | ".join(f"N={n}" for n in n_values) + " |",
        "|" + "---|" * (len(n_values) + 1),
    ]
    for ratio in penetration_ratios:
        cells = []
        for n_segments in n_values:
            row = next(
                item for item in rows
                if item["penetration_ratio"] == ratio and item["n_segments"] == n_segments
            )
            if row["contact_lost"]:
                cells.append("drop")
            else:
                cells.append(f"{100 * row['polyline_relative_error']:+.2f}%")
        lines.append(f"| {ratio:g} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Table 2 - dropout threshold versus closed-form sagitta",
        "",
        "| $R/r_p$ | N | measured dropout $\\delta$ (m) | sagitta $s$ (m) | relative difference |",
        "|---|---|---|---|---|",
    ]
    for row in payload["dropout"]:
        lines.append(
            "| {radius_ratio:g} | {n_segments} | {measured_dropout_penetration:.9e} | "
            "{predicted_sagitta:.9e} | {relative_difference:.2e} |".format(**row)
        )

    lines += [
        "",
        "### Table 3 - chords per 180 deg bow required for a 3% force error",
        "",
        "| $\\delta/r_p$ | " + " | ".join(
            f"$R/r_p$={ratio:g}" for ratio in sorted({row["radius_ratio"] for row in payload["tolerance_table"]})
        ) + " |",
        "|" + "---|" * (1 + len({row["radius_ratio"] for row in payload["tolerance_table"]})),
    ]
    ratios = sorted({row["radius_ratio"] for row in payload["tolerance_table"]})
    for penetration_ratio in sorted({row["penetration_ratio"] for row in payload["tolerance_table"]}):
        cells = []
        for ratio in ratios:
            row = next(
                item for item in payload["tolerance_table"]
                if item["radius_ratio"] == ratio
                and item["penetration_ratio"] == penetration_ratio
                and item["tolerance"] == 0.03
            )
            cells.append(str(row["n_segments_required"]))
        lines.append(f"| {penetration_ratio:g} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "### Table 4 - exact arc kernel versus the closed-form bow reference",
        "",
        "| quantity | measured | tolerance |",
        "|---|---|---|",
        f"| max relative force error | {payload['checks']['arc_kernel_max_relative_force_error']:.2e} | < 3% |",
        f"| max normal-direction error | {payload['checks']['arc_kernel_max_direction_error_deg']:.2e} deg | - |",
        "| max action-reaction residual | "
        f"{payload['checks']['arc_kernel_max_action_reaction_residual_N']:.2e} N | 0 |",
        "| polyline error vs closed form $-s/\\delta$ | "
        f"{payload['checks']['polyline_vs_closed_form_max_relative_error']:.2e} | < 1e-9 |",
        f"| dropout threshold vs sagitta | {payload['checks']['dropout_max_relative_difference']:.2e} | < 1e-5 |",
    ]
    text = "\n".join(lines) + "\n"
    (output / "report_tables.md").write_text(text, encoding="utf-8")
    return "report_tables.md"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/dem2d_hull_geometry")
    parser.add_argument("--particle-radius", type=float, default=1.0)
    parser.add_argument("--span-deg", type=float, default=180.0)
    parser.add_argument("--n-values", default="4,8,16,32,64,128,256,512")
    parser.add_argument("--radius-ratios", default="2,5,10,20")
    parser.add_argument("--penetration-ratios", default="0.002,0.005,0.01,0.02,0.05,0.1,0.2")
    parser.add_argument("--sweep-samples", type=int, default=81)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    particle_radius = args.particle_radius
    span = math.radians(args.span_deg)
    n_values = [int(item) for item in args.n_values.split(",")]
    radius_ratios = [float(item) for item in args.radius_ratios.split(",")]
    penetration_ratios = [float(item) for item in args.penetration_ratios.split(",")]

    mid_chord: list[dict[str, Any]] = []
    for radius_ratio in radius_ratios:
        mid_chord.extend(
            sweep_mid_chord(radius_ratio, particle_radius, span, n_values, penetration_ratios)
        )

    wash_n = 16
    wash_penetration_ratio = 0.05
    washboard = sweep_washboard(
        10.0, particle_radius, span, wash_n, wash_penetration_ratio, args.sweep_samples
    )

    dropout: list[dict[str, Any]] = []
    for radius_ratio in radius_ratios:
        for n_segments in n_values:
            measured = dropout_threshold(radius_ratio, particle_radius, span, n_segments)
            predicted = sagitta(radius_ratio * particle_radius, span, n_segments)
            dropout.append(
                {
                    "radius_ratio": radius_ratio,
                    "n_segments": n_segments,
                    "measured_dropout_penetration": measured,
                    "predicted_sagitta": predicted,
                    "relative_difference": relative_error(measured, predicted),
                }
            )

    tolerance_table: list[dict[str, Any]] = []
    for tolerance in (0.01, 0.03):
        for radius_ratio in radius_ratios:
            for penetration_ratio in penetration_ratios:
                tolerance_table.append(
                    {
                        "tolerance": tolerance,
                        "radius_ratio": radius_ratio,
                        "penetration_ratio": penetration_ratio,
                        "n_segments_required": segments_for_tolerance(
                            radius_ratio * particle_radius,
                            span,
                            penetration_ratio * particle_radius,
                            tolerance,
                        ),
                    }
                )

    def max_abs(rows: Sequence[dict[str, Any]], key: str, **filters: Any) -> float:
        values = [
            abs(row[key]) for row in rows
            if all(row[name] == value for name, value in filters.items())
        ]
        return max(values) if values else 0.0

    checks = {
        "arc_kernel_max_relative_force_error": max_abs(mid_chord, "arc_relative_error"),
        "arc_kernel_max_direction_error_deg": max_abs(mid_chord, "arc_direction_error_deg"),
        "arc_kernel_max_action_reaction_residual_N": max_abs(
            mid_chord, "action_reaction_residual_N"
        ),
        "polyline_vs_closed_form_max_relative_error": max(
            abs(row["polyline_relative_error"] - row["predicted_relative_error"])
            for row in mid_chord
        ),
        "dropout_max_relative_difference": max_abs(dropout, "relative_difference"),
        "washboard_max_polyline_relative_error": max_abs(washboard, "polyline_relative_error"),
        "washboard_max_corrected_relative_error": max_abs(
            washboard, "corrected_relative_error"
        ),
        "washboard_max_polyline_direction_error_deg": max(
            row["polyline_direction_error_deg"] for row in washboard
        ),
        "washboard_direction_bound_deg": 0.5 * math.degrees(span / wash_n),
    }

    # Where does the shipped polyline first satisfy the 3% tolerance?
    def first_pass(penetration_ratio: float) -> dict[str, Any]:
        rows = [
            row for row in mid_chord
            if row["penetration_ratio"] == penetration_ratio and row["radius_ratio"] == 10.0
        ]
        rows.sort(key=lambda row: row["n_segments"])
        for row in rows:
            if abs(row["polyline_relative_error"]) <= 0.03:
                return {"penetration_ratio": penetration_ratio,
                        "n_segments_for_3pct": row["n_segments"]}
        return {"penetration_ratio": penetration_ratio, "n_segments_for_3pct": None}

    verdict = {
        "tolerance": 0.03,
        "shipped_polyline_pass_by_penetration": [first_pass(r) for r in penetration_ratios],
        "shipped_polyline_max_error_at_n32_contacting": max(
            (abs(row["polyline_relative_error"]) for row in mid_chord
             if row["n_segments"] == 32 and row["radius_ratio"] == 10.0
             and row["predicted_force_ratio"] > 0.0),
            default=0.0,
        ),
        "shipped_polyline_dropout_cases_at_n32": sum(
            1 for row in mid_chord
            if row["n_segments"] == 32 and row["radius_ratio"] == 10.0
            and row["predicted_force_ratio"] == 0.0
        ),
        "corrected_polyline_max_error_at_n32": max_abs(
            mid_chord, "corrected_relative_error", n_segments=32, radius_ratio=10.0
        ),
        "arc_kernel_pass": checks["arc_kernel_max_relative_force_error"] < 0.03,
    }

    payload = {
        "campaign": "dem2d_hull_polyline_geometry_audit",
        "schema": "tensordem.hull_geometry.audit.v1",
        "settings": {
            "particle_radius_m": particle_radius,
            "bow_span_deg": args.span_deg,
            "n_values": n_values,
            "radius_ratios": radius_ratios,
            "penetration_ratios": penetration_ratios,
            "stiffness_N_per_m": 1.0,
            "washboard_n_segments": wash_n,
            "washboard_penetration_ratio": wash_penetration_ratio,
            "washboard_samples": args.sweep_samples,
        },
        "mid_chord": mid_chord,
        "washboard": washboard,
        "dropout": dropout,
        "tolerance_table": tolerance_table,
        "checks": checks,
        "verdict": verdict,
        "notes": [
            "All forces use stiffness 1 N/m, so force and penetration differ by a constant factor.",
            "The closed-form reference delta = r_p + R - r is evaluated in this script without "
            "calling either contact kernel.",
            "Geometry only: no ice sheet, fracture, buoyancy or integration is involved.",
        ],
    }

    with (output / "audit.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)

    with (output / "mid_chord_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mid_chord[0].keys()))
        writer.writeheader()
        writer.writerows(mid_chord)

    with (output / "washboard_sweep.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(washboard[0].keys()))
        writer.writeheader()
        writer.writerows(washboard)

    figures = make_figures(mid_chord, washboard, penetration_ratios, tolerance_table, output)
    figures.append(write_tables_markdown(payload, output))

    print(json.dumps({"checks": checks, "verdict": verdict, "figures": figures}, indent=2))


if __name__ == "__main__":
    main()