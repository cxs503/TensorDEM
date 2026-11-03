"""Audit the sagitta (facet-plane) bias of the 3-D hull-mesh contact kernel.

Why
---
``tensordem.hull_mesh_contact`` was merged with acceptance evidence measured on a
*flat* plate (``docs/m669_hull_mesh_contact.md``: "sphere_contact_force_error =
0.0"), where a triangle mesh is exact.  A real hull is curved, and any inscribed
triangle mesh replaces the true surface by facet planes that lie *inside* it.
For a section of radius ``R`` discretised into ``facets`` equal chords the bias
is the classical sagitta ``s = R (1 - cos(pi/facets))``:

    delta_facet = delta_true - s     =>     rel. force error = -s / delta_true,

so the penalty load is *systematically* underestimated and the contact is lost
altogether once ``delta_true <= s``.  This script quantifies that bias against an
independent closed-form reference, verifies the ``hull_radius = s`` correction,
and cross-checks the 3-D mesh kernel against the equivalent 2-D polyline kernel.

Nothing here integrates dynamics: it is pure geometry + the contact law, so the
reference values are exact (no time-stepping error to mask the bias).

Usage
-----
    python scripts/audit_hull_mesh_curvature_sagitta.py --output docs/hull-mesh-curvature
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensordem.hull_contact import polyline_contact_forces  # noqa: E402
from tensordem.hull_mesh_contact import (  # noqa: E402
    analytic_cylinder_penetration,
    facet_sagitta,
    make_cylinder_mesh,
    mesh_contact_forces,
    required_facets,
)

RATIOS = (2.0, 5.0, 10.0, 20.0)
FACETS = (4, 8, 16, 32, 64, 128, 256, 512)
DELTAS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
PSI_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)  # fraction of the facet span: 0 = vertex, 0.5 = mid-chord
R_PARTICLE = 1.0
STIFFNESS = 1.0  # N/m => force magnitude == penetration, in metres


def _point_segment_distance(px: float, py: float,
                            ax: float, ay: float,
                            bx: float, by: float) -> float:
    """Euclidean distance from a point to a closed segment (scalar algebra)."""
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0.0 else ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def closed_form_penetration(radius: float, facets: int, delta: float,
                            psi_fraction: float) -> float:
    """Exact penetration into the *inscribed polygon* section (analytic reference).

    The sphere centre sits at radial distance ``R + r_p - delta`` and at the angle
    ``2 pi psi_fraction / facets`` (``psi_fraction`` runs across one facet:
    0 = vertex, 0.5 = mid-chord).  The distance is the exact minimum over the
    ``facets`` closed chords of the section, computed with scalar algebra only,
    so the reference never calls the contact kernel.  For the mid-chord point the
    nearest feature is the chord interior and the result reduces to
    ``delta - s``; near a vertex the nearest feature is that vertex and the bias
    vanishes, which is the sawtooth visible in Figure 2.
    """
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be finite and positive")
    if isinstance(facets, bool) or int(facets) != facets or int(facets) < 3:
        raise ValueError("facets must be an integer >= 3")
    if not math.isfinite(delta):
        raise ValueError("delta must be finite")
    r = radius + R_PARTICLE - delta
    angle = psi_fraction * 2.0 * math.pi / facets
    px, py = r * math.cos(angle), r * math.sin(angle)
    best = math.inf
    for k in range(facets):
        a0 = 2.0 * math.pi * k / facets
        a1 = 2.0 * math.pi * (k + 1) / facets
        best = min(best, _point_segment_distance(
            px, py,
            radius * math.cos(a0), radius * math.sin(a0),
            radius * math.cos(a1), radius * math.sin(a1)))
    return max(0.0, R_PARTICLE - best)


def closed_form_angular(radius: float, delta: float, facets: int) -> float:
    """Facet half-angle in radians for the given facet count."""
    del delta
    return math.pi / facets


def measure_3d(radius: float, facets: int, delta: float, psi_fraction: float,
               *, offset: float = 0.0) -> dict:
    """Force magnitude from the shipped 3-D mesh kernel at one sphere position."""
    vertices, faces = make_cylinder_mesh(radius, length=4.0 * R_PARTICLE, facets=facets)
    psi = psi_fraction * 2.0 * math.pi / facets
    r = radius + R_PARTICLE - delta
    position = torch.tensor([[r * math.cos(psi), r * math.sin(psi), 0.0]], dtype=torch.float64)
    forces, _ = mesh_contact_forces(
        position,
        torch.zeros_like(position),
        vertices,
        faces,
        torch.zeros(3, dtype=torch.float64),
        particle_radius=R_PARTICLE,
        hull_radius=offset,
        stiffness=STIFFNESS,
        damping=0.0,
    )
    return {
        "force": float(torch.linalg.vector_norm(forces[0])),
        "normal": [float(x) for x in forces[0]],
        "position": [float(x) for x in position[0]],
    }


def measure_2d(radius: float, facets: int, delta: float, psi_fraction: float,
               *, offset: float = 0.0) -> float:
    """Force magnitude from the 2-D polyline kernel on the same section."""
    angles = 2.0 * math.pi * torch.arange(facets, dtype=torch.float64) / float(facets)
    ring = torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=-1)
    # Close the polyline so both neighbours exist at every sample point.
    vertices = torch.cat([ring, ring[:1]], dim=0)
    psi = psi_fraction * 2.0 * math.pi / facets
    r = radius + R_PARTICLE - delta
    position = torch.tensor([[r * math.cos(psi), r * math.sin(psi)]], dtype=torch.float64)
    forces, _ = polyline_contact_forces(
        position,
        torch.zeros_like(position),
        vertices,
        torch.zeros(2, dtype=torch.float64),
        particle_radius=R_PARTICLE,
        hull_radius=offset,
        stiffness=STIFFNESS,
        damping=0.0,
    )
    return float(torch.linalg.vector_norm(forces[0]))


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sweep_rows: list[dict] = []
    angular_rows: list[dict] = []
    checks: list[dict] = []

    # --- 1. closed form vs 3-D kernel over the whole parameter sweep ---------
    max_closed_form_error = 0.0
    max_2d_3d_difference = 0.0
    dropout_count = 0
    for ratio in RATIOS:
        radius = ratio * R_PARTICLE
        for facets in FACETS:
            s = facet_sagitta(radius, facets)
            for delta in DELTAS:
                reference = delta  # stiffness = 1 N/m
                errors = []
                for psi_fraction in PSI_FRACTIONS:
                    measured = measure_3d(radius, facets, delta, psi_fraction)["force"]
                    predicted = closed_form_penetration(radius, facets, delta, psi_fraction)
                    errors.append(measured / reference - 1.0)
                    if measured == 0.0 and reference > 0.0:
                        dropout_count += 1
                    max_closed_form_error = max(
                        max_closed_form_error, abs(measured - predicted))
                    two_d = measure_2d(radius, facets, delta, psi_fraction)
                    max_2d_3d_difference = max(max_2d_3d_difference, abs(two_d - measured))
                    angular_rows.append({
                        "R_over_rp": ratio,
                        "facets": facets,
                        "delta_over_rp": delta,
                        "psi_over_half_angle": psi_fraction,
                        "sagitta": s,
                        "delta_true": delta,
                        "delta_3d": measured,
                        "delta_2d": two_d,
                        "delta_closed_form": predicted,
                        "rel_error_3d": measured / reference - 1.0,
                        "rel_error_2d": two_d / reference - 1.0,
                        "dropout": measured == 0.0,
                    })
                mid = errors[2]  # f = 0.5 is the mid-chord
                sweep_rows.append({
                    "R_over_rp": ratio,
                    "facets": facets,
                    "delta_over_rp": delta,
                    "sagitta_over_delta": s / delta,
                    "mid_chord_rel_error": mid,
                    "min_rel_error_over_facet": min(errors),
                    "max_rel_error_over_facet": max(errors),
                    "closed_form_prediction": -s / delta,
                    "dropout_at_mid_chord": errors[2] <= -1.0,
                })

    # --- 2. sagitta offset correction (hull_radius = s) ---------------------
    offset_rows: list[dict] = []
    for ratio in RATIOS:
        radius = ratio * R_PARTICLE
        for facets in FACETS:
            s = facet_sagitta(radius, facets)
            for delta in DELTAS:
                if delta <= s:
                    continue  # no contact at mid-chord without the offset
                uncorrected_profile = [
                    measure_3d(radius, facets, delta, f)["force"] / delta - 1.0
                    for f in PSI_FRACTIONS]
                corrected_errors = [
                    measure_3d(radius, facets, delta, f, offset=s)["force"] / delta - 1.0
                    for f in PSI_FRACTIONS]
                uncorrected = uncorrected_profile[2]
                offset_rows.append({
                    "R_over_rp": ratio,
                    "facets": facets,
                    "delta_over_rp": delta,
                    "sagitta_over_delta": s / delta,
                    "mid_chord_error_uncorrected": uncorrected,
                    "mid_chord_error_offset": corrected_errors[2],
                    "vertex_error_offset": corrected_errors[0],
                    "uncorrected_profile": uncorrected_profile,
                    "corrected_profile": corrected_errors,
                    "max_abs_error_over_facet_offset": max(abs(e) for e in corrected_errors),
                    "mean_error_over_facet_offset": sum(corrected_errors) / len(corrected_errors),
                })

    # --- 3. dropout threshold vs sagitta ------------------------------------
    dropout_rows: list[dict] = []
    for ratio in RATIOS:
        radius = ratio * R_PARTICLE
        for facets in (8, 16, 32, 64):
            s = facet_sagitta(radius, facets)
            # bisect the mid-chord contact/no-contact boundary of the kernel
            lo, hi = 1e-9, 2.0 * s
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if measure_3d(radius, facets, mid, 0.5)["force"] > 0.0:
                    hi = mid
                else:
                    lo = mid
            dropout_rows.append({
                "R_over_rp": ratio,
                "facets": facets,
                "measured_dropout_delta": hi,
                "sagitta": s,
                "relative_difference": abs(hi - s) / s,
            })

    # --- 4. required-facet criterion: closed form vs measured ---------------
    criterion_rows: list[dict] = []
    for ratio in RATIOS:
        radius = ratio * R_PARTICLE
        for delta in DELTAS:
            needed = required_facets(radius, delta, 0.03)
            patient = next((f for f in range(3, 20000)
                            if facet_sagitta(radius, f) / delta <= 0.03), None)
            at_needed = measure_3d(radius, needed, delta, 0.5)["force"] / delta - 1.0
            below = needed - 1
            below_error = (None if below < 3 else
                           measure_3d(radius, below, delta, 0.5)["force"] / delta - 1.0)
            criterion_rows.append({
                "R_over_rp": ratio,
                "delta_over_rp": delta,
                "required_facets": needed,
                "brute_force_facets": patient,
                "rel_error_at_required": at_needed,
                "rel_error_below_required": below_error,
                "satisfies_3pct": abs(at_needed) <= 0.03,
            })

    # --- checks -------------------------------------------------------------
    checks.append({
        "name": "kernel_geometry_matches_closed_form_penetration",
        "max_abs_deviation_m": max_closed_form_error,
        "tolerance_m": 1e-9,
        "pass": max_closed_form_error < 1e-9,
    })
    checks.append({
        "name": "3d_mesh_kernel_equals_2d_polyline_kernel",
        "max_abs_difference_N": max_2d_3d_difference,
        "tolerance_N": 1e-9,
        "pass": max_2d_3d_difference < 1e-9,
    })
    max_dropout_dev = max(r["relative_difference"] for r in dropout_rows)
    checks.append({
        "name": "dropout_threshold_equals_sagitta",
        "max_relative_difference": max_dropout_dev,
        "tolerance": 1e-4,
        "pass": max_dropout_dev < 1e-4,
    })
    worst_mid = min(r["mid_chord_rel_error"] for r in sweep_rows)
    checks.append({
        "name": "worst_mid_chord_underestimate_over_sweep",
        "value": worst_mid,
        "note": "most negative relative force error found (systematic underload)",
        "pass": worst_mid <= -0.05,
    })
    offset_pass = all(abs(r["mid_chord_error_offset"]) < 1e-9 for r in offset_rows)
    checks.append({
        "name": "hull_radius_s_offset_cancels_mid_chord_bias",
        "max_abs_mid_chord_error_after_offset": max(
            (abs(r["mid_chord_error_offset"]) for r in offset_rows), default=0.0),
        "tolerance": 1e-9,
        "pass": offset_pass,
    })
    criterion_pass = all(r["satisfies_3pct"] for r in criterion_rows)
    checks.append({
        "name": "required_facets_criterion_delivers_3pct",
        "cases": len(criterion_rows),
        "max_abs_error_at_required": max(abs(r["rel_error_at_required"]) for r in criterion_rows),
        "pass": criterion_pass,
    })
    # The identity delta_facet = delta_true - s holds exactly wherever the facet
    # is still in contact; below the sagitta the clamp at zero takes over and the
    # quantity to check is the dropout threshold (previous check).
    analytic_deviation = max(
        abs(analytic_cylinder_penetration(ratio * R_PARTICLE + R_PARTICLE - delta,
                                         R_PARTICLE, ratio * R_PARTICLE)
            - closed_form_penetration(ratio * R_PARTICLE, facets, delta, 0.5)
            - facet_sagitta(ratio * R_PARTICLE, facets))
        for ratio in RATIOS for facets in FACETS for delta in DELTAS
        if delta > facet_sagitta(ratio * R_PARTICLE, facets))
    checks.append({
        "name": "analytic_reference_consistency",
        "note": "delta_facet = delta_true - s (cases with delta > s)",
        "max_abs_deviation_m": analytic_deviation,
        "tolerance_m": 1e-12,
    })
    checks[-1]["pass"] = checks[-1]["max_abs_deviation_m"] < 1e-12

    summary = {
        "protocol": "tensordem-dem3d-hull-mesh-curvature-v1",
        "parameters": {
            "R_over_particle_radius": list(RATIOS),
            "facets": list(FACETS),
            "delta_over_particle_radius": list(DELTAS),
            "psi_over_facet_half_angle": list(PSI_FRACTIONS),
            "particle_radius_m": R_PARTICLE,
            "stiffness_N_per_m": STIFFNESS,
        },
        "records": {
            "sweep": len(sweep_rows),
            "angular": len(angular_rows),
            "offset": len(offset_rows),
            "dropout": len(dropout_rows),
            "criterion": len(criterion_rows),
        },
        "dropout_samples_in_sweep": dropout_count,
        "checks": checks,
    }
    summary["all_checks_pass"] = all(c["pass"] for c in checks)

    _write_csv(output / "curvature_sweep.csv", sweep_rows)
    _write_csv(output / "angular_profiles.csv", angular_rows)
    _write_csv(output / "sagitta_offset_correction.csv", offset_rows)
    _write_csv(output / "dropout_threshold.csv", dropout_rows)
    _write_csv(output / "required_facets.csv", criterion_rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    figures = _figures(output, sweep_rows, offset_rows, criterion_rows)
    (output / "tables.md").write_text(_tables_md(sweep_rows, offset_rows,
                                                dropout_rows, criterion_rows,
                                                summary, figures))
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _figures(output: Path, sweep_rows, offset_rows, criterion_rows) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures: dict[str, str] = {}

    # Figure 1: mid-chord error vs facet count with the closed form overlay.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ratio in RATIOS:
        subset = [r for r in sweep_rows if r["R_over_rp"] == ratio and r["delta_over_rp"] == 0.05]
        subset.sort(key=lambda r: r["facets"])
        facets = [r["facets"] for r in subset]
        ax.loglog(facets, [-r["mid_chord_rel_error"] for r in subset], "o-",
                  label=f"measured R/rp={ratio:g}")
        ax.loglog(facets, [-r["closed_form_prediction"] for r in subset], "k--", lw=0.8)
    ax.axhline(0.03, color="crimson", ls=":", label="3% tolerance")
    ax.set_xlabel("facets per 2$\u03c0$")
    ax.set_ylabel("|relative force error| at mid-chord")
    ax.set_title("Inscribed-mesh sagitta bias, $\\delta/r_p=0.05$\n(dashed = closed form $s/\\delta$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    path = output / "fig1_error_vs_facets.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures["fig1"] = path.name

    # Figure 2: error across one facet, uncorrected vs hull_radius = sagitta.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    picks = [(10.0, 32, 0.05), (10.0, 64, 0.02)]
    for ax, (ratio, facets, delta) in zip(axes, picks):
        subset = [r for r in offset_rows
                  if r["R_over_rp"] == ratio and r["facets"] == facets
                  and r["delta_over_rp"] == delta]
        if not subset:
            continue
        row = subset[0]
        psi = list(PSI_FRACTIONS)
        uncorrected = row["uncorrected_profile"]
        corrected = row["corrected_profile"]
        ax.plot(psi, [100 * u for u in uncorrected], "o-", label="shipped mesh")
        ax.plot(psi, [100 * c for c in corrected], "s-", label="hull_radius = s")
        ax.axhline(0.0, color="k", lw=0.6)
        ax.set_xlabel("position across facet / half-angle")
        ax.set_ylabel("relative force error [%]")
        ax.set_title(f"R/rp={ratio:g}, facets={facets}, $\\delta/r_p$={delta:g}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Facet bias across one facet: constant offset converts a one-sided underload into a symmetric error")
    path = output / "fig2_angular_profile.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures["fig2"] = path.name

    # Figure 3: required facets vs overlap ratio for a 3% force error.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ratio in RATIOS:
        deltas = [r["delta_over_rp"] for r in criterion_rows if r["R_over_rp"] == ratio]
        needed = [r["required_facets"] for r in criterion_rows if r["R_over_rp"] == ratio]
        order = sorted(range(len(deltas)), key=lambda i: deltas[i])
        ax.loglog([deltas[i] for i in order], [needed[i] for i in order], "o-",
                  label=f"R/rp={ratio:g}")
    ax.axhline(16, color="crimson", ls=":", label="typical STL (16/circle)")
    ax.set_xlabel("$\\delta/r_p$")
    ax.set_ylabel("facets required for 3% force error")
    ax.set_title("Mesh resolution implied by the contact overlap")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    path = output / "fig3_required_facets.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures["fig3"] = path.name

    return figures


def _fmt(value: float, digits: int = 3) -> str:
    if value is None:
        return "-"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-4 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _tables_md(sweep_rows, offset_rows, dropout_rows, criterion_rows, summary, figures) -> str:
    lines: list[str] = ["# Generated tables and figures", ""]
    lines += ["All numbers below are written by `scripts/audit_hull_mesh_curvature_sagitta.py`; "
              "nothing is transcribed by hand.", ""]

    lines += ["## Table 1 - mid-chord relative force error (fraction of the true penalty load)",
              "", "Negative = systematic underload. `drop` = contact lost entirely.", ""]
    header = "| delta/rp | " + " | ".join(f"N={n}" for n in FACETS) + " |"
    lines += [header, "|" + "---|" * (len(FACETS) + 1)]
    for delta in DELTAS:
        for ratio in RATIOS:
            cells = []
            for facets in FACETS:
                row = next(r for r in sweep_rows
                           if r["R_over_rp"] == ratio and r["facets"] == facets
                           and r["delta_over_rp"] == delta)
                if row["dropout_at_mid_chord"]:
                    cells.append("drop")
                else:
                    cells.append(f"{100 * row['mid_chord_rel_error']:+.2f}%")
            lines.append(f"| {delta:g} (R/rp={ratio:g}) | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Table 2 - dropout threshold vs closed-form sagitta s = R (1 - cos(pi/N))", "",
              "| R/rp | facets | measured dropout delta [m] | sagitta s [m] | relative difference |",
              "|---|---|---|---|---|"]
    for row in dropout_rows:
        lines.append(
            f"| {row['R_over_rp']:g} | {row['facets']} | "
            f"{row['measured_dropout_delta']:.9e} | {row['sagitta']:.9e} | "
            f"{row['relative_difference']:.2e} |")
    lines.append("")

    lines += ["## Table 3 - facets required for a 3% force error", "",
              "| R/rp | delta/rp | closed-form facets | brute-force facets | error at N | error at N-1 | 3% met |",
              "|---|---|---|---|---|---|---|"]
    for row in criterion_rows:
        below = row["rel_error_below_required"]
        below_txt = "-" if below is None else f"{100 * below:+.2f}%"
        lines.append(
            f"| {row['R_over_rp']:g} | {row['delta_over_rp']:g} | {row['required_facets']} | "
            f"{row['brute_force_facets']} | {100 * row['rel_error_at_required']:+.3f}% | "
            f"{below_txt} | {'yes' if row['satisfies_3pct'] else 'NO'} |")
    lines.append("")

    lines += ["## Table 4 - hull_radius = s correction across one facet", "",
              "Selected cases; `error over facet` is the RMS-free max/min over five positions.",
              "",
              "| R/rp | facets | delta/rp | mid-chord before | mid-chord after | vertex after | max abs after | mean after |",
              "|---|---|---|---|---|---|---|---|"]
    picks = [(10.0, 32, 0.05), (10.0, 64, 0.02), (10.0, 128, 0.01),
             (5.0, 32, 0.02), (20.0, 128, 0.02), (2.0, 16, 0.05)]
    for ratio, facets, delta in picks:
        row = next((r for r in offset_rows
                    if r["R_over_rp"] == ratio and r["facets"] == facets
                    and r["delta_over_rp"] == delta), None)
        if row is None:
            continue
        lines.append(
            f"| {ratio:g} | {facets} | {delta:g} | "
            f"{100 * row['mid_chord_error_uncorrected']:+.2f}% | "
            f"{100 * row['mid_chord_error_offset']:+.3f}% | "
            f"{100 * row['vertex_error_offset']:+.2f}% | "
            f"{100 * row['max_abs_error_over_facet_offset']:.2f}% | "
            f"{100 * row['mean_error_over_facet_offset']:+.2f}% |")
    lines.append("")

    lines += ["## Checks", "", "| check | value | pass |", "|---|---|---|"]
    for check in summary["checks"]:
        key = [k for k in check if k not in ("name", "pass")]
        value = ", ".join(f"{k}={_fmt(check[k]) if isinstance(check[k], (int, float)) else check[k]}"
                          for k in key)
        lines.append(f"| {check['name']} | {value} | {'PASS' if check['pass'] else 'FAIL'} |")
    lines.append("")

    lines += ["## Figures", ""]
    captions = {
        "fig1": ("fig1_error_vs_facets.png",
                 "Mid-chord relative force error vs facet count for delta/rp = 0.05; "
                 "markers = engine, dashed = closed form s/delta."),
        "fig2": ("fig2_angular_profile.png",
                 "Error across one facet before/after the constant sagitta offset: "
                 "a one-sided underload becomes a symmetric ripple."),
        "fig3": ("fig3_required_facets.png",
                 "Facets required for a 3% force error versus the design overlap."),
    }
    for key, (name, caption) in captions.items():
        if key in figures:
            lines.append(f"- **{name}** - {caption}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/hull-mesh-curvature",
                        help="directory for CSV/JSON/PNG/tables.md outputs")
    args = parser.parse_args()
    summary = run(Path(args.output))
    print(json.dumps({"all_checks_pass": summary["all_checks_pass"],
                      "checks": summary["checks"]}, indent=2, default=float))
    return 0 if summary["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())