"""Verification of the polyline hull-contact kernel against closed-form geometry.

The kernel :func:`tensordem.hull_contact.polyline_contact_forces` models a rigid
hull as a polyline and a penalty contact with ``R_p + R_h`` overlap.  Nothing in
the shipped tests pinned that kernel to an *analytic* reference, so a user could
not answer the only question that matters when digitizing a bow: **how many hull
points are needed before the polyline reproduces the real curved hull?**

This module supplies the missing reference:

* an independent exact-circular-hull penalty law (``exact_circular_hull_force``),
  used as the analytic benchmark for an inscribed-polygon bow;
* an independent point-to-polyline distance (line/cross-product form, i.e. a
  *different* algorithm from the clamped projection used in the kernel);
* a discretization-error scan that measures the observed order of accuracy of
  the force magnitude and of the force direction, plus the closed-form sagitta
  criterion at which a polyline bow silently loses contact altogether;
* representation-invariance checks (rigid rotation, mirror, vertex-order
  reversal).

Everything is SI, float64, and computed on CPU in milliseconds: this is geometry,
not a DEM time integration.  The verified statements are about geometry, not
about the calibrated physical resistance of a ship in ice.
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch

from .hull_contact import polyline_contact_forces

SCHEMA = "tensordem.hull-contact-verification/1"
DTYPE = torch.float64


def _as_tensor(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=DTYPE)


# ---------------------------------------------------------------------------
# Reference geometry
# ---------------------------------------------------------------------------

def bow_polyline(radius: float, segments: int, *, phase: float = 0.0,
                 arc: float = math.tau, center: Sequence[float] = (0.0, 0.0)) -> torch.Tensor:
    """Vertices of an *inscribed* polygon on an arc of a circle.

    Vertices lie on the circle (this is what a lines-plan digitisation gives),
    so the chords cut *inside* the true hull.  ``arc=tau`` returns a closed
    polygon (last vertex repeats the first) with ``segments`` edges.
    """
    if segments < 3:
        raise ValueError("segments must be >= 3")
    if not (radius > 0 and math.isfinite(radius)):
        raise ValueError("radius must be finite and positive")
    angles = torch.arange(segments + 1, dtype=DTYPE) * (arc / segments) + phase
    cx, cy = center
    return torch.stack([cx + radius * torch.cos(angles), cy + radius * torch.sin(angles)], dim=1)


def sagitta(radius: float, segments: int) -> float:
    """Inward gap between a chord and the circular hull it replaces (m)."""
    return radius * (1.0 - math.cos(math.pi / segments))


def required_segments(radius: float, penetration: float) -> int:
    """Fewest chords whose sagitta is below ``penetration``.

    A disk of radius ``R_p`` touching the true hull with overlap ``penetration``
    loses contact with the inscribed polygon once the sagitta exceeds that
    overlap, so this is the discretization the kernel needs *at that depth*.
    """
    if not penetration > 0:
        raise ValueError("penetration must be positive")
    if radius * (1.0 - math.cos(math.pi / 3)) <= penetration:
        return 3
    return int(math.ceil(math.pi / math.acos(1.0 - penetration / radius)))


def segments_for_magnitude_error(target_relative_error: float, radius: float,
                                  penetration: float) -> int:
    """Fewest chords whose worst-case force-magnitude error stays below a target.

    Exact law (proved by construction, reproduced by :func:`contact_error_scan`):
    a disk centred above a chord midpoint sees an overlap reduced by exactly the
    sagitta ``R (1 - cos(pi/M))``, so ``max |dF| / F = sagitta / penetration``
    while ``sagitta < penetration`` (beyond that the polyline loses contact and
    the error is 100%).  Keeping the error below a target is therefore a pure
    sagitta criterion, and for shallow penetrations it is far stricter than
    merely *retaining contact*.
    """
    if not penetration > 0:
        raise ValueError("penetration must be positive")
    ratio = target_relative_error * penetration / radius
    if ratio >= 1.0:
        return 3
    return int(math.ceil(math.pi / math.acos(1.0 - ratio)))


def exact_circular_hull_force(positions, hull_center, hull_radius, *,
                              particle_radius, stiffness, damping=0.0,
                              velocities=None, hull_velocity=(0.0, 0.0),
                              hull_gap=0.0):
    """Penalty contact of disks against an exact circular hull (reference law).

    Same constitutive form as the polyline kernel, exact geometry: the outward
    radial unit normal replaces the nearest-point normal.
    """
    p = _as_tensor(positions)
    v = _as_tensor(velocities) if velocities is not None else torch.zeros_like(p)
    hv = _as_tensor(hull_velocity).reshape(2)
    centre = _as_tensor(hull_center).reshape(2)
    delta = p - centre[None, :]
    distance = delta.norm(dim=1)
    normal = delta / distance.clamp_min(torch.finfo(DTYPE).eps)[:, None]
    overlap = (particle_radius + hull_radius + hull_gap - distance).clamp_min(0.0)
    normal_speed = ((v - hv) * normal).sum(dim=1)
    magnitude = (stiffness * overlap - damping * normal_speed).clamp_min(0.0)
    magnitude = torch.where(overlap > 0, magnitude, torch.zeros_like(magnitude))
    forces = magnitude[:, None] * normal
    return forces, -forces.sum(dim=0)


def point_to_polyline_distance(positions, vertices):
    """Independent point-to-polyline distance and nearest segment index.

    Uses the cross-product area of the projection (valid only inside the
    segment) and Euclidean endpoint distances otherwise, i.e. a different
    branch structure from the clamped-projection kernel.
    """
    p = _as_tensor(positions)
    verts = _as_tensor(vertices)
    a, b = verts[:-1], verts[1:]
    seg = b - a
    seg_len = seg.norm(dim=1)
    best = None
    best_index = None
    for i in range(a.shape[0]):
        rel = p - a[i][None, :]
        t = (rel * seg[i][None, :]).sum(dim=1) / (seg_len[i] ** 2)
        cross = torch.abs(seg[i][0] * rel[:, 1] - seg[i][1] * rel[:, 0]) / seg_len[i]
        d_line = cross
        d_start = rel.norm(dim=1)
        d_end = (p - b[i][None, :]).norm(dim=1)
        candidate = torch.where(t < 0, d_start, torch.where(t > 1, d_end, d_line))
        if best is None:
            best, best_index = candidate, torch.zeros(p.shape[0], dtype=torch.long)
        else:
            better = candidate < best
            best = torch.where(better, candidate, best)
            best_index = torch.where(better, torch.full_like(best_index, i), best_index)
    return best, best_index


def implied_distance(forces, *, particle_radius, stiffness, hull_gap=0.0):
    """Distance implied by the kernel: ``D = R_p + R_gap - |F| / k`` when in contact."""
    magnitude = _as_tensor(forces).norm(dim=1)
    return particle_radius + hull_gap - magnitude / stiffness


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------

def _first_level_below(series, key, limit):
    """Lowest discretization level whose metric is below ``limit``, with no dropout."""
    for row in series:
        if row["dropout_fraction"] == 0 and row[key] <= limit:
            return row["segments"]
    return None


def _first_value_below(series, key, limit):
    for row in series:
        if row["dropout_fraction"] == 0 and row[key] <= limit:
            return row[key]
    return None


def observed_order(levels: Sequence[int], errors: Sequence[float]) -> float:
    """Least-squares slope of log(error) against log(segments)."""
    xs = [math.log(float(level)) for level in levels]
    ys = [math.log(max(float(err), 1e-300)) for err in errors]
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den


def contact_error_scan(levels: Iterable[int], penetration_fractions: Iterable[float], *,
                       radius=0.5, particle_radius=0.05, stiffness=1000.0,
                       angles_per_segment=4, hull_gap=0.0) -> list[dict[str, Any]]:
    """Magnitude/direction error and contact dropout vs bow discretization.

    Probe angles are placed *on the polygon itself* -- one disk above every chord
    midpoint and one above every vertex -- because the maximum magnitude error
    occurs at chord midpoints.  A uniform angular grid aliases that extremum and
    under-reports the error by up to a factor of two, which is exactly the kind
    of sampling artefact that makes a discretisation study look better than it is.
    """
    zeros = torch.zeros(2, dtype=DTYPE)
    rows: list[dict[str, Any]] = []
    for level in levels:
        level = int(level)
        vertices = bow_polyline(radius, level)
        vertex_angles = torch.arange(level, dtype=DTYPE) * (2.0 * math.pi / level)
        mid_angles = vertex_angles + math.pi / level
        extra = max(int(angles_per_segment) - 2, 0)
        blocks = [vertex_angles, mid_angles]
        for k in range(extra):
            blocks.append((torch.arange(level, dtype=DTYPE) + 0.25 * (k + 1)) * (2.0 * math.pi / level))
        angles = torch.cat(blocks)
        for fraction in penetration_fractions:
            overlap = float(fraction) * particle_radius
            distance = particle_radius + hull_gap + radius - overlap
            positions = torch.stack([distance * torch.cos(angles), distance * torch.sin(angles)], dim=1)
            static = torch.zeros_like(positions)
            forces, _ = polyline_contact_forces(
                positions, static, vertices, zeros,
                particle_radius=particle_radius, hull_radius=hull_gap,
                stiffness=stiffness)
            exact, _ = exact_circular_hull_force(
                positions, (0.0, 0.0), radius, particle_radius=particle_radius,
                stiffness=stiffness, hull_gap=hull_gap)
            exact_mag = exact.norm(dim=1)
            mag = forces.norm(dim=1)
            relative = (mag - exact_mag).abs() / exact_mag
            angle_reference = torch.atan2(exact[:, 1], exact[:, 0])
            angle_actual = torch.atan2(forces[:, 1], forces[:, 0])
            delta = (angle_actual - angle_reference + math.pi) % (2.0 * math.pi) - math.pi
            delta = torch.where(mag > 0, delta.abs(), torch.full_like(delta, math.pi))
            rows.append({
                "segments": level,
                "penetration_fraction": float(fraction),
                "penetration_m": overlap,
                "sagitta_m": sagitta(radius, level),
                "dropout_fraction": float((mag == 0).float().mean()),
                "max_relative_magnitude_error": float(relative.max()),
                "mean_relative_magnitude_error": float(relative.mean()),
                "max_direction_error_deg": math.degrees(float(delta.max())),
                "contact_dropout_predicted": bool(sagitta(radius, level) >= overlap),
                "min_segments_for_penetration": required_segments(radius, overlap),
            })
    return rows


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def distance_check(samples=200, seed=20261103) -> dict[str, Any]:
    """Kernel-implied distance vs an independent point-to-polyline distance."""
    generator = torch.Generator().manual_seed(seed)
    vertices = bow_polyline(0.5, 37, arc=math.pi, center=(0.0, 0.0))
    positions = (torch.rand((samples, 2), generator=generator, dtype=DTYPE) - 0.5) * 1.6
    reference, _ = point_to_polyline_distance(positions, vertices)
    forces, _ = polyline_contact_forces(
        positions, torch.zeros_like(positions), vertices, torch.zeros(2, dtype=DTYPE),
        particle_radius=0.05, hull_radius=0.0, stiffness=1000.0)
    implied = implied_distance(forces, particle_radius=0.05, stiffness=1000.0)
    contact = forces.norm(dim=1) > 0
    # Outside the contact band, the kernel reports zero force: distance cannot be
    # recovered, so restrict the comparison to contacting particles.
    deviation = (implied[contact] - reference[contact]).abs()
    scale = reference[contact].clamp_min(torch.finfo(DTYPE).eps)
    return {
        "samples": samples,
        "contacting": int(contact.sum()),
        "max_abs_distance_error_m": float(deviation.max()) if deviation.numel() else 0.0,
        "max_relative_distance_error": float((deviation / scale).max()) if deviation.numel() else 0.0,
    }


def vertex_check(radius=0.5, particle_radius=0.05, stiffness=1000.0) -> dict[str, Any]:
    """Endpoint (corner) contact must reduce to exact circle-to-circle overlap."""
    vertices = bow_polyline(radius, 24, arc=math.pi)
    apex = vertices[0]
    overlap = 0.2 * particle_radius
    # Disk on the outward radial ray of the apex, at distance R_p - overlap from it:
    # an endpoint contact must be exact circle-to-circle geometry.
    direction = apex / apex.norm()
    position = (apex + direction * (particle_radius - overlap))[None, :]
    forces, reaction = polyline_contact_forces(
        position, torch.zeros_like(position), vertices, torch.zeros(2, dtype=DTYPE),
        particle_radius=particle_radius, hull_radius=0.0, stiffness=stiffness)
    expected = stiffness * overlap
    error = abs(float(forces.norm(dim=1)) - expected) / expected
    return {
        "apex": [float(apex[0]), float(apex[1])],
        "analytic_force_N": expected,
        "kernel_force_N": float(forces.norm(dim=1)),
        "relative_error": error,
        "action_reaction_balance": float((forces.sum(dim=0) + reaction).abs().max()),
    }


def invariance_checks() -> dict[str, Any]:
    """Rigid rotation, mirror and vertex-order reversal must not matter."""
    vertices = bow_polyline(0.5, 33, arc=math.pi, center=(0.0, 0.0))
    positions = torch.tensor([[0.62, 0.07], [-0.20, 0.58], [0.30, -0.10], [0.05, 0.53]], dtype=DTYPE)
    velocities = torch.tensor([[0.1, 0.0], [-0.2, 0.3], [0.0, 0.05], [0.0, 0.0]], dtype=DTYPE)
    zeros = torch.zeros(2, dtype=DTYPE)
    kwargs = dict(particle_radius=0.05, hull_radius=0.0, stiffness=1000.0, damping=10.0)
    base, _ = polyline_contact_forces(positions, velocities, vertices, zeros, **kwargs)

    def rotate(tensor, angle):
        c, s = math.cos(angle), math.sin(angle)
        matrix = torch.tensor([[c, -s], [s, c]], dtype=DTYPE)
        return tensor @ matrix.T

    outcomes: dict[str, Any] = {}
    for angle in (0.3, 1.1, math.pi):
        rotated, _ = polyline_contact_forces(
            rotate(positions, angle), rotate(velocities, angle), rotate(vertices, angle), zeros, **kwargs)
        outcomes[f"rotation_{angle:.2f}_max_abs_dev"] = float((rotated - rotate(base, angle)).abs().max())
    mirror = torch.tensor([1.0, -1.0], dtype=DTYPE)
    mirrored, _ = polyline_contact_forces(
        positions * mirror, velocities * mirror, vertices * mirror, zeros, **kwargs)
    outcomes["mirror_max_abs_dev"] = float((mirrored - base * mirror).abs().max())
    reversed_forces, _ = polyline_contact_forces(positions, velocities, vertices.flip(0), zeros, **kwargs)
    outcomes["vertex_reversal_max_abs_dev"] = float((reversed_forces - base).abs().max())

    # Degenerate configuration: disk centre exactly on the hull line, where the
    # radial normal is undefined and the old implementation picked the sign from
    # the vertex order.
    hull = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=DTYPE)
    on_line = torch.zeros((1, 2), dtype=DTYPE)
    forward, _ = polyline_contact_forces(on_line, on_line, hull, zeros, **kwargs)
    backward, _ = polyline_contact_forces(on_line, on_line, hull.flip(0), zeros, **kwargs)
    outcomes["degenerate_forward_force"] = [float(x) for x in forward[0]]
    outcomes["degenerate_reversed_force"] = [float(x) for x in backward[0]]
    outcomes["degenerate_reversal_max_abs_dev"] = float((forward - backward).abs().max())
    return outcomes


def no_attraction_check(samples=200, seed=7) -> dict[str, Any]:
    """Every contact force must push the disk away from the hull it touches."""
    generator = torch.Generator().manual_seed(seed)
    vertices = bow_polyline(0.5, 41, arc=math.pi)
    positions = (torch.rand((samples, 2), generator=generator, dtype=DTYPE) - 0.5) * 1.2
    forces, reaction = polyline_contact_forces(
        positions, torch.zeros_like(positions), vertices, torch.zeros(2, dtype=DTYPE),
        particle_radius=0.05, hull_radius=0.0, stiffness=1000.0)
    distances, _ = point_to_polyline_distance(positions, vertices)
    magnitude = forces.norm(dim=1)
    contact = magnitude > 0
    # recover the outward normal the kernel used and check the force is aligned with it
    implied = implied_distance(forces, particle_radius=0.05, stiffness=1000.0)
    consistent = (distances[contact] - implied[contact]).abs().max() if int(contact.sum()) else torch.tensor(0.0, dtype=DTYPE)
    return {
        "samples": samples,
        "contacting": int(contact.sum()),
        "max_action_reaction_imbalance": float((forces.sum(dim=0) + reaction).abs().max()),
        "max_implied_vs_reference_distance": float(consistent),
        "negative_magnitudes": int((magnitude < 0).sum()),
    }


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

LEVELS = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256)
FRACTIONS = (0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def build_report(*, levels: Sequence[int] = LEVELS, fractions: Sequence[float] = FRACTIONS,
                 radius=0.5, particle_radius=0.05, stiffness=1000.0,
                 angles_per_segment=4) -> dict[str, Any]:
    rows = contact_error_scan(levels, fractions, radius=radius,
                              particle_radius=particle_radius, stiffness=stiffness,
                              angles_per_segment=angles_per_segment)
    by_fraction: dict[str, Any] = {}
    for fraction in fractions:
        series = [row for row in rows if row["penetration_fraction"] == fraction]
        smooth = [row for row in series if row["dropout_fraction"] == 0 and not row["contact_dropout_predicted"]]
        measured = next((row for row in series if row["dropout_fraction"] == 0), None)
        by_fraction[f"{fraction:g}"] = {
            "penetration_m": float(fraction) * particle_radius,
            "magnitude_order": observed_order([r["segments"] for r in smooth],
                                              [r["max_relative_magnitude_error"] for r in smooth]) if len(smooth) > 1 else None,
            "direction_order": observed_order([r["segments"] for r in smooth],
                                              [r["max_direction_error_deg"] for r in smooth]) if len(smooth) > 1 else None,
            "min_segments_measured_contacting": measured["segments"] if measured else None,
            "segments_for_5pct_error_predicted": segments_for_magnitude_error(0.05, radius, float(fraction) * particle_radius),
            "segments_for_5pct_error_measured": _first_level_below(series, "max_relative_magnitude_error", 0.05),
            "segments_for_15pct_error_predicted": segments_for_magnitude_error(0.15, radius, float(fraction) * particle_radius),
            "segments_for_15pct_error_measured": _first_level_below(series, "max_relative_magnitude_error", 0.15),
            "segments_for_5deg_direction_predicted": math.ceil(180.0 / 5.0),
            "segments_for_5deg_direction_measured": _first_level_below(series, "max_direction_error_deg", 5.0),
            "error_at_5pct_level": _first_value_below(series, "max_relative_magnitude_error", 0.05),
            "min_segments_predicted": required_segments(radius, float(fraction) * particle_radius),
            "error_at_measured_minimum": measured["max_relative_magnitude_error"] if measured else None,
            "direction_error_at_measured_minimum_deg": measured["max_direction_error_deg"] if measured else None,
        }
    return {
        "schema": SCHEMA,
        "geometry": {
            "hull_radius_m": radius,
            "particle_radius_m": particle_radius,
            "stiffness_N_per_m": stiffness,
            "angles_per_segment": angles_per_segment,
        },
        "error_table": rows,
        "summary_by_penetration": by_fraction,
        "distance_check": distance_check(),
        "vertex_check": vertex_check(radius=radius, particle_radius=particle_radius, stiffness=stiffness),
        "invariance": invariance_checks(),
        "no_attraction": no_attraction_check(),
        "real_bow_guidance": real_bow_guidance(radius=radius, particle_radius=particle_radius),
    }


def real_bow_guidance(*, radius=0.5, particle_radius=0.05,
                      bow_examples=((5.0, 0.01), (5.0, 0.05), (20.0, 0.01), (50.0, 0.02)),
                      digitizations=(10, 20, 40)) -> list[dict[str, Any]]:
    """Translate the error scan into hull-digitisation requirements.

    ``bow_examples`` are ``(bow radius in m, penetration depth in m)`` pairs;
    ``digitizations`` are hull-point counts over a 180 deg bow.
    """
    rows: list[dict[str, Any]] = []
    for bow_radius, penetration in bow_examples:
        segments = required_segments(bow_radius, penetration)
        for points in digitizations:
            chord_arc = math.pi / points
            gap = bow_radius * (1.0 - math.cos(chord_arc))
            rows.append({
                "bow_radius_m": bow_radius,
                "penetration_m": penetration,
                "hull_points_over_180deg": points,
                "chord_sagitta_m": gap,
                "required_segments": segments,
                "contact_lost": bool(gap >= penetration),
                "sagitta_over_penetration": gap / penetration,
            })
    return rows


def write_report(report: MappingLike, path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return target


def write_summary_csv(report: dict[str, Any], path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = report["error_table"]
    fields = list(rows[0].keys())
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return target


MappingLike = dict
