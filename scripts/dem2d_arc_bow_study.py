"""Verification campaign for the exact-arc bow integrated into :class:`IceDEM`.

Three campaigns, all driven through the *production* ``IceDEM`` object (not a
private copy of the contact law):

A. ``geometry``  -- an isolated disk at a prescribed penetration, force compared
   with the closed-form arc ``k * (r_p + R_pad - r)``.  Sweeps the chord count
   ``N`` and the dimensionless penetration ``delta / r_p``.
B. ``sheet``     -- a dynamic vertical indentation of a 33-disk bonded sheet by
   a moving bow, run with the exact arc and with ``N``-chord polyline bows.
   Reports peak tool reaction, contact-onset time and early-time RMS load.
C. ``floe``      -- the same indentation against one 4-disk floe, i.e. the
   small-contact-patch / brash-ice regime.

Outputs ``docs/dem2d-arc-bow/study.json`` plus the three figures used by the
mission report.

Run: ``PYTHONPATH=src python3 scripts/dem2d_arc_bow_study.py``
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

from tensordem import DEMConfig, IceDEM
from tensordem.hull_geometry import (
    CircularArc,
    analytic_arc_penetration,
    arc_polyline,
    sagitta,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "dem2d-arc-bow"

# ---- shared case definition ------------------------------------------------
BOW_RADIUS = 0.25          # m, bow radius (R / r_p = 10 for r_p = 0.025 m)
BOW_SPAN = math.pi         # rad, 180 deg bow
BOW_CENTER = (0.0, -0.30)  # m, bow centre at t = 0 (apex at y = -0.05 m)
HULL_SPEED = 0.2           # m/s, upward
CHORDS = (8, 16, 32, 64, 128, 512, 4096)
PENETRATIONS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)  # arc overlap / r_p


def bow_arc() -> CircularArc:
    return CircularArc(center=BOW_CENTER, radius=BOW_RADIUS,
                       start_angle=0.0, span=BOW_SPAN)


def bow_profile(n_chords: int) -> torch.Tensor:
    return arc_polyline(BOW_RADIUS, BOW_SPAN, n_chords,
                        start_angle=0.0, center=BOW_CENTER)


def make_sim(nx: int, ny: int, *, chords: int | None,
             static: bool = False) -> IceDEM:
    """Build an indentation case; ``static`` freezes the bow (pure geometry)."""
    cfg = DEMConfig(nx=nx, ny=ny, contact_damping=0.0 if static else 5.0,
                    tool_speed=HULL_SPEED)
    speed = 0.0 if static else HULL_SPEED
    kw = dict(hull_start=(0.0, 0.0), hull_velocity=(0.0, speed))
    if chords is None:
        return IceDEM(cfg, hull_arc=bow_arc(), **kw)
    return IceDEM(cfg, hull_profile=bow_profile(chords), **kw)


# ---------------------------------------------------------------------------
# Campaign A: isolated disk, prescribed penetration
# ---------------------------------------------------------------------------
def geometry_campaign() -> dict:
    """Force on one disk at a prescribed penetration, measured on IceDEM."""
    cfg = DEMConfig(nx=2, ny=2, contact_damping=0.0, tool_speed=HULL_SPEED)
    r_p = cfg.radius
    rows = []
    for ratio in PENETRATIONS:
        delta = ratio * r_p
        # Disk centre on the radial line through a *chord midpoint* (the worst
        # case of the secant discretisation, where chord penetration is exactly
        # delta - sagitta).  The apex itself is a polyline vertex and carries no
        # error, so it cannot be used to expose the bias.
        for chords in (None,) + CHORDS:
            angle = 0.5 * math.pi if chords is None else 0.5 * math.pi - 0.5 * BOW_SPAN / chords
            # ``delta`` is the *arc overlap* (what the kernel calls overlap), so
            # delta / r_p is directly comparable with the published polyline
            # audit, which quotes the error at a given overlap.
            radial = BOW_RADIUS + cfg.tool_radius + r_p - delta
            point = (BOW_CENTER[0] + radial * math.cos(angle),
                     BOW_CENTER[1] + radial * math.sin(angle))
            sim = make_sim(2, 2, chords=chords, static=True)
            sim.positions = torch.tensor(
                [list(point), [10.0, 10.0], [10.0, 20.0], [20.0, 10.0]],
                dtype=torch.float64)
            sim.velocities = torch.zeros_like(sim.positions)
            sim.alive = torch.zeros_like(sim.alive)   # isolate hull contact
            force, reaction = sim.forces()
            reference = cfg.contact_stiffness * analytic_arc_penetration(
                BOW_RADIUS + cfg.tool_radius, r_p, radial)
            magnitude = float(torch.linalg.vector_norm(force[0]))
            rows.append(dict(
                delta_over_rp=ratio,
                delta_m=delta,
                chords=chords,
                force_N=magnitude,
                closed_form_N=reference,
                relative_error=(magnitude - reference) / reference,
                sagitta_m=sagitta(BOW_RADIUS, BOW_SPAN, chords) if chords else 0.0,
                reaction_imbalance_N=float(torch.linalg.vector_norm(
                    force.sum(dim=0) + reaction)),
                direction_error_deg=float(torch.rad2deg(torch.acos(torch.clamp(
                    (force[0] / magnitude) @ torch.tensor(
                        [math.cos(angle), math.sin(angle)], dtype=torch.float64),
                    -1.0, 1.0)))),
            ))
    return dict(r_p_m=r_p, tool_radius_m=cfg.tool_radius,
                stiffness_N_m=cfg.contact_stiffness, rows=rows)


# ---------------------------------------------------------------------------
# Campaigns B/C: dynamic indentation
# ---------------------------------------------------------------------------
def indentation_campaign(nx: int, ny: int, *, chords: int | None,
                         steps: int) -> dict:
    sim = make_sim(nx, ny, chords=chords)
    history, contact_counts = [], []
    peak = 0.0
    for _ in range(steps):
        force, reaction = sim.forces()
        magnitude = -float(reaction[1])
        history.append(magnitude)
        contact_counts.append(int((torch.linalg.vector_norm(force, dim=1) > 0).sum()))
        peak = max(peak, magnitude)
        sim.step()
    onset = next((i * sim.dt for i, value in enumerate(history) if value > 1e-9), None)
    return dict(chords=chords, steps=steps, dt_s=sim.dt,
                peak_tool_force_N=peak, onset_time_s=onset,
                history_N=history, contact_counts=contact_counts,
                broken_bonds=sim.broken_bonds,
                energy_J=sim.mechanical_energy())


def dynamic_campaign(name: str, nx: int, ny: int, steps: int) -> dict:
    runs = {}
    for chords in (None,) + CHORDS:
        key = "arc" if chords is None else f"N{chords}"
        runs[key] = indentation_campaign(nx, ny, chords=chords, steps=steps)
    reference = runs[f"N{CHORDS[-1]}"]["peak_tool_force_N"]
    arc_peak = runs["arc"]["peak_tool_force_N"]
    for key, run in runs.items():
        run["peak_error_vs_arc"] = (run["peak_tool_force_N"] - arc_peak) / arc_peak
        run["peak_error_vs_finest"] = (run["peak_tool_force_N"] - reference) / reference
    early = int(round(0.30 / runs["arc"]["dt_s"]))
    arc_hist = runs["arc"]["history_N"][:early]
    arc_rms = math.sqrt(sum(v * v for v in arc_hist) / len(arc_hist)) or 1.0
    for key, run in runs.items():
        head = run["history_N"][:early]
        rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(head, arc_hist)) / len(head))
        run["early_rms_error"] = rms / arc_rms
    return dict(case=name, nx=nx, ny=ny, steps=steps,
                arc_equals_finest=(arc_peak - reference) / reference,
                runs=runs)


def main() -> int:
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        case=dict(bow_radius_m=BOW_RADIUS, bow_span_rad=BOW_SPAN,
                  bow_center_m=list(BOW_CENTER), hull_speed_m_s=HULL_SPEED,
                  chords=list(CHORDS), penetrations_delta_over_rp=list(PENETRATIONS)),
        geometry=geometry_campaign(),
        sheet=dynamic_campaign("sheet", 11, 3, 260),
        floe=dynamic_campaign("floe", 2, 2, 260),
    )
    (OUT / "study.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(dict(
        arc_vs_finest_sheet=payload["sheet"]["arc_equals_finest"],
        arc_vs_finest_floe=payload["floe"]["arc_equals_finest"],
        sheet_peak={k: round(v["peak_tool_force_N"], 3) for k, v in payload["sheet"]["runs"].items()},
        sheet_err={k: round(100 * v["peak_error_vs_arc"], 3) for k, v in payload["sheet"]["runs"].items()},
        floe_err={k: round(100 * v["peak_error_vs_arc"], 3) for k, v in payload["floe"]["runs"].items()},
    ), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())