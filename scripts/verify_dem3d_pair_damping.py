"""Closure audit for DEM3D particle/particle restitution after the m670 regression.

The particle/particle normal contact of :class:`tensordem.dem3d.IceDEM3D` must
honour two knobs:

* ``contact_damping`` -- the raw dashpot coefficient (N s/m), the convention the
  tool and platen branches already use;
* ``restitution_coefficient`` -- the requested rebound ratio, realised through
  the exact inverse of the *clipped* (unilateral) contact law.

This script measures the rebound produced by the engine in both modes and
compares it with ``tensordem.restitution.unilateral_restitution``, an
independently coded RK4 solution of the same law.  It also reproduces the two
defects the fix closed: (a) ``contact_damping`` used to be ignored for free
pairs, so every rebound was perfectly elastic; (b) the requested restitution
used to be inverted with the *bilateral* mapping, which under-prescribes the
dashpot and overshoots ``e`` at low targets.

Usage
-----
    python scripts/verify_dem3d_pair_damping.py [--output outputs/dem3d_pair_damping]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensordem.dem3d import DEM3DConfig, IceDEM3D
from tensordem.restitution import (
    classical_restitution,
    damping_for_restitution,
    effective_mass,
    unilateral_restitution,
)

RADIUS = 0.025
DENSITY = 917.0
STIFFNESS = 5_000.0
IMPACT_SPEED = 0.5
MASS = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
M_PAIR = effective_mass(MASS, MASS)

_IDLE_SITES = (
    (0.15, 0.15, 0.15), (-0.15, 0.15, 0.15), (0.15, -0.15, 0.15),
    (-0.15, -0.15, 0.15), (0.0, 0.15, 0.30), (0.0, -0.15, 0.30),
)


def measure_pair(damping: float, restitution_coefficient: float = 1.0,
                 dt_scale: float = 1.0, impact_speed: float = IMPACT_SPEED,
                 max_steps: int = 500_000) -> dict:
    """Head-on equal-mass impact inside the engine; returns the measured e."""
    torch.set_num_threads(1)
    kwargs = dict(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        contact_stiffness=STIFFNESS, contact_damping=damping,
        platen_stiffness=STIFFNESS, platen_damping=0.0,
        restitution_coefficient=restitution_coefficient,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=False, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
    )
    if dt_scale != 1.0:
        kwargs["dt"] = DEM3DConfig(**kwargs).recommended_dt * dt_scale
    sim = IceDEM3D(DEM3DConfig(**kwargs))
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0
    n = sim.positions.shape[0]
    with torch.no_grad():
        for idx in range(2, n):
            sim.positions[idx] = torch.tensor(_IDLE_SITES[(idx - 2) % len(_IDLE_SITES)],
                                              dtype=sim.dtype)
            sim.velocities[idx] = torch.zeros(3, dtype=sim.dtype)
        sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
        sim.positions[1] = torch.tensor([2.0 * RADIUS + 0.001, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([impact_speed, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1] = torch.tensor([-impact_speed, 0.0, 0.0], dtype=sim.dtype)

    def overlap() -> float:
        return 2.0 * RADIUS - float(
            torch.linalg.vector_norm(sim.positions[1] - sim.positions[0]))

    def relative_velocity() -> float:
        return float(sim.velocities[1, 0] - sim.velocities[0, 0])

    v_in = v_out = None
    peak = 0.0
    steps = 0
    for _ in range(max_steps):
        if overlap() > 0.0:
            if v_in is None:
                v_in = relative_velocity()
            peak = max(peak, overlap())
            steps += 1
        elif v_in is not None:
            v_out = relative_velocity()
            break
        sim.step()
    if v_in is None or v_out is None:
        raise RuntimeError("impact never completed")
    return {"restitution": v_out / (-v_in), "dt": sim.dt,
            "steps_in_contact": steps, "peak_overlap_m": peak,
            "v_in": v_in, "v_out": v_out}


def rel(value: float, reference: float) -> float:
    return (value - reference) / reference if reference else float("nan")


def build(dampings, targets, dt_scales, speeds):
    rows = []
    # (A) raw dashpot mode: engine vs reference solution of the clipped law.
    for c in dampings:
        r = measure_pair(c)
        ref = unilateral_restitution(c, STIFFNESS, M_PAIR)
        rows.append(dict(study="dashpot", damping=c, target=None,
                         measured=r["restitution"], reference=ref,
                         rel_error=rel(r["restitution"], ref),
                         bilateral_prediction=classical_restitution(c, STIFFNESS, M_PAIR),
                         duration_s=r["steps_in_contact"] * r["dt"],
                         peak_overlap_m=r["peak_overlap_m"]))
    # (B) requested restitution: clipped inverse vs the bilateral mapping.
    for e in targets:
        c_clip = damping_for_restitution(e, STIFFNESS, M_PAIR)
        r_clip = measure_pair(0.0, restitution_coefficient=e)
        from tensordem.contact_tangential import restitution_to_damping
        c_bi = restitution_to_damping(e, STIFFNESS, M_PAIR)
        measured_bi = measure_pair(c_bi)
        rows.append(dict(study="target", damping=c_clip, target=e,
                         measured=r_clip["restitution"], reference=e,
                         rel_error=rel(r_clip["restitution"], e),
                         bilateral_prediction=measured_bi["restitution"],
                         bilateral_rel_error=rel(measured_bi["restitution"], e),
                         duration_s=r_clip["steps_in_contact"] * r_clip["dt"],
                         peak_overlap_m=r_clip["peak_overlap_m"]))
    # (C) time-step and impact-speed sensitivity of the residual.
    sensitivity = []
    for scale in dt_scales:
        r = measure_pair(5.0, dt_scale=scale)
        sensitivity.append(dict(kind="dt_scale", value=scale,
                                measured=r["restitution"],
                                reference=unilateral_restitution(5.0, STIFFNESS, M_PAIR),
                                rel_error=rel(r["restitution"],
                                              unilateral_restitution(5.0, STIFFNESS, M_PAIR))))
    for v in speeds:
        r = measure_pair(5.0, impact_speed=v)
        ref = unilateral_restitution(5.0, STIFFNESS, M_PAIR, impact_speed=v)
        sensitivity.append(dict(kind="impact_speed", value=v,
                                measured=r["restitution"], reference=ref,
                                rel_error=rel(r["restitution"], ref)))
    return rows, sensitivity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="outputs/dem3d_pair_damping")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    dampings = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    targets = [0.9, 0.7, 0.5, 0.3]
    rows, sensitivity = build(dampings, targets, [1.0, 0.5, 0.25], [0.1, 0.5, 2.0])

    worst_dashpot = max(abs(r["rel_error"]) for r in rows if r["study"] == "dashpot")
    worst_target = max(abs(r["rel_error"]) for r in rows if r["study"] == "target")
    worst_bilateral = max(abs(r["bilateral_rel_error"]) for r in rows if r["study"] == "target")

    payload = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        platform=platform.platform(),
        setup=dict(radius=RADIUS, density=DENSITY, stiffness=STIFFNESS,
                   impact_speed=IMPACT_SPEED, mass=MASS, m_eff_pair=M_PAIR,
                   restitution_reference="tensordem.restitution.unilateral_restitution"),
        rows=rows, sensitivity=sensitivity,
        summary=dict(worst_dashpot_rel_error=worst_dashpot,
                     worst_target_rel_error=worst_target,
                     worst_bilateral_rel_error=worst_bilateral,
                     dashpot_under_3pct=worst_dashpot < 0.03,
                     target_under_3pct=worst_target < 0.03),
    )
    (out / "audit.json").write_text(json.dumps(payload, indent=2))

    with (out / "audit.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
        w.writeheader()
        w.writerows(rows)

    print(f"worst |rel err| dashpot mode : {worst_dashpot:.4%}")
    print(f"worst |rel err| target mode  : {worst_target:.4%}")
    print(f"worst |rel err| bilateral    : {worst_bilateral:.4%} (the defect)")
    for r in sensitivity:
        print(f"  {r['kind']}={r['value']:<6} e={r['measured']:.5f} "
              f"ref={r['reference']:.5f} rel={r['rel_error']:+.3%}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"matplotlib unavailable ({exc}); JSON/CSV written")
        return 0 if payload["summary"]["dashpot_under_3pct"] else 1

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    dp = [r for r in rows if r["study"] == "dashpot"]
    ax.plot([r["damping"] for r in dp], [r["measured"] for r in dp], "o-",
            label="engine (pair impact)")
    ax.plot([r["damping"] for r in dp], [r["reference"] for r in dp], "--",
            label="clipped-law RK4 reference")
    ax.plot([r["damping"] for r in dp], [r["bilateral_prediction"] for r in dp], ":",
            label="bilateral formula (wrong for this law)")
    ax.set_xlabel("contact_damping c_n [N s/m]")
    ax.set_ylabel("restitution e")
    ax.set_title("DEM3D pair rebound vs contact_damping")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "e_vs_contact_damping.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tg = [r for r in rows if r["study"] == "target"]
    xs = range(len(tg))
    ax.bar([x - 0.2 for x in xs], [r["rel_error"] * 100 for r in tg], width=0.4,
           label="clipped inverse (shipped)")
    ax.bar([x + 0.2 for x in xs], [r["bilateral_rel_error"] * 100 for r in tg],
           width=0.4, label="bilateral mapping (old)")
    ax.axhline(3.0, color="k", ls="--", lw=1, label="3% acceptance")
    ax.axhline(-3.0, color="k", ls="--", lw=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"e={r['target']}" for r in tg])
    ax.set_ylabel("relative error vs target [%]")
    ax.set_title("Requested restitution closure (sphere/sphere)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "target_closure.png", dpi=140)
    plt.close(fig)
    print(f"artifacts -> {out}")
    return 0 if payload["summary"]["dashpot_under_3pct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())