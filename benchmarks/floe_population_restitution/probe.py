"""Measure the restitution actually produced by the floe-population soft-sphere engine.

Reference (exact, no fitting)
-----------------------------
A pair of spheres in contact through a linear spring-dashpot

    F_n = k_n delta - c_n d(delta)/dt,   c_n = 2 zeta sqrt(k_n m_eff),

with ``m_eff = m_i m_j / (m_i + m_j)`` is a damped harmonic oscillator of the
*relative* coordinate, natural frequency ``w = sqrt(k_n / m_eff)`` and damping
ratio ``zeta``.  Two exact consequences follow and are used here as the
reference:

* contact duration (one half period) ``t_c = pi / (w sqrt(1 - zeta^2))``;
* restitution ``e = exp(-pi zeta / sqrt(1 - zeta^2))``,

so the damping ratio implemented by ``ContactModel.damping_ratio`` must invert
to the restitution the user asked for, *up to the integration error of the
velocity-Verlet march*.

Run::

    PYTHONPATH=src python benchmarks/floe_population_restitution/probe.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensordem.floe_population.soft_sphere import ContactModel, SphereAssembly

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "artifacts" / "floe_population_restitution" / "probe.json"

K_N = 2000.0
RADIUS = 0.5
V_REL = 0.1


def analytic(e: float, k_n: float = K_N, radius: float = RADIUS) -> dict:
    """Exact half-period and damping ratio for a prescribed restitution."""
    m = radius**3  # unit density, the engine's convention
    m_eff = m / 2.0
    zeta = ContactModel(k_n=k_n, restitution=e).damping_ratio
    w = math.sqrt(k_n / m_eff)
    return {
        "zeta": zeta,
        "omega_n": w,
        "t_contact": math.pi / (w * math.sqrt(1.0 - zeta * zeta)),
        "e_closed_form": math.exp(-math.pi * zeta / math.sqrt(1.0 - zeta * zeta)),
    }


def collide(e: float, dt: float, *, steps_cap: int = 400000,
            k_n: float = K_N, radius: float = RADIUS,
            v_rel: float = V_REL, gap: float = 0.5) -> dict:
    """Head-on two-sphere collision; returns the measured restitution."""
    box = np.array([8.0, 8.0, 8.0])
    m = radius**3
    position = np.array([[3.0, 4.0, 4.0], [3.0 + 2.0 * radius + gap, 4.0, 4.0]])
    velocity = np.array([[+0.5 * v_rel, 0.0, 0.0], [-0.5 * v_rel, 0.0, 0.0]])
    asm = SphereAssembly(
        radius=np.full(2, radius),
        position=position,
        velocity=velocity,
        box=box,
        model=ContactModel(k_n=k_n, restitution=e),
    )

    sep = asm.position[1, 0] - asm.position[0, 0]
    v_in = float(velocity[0, 0] - velocity[1, 0])
    v_out = None
    t = 0.0
    t_start = t_end = None
    overlap_max = 0.0
    momentum0 = float((asm.mass[:, None] * asm.velocity).sum())
    energy0 = float((0.5 * asm.mass[:, None] * asm.velocity**2).sum())
    n_steps = 0
    while n_steps < steps_cap:
        asm.advance(dt, 1, record_contacts=False)
        n_steps += 1
        t += dt
        sep = float(asm.position[1, 0] - asm.position[0, 0])
        overlap = 2.0 * radius - sep
        overlap_max = max(overlap_max, overlap)
        if overlap > 0.0 and t_start is None:
            t_start = t
        if t_start is not None and overlap <= 0.0:
            t_end = t
            v_out = float(asm.velocity[0, 0] - asm.velocity[1, 0])
            break
    if v_out is None:
        v_out = float(asm.velocity[0, 0] - asm.velocity[1, 0])
    momentum1 = float((asm.mass[:, None] * asm.velocity).sum())
    energy1 = float((0.5 * asm.mass[:, None] * asm.velocity**2).sum())
    ref = analytic(e, k_n=k_n, radius=radius)
    return {
        "e_prescribed": e,
        "dt": dt,
        "dt_over_t_contact": dt / ref["t_contact"],
        "steps_per_contact": ref["t_contact"] / dt,
        "e_measured": -v_out / v_in,
        "e_rel_error": -v_out / v_in / e - 1.0 if e > 0 else float("nan"),
        "t_contact_measured": (t_end - t_start) if (t_start and t_end) else None,
        "t_contact_analytic": ref["t_contact"],
        "overlap_max": overlap_max,
        "n_steps": n_steps,
        "gap": gap,
        "momentum_drift": momentum1 - momentum0,
        "energy_ratio": energy1 / energy0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(OUT))
    args = ap.parse_args(argv)

    e_values = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
    rel_dt = [0.4, 0.2, 0.1, 0.05, 0.025]
    phases = [0.05, 0.033, 0.017]
    rows = []
    for e in e_values:
        t_c = analytic(e)["t_contact"]
        for frac in rel_dt:
            for gap in phases:
                rows.append(collide(e, dt=frac * t_c, gap=gap))

    # What the engine's own recommended step gives for a *bond-free* assembly
    # (the dilute colliding ensemble the collision kernel is measured on).
    box = np.array([8.0, 8.0, 8.0])
    asm = SphereAssembly(
        radius=np.full(2, RADIUS),
        position=np.array([[3.0, 4.0, 4.0], [4.5, 4.0, 4.0]]),
        velocity=np.zeros((2, 3)),
        box=box,
        model=ContactModel(k_n=K_N, restitution=0.6),
    )
    dt_rec = asm.recommended_dt()
    recommended = [collide(e, dt=dt_rec, gap=g) for e in e_values for g in phases]

    summary = {}
    for frac in rel_dt:
        sel = [r for r in rows if abs(r["dt_over_t_contact"] - frac) < 0.02]
        summary[str(frac)] = {
            "n": len(sel),
            "max_abs_rel_error": max(abs(r["e_rel_error"]) for r in sel),
            "median_abs_rel_error": float(np.median([abs(r["e_rel_error"]) for r in sel])),
        }
    report = {
        "summary_by_dt_over_tc": summary,
        "recommended_max_abs_rel_error": max(abs(r["e_rel_error"]) for r in recommended),
        "schema": "floe-population-restitution-probe/1",
        "k_n": K_N,
        "radius": RADIUS,
        "v_rel": V_REL,
        "recommended_dt_bondfree": dt_rec,
        "recommended_overlap_ratio": {  # how coarse that step is per restitution
            str(e): dt_rec / analytic(e)["t_contact"] for e in e_values
        },
        "sweep": rows,
        "at_recommended_dt": recommended,
    }
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    print(f"recommended_dt (bond-free) = {dt_rec:.6e}")
    print(f"{'e':>5} {'dt/tc':>8} {'e_meas':>9} {'rel.err':>9} {'steps/contact':>13}")
    for r in recommended[:12]:
        print(f"{r['e_prescribed']:5.2f} {r['dt_over_t_contact']:8.3f} "
              f"{r['e_measured']:9.4f} {100 * r['e_rel_error']:8.2f}% "
              f"{r['steps_per_contact']:13.2f}")
    print("worst |rel.err| at recommended dt: "
          f"{100 * report['recommended_max_abs_rel_error']:.2f}%")
    print("\nby dt/tc (all e, all phases):")
    for k, v in summary.items():
        print(f"  {k:>7}: max {100 * v['max_abs_rel_error']:6.2f}%  median {100 * v['median_abs_rel_error']:6.2f}%")
    print("\nconvergence at e = 0.6:")
    for r in [r for r in rows if r["e_prescribed"] == 0.6 and r["gap"] == 0.5]:
        print(f"  dt/tc={r['dt_over_t_contact']:.4f}  e_meas={r['e_measured']:.5f}  "
              f"rel.err={100 * r['e_rel_error']:+.3f}%")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())