"""Audit TensorDEM3D's contact law against a reference impact solution.

The audit answers one question the repository could not answer before: *what
restitution does the solver actually produce for a given*
``DEM3DConfig.contact_damping``, and does the documented analytic mapping
(``contact_tangential.restitution_to_damping``) describe it?

Method
------
Two engine configurations exercise two different code branches and two
different effective masses:

* ``pair``   -- two free spheres collide head-on inside ``IceDEM3D`` (bond mask
  cleared, no tool/platen contact).  Both bodies move, so ``m_eff = m/2``.
* ``platen`` -- a single sphere rebounds off the infinite-mass bottom platen
  (``platen_stiffness``/``platen_damping``).  Only the sphere moves, so
  ``m_eff = m``.

For each configuration the measured ``e`` (and the contact duration and peak
overlap) is compared with

1. ``restitution.unilateral_restitution`` -- the reference RK4 solution of the
   *same* clipped law the engine implements (implementation-fidelity check);
2. ``contact_tangential.damping_to_restitution`` -- the bilateral textbook
   mapping that ``restitution_to_damping`` inverts (applicability check).

The script then inverts the reference solution
(``restitution.damping_for_restitution``) for a set of target restitution
values, feeds the resulting ``c_n`` back into the engine and measures the
closure error.  Finally it repeats one case at reduced time step and at
several impact speeds to show that the residual error is neither
discretisation nor a velocity artefact.

Usage
-----
    python scripts/verify_dem3d_restitution.py [--output outputs/dem3d_restitution]

Outputs ``audit.json`` plus two figures.  Plotting is skipped if matplotlib is
unavailable; all numbers are produced without it.
"""
from __future__ import annotations

import argparse
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
    unilateral_contact_detail,
    unilateral_restitution,
)

# --- fixed physical set-up -------------------------------------------------
RADIUS = 0.025
DENSITY = 917.0
STIFFNESS = 5_000.0
IMPACT_SPEED = 0.5

# Idle particles are parked in a compact cloud (0.15 m spacing) rather than far
# away: IceDEM3D's contact search builds a uniform cell list over the bounding
# box of all positions, so scattering particles over metres makes every step
# allocate a huge grid.
_IDLE_SITES = (
    (0.15, 0.15, 0.15), (-0.15, 0.15, 0.15), (0.15, -0.15, 0.15),
    (-0.15, -0.15, 0.15), (0.0, 0.15, 0.30), (0.0, -0.15, 0.30),
)


def _build_engine(damping: float, *, platen: bool, dt_scale: float = 1.0) -> IceDEM3D:
    """Build a 2x2x2 engine with all bonds removed and a quiet environment."""
    kwargs = dict(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        contact_stiffness=STIFFNESS, contact_damping=damping,
        platen_stiffness=STIFFNESS, platen_damping=damping,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=platen, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
    )
    if dt_scale != 1.0:
        kwargs["dt"] = DEM3DConfig(**kwargs).recommended_dt * dt_scale
    sim = IceDEM3D(DEM3DConfig(**kwargs))
    # Remove every bond without injecting the elastic energy that a fracture
    # event would release; contacts then govern all interactions.
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0
    return sim


def _park_idle(sim: IceDEM3D, start: int = 2) -> None:
    """Move the unused particles out of the impact zone (spacing 0.15 m).

    ``start`` is the first particle index the caller does *not* place itself;
    the two impact runners need different values because the pair case uses
    particles 0 and 1 while the platen case uses only particle 0.
    """
    n = sim.positions.shape[0]
    with torch.no_grad():
        for idx in range(start, n):
            site = _IDLE_SITES[(idx - start) % len(_IDLE_SITES)]
            sim.positions[idx] = torch.tensor(site, dtype=sim.dtype)
            sim.velocities[idx] = torch.zeros(3, dtype=sim.dtype)


def measure_pair(damping: float, *, impact_speed: float = IMPACT_SPEED,
                 dt_scale: float = 1.0) -> dict:
    """Head-on sphere/sphere impact inside IceDEM3D; returns measured e."""
    sim = _build_engine(damping, platen=False, dt_scale=dt_scale)
    _park_idle(sim)
    with torch.no_grad():
        sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
        sim.positions[1] = torch.tensor([2.0 * RADIUS + 0.001, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([impact_speed, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1] = torch.tensor([-impact_speed, 0.0, 0.0], dtype=sim.dtype)

    def overlap() -> float:
        return 2.0 * RADIUS - float(
            torch.linalg.vector_norm(sim.positions[1] - sim.positions[0]))

    def relative_velocity() -> float:
        return float(sim.velocities[1, 0] - sim.velocities[0, 0])

    result = _measure_impact(sim, overlap, relative_velocity)
    result["dt"] = sim.dt
    return result


def measure_platen(damping: float, *, impact_speed: float = IMPACT_SPEED,
                   dt_scale: float = 1.0) -> dict:
    """Sphere rebounding off the infinite-mass bottom platen."""
    sim = _build_engine(damping, platen=True, dt_scale=dt_scale)
    _park_idle(sim, start=1)
    platen_z = float(sim.bottom_platen_start_z)
    with torch.no_grad():
        # Start 0.5 mm above the contact height to keep the free-flight phase short.
        sim.positions[0] = torch.tensor([0.0, 0.0, RADIUS + platen_z + 0.0005], dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([0.0, 0.0, -impact_speed], dtype=sim.dtype)

    def overlap() -> float:
        return platen_z + RADIUS - float(sim.positions[0, 2])

    def relative_velocity() -> float:
        return float(sim.velocities[0, 2])

    result = _measure_impact(sim, overlap, relative_velocity)
    result["dt"] = sim.dt
    return result


def _measure_impact(sim: IceDEM3D, overlap, relative_velocity,
                    max_steps: int = 500_000) -> dict:
    """Step until the contact opens; return e, duration and peak overlap."""
    v_in = None
    v_out = None
    peak = 0.0
    steps_in_contact = 0
    for _ in range(max_steps):
        ov = overlap()
        if ov > 0.0:
            if v_in is None:
                v_in = relative_velocity()
            peak = max(peak, ov)
            steps_in_contact += 1
        elif v_in is not None:
            v_out = relative_velocity()
            break
        sim.step()
    if v_in is None or v_out is None:
        raise RuntimeError("impact never completed")
    return {
        "restitution": v_out / (-v_in),
        "duration_s": steps_in_contact * sim.dt,
        "peak_overlap_m": peak,
        "steps_in_contact": steps_in_contact,
        "v_in": v_in,
        "v_out": v_out,
    }


def _rel_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return float("nan")
    return (value - reference) / reference


def build_audit(damping_values: list[float], targets: list[float],
                dt_scales: list[float], speeds: list[float]) -> dict:
    mass = DENSITY * (4.0 / 3.0) * math.pi * RADIUS ** 3
    m_pair = effective_mass(mass, mass)
    m_wall = effective_mass(mass)

    print(f"particle mass = {mass:.6g} kg, m_eff(pair) = {m_pair:.6g} kg,"
          f" m_eff(wall) = {m_wall:.6g} kg")

    sweep = []
    for c in damping_values:
        row = {"contact_damping": c}
        for label, runner, m_eff in (("pair", measure_pair, m_pair),
                                     ("platen", measure_platen, m_wall)):
            measured = runner(c)
            ref = unilateral_contact_detail(c, STIFFNESS, m_eff, IMPACT_SPEED)
            bilateral = classical_restitution(c, STIFFNESS, m_eff)
            row[label] = {
                "e_engine": measured["restitution"],
                "duration_engine_s": measured["duration_s"],
                "peak_overlap_engine_m": measured["peak_overlap_m"],
                "e_reference_unilateral": ref.restitution,
                "duration_reference_s": ref.duration,
                "peak_overlap_reference_m": ref.peak_overlap,
                "detached_reference": ref.detached,
                "e_bilateral_classical": bilateral,
                "err_vs_unilateral": _rel_error(measured["restitution"], ref.restitution),
                "abs_err_vs_unilateral": measured["restitution"] - ref.restitution,
                "err_vs_bilateral": _rel_error(measured["restitution"], bilateral)
                if bilateral > 1e-12 else float("nan"),
                "abs_err_vs_bilateral": measured["restitution"] - bilateral,
                "duration_err_vs_reference": _rel_error(measured["duration_s"], ref.duration),
            }
        sweep.append(row)
        print(f"c={c:<8g} pair e={row['pair']['e_engine']:.5f}"
              f" (ref {row['pair']['e_reference_unilateral']:.5f},"
              f" classical {row['pair']['e_bilateral_classical']:.5f})"
              f"  platen e={row['platen']['e_engine']:.5f}"
              f" (ref {row['platen']['e_reference_unilateral']:.5f})")

    closure = []
    for target in targets:
        entry = {"target": target}
        for label, runner, m_eff in (("pair", measure_pair, m_pair),
                                     ("platen", measure_platen, m_wall)):
            c = damping_for_restitution(target, STIFFNESS, m_eff)
            measured = runner(c)
            if target < 1.0:
                zeta = (-math.log(target)) / math.sqrt(math.pi ** 2 + math.log(target) ** 2)
                c_classical = 2.0 * zeta * math.sqrt(STIFFNESS * m_eff)
            else:
                c_classical = 0.0
            measured_classical = runner(c_classical)
            entry[label] = {
                "contact_damping": c,
                "e_engine": measured["restitution"],
                "rel_error": _rel_error(measured["restitution"], target),
                "contact_damping_bilateral": c_classical,
                "e_engine_with_bilateral_damping": measured_classical["restitution"],
                "rel_error_with_bilateral_damping": _rel_error(
                    measured_classical["restitution"], target),
            }
        closure.append(entry)
        print(f"target {target:<5g} pair: c={entry['pair']['contact_damping']:.4g}"
              f" -> e={entry['pair']['e_engine']:.5f}"
              f" | platen: c={entry['platen']['contact_damping']:.4g}"
              f" -> e={entry['platen']['e_engine']:.5f}")

    refinement = []
    c_ref = damping_values[len(damping_values) // 2]
    for scale in dt_scales:
        measured = measure_pair(c_ref, dt_scale=scale)
        refinement.append({
            "dt_scale": scale,
            "dt": measured["dt"],
            "restitution": measured["restitution"],
            "steps_in_contact": measured["steps_in_contact"],
            "reference": unilateral_restitution(c_ref, STIFFNESS, m_pair, IMPACT_SPEED),
        })
        print(f"dt_scale={scale:<6g} dt={measured['dt']:.4g}"
              f" e={measured['restitution']:.6f}")

    speed_study = []
    for speed in speeds:
        measured = measure_pair(c_ref, impact_speed=speed)
        speed_study.append({
            "impact_speed": speed,
            "restitution": measured["restitution"],
            "reference": unilateral_restitution(c_ref, STIFFNESS, m_pair, speed),
        })
        print(f"v_impact={speed:<6g} e={measured['restitution']:.6f}")

    default_entry = next(
        (row for row in sweep if abs(row["contact_damping"] - 5.0) < 1e-12), None)
    summary = {
        "stiffness_N_per_m": STIFFNESS,
        "radius_m": RADIUS,
        "density_kg_per_m3": DENSITY,
        "particle_mass_kg": mass,
        "m_eff_pair_kg": m_pair,
        "m_eff_wall_kg": m_wall,
        "impact_speed_m_per_s": IMPACT_SPEED,
        "default_config_restitution": {
            "note": "TensorDEM3D defaults: contact_damping=5 N*s/m, "
                    "contact_stiffness=5000 N/m, radius=0.025 m, density=917 kg/m^3",
            "pair": default_entry["pair"] if default_entry else None,
            "platen": default_entry["platen"] if default_entry else None,
        } if default_entry else None,
        "max_abs_err_vs_unilateral": max(
            abs(row[label]["abs_err_vs_unilateral"])
            for row in sweep for label in ("pair", "platen")),
        "max_rel_err_vs_unilateral": max(
            abs(row[label]["err_vs_unilateral"])
            for row in sweep for label in ("pair", "platen")),
        "max_abs_err_vs_bilateral": max(
            abs(row[label]["abs_err_vs_bilateral"])
            for row in sweep for label in ("pair", "platen")
            if math.isfinite(row[label]["abs_err_vs_bilateral"])),
        "max_closure_rel_error": max(
            abs(entry[label]["rel_error"]) for entry in closure for label in ("pair", "platen")),
        "max_closure_rel_error_bilateral_damping": max(
            abs(entry[label]["rel_error_with_bilateral_damping"])
            for entry in closure for label in ("pair", "platen")),
        "duration_max_rel_error": max(
            abs(row[label]["duration_err_vs_reference"])
            for row in sweep for label in ("pair", "platen")),
        "dt_refinement_spread": max(r["restitution"] for r in refinement)
        - min(r["restitution"] for r in refinement),
        "speed_spread": max(s["restitution"] for s in speed_study)
        - min(s["restitution"] for s in speed_study),
    }

    return {
        "case": "TensorDEM3D contact-law restitution audit",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
        },
        "summary": summary,
        "damping_sweep": sweep,
        "mapping_closure": closure,
        "timestep_refinement": refinement,
        "impact_speed_study": speed_study,
        "reference_model": (
            "rk4 solution of d2(delta)/dt2 = -max(0, k delta + c d(delta)/dt)/m_eff "
            "with detach/exit event localisation"
        ),
    }


def write_figures(audit: dict, outdir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable ({exc}); skipping figures")
        return []

    paths = []
    sweep = audit["damping_sweep"]
    positive = [row for row in sweep if row["contact_damping"] > 0]
    pos = [row["contact_damping"] for row in positive]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, label, m_eff in ((axes[0], "pair", audit["summary"]["m_eff_pair_kg"]),
                             (axes[1], "platen", audit["summary"]["m_eff_wall_kg"])):
        ax.semilogx(pos, [row[label]["e_engine"] for row in positive],
                    "o", ms=6, label="engine (IceDEM3D)")
        ax.semilogx(pos, [unilateral_restitution(c, STIFFNESS, m_eff) for c in pos],
                    "-", lw=1.8, label="reference: unilateral clip")
        ax.semilogx(pos, [classical_restitution(c, STIFFNESS, m_eff) for c in pos],
                    "--", lw=1.8, label="classical bilateral")
        ax.set_xlabel("contact_damping c_n  [N s/m]")
        ax.set_ylabel("coefficient of restitution e")
        ax.set_title(f"{label} impact (m_eff = {m_eff:.4g} kg)")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("TensorDEM3D contact law vs reference solution")
    fig.tight_layout()
    path = outdir / "fig1_restitution_vs_damping.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].semilogx(pos, [abs(row["pair"]["abs_err_vs_unilateral"]) for row in positive],
                     "o-", label="pair vs unilateral ref")
    axes[0].semilogx(pos, [abs(row["platen"]["abs_err_vs_unilateral"]) for row in positive],
                     "s-", label="platen vs unilateral ref")
    axes[0].semilogx(pos, [abs(row["pair"]["abs_err_vs_bilateral"]) for row in positive],
                     "o--", label="pair vs bilateral")
    axes[0].set_xlabel("contact_damping c_n  [N s/m]")
    axes[0].set_ylabel("|e_engine - e_model|")
    axes[0].set_title("absolute restitution error")
    axes[0].grid(alpha=0.3)
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)

    targets = [row["target"] for row in audit["mapping_closure"]]
    axes[1].plot([0, 1], [0, 1], "k:", lw=1, label="target = measured")
    axes[1].plot(targets, [row["pair"]["e_engine"] for row in audit["mapping_closure"]],
                 "o-", label="engine, inverted clip law")
    axes[1].plot(targets, [row["pair"]["e_engine_with_bilateral_damping"]
                           for row in audit["mapping_closure"]],
                 "s--", label="engine, bilateral mapping")
    axes[1].set_xlabel("target e")
    axes[1].set_ylabel("measured e (pair impact)")
    axes[1].set_title("damping inversion closure")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "fig2_closure_and_error.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(str(path))
    print(f"figures written: {paths}")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/dem3d_restitution")
    args = parser.parse_args(argv)

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    damping_values = [0.0, 2.0, 5.0, 10.0, 20.0, 40.0]
    targets = [0.9, 0.7, 0.5, 0.3]
    dt_scales = [1.0, 0.5]
    speeds = [0.1, 0.5, 2.0]

    audit = build_audit(damping_values, targets, dt_scales, speeds)
    figures = write_figures(audit, outdir)
    audit["figures"] = figures

    path = outdir / "audit.json"
    path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    summary = audit["summary"]
    print("\n=== summary ===")
    print(f"max |e_engine - e_unilateral| = {summary['max_abs_err_vs_unilateral']:.4g}"
          f" (rel {summary['max_rel_err_vs_unilateral']:.3%})")
    print(f"max |e_engine - e_bilateral|   = {summary['max_abs_err_vs_bilateral']:.4g}")
    print(f"max closure error (clip inversion) = {summary['max_closure_rel_error']:.3%}")
    print(f"max closure error (bilateral map)  = "
          f"{summary['max_closure_rel_error_bilateral_damping']:.3%}")
    print(f"dt refinement spread = {summary['dt_refinement_spread']:.3g}")
    print(f"audit written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())