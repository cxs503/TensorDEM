"""Locate the gap between the documented and the executed pair-contact damping law.

Context
-------
``docs/dem3d_contact_restitution.md`` (merged) publishes a closure table: feeding
``c_n = restitution.damping_for_restitution(e_target, k_n, m_eff)`` into
``DEM3DConfig.contact_damping`` and colliding two free spheres inside
``IceDEM3D`` reproduces ``e_target`` to better than 0.28 %.  Two tests in
``tests/test_restitution.py::EngineClosureTests`` encode the same contract and
are red on ``main``:

    c=2.994 target=0.7 measured=0.99987   (rel. error 42.8 %)

This script localises the cause instead of guessing it.  Three independent
probes are run on the same two-sphere impact set-up:

``A. contact_damping sweep``
    ``contact_damping`` is swept over 0, 2, 5, 10, 20, 40 N s/m with
    ``restitution_coefficient = 1.0`` (the default).  If the documented
    ``k_n delta - c_n v_n`` law were executed, ``e`` would fall monotonically to
    ~0.07.  Measured ``e`` is reported next to both candidate reference laws
    (clipped RK4 solution, bilateral formula).

``B. restitution_coefficient sweep``
    Same impact, but the damping is requested through the *other* knob
    (``restitution_coefficient = e_target``).  This isolates whether the engine
    has dissipation at all, or only this one parameter is dead.

``C. platen branch control``
    ``platen_damping`` sweep on a sphere rebounding off the infinite-mass
    platen (a different code branch).  If the platen branch responds while the
    pair branch does not, the defect is a parameter-plumbing gap in the
    particle-particle branch, not a global absence of damping.

Outputs ``artifacts/dem3d_pair_damping_gap/audit.json`` + figure.  Nothing is
transcribed by hand: every number in the accompanying document comes from the
JSON.

Run: python3 scripts/audit_dem3d_pair_damping_gap.py
"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from tensordem.dem3d import DEM3DConfig, IceDEM3D  # noqa: E402
from tensordem.restitution import (  # noqa: E402
    classical_restitution,
    damping_for_restitution,
    effective_mass,
    unilateral_restitution,
)
import verify_dem3d_restitution as ref  # noqa: E402

STIFFNESS = 5_000.0
RADIUS = 0.025
DENSITY = 917.0
C_SWEEP = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
E_TARGETS = (0.9, 0.7, 0.5, 0.3)
OUT_DIR = ROOT / "artifacts" / "dem3d_pair_damping_gap"
M_EFF_PAIR = effective_mass(
    DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3,
    DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3,
)
M_EFF_WALL = effective_mass(DENSITY * (4.0 / 3.0) * math.pi * RADIUS**3)


def _custom_engine(contact_damping: float, restitution_coefficient: float) -> IceDEM3D:
    """Two-sphere impact set-up with both damping knobs under control."""
    config = DEM3DConfig(
        nx=2, ny=2, nz=2, radius=RADIUS, density=DENSITY,
        contact_stiffness=STIFFNESS, contact_damping=contact_damping,
        platen_stiffness=STIFFNESS, platen_damping=contact_damping,
        restitution_coefficient=restitution_coefficient,
        drag=0.0, fix_x_edges=False, fix_bottom=False,
        bottom_platen_enabled=False, top_platen_enabled=False,
        tool_gap=100.0, tool_speed=1e-9,
    )
    sim = IceDEM3D(config)
    sim.alive[:] = False
    sim.failure_energy_J[:] = 0.0
    return sim


def measure_pair_with(contact_damping: float, restitution_coefficient: float) -> dict:
    """Pair impact with an explicit (contact_damping, restitution_coefficient)."""
    sim = _custom_engine(contact_damping, restitution_coefficient)
    ref._park_idle(sim)
    speed = ref.IMPACT_SPEED
    with torch.no_grad():
        sim.positions[0] = torch.tensor([0.0, 0.0, 0.0], dtype=sim.dtype)
        sim.positions[1] = torch.tensor([2.0 * RADIUS + 0.001, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[0] = torch.tensor([speed, 0.0, 0.0], dtype=sim.dtype)
        sim.velocities[1] = torch.tensor([-speed, 0.0, 0.0], dtype=sim.dtype)

    def overlap() -> float:
        return 2.0 * RADIUS - float(
            torch.linalg.vector_norm(sim.positions[1] - sim.positions[0])
        )

    def relative_velocity() -> float:
        return float(sim.velocities[1, 0] - sim.velocities[0, 0])

    out = ref._measure_impact(sim, overlap, relative_velocity)
    out["dt"] = sim.dt
    return out


def probe_contact_damping() -> list[dict]:
    rows = []
    for c in C_SWEEP:
        measured = measure_pair_with(c, 1.0)["restitution"]
        rows.append(
            {
                "contact_damping_Ns_m": c,
                "engine_e": measured,
                "reference_clipped_e": unilateral_restitution(c, STIFFNESS, M_EFF_PAIR),
                "bilateral_e": classical_restitution(c, STIFFNESS, M_EFF_PAIR),
                "engine_minus_reference": measured
                - unilateral_restitution(c, STIFFNESS, M_EFF_PAIR),
            }
        )
    return rows


def probe_restitution_coefficient() -> list[dict]:
    rows = []
    for target in E_TARGETS:
        c = damping_for_restitution(target, STIFFNESS, M_EFF_PAIR)
        measured = measure_pair_with(0.0, target)["restitution"]
        rows.append(
            {
                "target_e": target,
                "implied_damping_Ns_m": c,
                "engine_e": measured,
                "rel_error": (measured - target) / target,
                "within_3pct": abs(measured - target) / target < 0.03,
            }
        )
    return rows


def probe_platen() -> list[dict]:
    rows = []
    for c in C_SWEEP:
        measured = ref.measure_platen(c)["restitution"]
        rows.append(
            {
                "platen_damping_Ns_m": c,
                "engine_e": measured,
                "reference_clipped_e": unilateral_restitution(c, STIFFNESS, M_EFF_WALL),
            }
        )
    return rows


def implied_pair_damping(engine_e: float) -> float | None:
    """Invert the reference clipped law: which c_n would give this e?"""
    if engine_e >= 1.0 - 1e-6:
        return 0.0
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if unilateral_restitution(mid, STIFFNESS, M_EFF_PAIR) > engine_e:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def run_audit(write_artifacts: bool = True) -> dict:
    torch.set_num_threads(1)
    a = probe_contact_damping()
    b = probe_restitution_coefficient()
    c = probe_platen()
    audit = {
        "script": "scripts/audit_dem3d_pair_damping_gap.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "setup": {
            "radius_m": RADIUS,
            "density_kg_m3": DENSITY,
            "contact_stiffness_N_m": STIFFNESS,
            "impact_speed_m_s": ref.IMPACT_SPEED,
            "effective_mass_pair_kg": M_EFF_PAIR,
            "effective_mass_wall_kg": M_EFF_WALL,
        },
        "A_contact_damping_sweep": a,
        "B_restitution_coefficient_sweep": b,
        "C_platen_damping_control": c,
        "verdicts": {
            "contact_damping_controls_pair_e": (
                "yes" if a[-1]["engine_e"] < a[0]["engine_e"] - 0.2 else "NO (parameter is dead)"
            ),
            "implied_pair_damping_Ns_m": implied_pair_damping(a[-1]["engine_e"]),
            "restitution_coefficient_closes": (
                "yes" if all(row["within_3pct"] for row in b) else "no"
            ),
            "platen_branch_responds": (
                "yes" if c[-1]["engine_e"] < c[0]["engine_e"] - 0.2 else "no"
            ),
        },
    }
    if write_artifacts:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "audit.json").write_text(json.dumps(audit, indent=2))
        plot(audit)
    return audit


def plot(audit: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    a = audit["A_contact_damping_sweep"]
    cs = [row["contact_damping_Ns_m"] for row in a]
    axes[0].plot(cs, [row["engine_e"] for row in a], "o-", color="tab:red", ms=7,
                 label="engine, contact_damping swept")
    axes[0].plot(cs, [row["reference_clipped_e"] for row in a], "k--", lw=1.6,
                 label="documented law (clipped RK4)")
    axes[0].plot(cs, [row["bilateral_e"] for row in a], ":", color="tab:gray",
                 label="bilateral formula")
    axes[0].set_xlabel(r"$c_n = $ `contact_damping` (N s/m)")
    axes[0].set_ylabel("measured restitution $e$")
    axes[0].set_title("(a) `contact_damping` is ignored for pairs")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    b = audit["B_restitution_coefficient_sweep"]
    et = [row["target_e"] for row in b]
    axes[1].plot([0, 1], [0, 1], "k-", lw=1, alpha=0.4)
    axes[1].plot(et, [row["engine_e"] for row in b], "s-", color="tab:green", ms=8,
                 label=r"engine, `restitution_coefficient`=target")
    axes[1].set_xlabel("requested $e$ (`restitution_coefficient`)")
    axes[1].set_ylabel("measured $e$")
    axes[1].set_title("(b) the other knob does close")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    c = audit["C_platen_damping_control"]
    ps = [row["platen_damping_Ns_m"] for row in c]
    axes[2].plot(ps, [row["engine_e"] for row in c], "o-", color="tab:blue", ms=7,
                 label="engine, platen branch")
    axes[2].plot(ps, [row["reference_clipped_e"] for row in c], "k--", lw=1.6,
                 label="documented law (clipped RK4)")
    axes[2].set_xlabel(r"$c_n = $ `platen_damping` (N s/m)")
    axes[2].set_ylabel("measured restitution $e$")
    axes[2].set_title("(c) platen control: branch does respond")
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "IceDEM3D pair-contact damping plumbing: where the documented closure breaks",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pair_damping_gap.png", dpi=160)
    plt.close(fig)


def render_markdown(audit: dict) -> str:
    lines = [
        "| c_n in `contact_damping` (N s/m) | engine e | documented law e | bilateral e |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in audit["A_contact_damping_sweep"]:
        lines.append(
            "| %.1f | %.5f | %.5f | %.5f |"
            % (
                row["contact_damping_Ns_m"],
                row["engine_e"],
                row["reference_clipped_e"],
                row["bilateral_e"],
            )
        )
    lines += [
        "",
        "| target e (`restitution_coefficient`) | engine e | rel. error | <3 % ? |",
        "| ---: | ---: | ---: | --- |",
    ]
    for row in audit["B_restitution_coefficient_sweep"]:
        lines.append(
            "| %.1f | %.5f | %+.3f %% | %s |"
            % (
                row["target_e"],
                row["engine_e"],
                100.0 * row["rel_error"],
                "yes" if row["within_3pct"] else "no",
            )
        )
    lines += [
        "",
        "| `platen_damping` (N s/m) | engine e | documented law e |",
        "| ---: | ---: | ---: |",
    ]
    for row in audit["C_platen_damping_control"]:
        lines.append(
            "| %.1f | %.5f | %.5f |"
            % (row["platen_damping_Ns_m"], row["engine_e"], row["reference_clipped_e"])
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps(result["verdicts"], indent=2))
    print(render_markdown(result))
    print("artifacts ->", OUT_DIR)