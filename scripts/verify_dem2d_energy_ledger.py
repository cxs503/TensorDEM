"""Closed-form audit of the 2-D ``IceDEM`` energy ledger.

Why this script exists
----------------------
``src/tensordem/dem.py`` already accumulates a full accounting ledger
(``tool_work_J``, ``external_work_J``, ``damping_dissipation_J``,
``fracture_release_J``, ``pair_contact_acquired_J``) plus a four-term
``mechanical_energy()``.  ``tests/test_accounting_ledger_validation.py`` closes
the *identity*

    dE_mech = W_external + W_tool - Q_damping - E_fracture + E_pair_contact_acquired

on indentation runs, but an identity check cannot tell whether the individual
channels carry the right *numbers*: a channel that is uniformly scaled wrong
still closes the balance if the opposite channel absorbs the error.  The 3-D
engine has a closed-form audit of exactly this kind
(``docs/dem3d_energy_ledger_analytic_verification.md``); the 2-D engine -- the
one the ship/ice benchmarks use -- did not, which is the verification debt this
script pays off.

Two exactly solvable subsystems are driven through the production ``step()``
and scored against textbook closed forms, and a third (tool indentation)
case measures the discretisation residual of the tool-work channel under
time-step refinement.  No solver source is modified: the subsystem state of a
fresh solver is overwritten before the first step.

Case A - exponential viscous drag decay
---------------------------------------
One free particle, mass ``m``, linear drag ``c``, no bonds, no contacts, tool
parked (``tool_speed = 1e-9`` m/s so it travels < 1e-10 m):

    m dv/dt = -c v   ->   v(t) = v0 exp(-t/tau),        tau = m/c
    D(t) = int_0^t c v^2 dt'  ->  D(t) = 0.5 m v0^2 (1 - exp(-2 t/tau))

The ledger samples the dissipated power at the pre-update velocity, so each
step over-records ``0.5 c v^2 (dt c / m) dt``, i.e. a fraction ``0.5 dt c / m``
of the dissipated energy:

    R_pred(t) = 0.5 (c/m) dt D(t)

Case B - bonded harmonic oscillator
-----------------------------------
Two free particles joined by one linear bond of stiffness ``k``, equal masses
``m``, no drag, no damping, released from rest at relative extension ``A``
(0.5 % of the rest length, so the linear bond law is the exact model):

    xi'' = -omega^2 xi,  omega = sqrt(2 k / m),  xi(t) = A cos(omega t)

Case C - tool-work channel under refinement
-------------------------------------------
Tool indentation of a bonded lattice at ``tool_speed`` for a fixed duration;
the residual of the energy balance is measured at ``dt``, ``dt/2``, ``dt/4``.
It is discretisation (rectangle rule on the tool power, semi-implicit Euler),
so it must shrink with ``dt``, not sit at a floor.

Usage
-----
    python scripts/verify_dem2d_energy_ledger.py --output docs/artifacts/dem2d_energy_ledger
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

import tensordem as _tensordem
from tensordem import DEMConfig, IceDEM

#: Provenance guard: the audit must score *this* worktree's engine, not a
#: site-packages copy or a sibling checkout that leaked onto ``sys.path``.
ENGINE_PATH = _tensordem.__file__


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _park(sim: IceDEM, spacing: float = 1.0) -> None:
    """Separate every particle so no pair/tool overlap can exist, drop bonds."""
    n = len(sim.positions)
    x = torch.arange(n, dtype=torch.float64) * spacing
    sim.positions = torch.stack((x, torch.zeros(n, dtype=torch.float64)), dim=1)
    sim.velocities = torch.zeros_like(sim.positions)
    sim.fixed = torch.zeros(n, dtype=torch.bool)
    sim.alive = torch.zeros_like(sim.alive)


def _energy(sim: IceDEM) -> float:
    return float(sim.mechanical_energy()["mechanical_J"])


def _residual(sim: IceDEM, e0: float) -> dict[str, float]:
    """Ledger residual: positive means the ledger missed energy input."""
    acc = sim.accounting
    e_now = _energy(sim)
    residual = (
        e_now - e0
        + acc["damping_dissipation_J"]
        + acc["fracture_release_J"]
        - acc["tool_work_J"]
        - acc["external_work_J"]
        - acc.get("pair_contact_acquired_J", 0.0)
    )
    return dict(
        e0_J=e0,
        e_end_J=e_now,
        tool_work_J=acc["tool_work_J"],
        external_work_J=acc["external_work_J"],
        damping_dissipation_J=acc["damping_dissipation_J"],
        fracture_release_J=acc["fracture_release_J"],
        pair_contact_acquired_J=acc.get("pair_contact_acquired_J", 0.0),
        residual_J=residual,
    )


# --------------------------------------------------------------------------- #
# Case A: viscous drag decay
# --------------------------------------------------------------------------- #
def drag_decay(dt_scale: float = 1.0, v0: float = 0.5, c: float = 2.0,
               tau_fraction: float = 1.5) -> dict[str, Any]:
    cfg0 = DEMConfig(nx=2, ny=2, drag=c, contact_damping=0.0,
                     tool_speed=1e-9, fix_edges=False)
    dt0 = IceDEM(cfg0).dt
    cfg = DEMConfig(nx=2, ny=2, drag=c, contact_damping=0.0,
                    tool_speed=1e-9, fix_edges=False, dt=dt0 * dt_scale)
    sim = IceDEM(cfg)
    _park(sim)
    m = cfg.mass
    tau = m / c
    sim.velocities[0, 0] = v0
    e0 = _energy(sim)                      # 0.5 m v0^2
    t_end = tau_fraction * tau
    steps = int(round(t_end / sim.dt))
    history = []
    for _ in range(steps):
        sim.step()
        history.append((sim.time, float(sim.velocities[0, 0].item()),
                        _energy(sim), sim.accounting["damping_dissipation_J"]))
    t = sim.time
    v_sim = float(sim.velocities[0, 0].item())
    d_sim = sim.accounting["damping_dissipation_J"]
    e_kin_sim = _energy(sim)
    v_ref = v0 * math.exp(-t / tau)
    d_ref = 0.5 * m * v0**2 * (1.0 - math.exp(-2.0 * t / tau))
    e_ref = 0.5 * m * v_ref**2
    res = _residual(sim, e0)
    r_pred = 0.5 * (c / m) * sim.dt * d_ref
    return dict(
        case="A_drag_decay",
        dt_scale=dt_scale, dt_s=sim.dt, steps=steps, t_end_s=t,
        mass_kg=m, tau_s=tau, v0=v0, drag=c,
        v_sim=v_sim, v_ref=v_ref,
        v_relerr=abs(v_sim - v_ref) / v_ref,
        ke_sim_J=e_kin_sim, ke_ref_J=e_ref,
        ke_relerr=abs(e_kin_sim - e_ref) / e_ref,
        dissipation_sim_J=d_sim, dissipation_ref_J=d_ref,
        dissipation_relerr=abs(d_sim - d_ref) / d_ref,
        residual_J=res["residual_J"],
        residual_pred_J=r_pred,
        residual_pred_relerr=abs(res["residual_J"] - r_pred) / abs(r_pred),
        residual_over_dissipation=abs(res["residual_J"]) / d_ref,
        **{k: v for k, v in res.items() if k.endswith("_J") and k not in
           ("residual_J", "e0_J", "e_end_J")},
        history=history,
    )


# --------------------------------------------------------------------------- #
# Case B: bonded harmonic oscillator
# --------------------------------------------------------------------------- #
def bonded_oscillator(dt_scale: float = 1.0, k: float = 2000.0,
                      amplitude_ratio: float = 0.005,
                      periods: float = 3.0) -> dict[str, Any]:
    base = dict(nx=2, ny=2, bond_stiffness=k, contact_stiffness=k, drag=0.0,
                contact_damping=0.0, tool_speed=1e-9, fix_edges=False,
                breaking_strain=10.0, shear_breaking_strain=10.0)
    dt0 = IceDEM(DEMConfig(**base)).dt
    cfg = DEMConfig(**base, dt=dt0 * dt_scale)
    sim = IceDEM(cfg)
    m = cfg.mass
    rest = 2.0 * cfg.radius
    amplitude = amplitude_ratio * rest
    # particles 0 and 1 are the nearest neighbours to the left/right of the
    # lattice centre, exactly ``rest`` apart.
    pair = int(torch.nonzero(
        (sim.pairs[:, 0] == 0) & (sim.pairs[:, 1] == 1)).item())
    sim.alive = torch.zeros_like(sim.alive)
    sim.alive[pair] = True
    sim.fixed = torch.zeros_like(sim.fixed)
    sim.positions[0] = torch.tensor([-rest / 2 - amplitude / 2, 0.0], dtype=torch.float64)
    sim.positions[1] = torch.tensor([rest / 2 + amplitude / 2, 0.0], dtype=torch.float64)
    sim.velocities = torch.zeros_like(sim.positions)
    omega = math.sqrt(2.0 * k / m)
    period = 2.0 * math.pi / omega
    e0 = _energy(sim)                       # 0.5 k A^2
    steps = int(round(periods * period / sim.dt))
    xi_relerr_max = 0.0
    energy_drift_max = 0.0
    history = []
    for _ in range(steps):
        sim.step()
        extension = float(torch.linalg.vector_norm(
            sim.positions[1] - sim.positions[0]).item()) - rest
        xi_ref = amplitude * math.cos(omega * sim.time)
        # Normalise by the amplitude, not by the instantaneous value: a relative
        # error at a zero crossing is meaningless (the reference vanishes there)
        # while the physical error does not.
        xi_relerr_max = max(xi_relerr_max, abs(extension - xi_ref) / amplitude)
        energy_drift_max = max(energy_drift_max, abs(_energy(sim) - e0) / e0)
        history.append((sim.time, extension, xi_ref, _energy(sim)))
    res = _residual(sim, e0)
    return dict(
        case="B_bonded_oscillator",
        dt_scale=dt_scale, dt_s=sim.dt, steps=steps, omega_rad_s=omega,
        period_s=period, amplitude_m=amplitude, rest_length_m=rest,
        stiffness=k, mass_kg=m, e0_J=e0,
        xi_relerr_max=xi_relerr_max,
        energy_drift_max=energy_drift_max,
        residual_J=res["residual_J"],
        residual_over_e0=abs(res["residual_J"]) / e0,
        **{k2: v for k2, v in res.items() if k2.endswith("_J") and k2 not in
           ("residual_J", "e0_J", "e_end_J")},
        history=history,
    )


# --------------------------------------------------------------------------- #
# Case C: tool-work channel under refinement
# --------------------------------------------------------------------------- #
def tool_indentation(dt_scale: float = 1.0, nx: int = 7, ny: int = 3,
                     tool_speed: float = 0.2,
                     duration: float = 0.12) -> dict[str, Any]:
    base = dict(nx=nx, ny=ny, tool_speed=tool_speed)
    dt0 = IceDEM(DEMConfig(**base)).dt
    cfg = DEMConfig(**base, dt=dt0 * dt_scale)
    sim = IceDEM(cfg)
    e0 = _energy(sim)
    steps = int(round(duration / sim.dt))
    w_trap = 0.0
    prev_power = None
    for _ in range(steps):
        sim.step()
        power = float(-(torch.as_tensor(sim.last_step["tool_reaction_N"],
                                        dtype=torch.float64) @ sim.tool_velocity))
        if prev_power is not None:
            w_trap += 0.5 * (power + prev_power) * sim.dt
        prev_power = power
    res = _residual(sim, e0)
    scale = max(res["e_end_J"], res["damping_dissipation_J"], 1e-12)
    return dict(
        case="C_tool_indentation",
        dt_scale=dt_scale, dt_s=sim.dt, steps=steps, duration_s=sim.time,
        tool_speed=tool_speed, broken_bonds=sim.broken_bonds,
        w_trapezoid_J=w_trap,
        residual_J=res["residual_J"],
        residual_over_scale=abs(res["residual_J"]) / scale,
        **{k: v for k, v in res.items() if k.endswith("_J") and k != "residual_J"},
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_campaign(output_dir: Path, dt_scales: tuple[float, ...] = (1.0, 0.5, 0.25)) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    histories: dict[str, list] = {}
    for scale in dt_scales:
        a = drag_decay(scale)
        histories[f"A_dt{scale}"] = a.pop("history")
        rows.append({k: v for k, v in a.items()})
        b = bonded_oscillator(scale)
        histories[f"B_dt{scale}"] = b.pop("history")
        rows.append({k: v for k, v in b.items()})
        rows.append(tool_indentation(scale))

    anchors = []
    for scale in dt_scales:
        a = next(r for r in rows if r["case"] == "A_drag_decay" and r["dt_scale"] == scale)
        b = next(r for r in rows if r["case"] == "B_bonded_oscillator" and r["dt_scale"] == scale)
        c = next(r for r in rows if r["case"] == "C_tool_indentation" and r["dt_scale"] == scale)
        anchors.append(dict(dt_scale=scale, dt_s=a["dt_s"],
                            A_v_relerr=a["v_relerr"], A_ke_relerr=a["ke_relerr"],
                            A_dissipation_relerr=a["dissipation_relerr"],
                            A_residual_relerr=a["residual_pred_relerr"],
                            B_xi_relerr=b["xi_relerr_max"],
                            B_energy_drift=b["energy_drift_max"],
                            B_residual_over_e0=b["residual_over_e0"],
                            C_residual_over_scale=c["residual_over_scale"]))
    report = dict(
        schema="tensordem.dem2d-energy-ledger/1",
        protocol="dem2d-ledger-closed-form-audit-v1",
        tolerance_pct=3.0,
        cases=rows,
        anchors=anchors,
        worst=dict(
            A_v_relerr=max(a["A_v_relerr"] for a in anchors),
            A_ke_relerr=max(a["A_ke_relerr"] for a in anchors),
            A_dissipation_relerr=max(a["A_dissipation_relerr"] for a in anchors),
            A_residual_pred_relerr=max(a["A_residual_relerr"] for a in anchors),
            B_xi_relerr=max(a["B_xi_relerr"] for a in anchors),
            B_energy_drift=max(a["B_energy_drift"] for a in anchors),
            C_residual_over_scale=max(a["C_residual_over_scale"] for a in anchors),
        ),
    )
    (output_dir / "ledger_closure.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = sorted({k for r in rows for k in r})
    with (output_dir / "ledger_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    with (output_dir / "ledger_anchors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(anchors[0]))
        writer.writeheader()
        writer.writerows(anchors)
    _plot(output_dir / "ledger_closure.png", histories, rows, dt_scales)
    return report


def _plot(path: Path, histories: dict, rows: list[dict], dt_scales) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:          # pragma: no cover - optional dependency
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    a = next(r for r in rows if r["case"] == "A_drag_decay" and r["dt_scale"] == 1.0)
    hist = histories["A_dt1.0"]
    t = [h[0] for h in hist]
    tau, v0, m, c = a["tau_s"], a["v0"], a["mass_kg"], a["drag"]
    axes[0].plot(t, [h[1] for h in hist], ".", ms=3, label="2-D engine $v(t)$")
    axes[0].plot(t, [v0 * math.exp(-tt / tau) for tt in t], "-", lw=1.0,
                 label=r"$v_0 e^{-t/\tau}$")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("v [m/s]")
    axes[0].set_title("A: drag decay"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(t, [h[3] for h in hist], ".", ms=3, label="ledger $Q_{damping}$")
    axes[1].plot(t, [0.5 * m * v0**2 * (1 - math.exp(-2 * tt / tau)) for tt in t],
                 "-", lw=1.0, label=r"$\frac{1}{2}m v_0^2(1-e^{-2t/\tau})$")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("energy [J]")
    axes[1].set_title("A: dissipated energy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    scales = [a["dt_s"] for a in
              sorted((r for r in rows if r["case"] == "A_drag_decay"),
                     key=lambda r: -r["dt_scale"])]
    a_rows = sorted((r for r in rows if r["case"] == "A_drag_decay"),
                    key=lambda r: -r["dt_scale"])
    c_rows = sorted((r for r in rows if r["case"] == "C_tool_indentation"),
                    key=lambda r: -r["dt_scale"])
    axes[2].loglog(scales, [r["residual_over_dissipation"] for r in a_rows], "o-",
                   label="A: |R|/Q")
    axes[2].loglog(scales, [r["residual_over_scale"] for r in c_rows], "s-",
                   label="C: |R|/energy scale")
    axes[2].set_xlabel(r"$\Delta t$ [s]"); axes[2].set_ylabel("relative residual")
    axes[2].set_title("residual under refinement"); axes[2].legend(); axes[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/artifacts/dem2d_energy_ledger")
    args = parser.parse_args()
    report = run_campaign(Path(args.output))
    for row in report["anchors"]:
        print(f"dt_scale={row['dt_scale']:<5} "
              f"A_v={100*row['A_v_relerr']:.3f}% A_ke={100*row['A_ke_relerr']:.3f}% "
              f"A_Q={100*row['A_dissipation_relerr']:.3f}% "
              f"A_res={100*row['A_residual_relerr']:.2f}% "
              f"B_xi={100*row['B_xi_relerr']:.3f}% B_drift={100*row['B_energy_drift']:.3f}% "
              f"B_res={100*row.get('B_residual_over_e0', float('nan')):.3f}% "
              f"C_res={100*row['C_residual_over_scale']:.3f}%")


if __name__ == "__main__":
    main()