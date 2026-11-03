"""Reproduce the 2-D impulse/energy ledger validation study.

Runs a small tool-indentation case, checks the integral momentum balance and the
integral energy balance against closed-form budgets, and writes machine-readable
numbers plus figures under ``outputs/accounting_ledger``.

Usage:  python scripts/validate_accounting_ledger.py [--output DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from tensordem import DEMConfig, IceDEM

NX, NY = 7, 3
DURATION_S = 0.12
IMPULSE_CHANNELS = ("external_impulse_Ns", "support_impulse_Ns",
                    "drag_impulse_Ns", "tool_impulse_on_ice_Ns")


def momentum_trace(dt_scale: float = 1.0, duration: float = DURATION_S) -> dict:
    """Per-step worst-case relative momentum-balance error."""
    dt0 = IceDEM(DEMConfig(nx=NX, ny=NY, tool_speed=0.2)).dt
    config = DEMConfig(nx=NX, ny=NY, tool_speed=0.2, dt=dt0 * dt_scale)
    sim = IceDEM(config)
    errors = []
    for _ in range(int(round(duration / sim.dt))):
        sim.step()
        momentum = config.mass * sim.velocities.sum(dim=0)
        ledger = sum(torch.as_tensor(sim.accounting[name]) for name in IMPULSE_CHANNELS)
        reference = max(float(momentum.abs().max()), float(ledger.abs().max()), 1e-30)
        errors.append(float((momentum - ledger).abs().max()) / reference)
    return dict(dt=sim.dt, steps=len(errors), max_relative_error=max(errors),
                mean_relative_error=sum(errors) / len(errors))


def energy_case(dt_scale: float = 1.0, duration: float = DURATION_S) -> dict:
    """Energy ledger terms at one time step, raw and after correcting the credit."""
    dt0 = IceDEM(DEMConfig(nx=NX, ny=NY, tool_speed=0.2)).dt
    config = DEMConfig(nx=NX, ny=NY, tool_speed=0.2, dt=dt0 * dt_scale)
    sim = IceDEM(config)
    energy0 = sim.mechanical_energy()["mechanical_J"]
    for _ in range(int(round(duration / sim.dt))):
        sim.step()
    energy1 = sim.mechanical_energy()["mechanical_J"]
    acc = dict(sim.accounting)
    raw = ((energy1 - energy0) - acc["tool_work_J"] - acc["external_work_J"]
           + acc["damping_dissipation_J"] + acc["fracture_release_J"])
    closed = raw - acc["pair_contact_acquired_J"]
    scale = (abs(acc["tool_work_J"]) + abs(acc["damping_dissipation_J"])
             + abs(energy1 - energy0))
    radius = config.radius
    overlap = (2 * radius - (sim.rest_lengths + sim.failure_extension_m)).clamp_min(0.0)
    broken = sim.failure_energy_J > 0.0
    analytic_retained = float(
        (0.5 * config.contact_stiffness * overlap[broken].square()).sum())
    return dict(
        dt=sim.dt, steps=sim.step_count, broken_bonds=sim.broken_bonds,
        delta_mechanical_J=energy1 - energy0,
        tool_work_J=acc["tool_work_J"],
        external_work_J=acc["external_work_J"],
        damping_dissipation_J=acc["damping_dissipation_J"],
        fracture_release_J=acc["fracture_release_J"],
        pair_contact_acquired_J=acc["pair_contact_acquired_J"],
        analytic_pair_contact_acquired_J=analytic_retained,
        retained_fraction_of_release=(
            acc["pair_contact_acquired_J"] / acc["fracture_release_J"]
            if acc["fracture_release_J"] else 0.0),
        scale_J=scale,
        raw_residual_J=raw, raw_residual_relative=raw / scale,
        closed_residual_J=closed, closed_residual_relative=closed / scale,
    )


def figure(momentum: dict, cases: list[dict], output: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax = axes[0]
    dts = [c["dt"] for c in cases]
    raw = [abs(c["raw_residual_relative"]) for c in cases]
    closed = [abs(c["closed_residual_relative"]) for c in cases]
    ax.loglog(dts, raw, "o-", label="raw ledger (fracture release as dissipated)")
    ax.loglog(dts, closed, "s-", label="+ pair-contact credit")
    ax.loglog(dts, [raw[0] * d / dts[0] for d in dts], "k--", lw=0.8, label="O(dt) reference")
    ax.set_xlabel("time step dt [s]")
    ax.set_ylabel(r"|energy residual| / energy scale")
    ax.set_title("Energy balance: dt-independence of the defect")
    ax.grid(True, which="both", ls=":", lw=0.5)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["reported\nfracture release", "retained as pair\ncontact", "net dissipated"]
    c = cases[0]
    values = [c["fracture_release_J"], -c["pair_contact_acquired_J"],
              c["fracture_release_J"] - c["pair_contact_acquired_J"]]
    ax.bar(labels, values, color=["tab:red", "tab:blue", "tab:green"])
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_ylabel("energy [J]")
    ax.set_title(f"Attribution (dt={c['dt']:.2e} s, {c['broken_bonds']} bonds)")
    for i, v in enumerate(values):
        ax.annotate(f"{v:+.3e}", (i, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    path = output / "ledger_validation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogy([1.0, 1.0], [momentum["min_relative_error"], momentum["max_relative_error"]])
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1.0])
    ax.set_xticklabels(["momentum balance"])
    ax.set_ylabel(r"relative error $|m\sum v - \sum I| / \max$")
    ax.set_title(f"Integral momentum identity over {momentum['steps']} steps (roundoff)")
    ax.grid(True, axis="y", ls=":", lw=0.5)
    fig.tight_layout()
    path = output / "momentum_balance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/accounting_ledger"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    momentum = momentum_trace()
    # Re-run the trace keeping the per-step extremes for the figure.
    dt0 = IceDEM(DEMConfig(nx=NX, ny=NY, tool_speed=0.2)).dt
    config = DEMConfig(nx=NX, ny=NY, tool_speed=0.2, dt=dt0)
    sim = IceDEM(config)
    errors = []
    for _ in range(int(round(DURATION_S / sim.dt))):
        sim.step()
        momentum_vector = config.mass * sim.velocities.sum(dim=0)
        ledger = sum(torch.as_tensor(sim.accounting[name]) for name in IMPULSE_CHANNELS)
        reference = max(float(momentum_vector.abs().max()), float(ledger.abs().max()), 1e-30)
        errors.append(float((momentum_vector - ledger).abs().max()) / reference)
    momentum["min_relative_error"] = min(errors)

    cases = [energy_case(1.0), energy_case(0.5), energy_case(0.25)]
    paths = figure(momentum, cases, args.output)
    report = dict(
        schema="tensordem.accounting-ledger-validation/1",
        case=dict(nx=NX, ny=NY, duration_s=DURATION_S, tool_speed_m_s=0.2),
        momentum_balance=momentum,
        energy_cases=cases,
        raw_residual_ratios=[abs(cases[i]["raw_residual_J"]) / abs(cases[i + 1]["raw_residual_J"])
                             for i in range(len(cases) - 1)],
        closed_residual_ratios=[abs(cases[i]["closed_residual_J"]) / abs(cases[i + 1]["closed_residual_J"])
                                for i in range(len(cases) - 1)],
        figures=[p.name for p in paths],
    )
    (args.output / "study.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(
        momentum_max_rel=momentum["max_relative_error"],
        raw_rel=[c["raw_residual_relative"] for c in cases],
        closed_rel=[c["closed_residual_relative"] for c in cases],
        retained_fraction=[c["retained_fraction_of_release"] for c in cases],
        analytic_match=max(abs(c["pair_contact_acquired_J"] - c["analytic_pair_contact_acquired_J"])
                           for c in cases),
        raw_ratios=report["raw_residual_ratios"],
        closed_ratios=report["closed_residual_ratios"],
    ), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())