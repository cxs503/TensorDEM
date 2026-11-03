"""Audit the two public energy-reporting APIs of the DEM3D solver.

Motivation
----------
``DEM3D`` exposes the same energy channels twice:

* ``diagnostics()`` -- the per-step report consumed by the CLI / CSV writers;
* ``energy()``      -- the standalone snapshot used by scripts and by the
  calibration loop.

Both dictionaries name a channel ``released_bond_energy_J``, but they computed it
from two different quantities:

* ``diagnostics()``: ``self.failure_energy_J.sum()``, i.e. the *elastic energy
  stored in each bond at the instant it broke* (an irreversible, monotone
  accumulator written by ``_commit_fracture``);
* ``energy()``: ``0.5 * k * (extension[~alive] ** 2).sum()``, i.e. the energy of
  the *current* length of every dead bond.

After a bond breaks its two particles are free; the separation keeps growing, so
the second expression grows without bound and is not a released energy at all.
This script measures the divergence on a real indentation run and writes the
evidence used in ``docs/dem3d_energy_api_consistency.md``.

Outputs (``artifacts/dem3d_energy_api/``):
    energy_api_divergence.csv   per-sample channel comparison
    energy_api_audit.json       protocol + verdict + summary numbers
    energy_api_divergence.png   released-energy channels and their drift
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tensordem.dem3d import IceDEM3D, DEM3DConfig  # noqa: E402

OUT = ROOT / "artifacts" / "dem3d_energy_api"

SHARED_CHANNELS = (
    "broken_bonds",
    "particle_count",
    "kinetic_energy_J",
    "bond_elastic_energy_J",
    "particle_contact_energy_J",
    "tool_contact_energy_J",
    "platen_contact_energy_J",
    "released_bond_energy_J",
    "mechanical_energy_J",
)


def make_solver() -> IceDEM3D:
    """Indentation case that reliably breaks a handful of bonds."""
    cfg = DEM3DConfig(
        nx=6,
        ny=3,
        nz=3,
        breaking_strain=0.004,       # low threshold: first breakage within ~1e2 steps
        contact_damping=0.0,
        drag=0.0,
        tool_speed=0.5,
    )
    return IceDEM3D(cfg)


def released_from_extension(sim: IceDEM3D) -> float:
    """The (incorrect) current-extension proxy formerly reported by ``energy()``."""
    extension = torch.linalg.vector_norm(
        sim.positions[sim.pairs[:, 1]] - sim.positions[sim.pairs[:, 0]], dim=1
    ) - sim.rest_lengths
    return 0.5 * sim.config.bond_stiffness * (extension[~sim.alive] ** 2).sum().item()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sim = make_solver()
    max_steps = 3000
    post_break_samples = 900

    first_break_step: int | None = None
    rows: list[dict[str, float]] = []
    for step in range(max_steps):
        if sim.broken_bonds and first_break_step is None:
            first_break_step = step
        if first_break_step is not None and step - first_break_step > post_break_samples:
            break
        sim.step()
        if first_break_step is None:
            if step % 50:
                continue
        elif (step - first_break_step) % 10:
            continue

        diag = sim.diagnostics()
        snap = sim.energy()
        row: dict[str, float] = {
            "step": step,
            "steps_after_first_break": -1 if first_break_step is None else step - first_break_step,
            "broken_bonds": diag["broken_bonds"],
            "released_ledger_J": diag["released_bond_energy_J"],
            "released_extension_J": released_from_extension(sim),
            "energy_api_released_J": snap["released_bond_energy_J"],
            "mechanical_energy_diag_J": diag["mechanical_energy_J"],
            "mechanical_energy_api_J": snap["mechanical_energy_J"],
        }
        for name in SHARED_CHANNELS:
            if name in ("released_bond_energy_J",):
                continue
            a, b = diag[name], snap[name]
            if isinstance(a, float) or isinstance(b, float):
                row[f"abs_diff_{name}"] = abs(float(a) - float(b))
            else:
                row[f"abs_diff_{name}"] = float(abs(float(a) - float(b)))
        rows.append(row)

    if first_break_step is None:
        raise SystemExit("audit case never broke a bond; nothing to compare")

    # Channel-by-channel agreement on the last sample (identical solver state).
    final_diag = sim.diagnostics()
    final_snap = sim.energy()
    channel_max_abs_diff = {
        name: abs(float(final_diag[name]) - float(final_snap[name])) for name in SHARED_CHANNELS
    }

    post = [r for r in rows if r["steps_after_first_break"] >= 0]
    ledger0 = post[0]["released_ledger_J"]
    extension_drift = post[-1]["released_extension_J"] - post[0]["released_extension_J"]
    ledger_drift = post[-1]["released_ledger_J"] - post[0]["released_ledger_J"]
    # The ledger may still grow while *new* bonds break; the proxy grows after
    # breakage stops as well, which is the defect.  Quantify both.
    frozen = [r for r in post[-50:]]
    late_proxy_growth = frozen[-1]["released_extension_J"] - frozen[0]["released_extension_J"]
    late_ledger_growth = frozen[-1]["released_ledger_J"] - frozen[0]["released_ledger_J"]

    verdict = {
        "shared_channels_agree": all(v <= 1e-9 * max(1.0, abs(float(final_diag[k])))
                                    for k, v in channel_max_abs_diff.items()),
        "channel_max_abs_diff": channel_max_abs_diff,
        "first_break_step": first_break_step,
        "broken_bonds_final": sim.broken_bonds,
        "released_ledger_first_J": ledger0,
        "released_ledger_last_J": post[-1]["released_ledger_J"],
        "proxy_first_after_break_J": post[0]["released_extension_J"],
        "proxy_last_J": post[-1]["released_extension_J"],
        "proxy_total_drift_J": extension_drift,
        "ledger_total_drift_J": ledger_drift,
        "late_window_steps": len(frozen),
        "late_window_proxy_growth_J": late_proxy_growth,
        "late_window_ledger_growth_J": late_ledger_growth,
        "proxy_relative_excess": (post[-1]["released_extension_J"] / ledger0 - 1.0) if ledger0 else None,
    }

    with (OUT / "energy_api_divergence.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "protocol": "tensordem-dem3d-energy-api-consistency-audit-v1",
        "config": {k: v for k, v in sim.config.__dict__.items() if not k.startswith("_")},
        "verdict": verdict,
        "notes": [
            "released_ledger_J is IceDEM3D.failure_energy_J.sum(): per-bond elastic energy at failure time",
            "released_extension_J is the former energy() proxy: 0.5*k*sum(extension^2 over dead bonds)",
            "energy_api_released_J is IceDEM3D.energy()['released_bond_energy_J'] as shipped",
        ],
    }
    (OUT / "energy_api_audit.json").write_text(json.dumps(payload, indent=2, default=str))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [r["steps_after_first_break"] for r in post]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
        ax = axes[0]
        ax.plot(steps, [r["released_ledger_J"] for r in post], label="ledger (failure_energy_J.sum)")
        ax.plot(steps, [r["released_extension_J"] for r in post], "--",
                label="extension proxy (former energy())")
        ax.set_xlabel("steps after first bond break")
        ax.set_ylabel("released bond energy (J)")
        ax.set_title("released-energy channel: ledger vs length proxy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1]
        ax.plot(steps, [r["released_extension_J"] - r["released_ledger_J"] for r in post], color="crimson")
        ax.set_xlabel("steps after first bond break")
        ax.set_ylabel("proxy - ledger (J)")
        ax.set_title("spurious released energy reported by energy()")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "energy_api_divergence.png", dpi=140)
    except Exception as exc:  # pragma: no cover - figure is optional
        payload["figure_error"] = repr(exc)
        (OUT / "energy_api_audit.json").write_text(json.dumps(payload, indent=2, default=str))

    print(json.dumps(verdict, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())