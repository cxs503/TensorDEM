"""Generate the m4790 evidence figure + machine-readable residual table.

This is a test-verification (backend) task with no physical field quantity, so
the evidence figure is the *residual sensitivity* of the 2-D DEM energy /
fracture ledger balance: the uncorrupted ledger must close (<0.5 %), and every
deliberately corrupted channel must push the residual above the tolerance
(i.e. the gate must detect it).

The same numbers are written to ``artifacts/m4790/sensitivity.json`` so that
reports can inject them programmatically instead of transcribing them.

Usage::

    python scripts/plot_m4790_sensitivity.py
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem import DEMConfig, IceDEM  # noqa: E402

DURATION = 0.12
LATTICE = (7, 3)
TOL = 0.005  # 0.5 %
OUT_DIR = ROOT / "artifacts" / "m4790"

#: (label, channel, kind, value); kind is "factor" or "offset"
CASES = [
    ("fracture_release\nx0", "fracture_release_J", "factor", 0.0),
    ("fracture_release\nx3", "fracture_release_J", "factor", 3.0),
    ("pair_contact\nx0", "pair_contact_acquired_J", "factor", 0.0),
    ("pair_contact\nx3", "pair_contact_acquired_J", "factor", 3.0),
    ("tool_work\nx0", "tool_work_J", "factor", 0.0),
    ("damping\nx0", "damping_dissipation_J", "factor", 0.0),
    ("external_work\n+W_tool", "external_work_J", "offset", None),
]


def run_indentation() -> IceDEM:
    """Run the tool-indentation scenario that breaks bonds."""
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    sim = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0))
    for _ in range(int(round(DURATION / sim.dt))):
        sim.step()
    return sim


def initial_mechanical_energy() -> float:
    """Mechanical energy of the untouched configuration (residual datum)."""
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    return IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0)) \
        .mechanical_energy()["mechanical_J"]


def residual(sim: IceDEM, e0: float) -> float:
    """dE_mech - W_tool - W_external + Q_damping + E_fracture - E_pair_contact."""
    e1 = sim.mechanical_energy()["mechanical_J"]
    a = sim.accounting
    return (e1 - e0) - a["tool_work_J"] - a["external_work_J"] \
        + a["damping_dissipation_J"] + a["fracture_release_J"] \
        - a["pair_contact_acquired_J"]


def scale(sim: IceDEM) -> float:
    """Reference scale of the balance (the three active magnitude channels)."""
    a = sim.accounting
    return abs(a["tool_work_J"]) + abs(a["damping_dissipation_J"]) \
        + abs(a["fracture_release_J"])


def rel_err(sim: IceDEM, e0: float) -> float:
    return abs(residual(sim, e0)) / max(scale(sim), 1e-18)


def main() -> None:
    sim = run_indentation()
    e0 = initial_mechanical_energy()
    base = rel_err(sim, e0)
    channels = dict(sim.accounting)

    rows, labels, values, detected = [], [], [], []
    for label, channel, kind, value in CASES:
        c = copy.deepcopy(sim)
        injected = channels["tool_work_J"] if value is None else value
        if kind == "factor":
            c.accounting[channel] *= injected
        else:
            c.accounting[channel] += injected
        vacuous = c.accounting[channel] == channels[channel]
        err = rel_err(c, e0)
        rows.append({
            "label": label.replace("\n", " "),
            "channel": channel,
            "fault_kind": kind,
            "fault_value": injected,
            "channel_value": channels[channel],
            "corrupted_channel_value": c.accounting[channel],
            "vacuous": vacuous,
            "rel_err": err,
            "detected": (err >= TOL) and not vacuous,
        })
        labels.append(label)
        values.append(err)
        detected.append((err >= TOL) and not vacuous)

    table = {
        "generated_by": "scripts/plot_m4790_sensitivity.py",
        "scenario": {
            "case": "2-D ice tool indentation (bonded particle)",
            "lattice": list(LATTICE),
            "tool_speed_m_s": 0.2,
            "duration_s": DURATION,
            "broken_bonds": sim.broken_bonds,
        },
        "tolerance_rel": TOL,
        "reference_rel": 0.03,
        "uncorrupted_rel_err": base,
        "channels_J": channels,
        "cases": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUT_DIR / "sensitivity.json"
    json_path.write_text(json.dumps(table, indent=1) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=130)
    colors = ["#d62728" if d else "#2ca02c" for d in detected]
    bars = ax.bar(labels, values, color=colors, edgecolor="k", linewidth=0.6)
    ax.axhline(TOL, color="k", linestyle="--", linewidth=1.2,
               label="tolerance 0.5 %")
    ax.axhline(0.03, color="#1f77b4", linestyle=":", linewidth=1.2,
               label="3 % reference")
    for b, v in zip(bars, values):
        ax.annotate(f"{v*100:.1f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("relative residual  |dE - W + Q + E_f - E_p| / scale")
    ax.set_title("2-D DEM energy/fracture ledger: corruption sensitivity\n"
                 f"uncorrupted residual = {base*100:.2f} %  (must be < 0.5 %)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(0, max(values) * 1.15)
    fig.tight_layout()
    out = OUT_DIR / "ledger_corruption_sensitivity.png"
    fig.savefig(out)

    print("saved", out)
    print("saved", json_path)
    print("uncorrupted_rel_err_pct", round(base * 100, 4))
    for lab, v, d in zip(labels, values, detected):
        print(f"  {lab.replace(chr(10), ' ')}: {v*100:.2f}%  detected={d}")


if __name__ == "__main__":
    main()