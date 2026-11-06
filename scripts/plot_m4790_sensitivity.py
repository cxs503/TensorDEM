"""Generate the m4790 sensitivity figure: ledger residual per corrupted channel.

This is a test-fix (backend) task with no physical field quantity, so the
evidence figure is the *residual sensitivity* of the energy/fracture ledger
balance: the uncorrupted ledger must close (<0.5 %), and each deliberately
corrupted channel must push the residual above the tolerance (detected).
"""
from __future__ import annotations

import copy
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


def run_indentation() -> IceDEM:
    nx, ny = LATTICE
    dt0 = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2)).dt
    sim = IceDEM(DEMConfig(nx=nx, ny=ny, tool_speed=0.2, dt=dt0))
    for _ in range(int(round(DURATION / sim.dt))):
        sim.step()
    return sim


def residual(sim: IceDEM, e0: float) -> float:
    e1 = sim.mechanical_energy()["mechanical_J"]
    a = sim.accounting
    return (e1 - e0) - a["tool_work_J"] - a["external_work_J"] \
        + a["damping_dissipation_J"] + a["fracture_release_J"] \
        - a["pair_contact_acquired_J"]


def scale(sim: IceDEM) -> float:
    a = sim.accounting
    return abs(a["tool_work_J"]) + abs(a["damping_dissipation_J"]) + abs(a["fracture_release_J"])


def main() -> None:
    sim = run_indentation()
    sim0 = IceDEM(DEMConfig(nx=LATTICE[0], ny=LATTICE[1], tool_speed=0.2,
                           dt=IceDEM(DEMConfig(nx=LATTICE[0], ny=LATTICE[1], tool_speed=0.2)).dt))
    e0 = sim0.mechanical_energy()["mechanical_J"]

    base = abs(residual(sim, e0)) / scale(sim)

    cases = [
        ("fracture_release\nx0", "fracture_release_J", 0.0),
        ("fracture_release\nx3", "fracture_release_J", 3.0),
        ("pair_contact\nx0", "pair_contact_acquired_J", 0.0),
        ("pair_contact\nx3", "pair_contact_acquired_J", 3.0),
        ("tool_work\nx0", "tool_work_J", 0.0),
        ("damping\nx0", "damping_dissipation_J", 0.0),
    ]
    labels, values = [], []
    for label, ch, f in cases:
        c = copy.deepcopy(sim)
        c.accounting[ch] *= f
        values.append(abs(residual(c, e0)) / max(scale(c), 1e-18))
        labels.append(label)

    fig, ax = plt.subplots(figsize=(8.0, 4.6), dpi=130)
    colors = ["#d62728" if v >= TOL else "#2ca02c" for v in values]
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
    out = ROOT / "artifacts" / "m4790" / "ledger_corruption_sensitivity.png"
    fig.savefig(out)
    print("saved", out)
    print("uncorrupted_rel_err_pct", round(base * 100, 4))
    for l, v in zip(labels, values):
        print(f"  {l.replace(chr(10), ' ')}: {v*100:.2f}%  detected={v >= TOL}")


if __name__ == "__main__":
    main()
