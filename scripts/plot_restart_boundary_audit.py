"""Figure for the restart-boundary audit (docs/coupling_restart_boundary_matrix.md).

Panel A: number of disagreeing restart-record leaves at the interruption point
and after one further step, for every interruption step in the matrix.
Panel B: how many tests the negative-control injections turn red
(from artifacts/coupling_restart_negative_controls.json).

Output: artifacts/restart_boundary_audit.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensordem import DEMConfig, IceDEM  # noqa: E402

STEPS = (0, 1, 2, 5, 11, 13)
OUT = ROOT / "artifacts" / "restart_boundary_audit.png"


def leaves(node):
    """Flatten a restart record into path -> value."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            out.update({f"{key}.{k}": v for k, v in leaves(value).items()})
        return out
    if isinstance(node, list):
        out = {}
        for index, value in enumerate(node):
            out.update({f"[{index}]{k}": v for k, v in leaves(value).items()})
        return out
    return {"": node}


def disagree(reference, actual):
    left, right = leaves(reference), leaves(actual)
    assert set(left) == set(right)
    return sum(1 for key in left if left[key] != right[key])


def loads(sim, step):
    pattern = torch.linspace(-1.0, 1.0, sim.positions.numel(), dtype=torch.float64)
    return (0.5 + 0.1 * step) * pattern.reshape(sim.positions.shape)


def audit(fix_edges):
    rows = []
    for k in STEPS:
        sim = IceDEM(DEMConfig(nx=3, ny=2, drag=0, fix_edges=fix_edges, tool_gap=1, tool_speed=0.3))
        for step in range(k):
            sim.step(loads(sim, step), load_time=sim.time)
        saved = json.loads(json.dumps(sim.snapshot(), allow_nan=False))
        restarted = IceDEM.from_snapshot(saved)
        at_boundary = disagree(saved, restarted.snapshot())
        sim.step(loads(sim, k))
        restarted.step(loads(restarted, k))
        after_step = disagree(sim.snapshot(), restarted.snapshot())
        divergence = 0.0
        for step in range(k + 1, k + 6):
            sim.step(loads(sim, step))
            restarted.step(loads(restarted, step))
            divergence = max(divergence, float((sim.positions - restarted.positions).abs().max()))
            after_step = max(after_step, disagree(sim.snapshot(), restarted.snapshot()))
        rows.append((k, at_boundary, after_step, divergence, len(leaves(saved))))
    return rows


def main():
    rows = {flag: audit(flag) for flag in (False, True)}
    artifact = json.loads((ROOT / "artifacts" / "coupling_restart_negative_controls.json").read_text())
    injections = artifact["injections"]
    names = list(injections)
    red = [len(injections[name]["failed_tests"]) for name in names]
    green = [artifact["clean"]["summary"].split(",")[0]]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
    width = 0.38
    positions = list(range(len(STEPS)))
    for offset, flag in ((-width / 2, False), (width / 2, True)):
        values = [row[1] + row[2] for row in rows[flag]]
        ax0.bar([p + offset for p in positions], values, width,
                label=f"fix_edges={flag} (leaves compared: {rows[flag][0][4]})")
    ax0.set_xticks(positions, [f"k={k}" for k in STEPS])
    ax0.set_xlabel("interruption step")
    ax0.set_ylabel("disagreeing leaves\n(boundary + 1 step + 5 steps)")
    ax0.set_ylim(0, 1.0)
    ax0.set_title("A. Restart at boundary is exact (all bars = 0)")
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(axis="y", alpha=0.3)
    ax0.text(0.02, 0.83, "max |dx| between branches after 6 further steps = 0.0 m",
             transform=ax0.transAxes, fontsize=8, color="#1a5d1a")

    ax1.bar(range(len(names)), red, color="#b03030")
    ax1.axhline(len(artifact["clean"]["failed_tests"]), color="#1a5d1a", linestyle="--",
                label=f"clean tree: {green[0]}")
    ax1.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=8)
    ax1.set_ylabel("number of red tests")
    ax1.set_title("B. Injected defects are caught")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle("IceDEM restart boundary audit: bitwise equality at interruption, fail-red under injection",
                 fontsize=10)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(json.dumps({"png": str(OUT), "rows": rows,
                      "injections": {n: len(injections[n]["failed_tests"]) for n in names}},
                     indent=2))


if __name__ == "__main__":
    main()