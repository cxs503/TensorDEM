#!/usr/bin/env python
"""Figures for the LBM--DEM drag-closure calibration report.

Reads ``result.json`` (written by ``run.py --merge``) and produces the PNGs that
are embedded in the campaign report.  The field snapshot is regenerated here
because a full 3-D field dump would dwarf the result JSON; the case is the same
one that is gated (Stokes, d = 6, L/d = 8).

    python benchmarks/verified/lbm_dem_drag_calibration/figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensordem.lbmdem import Lattice, LatticeShape, MomentumExchangeIBM, Sphere  # noqa: E402
from tensordem.lbmdem.drag import schiller_naumann_cd  # noqa: E402

FIGDIR = HERE / "figures"


def _load() -> dict:
    path = HERE / "result.json"
    if not path.exists():
        raise SystemExit("run.py --merge must be executed first")
    return json.loads(path.read_text())


def fig_drag_vs_re(result: dict) -> Path:
    reich = [case for case in result["cases"] if case["label"].startswith("reich")]
    reich.sort(key=lambda case: case["re_measured"])
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    re_ref = [10 ** (-1.5 + 3.5 * i / 200) for i in range(201)]
    ax.loglog(re_ref, [schiller_naumann_cd(re) for re in re_ref], "k-",
              label="Schiller-Naumann $24/Re\\,(1+0.15Re^{0.687})$")
    ax.loglog(24.0 / torch.tensor(re_ref).numpy(), "k--", lw=1, label="Stokes $24/Re$")
    if reich:
        ax.loglog([case["re_measured"] for case in reich],
                  [case["cd_measured"] for case in reich], "o", ms=6, color="C3",
                  label="LBM-IBM (this work)")
        for case in reich:
            ax.annotate(f"{case['cd_rel_error'] * 100:+.1f}%",
                        (case["re_measured"], case["cd_measured"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.set_xlabel("Re = U d / $\\nu$")
    ax.set_ylabel("$C_D$")
    ax.set_title("Drag coefficient: MEM-LBM vs Schiller-Naumann")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIGDIR / "drag_vs_re.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_stokes_gates(result: dict) -> Path:
    cases = {case["label"]: case for case in result["cases"]}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    # (a) resolution study
    res = [(cases[label]["diameter"], cases[label]["stokes_ratio_corrected"],
            cases[label]["stokes_ratio"]) for label in ("stokes_d4", "stokes_d6", "stokes_d8")
           if label in cases]
    if res:
        res.sort()
        ax = axes[0]
        ax.plot([r[0] for r in res], [r[1] for r in res], "o-", label="corrected (array)")
        ax.plot([r[0] for r in res], [r[2] for r in res], "s--", label="raw (finite domain)")
        ax.axhline(1.0, color="k", lw=1, ls=":")
        ax.axhspan(0.98, 1.02, color="C0", alpha=0.12, label="±2 % gate")
        ax.set_xlabel("sphere diameter d [cells]")
        ax.set_ylabel("$F/(3\\pi\\mu d U)$")
        ax.set_title("(a) resolution at L/d = 8")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    # (b) Cs sensitivity
    cs = [(cases[label]["smagorinsky_cs"], cases[label]["stokes_ratio_corrected"])
          for label in ("stokes_d6", "stokes_cs0.5", "stokes_cs1", "stokes_cs2") if label in cases]
    if cs:
        cs.sort()
        ax = axes[1]
        ax.plot([c[0] for c in cs], [c[1] for c in cs], "o-", color="C2")
        ax.axhline(1.0, color="k", lw=1, ls=":")
        ax.set_xlabel("Smagorinsky $C_s$")
        ax.set_ylabel("$F/(3\\pi\\mu d U)$ (corrected)")
        ax.set_title("(b) SGS insensitivity at Re$\\to$0")
        ax.grid(alpha=0.3)
    # (c) domain sensitivity
    ld = [(cases[label]["domain_ratio"], cases[label]["stokes_ratio_corrected"])
          for label in ("stokes_ld8", "stokes_ld16") if label in cases]
    if ld:
        ld.sort()
        ax = axes[2]
        ax.plot([c[0] for c in ld], [c[1] for c in ld], "o-", color="C4")
        ax.axhline(1.0, color="k", lw=1, ls=":")
        ax.axhspan(0.99, 1.01, color="C4", alpha=0.12, label="±1 % gate")
        ax.set_xlabel("L/d")
        ax.set_ylabel("corrected $F/(3\\pi\\mu d U)$")
        ax.set_title("(c) domain independence")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "stokes_gates.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_settling(result: dict) -> Path:
    cases = [case for case in result["cases"] if case["kind"] == "settling"]
    if not cases:
        raise SystemExit("no settling case in result.json")
    case = cases[0]
    history = case["history"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    ax.plot([row["step"] for row in history], [row["velocity"] for row in history], lw=1.2,
            label="particle velocity $v$")
    ax.plot([row["step"] for row in history], [row["relative_velocity"] for row in history],
            lw=1.2, label="relative $v-u_{fluid}$")
    ax.axhline(case["u_analytic_schiller_naumann"], color="k", ls="--", lw=1,
               label="$U_t$ Schiller-Naumann")
    ax.axhline(case["u_relative_terminal_corrected"], color="C3", ls=":", lw=1,
               label="$U_t$ corrected (array)")
    ax.set_xlabel("step")
    ax.set_ylabel("velocity [lattice]")
    ax.set_title("(a) settling transient, d = %g" % case["diameter"])
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot([row["step"] for row in history], [row["drag"] for row in history], lw=1.2)
    ax.axhline(case["drive_force"], color="k", ls="--", lw=1, label="effective weight")
    ax.set_xlabel("step")
    ax.set_ylabel("$F_{drag}$")
    ax.set_title("(b) drag history, min = %.3g > 0" % case["drag_min"])
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGDIR / "settling.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_field_snapshot(diameter: float = 6.0, ratio: float = 8.0, steps: int = 400) -> Path:
    """Mid-plane speed and streamlines for the gated Stokes case (regenerated)."""
    length = int(round(ratio * diameter))
    lattice = Lattice(LatticeShape(length, length, length), tau0=0.8, body_force=(4e-5, 0, 0),
                      dtype="float32")
    lattice.initialize_velocity((8.3e-4, 0.0, 0.0))
    ibm = MomentumExchangeIBM(lattice, Sphere((length / 2, length / 2, length / 2), diameter / 2))
    for _ in range(steps):
        lattice.step(post_collision_hook=ibm.apply)
    _, u = lattice.macroscopic(lattice.f)
    speed = torch.sqrt((u * u).sum(dim=-1)).numpy()
    plane = speed[:, :, length // 2].T
    ux = u[:, :, length // 2, 0].T.numpy()
    uy = u[:, :, length // 2, 1].T.numpy()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    im = axes[0].imshow(plane, origin="lower", cmap="viridis")
    fig.colorbar(im, ax=axes[0], label="|u| [lattice]")
    axes[0].set_title("(a) |u|, z mid-plane")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    step = 2
    axes[1].streamplot(
        [i for i in range(0, length, step)], [j for j in range(0, length, step)],
        ux[::step, ::step], uy[::step, ::step], color=plane[::step, ::step], cmap="viridis",
        density=1.1, linewidth=0.6,
    )
    axes[1].set_title("(b) streamlines (vortex-free Stokes field)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.tight_layout()
    out = FIGDIR / "flow_field.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    FIGDIR.mkdir(exist_ok=True)
    result = _load()
    made = [fig_drag_vs_re(result), fig_stokes_gates(result), fig_settling(result)]
    torch.set_num_threads(4)
    made.append(fig_field_snapshot())
    for path in made:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())