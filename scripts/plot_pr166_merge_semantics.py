"""Figures for the PR #166 (agent/m2942) merge resolution.

All data are read from ``artifacts/pr166/verify_pr166.json`` (written by
``scripts/verify_pr166_merge_semantics.py``) or recomputed with the same public
API -- no number in the plots is transcribed by hand.

Panels
  (a) footprint map: the 8 x 8 crossing points of the nearest-neighbour bonds that
      straddle the face plane ``x = 1`` in the (y, z) plane, coloured by whether
      the resolved ``footprint_aware`` criterion severs them, with the footprint
      disc ``R = 0.30 m``.  This is the "field" the criterion actually acts on.
  (b) severed-bond count vs footprint radius: implemented criterion against the
      exact lattice enumeration (it must coincide) and against the continuum
      ``pi R^2 / h^2`` (it must approach it), plus the frozen infinite-plane
      criterion which is area independent.
  (c) mass mapping: ``m / (rho h^3)`` for ``mass_mode="cell"`` and ``"sphere"``
      with the analytic ``pi/6`` line of the frozen deficit.

Run:  python3 scripts/plot_pr166_merge_semantics.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensordem.fracture_gen import (  # noqa: E402
    FailureFace,
    build_bond_topology,
    generate_particle_lattice,
    identify_bonds_crossing_face,
)

OUT = ROOT / "artifacts" / "pr166"
RADIUS = 0.125
SPACING = 2.0 * RADIUS
DOMAIN = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
MAP_RADIUS = 0.30


def footprint_map() -> dict:
    positions = generate_particle_lattice(DOMAIN, RADIUS)
    bond_indices, _ = build_bond_topology(positions, RADIUS)
    face = FailureFace(
        id=0,
        area=math.pi * MAP_RADIUS**2,
        normal=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        centroid=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
        G=1.0,
        released_energy=1.0,
        extent_radius=MAP_RADIUS,
    )
    crosses = identify_bonds_crossing_face(
        bond_indices, positions, face, footprint_aware=False
    )
    pierced = identify_bonds_crossing_face(
        bond_indices, positions, face, footprint_aware=True
    )
    mid = 0.5 * (positions[bond_indices[:, 0]] + positions[bond_indices[:, 1]])
    mid_plane = mid[crosses]
    mask = pierced[crosses]
    return {
        "y": mid_plane[:, 1].tolist(),
        "z": mid_plane[:, 2].tolist(),
        "severed": mask.tolist(),
        "n_plane": int(crosses.sum().item()),
        "n_footprint": int(pierced.sum().item()),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((OUT / "verify_pr166.json").read_text())
    sweep = report["P4_bond_count_sweep"]["rows"]
    mass = report["P3_mass_mapping"]
    fmap = footprint_map()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # (a) footprint map -------------------------------------------------------
    ax = axes[0]
    ys, zs, sev = fmap["y"], fmap["z"], fmap["severed"]
    ax.scatter(
        [y for y, s in zip(ys, sev) if not s],
        [z for z, s in zip(zs, sev) if not s],
        s=48, c="#9aa4b2", marker="o", label=f"kept ({fmap['n_plane'] - fmap['n_footprint']})",
    )
    ax.scatter(
        [y for y, s in zip(ys, sev) if s],
        [z for z, s in zip(zs, sev) if s],
        s=70, c="#c0392b", marker="X", label=f"severed ({fmap['n_footprint']})",
    )
    theta = [2 * math.pi * k / 400 for k in range(401)]
    ax.plot(
        [1.0 + MAP_RADIUS * math.cos(t) for t in theta],
        [1.0 + MAP_RADIUS * math.sin(t) for t in theta],
        "k--", lw=1.2, label=f"footprint disc R={MAP_RADIUS:.2f} m",
    )
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("(a) crossing points on face plane x=1 m\n"
                 "footprint-aware criterion (h=0.25 m lattice)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (b) bond count sweep ----------------------------------------------------
    ax = axes[1]
    r = [row["footprint_radius"] for row in sweep]
    ax.plot(r, [row["bond_count_disc"] for row in sweep], "o-", c="#c0392b",
            label="implemented, footprint_aware=True")
    ax.plot(r, [row["exact_lattice_enumeration"] for row in sweep], "s", ms=11,
            mfc="none", mec="#1f77b4", mew=1.6, label="exact lattice enumeration")
    ax.plot(r, [row["analytic_pi_r2_over_h2"] for row in sweep], "--", c="#1f77b4",
            label=r"continuum $\pi R^2/h^2$")
    ax.plot(r, [row["bond_count_plane"] for row in sweep], ":", c="#555555",
            label="frozen infinite-plane criterion")
    ax.set_xlabel("footprint radius R [m]")
    ax.set_ylabel("severed nearest-neighbour bonds")
    ax.set_title("(b) severed-bond count vs footprint radius\n"
                 "implemented vs exact enumeration: max rel. err = "
                 f"{report['P4_bond_count_sweep']['max_error_vs_enumeration']*100:.2f}%")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # (c) mass mapping -------------------------------------------------------
    ax = axes[2]
    labels = ["mass_mode=cell\n(trunk default)", 'mass_mode=sphere\n(frozen SI-2)']
    ratios = [
        mass["cell_mode_mass"] / mass["continuum_mass_density_times_h3"],
        mass["sphere_mode_mass"] / mass["continuum_mass_density_times_h3"],
    ]
    bars = ax.bar(labels, ratios, color=["#2e7d32", "#c0392b"], width=0.55)
    ax.axhline(1.0, color="k", lw=1.0, label="continuum  $\\rho h^3$")
    ax.axhline(math.pi / 6.0, color="#c0392b", ls="--", lw=1.2,
               label=r"analytic $\pi/6$ = 0.5236")
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.6f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(r"$m \,/\, (\rho h^3)$")
    ax.set_title("(c) particle mass vs represented ice volume\n"
                 f"sphere deficit = {mass['sphere_mode_relative_deficit']*100:.2f}% "
                 f"(analytic {mass['analytic_pi_over_6_minus_1']*100:.2f}%)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "TensorDEM PR #166 (agent/m2942) merge resolution — semantic verification of "
        "fracture_gen union", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "pr166_merge_semantics.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())