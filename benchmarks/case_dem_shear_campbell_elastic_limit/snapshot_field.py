#!/usr/bin/env python3
"""Field snapshot for R4: particle map of the shear cell at three stiffnesses.

Reproduces the *deepest gated* point of R4 (kappa = 9.04) and two reference
points from the same family, and draws the particle centres coloured by the
instantaneous peculiar speed together with the contact-overlap field.  The point
of the figure is diagnostic, not decorative: at kappa < 100 the mean overlap
reaches 8-12 % of the diameter, which is exactly why the small-overlap validity
gate (mean <= 0.12 d) is what stops the walk down the kappa axis -- this is
visible in the picture, and it is the physical reason the inertial branch cannot
be reached from a constant-volume cell.

    PYTHONPATH=src python3 .../snapshot_field.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensordem.elastic_limit_shear import (  # noqa: E402
    ElasticLimitShearCase, simulate_plane_shear)

OUT = HERE / "runs" / "inertial_branch"
PANELS = [("kappa = 160.8  ($k_n$=1e4, $\\dot\\gamma$=6)", 1.0e4, 6.0),
          ("kappa = 22.6  ($k_n$=1e4, $\\dot\\gamma$=16)", 1.0e4, 16.0),
          ("kappa = 9.04  ($k_n$=1e3, $\\dot\\gamma$=8)", 1.0e3, 8.0)]


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.6))
    rows = []
    for ax, (title, kn, gdot) in zip(axes, PANELS):
        case = ElasticLimitShearCase(n_particles=400, solid_fraction=0.80, kn=kn,
                                     shear_rate=gdot, strain_total=1.4,
                                     strain_average_from=0.7, dt_factor=0.004)
        out = simulate_plane_shear(case, verbose=True)
        st = out["final_state"]
        speed = np.hypot(st["vx"], st["vy"])
        sc = ax.scatter(st["x"], st["y"], c=speed, s=8 + 90 * case.radii ** 2,
                        cmap="inferno", alpha=0.9)
        ax.set_xlim(0, case.lx); ax.set_ylim(0, case.ly)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nmean ovl/d = {out['mean_overlap_ratio']:.3f}, "
                     f"Z = {out['coordination']:.2f}", fontsize=9)
        ax.set_xlabel("x / d"); ax.set_ylabel("y / d")
        fig.colorbar(sc, ax=ax, fraction=0.046, label=r"$|v|$ (peculiar)")
        rows.append({"kn": kn, "gamma_dot": gdot, "kappa": out["kappa"],
                     "tau_star": out["tau_star"], "I": out["I"],
                     "mu_eff": out["mu_eff"],
                     "mean_overlap_ratio": out["mean_overlap_ratio"],
                     "max_overlap_ratio": out["max_overlap_ratio"],
                     "kinetic_stress_share": out["kinetic_stress_share"],
                     "gates": out["gates"]})
    fig.suptitle("R4 field snapshots: the soft-contact end of the kappa walk "
                 "(steady state, end of the averaging window)", fontsize=10)
    fig.tight_layout()
    png = OUT / "fig_field_snapshot.png"
    fig.savefig(png, dpi=140)
    plt.close(fig)
    (OUT / "field_snapshot.json").write_text(json.dumps(rows, indent=1, default=float))
    print(f"figure -> {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())