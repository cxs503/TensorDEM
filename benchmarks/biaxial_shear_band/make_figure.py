"""Regenerate the shear-band figure from a fresh run of the reference config.

Writes ``artifacts/biaxial_shear_band/shear_band.png`` with

(a) the local maximum-shear-strain field of the 24x48 / 100 kPa specimen at the
    end of the run, the fitted band line and the excluded wall layers;
(b) measured inclination (mean +- spread over the late post-peak snapshots)
    against the 60 deg literature anchor and the three theoretical laws.

Run: ``PYTHONPATH=src python benchmarks/biaxial_shear_band/make_figure.py``
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensordem.biaxial_shear_band import (
    _make_specimen,
    local_shear_strain,
    measure_band,
    run_biaxial_shear_band,
)
from tensordem.ridge_dem import RidgeMaterial, _advance_planes

OUT = "artifacts/biaxial_shear_band/shear_band.png"
FIELD = "artifacts/biaxial_shear_band/final_field.npz"


def main() -> int:
    torch.set_num_threads(1)
    columns, rows, confining = 24, 48, 100.0e3
    result = run_biaxial_shear_band(
        confining_stress=confining, columns=columns, rows=rows, strain_rate=0.05,
        strain_target=0.12, consolidation_steps=2000, snapshot_strain=0.01,
        servo_gain=3.0e-4, servo_velocity_limit=2.0,
        material_overrides={"freeze_bond_fraction": 0.35})

    # rebuild the final configuration to keep the particle field for the figure
    material = RidgeMaterial(particle_radius=0.75, freeze_bond_fraction=0.35)
    assembly, geometry = _make_specimen(material, columns=columns, rows=rows,
                                        confining_stress=confining, jitter=0.03,
                                        servo_gain=3.0e-4, servo_velocity_limit=2.0)
    dt = material.recommended_dt
    for _ in range(2000):
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)
    reference = assembly.pos.detach().numpy().copy()
    from dataclasses import replace
    height0 = float(assembly.planes[3].point[1] - assembly.planes[2].point[1])
    strain = 0.0
    while strain < 0.12:
        left, right, bottom, top = assembly.planes
        height = float(top.point[1] - bottom.point[1])
        assembly.planes[3] = replace(top, velocity=float(0.05 * height))
        out = assembly.step(dt)
        assembly.last_plane_forces = out["plane_forces"]
        _advance_planes(assembly, dt)
        left, right, bottom, top = assembly.planes
        strain = (height0 - float(top.point[1] - bottom.point[1])) / height0
    final = assembly.pos.detach().numpy().copy()

    diameter = 2.0 * material.particle_radius
    bounds = (float(left.point[0]) + 2 * diameter, float(right.point[0]) - 2 * diameter,
              float(bottom.point[1]) + 2 * diameter, float(top.point[1]) - 2 * diameter)
    inside = ((reference[:, 0] >= bounds[0]) & (reference[:, 0] <= bounds[1])
              & (reference[:, 1] >= bounds[2]) & (reference[:, 1] <= bounds[3]))
    gamma = local_shear_strain(reference[inside], final[inside], k=8)
    angle = result["band"]["angle_deg"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(FIELD, x=reference[inside, 0], y=reference[inside, 1],
                        gamma=gamma, angle_deg=angle,
                        strain=float(result["loading"]["final_axial_strain"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    ax = axes[0]
    order = np.argsort(gamma)
    sc = ax.scatter(reference[inside, 0][order], reference[inside, 1][order],
                    c=gamma[order], s=9, cmap="inferno")
    x0 = 0.5 * (bounds[0] + bounds[1])
    for sign in (+1, -1):
        dx = sign * np.cos(np.radians(angle))
        dy = sign * np.sin(np.radians(angle))
        ax.plot([x0 - 0.45 * dx * (bounds[3] - bounds[2]),
                 x0 + 0.45 * dx * (bounds[3] - bounds[2])],
                [bounds[2] + 0.1 * (bounds[3] - bounds[2]),
                 bounds[3] - 0.1 * (bounds[3] - bounds[2])], "c--", lw=1.2)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]  ($\\sigma_1$ axis)")
    ax.set_title("local max shear strain, $\\epsilon_a=%.2f$\nfitted band %.1f$^\\circ$ from $\\sigma_3$"
                 % (result["loading"]["final_axial_strain"], angle))
    fig.colorbar(sc, ax=ax, label="$\\gamma_{max}$ [-]")

    ax = axes[1]
    histories = [b for b in result["band"]["history"] if b["axial_strain"] >= 0.05]
    strains = [b["axial_strain"] for b in histories]
    incremental = [b["folded_angle_deg"] for b in histories]
    cumulative = [b["cumulative"]["folded_angle_deg"] for b in histories]
    ax.plot(strains, incremental, "o-", label="incremental field")
    ax.plot(strains, cumulative, "s--", alpha=0.6, label="cumulative field")
    final_angle = result["band"]["angle_deg"]
    ax.axhline(60.0, color="k", lw=1.2, label="literature anchor 60$^\\circ$ (Desrues 2004)")
    ax.axhspan(51.0, 69.0, color="k", alpha=0.08, label="15 % tolerance band")
    for key, value in result["theory"].items():
        ax.axhline(value, lw=0.8, ls=":", label=f"{key} {value:.1f}$^\\circ$")
    if np.isfinite(final_angle):
        ax.axhline(final_angle, color="tab:red", lw=1.5,
                   label=f"measured {final_angle:.1f}$^\\circ$")
    ax.set_xlabel("axial strain [-]")
    ax.set_ylabel("band inclination from $\\sigma_3$ [deg]")
    ax.set_title("measured vs literature anchor and theory\n$\\phi=%.1f^\\circ$, $\\psi=%.1f^\\circ$"
                 % (result["peak"]["friction_angle_deg"], result["peak"]["dilation_angle_deg"]))
    ax.legend(fontsize=6.5, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=140)
    print(json.dumps({"figure": OUT, "angle_deg": final_angle,
                      "gamma_max": float(gamma.max())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())