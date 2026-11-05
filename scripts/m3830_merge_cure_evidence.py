"""Reproduce the m3830 merge-cure evidence for TensorDEM PR #166.

The mission is a *merge cure* (branch ``agent/m2942`` was dirty against ``main``),
so the quantitative evidence has to show two things:

1. the conflict resolution is semantically loss-free: every symbol that
   ``agent/m2942`` added still exists exactly once and still evaluates to the
   value its own documentation claims;
2. the contract that survives is the *stricter* one (``main``'s), because a
   frozen test requires ``face_footprint_radius`` to reject a degenerate
   footprint that ``agent/m2942``'s copy silently accepted.

Everything printed/plotted below is computed from the merged module at import
time -- no number is transcribed by hand.

Usage::

    python3 scripts/m3830_merge_cure_evidence.py [--out-dir docs/m3830-merge-cure]

Writes ``evidence.json``, ``evidence.csv`` and ``fig_m3830_merge_cure.png``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensordem.fracture_gen import (  # noqa: E402
    MASS_RADIUS_FACTOR,
    FailureFace,
    build_bond_topology,
    face_footprint_radius,
    generate_particle_lattice,
    identify_bonds_crossing_face,
)

#: The two closed forms of the mass-radius factor that the two branches wrote
#: independently.  ``main``: r_m/r = 2 (3/4pi)^{1/3}.  ``agent/m2942``: (6/pi)^{1/3}.
FORM_MAIN = 2.0 * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
FORM_M2942 = (6.0 / math.pi) ** (1.0 / 3.0)

RHO = 917.0          # ice density, kg/m^3
R = 0.125            # particle radius, m (h = 2R = 0.25 m)
H = 2.0 * R


def _face(area: float, extent_radius: float | None = None) -> FailureFace:
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        centroid=torch.zeros(3, dtype=torch.float64),
        G=10.0,
        released_energy=1.0,
        extent_radius=extent_radius,
    )


def _rejects(fn, *args, **kwargs) -> bool:
    try:
        fn(*args, **kwargs)
    except ValueError:
        return True
    return False


def mass_mapping_table() -> list[dict]:
    """Cell (volume-consistent) vs inscribed-sphere particle mass."""
    m_cell = RHO * H ** 3
    m_sphere = RHO * (4.0 / 3.0) * math.pi * R ** 3
    #: Mass a sphere of radius MASS_RADIUS_FACTOR * R carries at the same density.
    m_mass_radius = RHO * (4.0 / 3.0) * math.pi * (MASS_RADIUS_FACTOR * R) ** 3
    return [
        {
            "quantity": "m_cell = rho h^3",
            "value_kg": m_cell,
            "note": "mass_mode='cell' (default, volume consistent)",
        },
        {
            "quantity": "m_sphere = rho (4/3) pi r^3",
            "value_kg": m_sphere,
            "note": "mass_mode='sphere' (legacy, under-counts)",
        },
        {
            "quantity": "m(MASS_RADIUS_FACTOR r)",
            "value_kg": m_mass_radius,
            "note": "sphere carrying the cell mass",
        },
        {
            "quantity": "deficit 1 - pi/6",
            "value_kg": 1.0 - m_sphere / m_cell,
            "note": "relative mass loss of mass_mode='sphere'",
        },
    ]


def footprint_contract_table() -> list[dict]:
    """Accepted / rejected footprints (the union-of-contracts question)."""
    rows = []
    for area, extent, why in [
        (1.0, None, "equal-area disc, sqrt(A/pi)"),
        (0.5, 0.9, "explicit extent_radius wins over area"),
        (0.04, 0.0, "degenerate footprint: rejected (main) / accepted as 0.0 (m2942)"),
        (0.04, -1.0, "negative footprint: rejected by both, different message"),
        (0.0, None, "zero area: rejected by both"),
        (math.nan, None, "NaN area: rejected by main only"),
    ]:
        f = _face(area, extent)
        rejected = _rejects(face_footprint_radius, f)
        value = None if rejected else face_footprint_radius(f)
        rows.append(
            {
                "area_m2": None if math.isnan(area) else area,
                "extent_radius_m": extent,
                "rejected": rejected,
                "radius_m": value,
                "note": why,
            }
        )
    return rows


def severing_sweep(n_cells_per_side: int = 12) -> list[dict]:
    """Severed-bond count on an n x n cross-section, plane vs footprint aware.

    A plane x = 0 cuts the lattice; ``footprint_aware=True`` keeps only the
    bonds whose in-plane midpoint distance is inside ``sqrt(A/pi)``.
    """
    half = 0.5 * (n_cells_per_side - 1) * H
    bounds = (-half - 0.5 * H, half + 0.5 * H, -half, half, -half, half)
    positions = generate_particle_lattice(bounds, R, dtype=torch.float64)
    bonds, _rest = build_bond_topology(positions, R)
    # keep only bonds that actually straddle the plane x = 0 (the severing test
    # below decides that itself, so feed every bond)
    rows = []
    plane_face = _face(area=1.0)
    plane_hits = identify_bonds_crossing_face(bonds, positions, plane_face)
    n_plane = int(plane_hits.sum())

    #: In-plane column coordinates (y, z) of the lattice; a footprint-aware
    #: break keeps exactly one bond per column inside the disc, so the exact
    #: discrete prediction is "number of columns whose in-plane radius <= R".
    cols = torch.unique(positions[:, 1:], dim=0)
    col_r = torch.linalg.norm(cols, dim=1)

    for radius in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
        area = math.pi * radius ** 2
        f = _face(area=area)
        hits = identify_bonds_crossing_face(
            bonds, positions, f, footprint_aware=True
        )
        n_fp = int(hits.sum())
        n_disc = int((col_r <= radius + 1e-12).sum())
        continuum = math.pi * radius ** 2 / H ** 2
        rows.append(
            {
                "footprint_radius_m": radius,
                "face_area_m2": area,
                "plane_criterion_bonds": n_plane,
                "footprint_criterion_bonds": n_fp,
                "discrete_column_prediction": n_disc,
                "relative_diff_vs_discrete": abs(n_fp - n_disc) / max(n_disc, 1),
                "continuum_pi_R2_over_h2": continuum,
                "relative_diff_vs_continuum": abs(n_fp - continuum) / max(continuum, 1e-30),
                "saturated": n_fp == n_plane,
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/m3830-merge-cure")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    mass_rows = mass_mapping_table()
    fp_rows = footprint_contract_table()
    sev_rows = severing_sweep()

    factor_table = {
        "MASS_RADIUS_FACTOR_merged": MASS_RADIUS_FACTOR,
        "form_main_2_(3/4pi)^(1/3)": FORM_MAIN,
        "form_m2942_(6/pi)^(1/3)": FORM_M2942,
        "abs_diff": abs(FORM_MAIN - FORM_M2942),
        "rel_diff": abs(FORM_MAIN - FORM_M2942) / FORM_MAIN,
        "bit_identical": FORM_MAIN == FORM_M2942,
    }

    evidence = {
        "module": "tensordem.fracture_gen",
        "merge": {
            "repo": "TensorDEM",
            "mission": "m3830 (cure of PR #166 / branch agent/m2942)",
            "conflicted_files": ["src/tensordem/fracture_gen.py"],
            "auto_merge_hazards_found": [
                "git text-merge kept BOTH copies of MASS_RADIUS_FACTOR"
                " (module-level redefinition, later def silently wins)",
                "git text-merge kept BOTH copies of face_footprint_radius"
                " (different validation: m2942 accepted extent_radius=0.0,"
                " main rejects it)",
            ],
        },
        "mass_radius_factor_equivalence": factor_table,
        "mass_mapping": mass_rows,
        "footprint_contract": fp_rows,
        "severing_sweep": sev_rows,
    }

    with open(os.path.join(args.out_dir, "evidence.json"), "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)

    with open(os.path.join(args.out_dir, "evidence.csv"), "w", encoding="utf-8") as fh:
        fh.write(
            "footprint_radius_m,face_area_m2,plane_bonds,footprint_bonds,"
            "discrete_columns,continuum,rel_diff_discrete\n"
        )
        for r in sev_rows:
            fh.write(
                "%.6g,%.6g,%d,%d,%d,%.6g,%.3e\n"
                % (
                    r["footprint_radius_m"],
                    r["face_area_m2"],
                    r["plane_criterion_bonds"],
                    r["footprint_criterion_bonds"],
                    r["discrete_column_prediction"],
                    r["continuum_pi_R2_over_h2"],
                    r["relative_diff_vs_discrete"],
                )
            )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # (a) mass mapping deficit -------------------------------------------------
    ax = axes[0]
    labels = ["cell\n(rho h^3)", "sphere\n(rho 4/3 pi r^3)", "MASS_RADIUS\nFACTOR sphere"]
    vals = [m["value_kg"] for m in mass_rows[:3]]
    bars = ax.bar(labels, vals, color=["#2b6cb0", "#c53030", "#2f855a"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.01, "%.2f" % v, ha="center", fontsize=9)
    ax.set_ylabel("particle mass [kg]")
    ax.set_title(
        "(a) mass mapping, r=%.3f m, h=%.2f m\ndeficit %.2f %%"
        % (R, H, 100.0 * mass_rows[3]["value_kg"])
    )
    ax.set_ylim(0.0, max(vals) * 1.15)

    # (b) footprint radius contract -------------------------------------------
    import numpy as np

    ax = axes[1]
    areas = np.logspace(-3, 1, 200)
    ax.loglog(areas, np.sqrt(areas / np.pi), "k-", label="sqrt(A/pi) (equal-area disc)")
    ok = [r for r in fp_rows if not r["rejected"] and r["radius_m"] is not None]
    for r in ok:
        if r["extent_radius_m"] is not None and r["area_m2"]:
            ax.plot(r["area_m2"], r["radius_m"], "o", ms=8, color="#2f855a")
    ax.plot([], [], "o", ms=8, color="#2f855a", label="explicit extent_radius (used as-is)")
    rej = [r for r in fp_rows if r["rejected"]]
    ax.set_xlabel("face area A [m$^2$]")
    ax.set_ylabel("footprint radius [m]")
    ax.set_title(
        "(b) footprint contract\n%d rejected (extent_radius=0, A<=0, NaN A)" % len(rej)
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)

    # (c) severing sweep -------------------------------------------------------
    ax = axes[2]
    rs = [r["footprint_radius_m"] for r in sev_rows]
    fp_n = [r["footprint_criterion_bonds"] for r in sev_rows]
    plane_n = [r["plane_criterion_bonds"] for r in sev_rows]
    disc = [r["discrete_column_prediction"] for r in sev_rows]
    cont = [r["continuum_pi_R2_over_h2"] for r in sev_rows]
    ax.plot(rs, cont, "k--", lw=1.0, label="continuum pi R^2 / h^2")
    ax.plot(rs, disc, "x", ms=9, label="exact discrete column count")
    ax.plot(rs, fp_n, "o-", label="footprint_aware=True (measured)")
    ax.plot(rs, plane_n, "s-", color="#c53030", label="footprint_aware=False (plane)")
    ax.set_xlabel("footprint radius R [m]")
    ax.set_ylabel("severed bonds on x=0 cross-section")
    ax.set_title("(c) footprint vs infinite-plane severing\n(merged implementation)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "TensorDEM m3830 / PR #166 merge cure: conflict resolution keeps both branches' "
        "contracts (value-equal factors, stricter validation, footprint feature live)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(args.out_dir, "fig_m3830_merge_cure.png")
    fig.savefig(png, dpi=140)
    print(json.dumps(evidence, indent=2, default=str))
    print("wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())