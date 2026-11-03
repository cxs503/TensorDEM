"""Independent re-verification of the restored fracture-gen audit API.

The implementation restored here (see docs/fracture_gen_restore_verification.md)
came from the m2245/m2287 audit that the merge gate had dropped.  This script
re-derives the three analytic baselines with *closed-form* expressions that
never call the module's own mass/area helpers, then compares them with what the
module now reports.  Everything is written to docs/fracture_gen_restore/ as
JSON (machine readable) and PNG (report figure).

Run:  python scripts/verify_fracture_gen_restore.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from tensordem.fracture_gen import (  # noqa: E402
    MASS_RADIUS_FACTOR,
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    face_footprint_radius,
    generate_fracture_field,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "fracture_gen_restore"

RADIUS = 0.125
SPACING = 2.0 * RADIUS
DENSITY = 917.0
N = 8
DOMAIN = (0.0, N * SPACING, 0.0, N * SPACING, 0.0, N * SPACING)
PLANE_AREA = (N * SPACING) ** 2
CONTINUUM_VOLUME = (N * SPACING) ** 3


def face(area: float = PLANE_AREA, extent_radius: float | None = None, z: float = 0.5) -> FailureFace:
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([1.0, 1.0, z], dtype=torch.float64),
        G=100.0,
        released_energy=1.0,
        extent_radius=extent_radius,
    )


def field(faces, **kw):
    cfg = FractureGenConfig(density=DENSITY, particle_radius=RADIUS, domain_bounds=DOMAIN, **kw)
    return generate_fracture_field(faces, cfg)


def enumerated_columns(radius: float | None) -> int:
    """Closed-form-by-brute-force count of lattice columns inside a disc.

    ``radius=None`` means the infinite plane: every column of the lattice.
    """
    coords = [RADIUS + i * SPACING for i in range(N)]
    c = 0.5 * N * SPACING
    total = 0
    for x in coords:
        for y in coords:
            if radius is None or math.hypot(x - c, y - c) <= radius:
                total += 1
    return total


def mass_block() -> dict:
    exact_frozen = math.pi / 6.0 - 1.0  # (4/3 pi r^3) / (2r)^3 - 1
    # The repository default is the mass-conserving cell mapping, so the frozen
    # legacy mapping under audit has to be requested explicitly.
    frozen = FractureLedger(
        field([face()], mass_mapping="equal_volume_spheres"), [face()]
    ).mass_accounting(CONTINUUM_VOLUME)
    cell = FractureLedger(field([face()], mass_mapping="lattice_cell"), [face()]).mass_accounting(
        CONTINUUM_VOLUME
    )
    # Negative control: L = 0.5 m is not a multiple of 2r = 0.25 m, so even the
    # mass-conserving mapping must report the lattice quantisation deficit.
    # r = 0.1 m -> spacing 0.2 m; L = 0.5 m is not a multiple of 0.2 m, so the
    # lattice covers only 0.4 m per axis even under the mass-conserving mapping.
    cfg = FractureGenConfig(
        density=DENSITY,
        particle_radius=0.1,
        domain_bounds=(0.0, 0.5, 0.0, 0.5, 0.0, 0.5),
        mass_mapping="cell_volume",
    )
    quant = FractureLedger(generate_fracture_field([], cfg), []).mass_accounting(0.5**3)
    exact_quant = -abs(0.4**3 - 0.5**3) / 0.5**3
    return {
        "frozen_mapping": {
            "measured_relative_error": frozen["relative_error"],
            "analytic_pi_over_6_minus_1": exact_frozen,
            "abs_deviation": abs(frozen["relative_error"] - exact_frozen),
            "within_15pct_flag": frozen["within_15pct"],
        },
        "lattice_cell_mapping": {
            "measured_relative_error": cell["relative_error"],
            "abs_deviation_from_zero": abs(cell["relative_error"]),
            "within_15pct_flag": cell["within_15pct"],
            "mass_radius_over_radius": field([face()], mass_mapping="lattice_cell").mass_radius / RADIUS,
            "analytic_mass_radius_factor": MASS_RADIUS_FACTOR,
            "particle_mass": field([face()], mass_mapping="lattice_cell").particle_mass,
            "analytic_particle_mass": DENSITY * SPACING**3,
        },
        "quantisation_negative_control": {
            "measured_relative_error": quant["relative_error"],
            "analytic": exact_quant,
            "abs_deviation": abs(quant["relative_error"] - exact_quant),
        },
    }


def footprint_block() -> dict:
    plane = field([face()])
    plane_broken = plane.num_broken_bonds
    analytic_plane = PLANE_AREA / SPACING**2
    rows = []
    for ratio in (0.5, 1.0, 2.0, 3.0, 5.0, 10.0):
        r = ratio * SPACING
        disc = field([face(area=math.pi * r**2, extent_radius=r)], face_extent="disc")
        expected = enumerated_columns(r)
        rows.append(
            {
                "extent_radius_over_spacing": ratio,
                "extent_radius": r,
                "footprint_radius": face_footprint_radius(face(area=math.pi * r**2, extent_radius=r)),
                "enumerated_columns": expected,
                "module_broken_bonds": disc.num_broken_bonds,
                "absolute_deviation": abs(disc.num_broken_bonds - expected),
                "plane_over_disc_oversevering": plane_broken / max(disc.num_broken_bonds, 1),
            }
        )
    # Consistency control: a disc large enough to cover the whole section must
    # reproduce the infinite-plane count exactly.
    big = 1.5 * N * SPACING
    whole = field([face(area=math.pi * big**2, extent_radius=big)], face_extent="disc")
    return {
        "full_plane": {
            "module_broken_bonds": plane_broken,
            "analytic_area_over_spacing_squared": analytic_plane,
            "absolute_deviation": abs(plane_broken - analytic_plane),
        },
        "disc_sweep": rows,
        "disc_covers_section_control": {
            "extent_radius": big,
            "module_broken_bonds": whole.num_broken_bonds,
            "plane_broken_bonds": plane_broken,
            "absolute_deviation": abs(whole.num_broken_bonds - plane_broken),
        },
    }


def figure(mass: dict, fp: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))

    labels = ["frozen default\n(equal_volume_spheres)", "opt-in\n(lattice_cell)"]
    measured = [
        mass["frozen_mapping"]["measured_relative_error"] * 100.0,
        mass["lattice_cell_mapping"]["measured_relative_error"] * 100.0,
    ]
    analytic = [mass["frozen_mapping"]["analytic_pi_over_6_minus_1"] * 100.0, 0.0]
    x = range(len(labels))
    ax1.bar([i - 0.18 for i in x], measured, 0.36, label="module", color="tab:blue")
    ax1.bar([i + 0.18 for i in x], analytic, 0.36, label="analytic", color="tab:orange")
    ax1.axhspan(-15.0, 15.0, color="tab:green", alpha=0.12, label="±15 % acceptance band")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("mass relative error vs $\\rho V_{FEM}$ [%]")
    ax1.set_title("(a) mass mapping: frozen deficit $-47.64\\,\\%$, opt-in $0$")
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3, axis="y")

    rows = fp["disc_sweep"]
    ratio = [r["extent_radius_over_spacing"] for r in rows]
    model = [r["module_broken_bonds"] for r in rows]
    enum = [r["enumerated_columns"] for r in rows]
    ax2.plot(ratio, enum, "o-", color="tab:orange", label="enumerated disc columns (baseline)")
    ax2.plot(ratio, model, "s--", color="tab:blue", label="module, face_extent='disc'")
    ax2.axhline(
        fp["full_plane"]["module_broken_bonds"],
        color="tab:red",
        ls=":",
        label="legacy infinite plane (over-severs)",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("$R_{face} / s$  (disc extent / lattice spacing)")
    ax2.set_ylabel("severed bonds")
    ax2.set_title("(b) footprint-aware breakage vs exact column count")
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "fracture_gen_restore.png", dpi=150)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mass = mass_block()
    fp = footprint_block()
    result = {"mass": mass, "footprint": fp}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2))
    figure(mass, fp)

    print(json.dumps(result, indent=2))
    ok = (
        mass["frozen_mapping"]["abs_deviation"] < 1e-12
        and mass["lattice_cell_mapping"]["abs_deviation_from_zero"] < 1e-12
        and mass["quantisation_negative_control"]["abs_deviation"] < 1e-12
        and fp["full_plane"]["absolute_deviation"] == 0
        and all(r["absolute_deviation"] == 0 for r in fp["disc_sweep"])
        and fp["disc_covers_section_control"]["absolute_deviation"] == 0
    )
    print("\nALL CRITERIA PASS" if ok else "\nCRITERIA FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())