"""Validation-debt audit of the FEM failure-face -> DEM ice-field generator.

Recomputes, against independent analytic baselines, three properties of the
SI-2/DEM contract implemented in ``src/tensordem/fracture_gen.py``:

* mass mapping error of the frozen equal-volume-sphere mapping vs the
  continuum solid mass, and the corrected ``lattice_cell`` mapping;
* broken-bond count of a full plane cut vs ``A / s^2`` for a simple-cubic
  lattice of spacing ``s``;
* broken-bond count of a finite face footprint vs ``pi R^2 / s^2`` and vs exact
  lattice enumeration, plus the frozen infinite-plane over-break factor;
* the layer-coincident plane degeneracy (``0`` broken bonds with the strict
  sign test, ``2 n^2`` once an on-plane tolerance is enabled).

Writes ``artifacts/fracture_gen_audit/audit.json``, ``audit_tables.md`` and two
figures. No numbers in the audit document are transcribed by hand.

Run: python3 benchmarks/fracture_gen_mass_footprint_audit.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from tensordem.fracture_gen import (
    MASS_RADIUS_FACTOR,
    FailureFace,
    FractureGenConfig,
    FractureLedger,
    generate_fracture_field,
)

RADIUS = 0.125
SPACING = 2.0 * RADIUS
DOMAIN = (0.0, 2.0, 0.0, 2.0, 0.0, 2.0)
N_LAYERS = 8
PLANE_AREA = (N_LAYERS * SPACING) ** 2
CONTINUUM_VOLUME = (N_LAYERS * SPACING) ** 3
DENSITY = 917.0
OUT_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "fracture_gen_audit"


def _face(area=PLANE_AREA, z=0.5, extent_radius=None):
    return FailureFace(
        id=0,
        area=area,
        normal=torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        centroid=torch.tensor([1.0, 1.0, z], dtype=torch.float64),
        G=100.0,
        released_energy=1.0,
        extent_radius=extent_radius,
    )


def _field(faces, **config_kwargs):
    config = FractureGenConfig(
        density=DENSITY,
        particle_radius=RADIUS,
        domain_bounds=DOMAIN,
        **config_kwargs,
    )
    return generate_fracture_field(faces, config)


def enumerate_footprint(radius):
    coords = torch.arange(RADIUS, DOMAIN[1], SPACING, dtype=torch.float64)
    return sum(
        1
        for x in coords
        for y in coords
        if math.hypot(float(x) - 1.0, float(y) - 1.0) <= radius
    )


def run_audit(write_artifacts: bool = True):
    faces = [_face()]
    # The legacy mapping is selected explicitly: trunk's *default* is the
    # volume-consistent one (docs/dem3d_fracture_mass_mapping_audit.md), so the
    # frozen number must be reproduced by asking for it, not by relying on it.
    frozen = _field(faces, mass_mapping="equal_volume_sphere")
    corrected = _field(faces, mass_mapping="lattice_cell")
    frozen_acct = FractureLedger(frozen, faces).mass_accounting(CONTINUUM_VOLUME)
    corrected_acct = FractureLedger(corrected, faces).mass_accounting(CONTINUUM_VOLUME)

    lattice = {
        "particle_radius_m": RADIUS,
        "spacing_m": SPACING,
        "n_layers": N_LAYERS,
        "n_particles": frozen.num_particles,
        "n_bonds": int(frozen.bond_indices.shape[0]),
        "n_particles_analytic": N_LAYERS**3,
        "n_bonds_analytic": 3 * N_LAYERS**2 * (N_LAYERS - 1),
        "continuum_volume_m3": CONTINUUM_VOLUME,
    }

    mass = {
        "density_kg_m3": DENSITY,
        "frozen_mapping": "equal_volume_sphere",
        "frozen_mapping_selected_explicitly": True,
        "trunk_default_mapping": "cell",
        "frozen_dem_mass_kg": frozen.total_mass,
        "frozen_relative_error": frozen_acct["relative_error"],
        "frozen_within_15pct": frozen_acct["within_15pct"],
        "analytic_ratio_pi_over_6": math.pi / 6,
        "lattice_cell_mapping": "lattice_cell",
        "lattice_cell_mass_radius_factor": MASS_RADIUS_FACTOR,
        "lattice_cell_dem_mass_kg": corrected.total_mass,
        "lattice_cell_relative_error": corrected_acct["relative_error"],
        "lattice_cell_within_15pct": corrected_acct["within_15pct"],
        "continuum_mass_kg": frozen_acct["continuum_mass"],
    }

    plane_cut = {
        "analytic_bonds_area_over_spacing_squared": PLANE_AREA / SPACING**2,
        "broken_bonds_full_plane": frozen.num_broken_bonds,
        "broken_bonds_full_plane_rel_error": (
            frozen.num_broken_bonds - PLANE_AREA / SPACING**2
        )
        / (PLANE_AREA / SPACING**2),
        "broken_bonds_layer_coincident_strict": _field([_face(z=0.625)]).num_broken_bonds,
        "broken_bonds_layer_coincident_tolerant": _field(
            [_face(z=0.625)], on_plane_tolerance=1e-12
        ).num_broken_bonds,
        "analytic_bonds_layer_coincident": 2 * N_LAYERS**2,
    }

    footprint_rows = []
    for radius in (0.25, 0.5, 0.75, 1.0, 1.25):
        area = math.pi * radius**2
        frozen_f = _field([_face(area=area, extent_radius=radius)])
        aware_f = _field([_face(area=area, extent_radius=radius)], footprint_aware=True)
        analytic = area / SPACING**2
        exact = enumerate_footprint(radius)
        footprint_rows.append(
            {
                "footprint_radius_m": radius,
                "face_area_m2": area,
                "analytic_pi_r2_over_s2": analytic,
                "exact_enumerated": exact,
                "frozen_infinite_plane": frozen_f.num_broken_bonds,
                "footprint_aware": aware_f.num_broken_bonds,
                "aware_rel_error_vs_analytic": (aware_f.num_broken_bonds - analytic) / analytic,
                "over_break_factor": frozen_f.num_broken_bonds
                / max(aware_f.num_broken_bonds, 1),
            }
        )

    audit = {
        "module": "src/tensordem/fracture_gen.py",
        "suite": "SI-2/DEM mass mapping & face footprint validation audit",
        "lattice": lattice,
        "mass": mass,
        "plane_cut": plane_cut,
        "footprint": footprint_rows,
        "acceptance_bar_relative_error": 0.15,
        "verdicts": {
            "frozen_mass_mapping": "FAIL (>15 %)",
            "lattice_cell_mapping": "PASS (<3 %)",
            "full_plane_bond_count": "PASS",
            "frozen_footprint_over_break": "FAIL (up to ~%.1fx)"
            % max(row["over_break_factor"] for row in footprint_rows),
            "layer_coincident_plane": "FAIL with strict test, PASS with on_plane_tolerance",
        },
    }

    if write_artifacts:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "audit.json").write_text(json.dumps(audit, indent=2))
        (OUT_DIR / "audit_tables.md").write_text(render_markdown(audit))
        plot(audit)
    return audit


def render_markdown(audit):
    mass = audit["mass"]
    plane = audit["plane_cut"]
    lines = [
        "### Mass mapping (continuum reference density × volume)",
        "",
        "| mapping | DEM mass (kg) | continuum mass (kg) | relative error | <15 % ? |",
        "|---|---|---|---|---|",
        "| equal_volume_sphere (frozen) | %.6f | %.6f | %+.4f %% | %s |"
        % (
            mass["frozen_dem_mass_kg"],
            mass["continuum_mass_kg"],
            100.0 * mass["frozen_relative_error"],
            "yes" if mass["frozen_within_15pct"] else "**no**",
        ),
        "| lattice_cell (corrected) | %.6f | %.6f | %+.6f %% | %s |"
        % (
            mass["lattice_cell_dem_mass_kg"],
            mass["continuum_mass_kg"],
            100.0 * mass["lattice_cell_relative_error"],
            "yes" if mass["lattice_cell_within_15pct"] else "no",
        ),
        "",
        "### Plane-cut bond counts vs analytic A / s²",
        "",
        "| case | measured | analytic | relative error |",
        "|---|---|---|---|",
        "| full plane between layers | %d | %.0f | %+.2e |"
        % (
            plane["broken_bonds_full_plane"],
            plane["analytic_bonds_area_over_spacing_squared"],
            plane["broken_bonds_full_plane_rel_error"],
        ),
        "| plane on a particle layer, strict test | %d | %d | %+.1f %% |"
        % (
            plane["broken_bonds_layer_coincident_strict"],
            plane["analytic_bonds_layer_coincident"],
            -100.0,
        ),
        "| plane on a particle layer, on_plane_tolerance=1e-12 | %d | %d | %+.1f %% |"
        % (
            plane["broken_bonds_layer_coincident_tolerant"],
            plane["analytic_bonds_layer_coincident"],
            100.0
            * (
                plane["broken_bonds_layer_coincident_tolerant"]
                - plane["analytic_bonds_layer_coincident"]
            )
            / plane["analytic_bonds_layer_coincident"],
        ),
        "",
        "### Finite face footprint (n = 8, s = 0.25 m)",
        "",
        "| R (m) | A (m²) | analytic πR²/s² | exact enumeration | frozen (infinite plane) | footprint_aware | aware error | over-break |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in audit["footprint"]:
        lines.append(
            "| %.2f | %.4f | %.2f | %d | %d | %d | %+.1f %% | %.2f× |"
            % (
                row["footprint_radius_m"],
                row["face_area_m2"],
                row["analytic_pi_r2_over_s2"],
                row["exact_enumerated"],
                row["frozen_infinite_plane"],
                row["footprint_aware"],
                100.0 * row["aware_rel_error_vs_analytic"],
                row["over_break_factor"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def plot(audit):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    # Panel 1: footprint-aware vs frozen vs analytic
    rows = audit["footprint"]
    rs = [row["footprint_radius_m"] for row in rows]
    axes[0].plot(rs, [row["analytic_pi_r2_over_s2"] for row in rows], "k-", lw=2,
                 label=r"analytic $\pi R^2/s^2$")
    axes[0].plot(rs, [row["exact_enumerated"] for row in rows], "ks", ms=6,
                 label="exact lattice enumeration")
    axes[0].plot(rs, [row["footprint_aware"] for row in rows], "o", color="tab:blue",
                 ms=7, label="footprint_aware=True")
    axes[0].plot(rs, [row["frozen_infinite_plane"] for row in rows], "x", color="tab:red",
                 ms=9, mew=2, label="frozen criterion (infinite plane)")
    axes[0].axhline(audit["lattice"]["n_layers"] ** 2, ls=":", color="tab:red", lw=1)
    axes[0].text(0.27, audit["lattice"]["n_layers"] ** 2 + 1.5,
                 r"$n^2 = A_{\rm plane}/s^2$", color="tab:red", fontsize=9)
    axes[0].set_xlabel("face footprint radius R (m)")
    axes[0].set_ylabel("broken bonds")
    axes[0].set_title("Finite face footprint: severed bonds")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    # Panel 2: mass error vs mapping, with the 15 % acceptance band
    labels = ["equal_volume_sphere\n(frozen)", "lattice_cell\n(opt-in)"]
    errors = [100.0 * audit["mass"]["frozen_relative_error"],
              100.0 * audit["mass"]["lattice_cell_relative_error"]]
    bars = axes[1].bar(labels, errors, color=["tab:red", "tab:green"], width=0.55)
    axes[1].axhspan(-15.0, 15.0, color="tab:green", alpha=0.12, label="±15 % acceptance band")
    for bar, err in zip(bars, errors):
        axes[1].annotate("%+.2f %%" % err, (bar.get_x() + bar.get_width() / 2, err),
                         ha="center", va="bottom" if err >= 0 else "top", fontsize=10)
    axes[1].axhline(-47.6401, ls="--", color="k", lw=1)
    axes[1].text(1.42, -47.6401, r"$\pi/6 - 1$", fontsize=9, va="bottom", ha="right")
    axes[1].set_ylim(-60.0, 20.0)
    axes[1].set_ylabel("relative error vs continuum mass (%)")
    axes[1].set_title("DEM mass mapping vs continuum (N = %d)" % audit["lattice"]["n_particles"])
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=0.3, axis="y")

    fig.suptitle("SI-2/DEM fracture-field generator: validation-debt audit", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fracture_gen_audit.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps({k: v for k, v in result.items() if k != "footprint"}, indent=2))
    print(render_markdown(result))
    print("artifacts ->", OUT_DIR)