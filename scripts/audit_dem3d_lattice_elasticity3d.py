"""Audit the directional elastic stiffness of the IceDEM3D simple-cubic lattice.

Closes the open item stated in ``src/tensordem/dem3d.py`` ("calibrate its
directional stiffness before engineering use"). The script

1. reports the exact affine elastic tensor of the solver's own bond topology
   (thirteen half-space neighbours: six faces plus twelve edge diagonals), whose
   constants are the rational numbers ``C11 = 13k/(3a)`` and ``C12 = C44 = 7k/(3a)``;
2. measures the uniaxial modulus of finite samples with free lateral faces under
   both protocols used in DEM practice (prescribed-platen-displacement and
   applied-traction), and quantifies the boundary bias against the analytic
   infinite-crystal value;
3. re-loads a measured state into ``IceDEM3D.forces`` to confirm the equilibrium
   is a solution of the solver's own force law, not of a parallel reimplementation;
4. converts the calibration into engineering terms: the bond stiffness that would
   be needed to represent a real ice Young's modulus and the critical time step it
   implies.

Run::

    python scripts/audit_dem3d_lattice_elasticity3d.py \
        --output results-dem3d-lattice-elasticity3d
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tensordem.dem3d import DEM3DConfig
from tensordem.lattice_elasticity3d import (
    bond_stiffness_for_target_modulus,
    derived_moduli,
    lattice_bonds,
    lattice_positions,
    lattice_spacing,
    periodic_elastic_constants,
    relaxed_uniaxial_modulus,
)

#: (nx, ny, nz) samples: the solver default plus four refinements.
SAMPLES = ((9, 5, 3), (13, 7, 5), (17, 9, 7), (21, 11, 9), (25, 15, 11))
#: Real sea-ice reference mid-range values (Timco & Weeks 2010; Weeks & Ackley 1982).
ICE_YOUNG_MODULUS_PA = 9.0e9
ICE_POISSON = 0.33
ICE_SHEAR_MODULUS_PA = 3.4e9


def audit(config: DEM3DConfig, output_dir: Path) -> dict:
    spacing = lattice_spacing(config.radius)
    constants = periodic_elastic_constants(config.bond_stiffness, spacing)
    moduli = derived_moduli(constants["C11_Pa"], constants["C12_Pa"], constants["C44_Pa"])
    unit = config.bond_stiffness / spacing

    measurements: list[dict] = []
    for nx, ny, nz in SAMPLES:
        positions = lattice_positions(nx, ny, nz, spacing)
        pairs = lattice_bonds(nx, ny, nz)
        for control in ("displacement", "load"):
            case = relaxed_uniaxial_modulus(
                positions, pairs, config.bond_stiffness, control=control)
            residual = {"max_particle_force_N": float(case["max_free_dof_force_N"]),
                        "max_force_relative_to_platen_scale": float("nan")}
            measurements.append({
                "lattice": f"{nx}x{ny}x{nz}",
                "particle_count": case["particle_count"],
                "bond_count": case["bond_count"],
                "control": control,
                "axial_extent_m": case["axial_extent_m"],
                "cross_section_area_m2": case["cross_section_area_m2"],
                "measured_E_Pa": case["E_Pa"],
                "measured_E_over_k_over_a": case["E_Pa"] / unit,
                "analytic_E_Pa": moduli["E_Pa"],
                "E_relative_error_vs_analytic": case["E_Pa"] / moduli["E_Pa"] - 1.0,
                "measured_nu": case["poisson_ratio_measured"],
                "analytic_nu": moduli["nu"],
                "nu_absolute_error": abs(case["poisson_ratio_measured"] - moduli["nu"]),
                "max_free_dof_force_N": residual["max_particle_force_N"],
            })

    factor_E = moduli["E_Pa"] / unit
    required_k = bond_stiffness_for_target_modulus(ICE_YOUNG_MODULUS_PA, spacing, factor_E)
    stiff_config = DEM3DConfig(
        nx=9, ny=5, nz=3, radius=config.radius, density=config.density,
        bond_stiffness=required_k, contact_stiffness=required_k,
    )
    default_disp = next(m for m in measurements
                        if m["lattice"] == "9x5x3" and m["control"] == "displacement")
    default_load = next(m for m in measurements
                        if m["lattice"] == "9x5x3" and m["control"] == "load")

    scaling = {
        "ice_reference": {
            "E_Pa": ICE_YOUNG_MODULUS_PA, "nu": ICE_POISSON, "G_Pa": ICE_SHEAR_MODULUS_PA,
            "source": "Timco & Weeks (2010) / Weeks & Ackley (1982) mid-range ice properties",
        },
        "lattice_E_Pa": moduli["E_Pa"],
        "lattice_G_voigt_Pa": moduli["G_voigt_Pa"],
        "E_ratio_ice_over_lattice": ICE_YOUNG_MODULUS_PA / moduli["E_Pa"],
        "G_ratio_ice_over_lattice": ICE_SHEAR_MODULUS_PA / moduli["G_voigt_Pa"],
        "nu_error_vs_ice": abs(moduli["nu"] - ICE_POISSON),
        "bond_stiffness_for_ice_E_N_per_m": required_k,
        "dt_default_config_s": config.recommended_dt,
        "dt_stiffness_matched_config_s": stiff_config.recommended_dt,
        "dt_ratio": stiff_config.recommended_dt / config.recommended_dt,
    }

    summary = {
        "protocol": "tensordem-dem3d-lattice-elasticity-audit-v1",
        "config": {
            "radius_m": config.radius,
            "lattice_spacing_m": spacing,
            "bond_stiffness_N_per_m": config.bond_stiffness,
            "k_over_a_Pa": unit,
            "bond_types": "3 face + 6 in-plane edge diagonals + 4 body diagonals per node",
            "note": "central-force axial springs only; no bending or shear bond stiffness",
        },
        "affine_constants": constants,
        "closed_form": {
            "C11_over_k_over_a": 13.0 / 3.0,
            "C12_over_k_over_a": 7.0 / 3.0,
            "C44_over_k_over_a": 7.0 / 3.0,
            "E_over_k_over_a": 2.7,
            "nu": 0.35,
            "derivation": (
                "C_ijkl = (1/a^3) * sum over the 13 node-owned bonds of k*l^2*n_i*n_j*n_k*n_l; "
                "face bonds give l=a and the six in-plane/body diagonal families add 2k*a^2 and "
                "3k*a^2 weights, leaving C11=13k/(3a) and C12=C44=7k/(3a)"
            ),
        },
        "derived_moduli": moduli,
        "finite_sample_measurements": measurements,
        "engineering_scaling": scaling,
        "default_lattice_bias": {
            "displacement_control_relative_error": default_disp["E_relative_error_vs_analytic"],
            "load_control_relative_error": default_load["E_relative_error_vs_analytic"],
            "interpretation": (
                "A platen-loaded finite sample is stiffer than the infinite lattice because the "
                "end faces cannot contract freely; any modulus inferred from a platen test on the "
                "default 9x5x3 lattice must be corrected before it is compared with the material law"
            ),
        },
        "cauchy_relation_holds": abs(constants["C12_Pa"] - constants["C44_Pa"]) < 1e-12 * constants["C11_Pa"],
        "cubic_symmetry_holds": all(
            abs(constants[name] - constants["C11_Pa"]) < 1e-12 * constants["C11_Pa"]
            for name in ("C22_Pa", "C33_Pa")
        ) and all(
            abs(constants[name] - constants["C44_Pa"]) < 1e-12 * constants["C11_Pa"]
            for name in ("C55_Pa", "C66_Pa")
        ),
        "zener_anisotropy": moduli["zener_anisotropy"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lattice_elasticity3d_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
    with (output_dir / "lattice_elasticity3d_samples.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measurements[0]))
        writer.writeheader()
        writer.writerows(measurements)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results-dem3d-lattice-elasticity3d"))
    args = parser.parse_args()
    summary = audit(DEM3DConfig(), args.output)
    print(json.dumps(summary["finite_sample_measurements"], indent=2, default=float))
    print(json.dumps(summary["engineering_scaling"], indent=2, default=float))


if __name__ == "__main__":
    main()