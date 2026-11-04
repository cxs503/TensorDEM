"""Attribution of the Brazilian-disc centre field miss: anisotropy or discretisation?

`docs/dem3d_brazilian_hondros_closure.md` measured the relaxed bonded-sphere
specimen against the Hondros point-load solution and got ``sigma_zz/sigma_xx =
-2.015`` instead of ``-3`` (centre tension 30.5 % high).  It proved the miss is not
a transient, and left its *cause* open:

    "未做晶格分辨/接触弧加密扫描（recipe 已写进 §6），故"加密后能否进 15%"未知"

This driver answers that question by refinement *and* by an independent model:

1. **DEM refinement** -- the same static-equilibrium solve at ``nx = nz = 15, 21,
   27`` (``ny = 3``), i.e. lattice spacings ``D/14, D/20, D/26``.
2. **Anisotropic continuum** -- `scripts/brazilian_aniso_continuum.py` on the same
   diameter, thickness and measured contact-arc ratio, with the Born (affine)
   cubic constants of the bond network.  This model has no lattice spacing, so
   whatever miss survives in it cannot be a discretisation artefact.
3. **Isotropic continuum** -- the same finite-element code in the isotropic limit,
   which *must* reproduce Hondros (``-3`` and ``2P/(pi D t)``).  That is the check
   that licenses using the anisotropic branch as a reference at all.

The recorded quantities are the ones the manifest already judges
(``centre_principal_ratio``, ``centre_tension_rel_error``,
``axial_uniformity_rel``), so nothing is re-defined to make a verdict easier.

Outputs (all under ``artifacts/dem3d_brazilian_aniso/``):

* ``attribution.json`` -- every case, every metric, plus the born constants and
  the manifest acceptance tolerances;
* ``fig1_ratio_convergence.png`` -- centre principal ratio versus refinement for
  the lattice and for both continuum branches;
* ``fig2_axial_profile.png`` -- ``sigma_xx`` along the loaded diameter.

Scope: numerical verification of the apparatus.  Nothing here calibrates an ice
material or certifies a tensile strength.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmark_dem3d_brazilian import MANIFEST, _build, contact_arc, specimen_geometry  # noqa: E402
from brazilian_aniso_continuum import run_case as fem_case  # noqa: E402
from brazilian_lattice_static import lattice_moduli, solve_static_equilibrium  # noqa: E402

OUTPUT = ROOT / "artifacts" / "dem3d_brazilian_aniso"
REFERENCE_RATIO = -3.0
REFERENCE_TENSION = 2.0 / np.pi          # sigma_xx(0,0) / (P / (D t)) for a point load


def _manifest_case(nx: int) -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    case = deepcopy(next(c for c in manifest["cases"] if c["id"] == "brazil_elastic_field"))
    case["config"]["nx"] = nx
    case["config"]["nz"] = nx
    return case


def dem_case(nx: int, travel: float) -> dict[str, Any]:
    """Relaxed bonded-sphere specimen at one lattice resolution."""
    case = _manifest_case(nx)
    started = time.time()
    result = solve_static_equilibrium(case, travel)
    elapsed = time.time() - started
    _, geometry = _build(case)
    normalisation = result["mean_platen_load_N"] / (
        geometry["diameter_m"] * geometry["thickness_m"])
    return {
        "model": "dem-lattice-static",
        "nx": int(nx),
        "lattice_spacing_m": 2.0 * float(case["config"]["radius_m"]),
        "lattice_spacing_over_D": 2.0 * float(case["config"]["radius_m"]) / geometry["diameter_m"],
        "particle_count": int(geometry["particle_count"]),
        "bond_count": int(geometry["bond_count"]),
        "diameter_m": geometry["diameter_m"],
        "thickness_m": geometry["thickness_m"],
        "contact_arc_ratio": result["contact_arc_ratio"],
        "top_contact_particles": result["top_contact_particles"],
        "centre_particle_count": result["centre_particle_count"],
        "centre_sigma_xx_Pa": result["centre_sigma_xx_Pa"],
        "centre_sigma_zz_Pa": result["centre_sigma_zz_Pa"],
        "centre_principal_ratio": result["centre_principal_ratio"],
        "centre_sigma_xx_over_P_on_Dt": result["centre_sigma_xx_Pa"] / normalisation,
        "centre_tension_rel_error": abs(result["centre_sigma_xx_Pa"] / normalisation
                                        / REFERENCE_TENSION - 1.0),
        "centre_principal_ratio_rel_error": abs(result["centre_principal_ratio"] / REFERENCE_RATIO
                                                - 1.0),
        "axial_uniformity_rel": result["axial_uniformity_rel"],
        "axial_profile": result["axial_profile"],
        "equilibrium_residual_rel": result["equilibrium_residual_rel"],
        "wall_clock_s": elapsed,
    }


def continuum_case(*, constitutive: str, constants: tuple[float, float, float],
                   patch_ratio: float, rings: int, sectors: int | None = None,
                   thickness: float = 0.06, radius: float = 0.15,
                   label: str = "") -> dict[str, Any]:
    result = fem_case(radius=radius, thickness=thickness, rings=rings,
                      sectors=sectors if sectors is not None else rings,
                      patch_ratio=patch_ratio, total_load=1.0,
                      constitutive=constitutive, constants=constants)
    result["model"] = label or f"fem-{constitutive}"
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[15, 21, 27])
    parser.add_argument("--rings", type=int, nargs="+", default=[20, 40, 80])
    parser.add_argument("--travel", type=float, default=7.0913e-4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-dem", action="store_true")
    args = parser.parse_args(argv)

    reference_arc = 0.0668693921916482
    dem_results = [] if args.skip_dem else [dem_case(nx, args.travel) for nx in args.resolutions]

    born = lattice_moduli(_manifest_case(args.resolutions[0]))
    cubic = (born["C11_Pa"], born["C12_Pa"], born["C44_Pa"])
    isotropic = (lambda_plus_2mu(born), lam(born), mu(born))

    isotropic_sweep = [
        continuum_case(constitutive="isotropic-plane-stress", constants=isotropic,
                       patch_ratio=reference_arc, rings=rings,
                       label="fem-isotropic-plane-stress")
        for rings in args.rings
    ]
    arc_sweep = [
        continuum_case(constitutive="isotropic-plane-stress", constants=isotropic,
                       patch_ratio=ratio, rings=max(args.rings),
                       label="fem-isotropic-arc-sweep")
        for ratio in (reference_arc, 0.01, 0.002)
    ]
    cubic_sweep = [
        continuum_case(constitutive="cubic-plane-stress", constants=cubic,
                       patch_ratio=reference_arc, rings=rings,
                       label="fem-cubic-plane-stress")
        for rings in args.rings
    ]
    plane_strain = continuum_case(constitutive="cubic-plane-strain", constants=cubic,
                                  patch_ratio=reference_arc, rings=max(args.rings),
                                  label="fem-cubic-plane-strain")
    matched = [
        continuum_case(constitutive="cubic-plane-stress", constants=cubic,
                       patch_ratio=float(case["contact_arc_ratio"]),
                       rings=max(args.rings), radius=0.5 * float(case["diameter_m"]),
                       thickness=float(case["thickness_m"]),
                       label=f"fem-cubic-matched-nx{case['nx']}")
        for case in dem_results
    ]

    payload = {
        "protocol": "tensordem-brazilian-anisotropy-attribution-v1",
        "question": ("is the Brazilian centre-field miss against Hondros caused by lattice "
                     "discretisation/contact-arc width, or by the cubic anisotropy of a "
                     "central-force bond network?"),
        "reference": {
            "hondros_centre_principal_ratio": REFERENCE_RATIO,
            "hondros_centre_sigma_xx_over_P_on_Dt": REFERENCE_TENSION,
            "manifest_tolerances": {"centre_principal_ratio_tol": 0.15,
                                     "centre_tension_rel_tol": 0.15},
        },
        "born_constants_Pa": {key: float(born[key]) for key in
                              ("C11_Pa", "C12_Pa", "C44_Pa", "E_Pa", "nu",
                               "zener_anisotropy", "cauchy_residual_Pa")},
        "dem_refinement": dem_results,
        "fem_isotropic_mesh_sweep": isotropic_sweep,
        "fem_isotropic_arc_sweep": arc_sweep,
        "fem_cubic_mesh_sweep": cubic_sweep,
        "fem_cubic_plane_strain": plane_strain,
        "fem_cubic_matched_to_dem": matched,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "attribution.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(_summary(payload), indent=2))
    _figures(payload, args.output)


def lam(born: dict[str, float]) -> float:
    """Lame lambda of the isotropic-equivalent solid from C11 - 2 mu."""
    return float(born["C11_Pa"] - 2.0 * mu(born))


def mu(born: dict[str, float]) -> float:
    """Isotropic-equivalent shear modulus: Voigt (arithmetic) average of cubic C44."""
    return float(born["C44_Pa"])


def lambda_plus_2mu(born: dict[str, float]) -> float:
    return float(born["C11_Pa"])


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "born": payload["born_constants_Pa"],
        "dem": [{key: case[key] for key in
                 ("nx", "lattice_spacing_over_D", "contact_arc_ratio", "particle_count",
                  "centre_principal_ratio", "centre_sigma_xx_over_P_on_Dt",
                  "centre_tension_rel_error", "axial_uniformity_rel", "wall_clock_s")
                 } for case in payload["dem_refinement"]],
        "fem_isotropic": [{key: case[key] for key in
                           ("rings", "patch_ratio", "centre_principal_ratio",
                            "centre_sigma_xx_over_P_on_Dt", "centre_principal_ratio_rel_error")}
                          for case in payload["fem_isotropic_mesh_sweep"]],
        "fem_isotropic_arc": [{key: case[key] for key in
                               ("patch_ratio", "centre_principal_ratio",
                                "centre_sigma_xx_over_P_on_Dt")}
                              for case in payload["fem_isotropic_arc_sweep"]],
        "fem_cubic": [{key: case[key] for key in
                       ("rings", "patch_ratio", "centre_principal_ratio",
                        "centre_sigma_xx_over_P_on_Dt", "centre_principal_ratio_rel_error")}
                      for case in payload["fem_cubic_mesh_sweep"]],
        "fem_cubic_plane_strain": {key: payload["fem_cubic_plane_strain"][key] for key in
                                   ("centre_principal_ratio", "centre_sigma_xx_over_P_on_Dt")},
        "matched": [{key: case[key] for key in
                     ("model", "patch_ratio", "centre_principal_ratio",
                      "centre_sigma_xx_over_P_on_Dt", "centre_tension_rel_error")}
                    for case in payload["fem_cubic_matched_to_dem"]],
    }


def _figures(payload: dict[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    dem = payload["dem_refinement"]
    iso = payload["fem_isotropic_mesh_sweep"]
    cub = payload["fem_cubic_mesh_sweep"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.axhline(REFERENCE_RATIO, color="k", ls="--", lw=1.0, label="Hondros point load (-3)")
    if dem:
        ax.plot([c["lattice_spacing_over_D"] for c in dem],
                [c["centre_principal_ratio"] for c in dem], "o-", color="#b2182b",
                label="DEM lattice (static equilibrium)")
    ax.plot([1.0 / c["rings"] for c in iso], [c["centre_principal_ratio"] for c in iso],
            "s-", color="#2166ac", label="isotropic continuum (FEM)")
    ax.plot([1.0 / c["rings"] for c in cub], [c["centre_principal_ratio"] for c in cub],
            "^-", color="#1b7837", label="cubic Born-tensor continuum (FEM)")
    ax.set_xscale("log")
    ax.set_xlabel("normalised resolution  (lattice spacing / D  or  1 / rings)")
    ax.set_ylabel(r"centre principal ratio  $\sigma_{zz}/\sigma_{xx}$")
    ax.set_title("centre field vs refinement")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(REFERENCE_TENSION, color="k", ls="--", lw=1.0,
               label=r"Hondros $2P/(\pi D t)$ = 0.637")
    for case, colour in zip(dem, ("#b2182b", "#ef8a62", "#67001f")):
        profile = case["axial_profile"]
        ax.plot([row["z_over_D"] for row in profile],
                [row["sigma_xx_Pa"] / (case["centre_sigma_xx_Pa"]
                                       / case["centre_sigma_xx_over_P_on_Dt"]) for row in profile],
                "o-", color=colour, label=f"DEM nx={case['nx']}")
    for case, style in ((cub[0], ":"), (cub[-1], "-")):
        profile = case["axial_profile"]
        scale = case["centre_sigma_xx_over_P_on_Dt"]
        ax.plot([row["z_over_D"] for row in profile],
                [row["sigma_xx_Pa"] / (case["centre_sigma_xx_Pa"] / scale) for row in profile],
                style, color="#1b7837", label=f"cubic FEM rings={case['rings']}")
    ax.set_xlabel("z / D  (along the loaded diameter)")
    ax.set_ylabel(r"$\sigma_{xx}$ / (P / D t)")
    ax.set_title("axial tension profile")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "fig1_ratio_convergence.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()